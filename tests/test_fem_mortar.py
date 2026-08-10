"""The interface frame that tied/periodic ties project through.

Matching a slave node against the master face means comparing the two **in the interface**, with the
across-interface coordinate removed. That used to be done by dropping the single global axis whose tag
means differed most, which assumes the two faces are planar, axis-aligned and separated by a pure
translation along that axis. :func:`_interface_frame` replaces it with an SVD fit of the master face's
own tangent plane.

Two things are checked here:

* **No regression.** On a planar axis-aligned pair the frame spans the same plane the transverse axes
  did, so the conforming and non-matching periodic paths are unchanged (``tests/test_fem_periodic*``
  cover those end-to-end; the equivalence is asserted directly below).
* **The new case.** A **coincident** (tied) interface — both faces at the same location, as two
  independently meshed bodies glued along a shared surface. The mean difference is then ~0, so the old
  axis-drop selected whichever coordinate carried the largest rounding error; when that landed on an
  in-plane axis the projection collapsed the face onto a line and produced silently wrong weights.

**Not** covered, because it is not supported by either the old or the new code: a *sheared* lattice,
where the offset between the two faces has a component **along** the interface. Projecting onto the
tangent plane removes the across-interface offset only, so an in-plane shift would tie each slave to
the wrong master location. A periodic cell whose lattice vector is normal to its faces is fine.
"""

import jax
import numpy as np
import pytest

from jno.utils.solver.fem_utils import (
    _as_ccw,
    _clip_convex,
    _dual_coeffs,
    _edge_shape,
    _faces_span_the_same_extent,
    _facet_dual_coeffs,
    _interface_frame,
    _master_covers_slave_3d,
    _mortar_rows_2d,
    _mortar_rows_3d,
    _periodic_facet_weights,
    _signed_area,
    _tri_bary,
    _tri_dual_available,
    _tri_quadrature,
    _tri_shape,
    build_periodic_prolongation,
    interface_gap_data,
    master_trace_weights,
)


@pytest.fixture(autouse=True)
def _x64():
    """x64 so the reduction is exact: the prolongation weights are not binary fractions once the
    interface is tilted, and float32 leaves ~1e-7 of noise in an otherwise exact transfer."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _plane_frame(normal):
    """An orthonormal ``(t1, t2, n)`` with ``n`` along ``normal`` — to *build* test planes with."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, seed)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1), n


def _square_patch(origin, t1, t2):
    """Unit square as 4 corners + 2 triangles, embedded in the plane ``origin + s*t1 + q*t2``."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = origin + uv[:, :1] * t1 + uv[:, 1:] * t2
    return pts, np.array([[0, 1, 2], [0, 2, 3]])


# --------------------------------------------------------------------------- the frame itself


def test_frame_spans_the_face_plane_when_axis_aligned():
    """A z = const face: the two frame rows are orthonormal and orthogonal to the face normal, so
    they span exactly the plane the old transverse axes (x, y) did."""
    m = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    frame, origin = _interface_frame(m, m)
    assert frame.shape == (2, 3)
    assert np.allclose(frame @ frame.T, np.eye(2), atol=1e-12)  # orthonormal rows
    assert np.allclose(frame @ np.array([0.0, 0.0, 1.0]), 0.0, atol=1e-12)  # orthogonal to n
    assert np.allclose(origin, [0.5, 0.5, 1.0])


def test_frame_projection_preserves_in_plane_distances():
    """The frame is only defined up to a rotation *within* the plane, and every consumer (nearest-
    neighbour matching, edge parameters, barycentric weights) is rotation-invariant. Pin that down:
    projected pairwise distances equal the true in-plane distances."""
    t1, t2, n = _plane_frame([1.0, 2.0, -0.5])
    pts, _ = _square_patch(np.array([0.3, -0.2, 0.7]), t1, t2)
    frame, origin = _interface_frame(pts, pts)
    loc = (pts - origin) @ frame.T
    d_full = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d_proj = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=-1)
    assert np.allclose(d_full, d_proj, atol=1e-12)  # the face is planar -> projection is an isometry


def test_frame_fits_a_face_whose_normal_is_not_a_global_axis():
    t1, t2, n = _plane_frame([1.0, 1.0, 1.0])
    pts, _ = _square_patch(np.zeros(3), t1, t2)
    frame, _origin = _interface_frame(pts, pts)
    assert np.allclose(frame @ n, 0.0, atol=1e-12)


def test_frame_handles_coincident_faces():
    """The tied case: master and slave occupy the SAME plane, so the mean difference is ~0 and the
    old axis-drop had nothing to key on. The frame comes from the master face's own geometry, so it
    is unaffected."""
    m = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    s = np.array([[0.3, 0.2, 0.0], [0.6, 0.7, 0.0]])
    delta = m.mean(axis=0) - s.mean(axis=0)
    assert delta[2] == 0.0  # coincident: nothing separates the faces along the normal ...
    assert int(np.argmax(np.abs(delta))) in (0, 1)  # ... so the old axis-drop lands IN the plane
    frame, _ = _interface_frame(m, s)
    assert np.allclose(frame @ np.array([0.0, 0.0, 1.0]), 0.0, atol=1e-12)


def test_frame_is_empty_in_1d():
    """A 1-D interface is a single point: no in-interface coordinate exists, and every slave ties to
    the master exactly (distance 0), which is what dropping the only axis did."""
    frame, origin = _interface_frame(np.array([[1.0]]), np.array([[0.0]]))
    assert frame.shape == (0, 1)
    loc = (np.array([[0.0], [1.0]]) - origin) @ frame.T
    assert loc.shape == (2, 0)
    d2 = np.sum((loc[:, None, :] - loc[None, :, :]) ** 2, axis=-1)
    assert np.allclose(d2, 0.0)  # every pair is at distance 0 -> exact tie


def test_frame_refuses_a_collinear_3d_face():
    """A 3-D tie whose 'face' is an edge has no tangent plane. Refuse loudly rather than fit noise."""
    line = np.column_stack([np.linspace(0, 1, 5), np.zeros(5), np.zeros(5)])
    with pytest.raises(ValueError, match="collinear"):
        _interface_frame(line, line)


def test_frame_refuses_a_point_face():
    pt = np.zeros((3, 2))
    with pytest.raises(ValueError, match="single point"):
        _interface_frame(pt, pt)


# ------------------------------------------------------- end-to-end through the prolongation


def _coincident_faces(slave_uv, *, z=0.0):
    """A master unit square (4 corners, 2 triangles) and a slave node set in the SAME plane."""
    master = np.array([[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    slave = np.column_stack([slave_uv, np.full(len(slave_uv), z)])
    pts = np.vstack([master, slave])
    tags = {"a": np.arange(4), "b": np.arange(4, 4 + len(slave))}
    return pts, tags, {"a": tris}


def test_tied_coincident_interface_reproduces_constant_and_linear():
    """The Phase-B enabler: two faces at the SAME location with different node layouts. The
    prolongation must be a partition of unity and reproduce a linear field exactly."""
    uv = np.array([[0.3, 0.2], [0.7, 0.6], [0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
    pts, tags, facets = _coincident_faces(uv)
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets=facets)
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])

    assert res["n_red"] == 4  # the 5 slave nodes are eliminated onto the 4 master corners
    assert np.allclose(P.sum(axis=1), 1.0)
    for a, b, c in [(0.0, 0.0, 1.0), (2.0, -1.5, 0.3)]:
        field = a * pts[:, 0] + b * pts[:, 1] + c
        assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_tied_interface_on_a_tilted_plane_reproduces_linear():
    """Same, on a plane whose normal is not a global axis — the frame has to be fitted, not guessed."""
    t1, t2, n = _plane_frame([1.0, 1.0, 1.0])
    master, tris = _square_patch(np.zeros(3), t1, t2)
    uv = np.array([[0.3, 0.2], [0.7, 0.6], [0.25, 0.65]])
    slave = uv[:, :1] * t1 + uv[:, 1:] * t2
    pts = np.vstack([master, slave])
    tags = {"a": np.arange(4), "b": np.arange(4, 4 + len(slave))}
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets={"a": tris})
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    for coef in [(0.0, 0.0, 0.0, 1.0), (2.0, -1.5, 0.4, 0.3)]:
        field = pts @ np.array(coef[:3]) + coef[3]
        assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_tied_interface_survives_a_high_density_ratio():
    """Extreme: one master facet pair against a 1:8-refined slave face (81 nodes)."""
    g = np.linspace(0.02, 0.98, 9)
    uv = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    pts, tags, facets = _coincident_faces(uv)
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets=facets)
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert res["n_red"] == 4 and len(kept) == 4
    assert np.allclose(P.sum(axis=1), 1.0)
    field = 1.3 * pts[:, 0] - 0.7 * pts[:, 1] + 2.0
    assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_tied_interface_with_a_slave_node_on_a_master_node():
    """Degenerate overlap: a slave node coincident with a master node ties exactly (distance 0),
    the rest interpolate. Both branches must coexist in one tie."""
    uv = np.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.3]])  # first two land ON master corners
    pts, tags, facets = _coincident_faces(uv)
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets=facets)
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    field = 0.9 * pts[:, 0] + 1.7 * pts[:, 1] - 0.4
    assert np.allclose(P @ field[kept], field, atol=1e-10)


# ------------------------------------------------------------------- 2-D dual mortar
#
# What the mortar coupling does and does not buy, measured rather than assumed:
#
# * A **linear** field transfers exactly under BOTH mortar and node-to-segment collocation, so the
#   2-D linear patch test does not tell them apart. (It still gates the new code's correctness.)
# * When the master nodes are a subset of the slave nodes -- e.g. 5 master against 9 slave nodes on
#   the same interval -- the master basis lies *inside* the slave space, the dual basis reproduces it
#   pointwise, and the two couplings are identical to machine precision. Any test built on nested
#   meshes is therefore vacuous.
# * The difference appears on **non-nested** meshes for a field the master space cannot represent:
#   mortar returns the L2 projection, collocation the pointwise value. Measured below.
#
# The coupling's real payoff is 3-D, where point-in-triangle collocation is not a projection at all.


def _edge_faces(n_master, n_slave, *, order=1):
    """Master (x=1) and slave (x=0) edge faces, both spanning y in [0, 1], facets for BOTH sides."""

    def side(x, n):
        y = np.linspace(0.0, 1.0, n)
        pts = np.column_stack([np.full(n, x), y])
        if order == 1:
            return pts, None
        mids = np.column_stack([np.full(n - 1, x), 0.5 * (y[:-1] + y[1:])])
        return pts, mids

    m_pts, m_mid = side(1.0, n_master)
    s_pts, s_mid = side(0.0, n_slave)
    blocks = [m_pts, s_pts] if order == 1 else [m_pts, m_mid, s_pts, s_mid]
    pts = np.vstack(blocks)
    off = np.cumsum([0] + [len(b) for b in blocks])
    m_ids, s_ids = np.arange(off[0], off[1]), (np.arange(off[1], off[2]) if order == 1 else np.arange(off[2], off[3]))
    if order == 1:
        mf = np.column_stack([m_ids[:-1], m_ids[1:]])
        sf = np.column_stack([s_ids[:-1], s_ids[1:]])
        tags = {"r": m_ids, "l": s_ids}
    else:
        m_mid_ids, s_mid_ids = np.arange(off[1], off[2]), np.arange(off[3], off[4])
        mf = np.column_stack([m_ids[:-1], m_ids[1:], m_mid_ids])
        sf = np.column_stack([s_ids[:-1], s_ids[1:], s_mid_ids])
        tags = {"r": np.concatenate([m_ids, m_mid_ids]), "l": np.concatenate([s_ids, s_mid_ids])}
    return pts, tags, {"r": mf, "l": sf}


def _apply(rows, master_vals_by_id):
    """Evaluate a row dict {slave: [(master, w)]} against master nodal values keyed by node id."""
    return {s: sum(w * master_vals_by_id[m] for m, w in ws) for s, ws in rows.items()}


def test_dual_basis_is_biorthogonal():
    """The invariant D^-1 relies on: int psi_i N_j = delta_ij int N_i, element-locally."""
    qp, qw = np.polynomial.legendre.leggauss(6)
    qp, qw = 0.5 * (qp + 1.0), 0.5 * qw
    for k in (2, 3):
        n = _edge_shape(qp, k)
        a = _facet_dual_coeffs(k, qp, qw)
        psi = n @ a.T
        pairing = psi.T @ (qw[:, None] * n)  # int psi_i N_j
        assert np.allclose(pairing, np.diag(qw @ n), atol=1e-12)
    # P1 recovers the textbook dual functions psi_0 = 2 - 3xi, psi_1 = 3xi - 1 (Wohlmuth 2000, §3)
    a1 = _facet_dual_coeffs(2, qp, qw)
    assert np.allclose(_edge_shape(np.array([0.0, 1.0]), 2) @ a1.T, [[2.0, -1.0], [-1.0, 2.0]], atol=1e-12)


def test_mortar_is_selected_and_reported():
    """Both faces faceted and co-extensive -> mortar. Master facets only -> collocation. The
    ``coupling`` key says which, so a caller never has to infer it."""
    pts, tags, facets = _edge_faces(5, 9)
    assert build_periodic_prolongation(pts, [("r", "l")], tags, facets=facets)["coupling"] == "mortar"
    only_master = {"r": facets["r"]}
    assert build_periodic_prolongation(pts, [("r", "l")], tags, facets=only_master)["coupling"] == "collocated"


@pytest.mark.parametrize("order,field", [(1, lambda y: 2.0 * y - 0.5), (2, lambda y: 0.7 * y**2 - 0.2 * y + 0.1)])
def test_mortar_transfers_the_master_space_exactly(order, field):
    """Correctness gate: a field the master facets represent exactly transfers exactly (linear for
    P1 edges, quadratic for P2). Requires the assembled D to be genuinely diagonal."""
    pts, tags, facets = _edge_faces(5, 8, order=order)
    res = build_periodic_prolongation(pts, [("r", "l")], tags, facets=facets)
    assert res["coupling"] == "mortar"
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    f = field(pts[:, 1])
    assert np.allclose(P @ f[kept], f, atol=1e-10)


def test_mortar_is_the_l2_projection_not_collocation():
    """The honest discriminator. On NON-nested meshes and a field outside the master space the two
    couplings differ, and mortar is the L2 projection -- closer to the true field than the pointwise
    value. On nested meshes (master nodes a subset of slave nodes) they provably coincide."""
    f = lambda y: np.sin(3.0 * y)  # noqa: E731

    def compare(n_master, n_slave):
        pts, _tags, fc = _edge_faces(n_master, n_slave)
        loc = pts[:, 1:2]
        rows = _mortar_rows_2d(fc["l"], fc["r"], loc, span=1.0)
        vals = {int(i): f(pts[i, 1]) for i in np.unique(fc["r"])}
        mortar = _apply(rows, vals)
        colloc = _apply({int(s): _periodic_facet_weights(loc[int(s)], fc["r"], loc) for s in np.unique(fc["l"])}, vals)
        exact = {s: f(pts[s, 1]) for s in mortar}
        gap = max(abs(mortar[s] - colloc[s]) for s in mortar)
        return gap, max(abs(mortar[s] - exact[s]) for s in mortar), max(abs(colloc[s] - exact[s]) for s in mortar)

    gap, e_mortar, e_colloc = compare(5, 8)  # non-nested
    assert gap > 1e-3, "non-nested meshes must separate the two couplings"
    assert e_mortar < e_colloc, "the L2 projection should beat pointwise collocation"

    gap_nested, _, _ = compare(5, 9)  # 9 = 2*5-1 -> every master node is a slave node
    assert gap_nested < 1e-12, "nested meshes make the master basis a subset of the slave space"


def test_mortar_rows_match_an_independent_fine_quadrature():
    """Validate the segmentation: rebuild D and M by midpoint quadrature on a fine uniform grid --
    no clipping, no segment geometry -- and compare. Different decomposition, same integral."""
    pts, _tags, fc = _edge_faces(4, 7)
    t = pts[:, 1]
    rows = _mortar_rows_2d(fc["l"], fc["r"], pts[:, 1:2], span=1.0)

    ks, km = fc["l"].shape[1], fc["r"].shape[1]
    qp, qw = np.polynomial.legendre.leggauss(max(ks, km) + 2)
    dual = _facet_dual_coeffs(ks, 0.5 * (qp + 1.0), 0.5 * qw)
    n = 200_000
    h = 1.0 / n
    tq = (np.arange(n) + 0.5) * h
    s_nodes, m_nodes = np.unique(fc["l"]), np.unique(fc["r"])
    s_at = {int(v): i for i, v in enumerate(s_nodes)}
    m_at = {int(v): i for i, v in enumerate(m_nodes)}
    D, M = np.zeros(len(s_nodes)), np.zeros((len(s_nodes), len(m_nodes)))
    for e in range(len(fc["l"])):
        a, b = t[fc["l"][e, 0]], t[fc["l"][e, 1]]
        inside = (tq >= min(a, b)) & (tq <= max(a, b))
        x = tq[inside]
        n_s = _edge_shape((x - a) / (b - a), ks)
        psi = n_s @ dual.T
        r = [s_at[int(v)] for v in fc["l"][e]]
        D[r] += np.einsum("qi,qi->i", psi, n_s) * h
        for g in range(len(fc["r"])):
            c, d = t[fc["r"][g, 0]], t[fc["r"][g, 1]]
            sel = (x >= min(c, d)) & (x <= max(c, d))
            if not sel.any():
                continue
            n_m = _edge_shape((x[sel] - c) / (d - c), km)
            M[np.ix_(r, [m_at[int(v)] for v in fc["r"][g]])] += np.einsum("qi,qj->ij", psi[sel], n_m) * h

    for node, i in s_at.items():
        ref = M[i] / D[i]
        got = np.zeros(len(m_nodes))
        for m, w in rows[node]:
            got[m_at[m]] = w
        assert np.allclose(got, ref, atol=2e-4), f"slave {node}: {got} vs {ref}"


def test_extent_mismatch_keeps_collocation():
    """A face tagged without its corners is shorter than its partner; the master then does not cover
    the slave and an integral over the slave face is not well posed. That tie must fall back to
    collocation and REPORT it, not integrate over a domain the master does not span."""
    pts, tags, facets = _edge_faces(5, 9)
    trimmed = facets["r"][1:-1]  # drop the master's two end facets -> master no longer spans the slave
    keep = np.unique(trimmed)
    res = build_periodic_prolongation(
        pts, [("r", "l")], {"r": keep, "l": tags["l"]}, facets={"r": trimmed, "l": facets["l"]}
    )
    assert res["coupling"] == "collocated"
    assert not _faces_span_the_same_extent(facets["l"], trimmed, pts[:, 1:2], span=1.0)
    assert _faces_span_the_same_extent(facets["l"], facets["r"], pts[:, 1:2], span=1.0)


def test_hole_in_the_master_face_raises():
    """Same extent but a missing interior facet: the slave face IS covered at its ends, so the
    extent gate passes and the per-facet coverage check has to catch the hole."""
    pts, _tags, facets = _edge_faces(6, 11)
    holed = np.delete(facets["r"], 2, axis=0)  # remove an interior master facet
    with pytest.raises(ValueError, match="HOLE in the master face"):
        _mortar_rows_2d(facets["l"], holed, pts[:, 1:2], span=1.0)


def test_mortar_survives_a_high_density_ratio_either_way():
    """Extremes: 1:8 refinement, the reverse (master finer than slave), and a single master facet.

    Node counts are chosen so the slave nodes do NOT all land on master nodes -- e.g. 17 master
    against 3 slave nodes is fully conforming (0, 0.5, 1 are all master nodes) and would exercise
    nothing."""
    for n_master, n_slave in [(3, 17), (16, 7), (2, 25)]:
        pts, tags, facets = _edge_faces(n_master, n_slave)
        res = build_periodic_prolongation(pts, [("r", "l")], tags, facets=facets)
        assert res["coupling"] == "mortar", f"master {n_master}, slave {n_slave}"
        P = np.asarray(res["P_node"].todense())
        kept = np.asarray(res["kept_nodes"])
        assert np.allclose(P.sum(axis=1), 1.0)
        f = 2.0 * pts[:, 1] - 0.5
        assert np.allclose(P @ f[kept], f, atol=1e-10)


# ------------------------------------------------------------------- 3-D dual mortar
#
# The 3-D interface is where the segmentation is real geometry: triangle-against-triangle polygon
# clipping rather than an interval intersection.
#
# It is NOT, however, where the patch test starts to discriminate. jNO enforces a tie by master-slave
# elimination through a prolongation P, and such a scheme reproduces a linear solution exactly
# whenever P does -- which node-to-segment barycentric interpolation does, in 3-D as in 2-D. The
# textbook "node-to-segment fails the patch test" result concerns contact formulations that distribute
# nodal forces, not a linearly-complete MPC elimination. The linear patch test below is therefore a
# correctness gate on the new code, not evidence that mortar beats collocation.
#
# What separates them is that mortar imposes the INTEGRAL constraint: asserted directly below by
# checking that the collocated weights leave a non-zero mortar residual while the mortar weights do
# not, and quantified as a 4-40% lower RMS error on fields the master space cannot represent.


def _tri_grid(n, z, base):
    """An ``n x n`` structured triangulation of the unit square at height ``z``, ids from ``base``."""
    g = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.full((n + 1) ** 2, float(z))])
    ids = base + np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    tris = [
        t
        for i in range(n)
        for j in range(n)
        for t in ([ids[i, j], ids[i + 1, j], ids[i + 1, j + 1]], [ids[i, j], ids[i + 1, j + 1], ids[i, j + 1]])
    ]
    return pts, np.array(tris)


def _tri_faces(n_master, n_slave):
    """Master (z=1) and slave (z=0) triangulated faces, both covering the unit square."""
    mp, mt = _tri_grid(n_master, 1.0, 0)
    sp, st = _tri_grid(n_slave, 0.0, len(mp))
    pts = np.vstack([mp, sp])
    tags = {"top": np.unique(mt), "bot": np.unique(st)}
    return pts, tags, {"top": mt, "bot": st}


def test_duffy_quadrature_is_exact_on_monomials():
    """The reference-triangle rule, verified against closed-form integrals rather than a table."""
    bary, w = _tri_quadrature(4)
    x, y = bary[..., 1], bary[..., 2]
    for (i, j), exact in {
        (0, 0): 0.5,
        (1, 0): 1 / 6,
        (0, 1): 1 / 6,
        (2, 0): 1 / 12,
        (1, 1): 1 / 24,
        (3, 0): 1 / 20,
    }.items():
        assert abs(float(w @ (x**i * y**j)) - exact) < 1e-14, (i, j)


def test_convex_clipping_handles_overlap_and_degeneracy():
    """Sutherland-Hodgman on the cases that break clippers: identical, partial, disjoint, and the
    degenerate contacts (a shared corner, a shared edge) that carry zero area."""
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert abs(abs(_signed_area(_clip_convex(_as_ccw(tri), _as_ccw(tri)))) - 0.5) < 1e-14
    other = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # the square's other half
    assert abs(abs(_signed_area(_clip_convex(_as_ccw(tri), _as_ccw(other)))) - 0.25) < 1e-14
    assert _clip_convex(_as_ccw(tri), _as_ccw(tri + 5.0)).shape[0] == 0  # disjoint
    assert _clip_convex(_as_ccw(tri), _as_ccw(tri + np.array([1.0, 1.0]))).shape[0] == 0  # corner touch
    quad = np.array([[-0.5, 0.25], [0.75, -0.4], [1.2, 0.5], [0.1, 0.9]])
    assert _clip_convex(_as_ccw(tri), _as_ccw(quad)).shape[0] == 5  # a genuine 5-gon overlap


def test_clipping_is_orientation_independent():
    """Facet node order is whatever the mesh gives; the overlap area must not depend on it."""
    tri = np.array([[0.1, 0.0], [1.0, 0.2], [0.3, 0.9]])
    other = np.array([[0.0, 0.1], [0.9, 0.0], [0.5, 1.0]])
    ref = abs(_signed_area(_clip_convex(_as_ccw(tri), _as_ccw(other))))
    for a in (tri, tri[::-1]):
        for b in (other, other[::-1]):
            assert abs(abs(_signed_area(_clip_convex(_as_ccw(a), _as_ccw(b)))) - ref) < 1e-14


def test_triangle_dual_basis_is_biorthogonal():
    bary, w = _tri_quadrature(5)
    for k in (3, 6):
        n = _tri_shape(bary, k)
        pairing = (n @ _dual_coeffs(n, w).T).T @ (w[:, None] * n)
        assert np.allclose(pairing, np.diag(w @ n), atol=1e-13), k


def test_mortar_3d_is_selected_and_reported():
    pts, tags, facets = _tri_faces(3, 5)
    assert build_periodic_prolongation(pts, [("top", "bot")], tags, facets=facets)["coupling"] == "mortar"
    assert (
        build_periodic_prolongation(pts, [("top", "bot")], tags, facets={"top": facets["top"]})["coupling"] == "collocated"
    )


def test_mortar_3d_transfers_a_linear_field_exactly():
    """Correctness gate on the clipping and quadrature: a linear field must survive the segmentation
    to machine precision. (Collocation passes this too -- see the note above.)"""
    pts, tags, facets = _tri_faces(3, 7)
    res = build_periodic_prolongation(pts, [("top", "bot")], tags, facets=facets)
    assert res["coupling"] == "mortar"
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    for coef in [(0.0, 0.0, 1.0), (2.0, -1.5, 0.3)]:
        field = coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2]
        assert np.allclose(P @ field[kept], field, atol=1e-10)


def _mortar_residual(rows, s_facets, m_facets, loc, field, nq=20):
    """``int psi_i (u_s - u_m.Phi) dG`` by independent per-slave-facet quadrature: every point is
    located in the master face by barycentric search, so no clipping is involved.

    The integrand is only piecewise smooth (the master field kinks at every master facet edge), so
    this converges in ``nq`` rather than being exact -- which is why the test below asserts a ratio
    and a convergence trend instead of an absolute threshold."""
    xy = np.asarray(loc, float)
    ks, km = s_facets.shape[1], m_facets.shape[1]
    bary, w = _tri_quadrature(nq)
    dual = _dual_coeffs(_tri_shape(bary, ks), w)
    resid = {int(n): 0.0 for n in np.unique(s_facets)}
    for e in range(len(s_facets)):
        sv = xy[s_facets[e, :3]]
        area = abs(_signed_area(sv))
        xq = bary @ sv
        n_s = _tri_shape(_tri_bary(xq, sv), ks)
        psi = n_s @ dual.T
        u_s = n_s @ np.array([rows[int(n)] for n in s_facets[e]])
        u_m = np.zeros(len(xq))
        for f in range(len(m_facets)):
            mv = xy[m_facets[f, :3]]
            b = _tri_bary(xq, mv)
            sel = b.min(axis=1) >= -1e-12
            if sel.any():
                u_m[sel] = _tri_shape(b[sel], km) @ np.array([field[int(n)] for n in m_facets[f]])
        contrib = psi.T @ ((u_s - u_m) * w * (area / 0.5))
        for a, n in enumerate(s_facets[e]):
            resid[int(n)] += float(contrib[a])
    return np.array([resid[int(n)] for n in np.unique(s_facets)])


def test_mortar_satisfies_the_integral_constraint_and_collocation_does_not():
    """The defining difference, asserted rather than described. Mortar solves
    ``int psi (u_s - u_m.Phi) = 0``; collocation solves ``u_s(x_i) = u_m(x_i)``, which leaves that
    integral non-zero. Both are consistent; only one is the variational constraint."""
    pts, _tags, fc = _tri_faces(3, 5)
    loc = pts[:, :2]
    field = {int(i): np.sin(3.0 * pts[i, 0]) * np.cos(2.5 * pts[i, 1]) for i in np.unique(fc["top"])}

    rows = _mortar_rows_3d(fc["bot"], fc["top"], loc, span=1.0)
    u_mortar = {s: sum(w * field[m] for m, w in ws) for s, ws in rows.items()}
    u_colloc = {
        int(s): sum(w * field[m] for m, w in _periodic_facet_weights(loc[int(s)], fc["top"], loc))
        for s in np.unique(fc["bot"])
    }
    r_mortar, r_colloc = ({}, {})
    for nq in (8, 20):
        r_mortar[nq] = np.abs(_mortar_residual(u_mortar, fc["bot"], fc["top"], loc, field, nq)).max()
        r_colloc[nq] = np.abs(_mortar_residual(u_colloc, fc["bot"], fc["top"], loc, field, nq)).max()

    assert r_colloc[20] > 100 * r_mortar[20], "the two must enforce measurably different constraints"
    # Mortar's residual is the REFERENCE integrator's quadrature error, so it shrinks as that rule is
    # refined; collocation's is a genuine violation, so it does not. This is what makes the first
    # assertion meaningful rather than a threshold that happens to hold.
    assert r_mortar[20] < 0.5 * r_mortar[8], "mortar residual must converge away under refinement"
    assert r_colloc[20] > 0.9 * r_colloc[8], "collocation residual must persist under refinement"


@pytest.mark.parametrize("n_master,n_slave", [(3, 7), (4, 7), (5, 11), (7, 11)])
def test_mortar_3d_is_more_accurate_than_collocation(n_master, n_slave):
    """Quantify the gain honestly: RMS error over the slave nodes, on a field neither space
    represents. Mortar is the L2 projection, so it should win -- measured 4-40% here."""
    pts, _tags, fc = _tri_faces(n_master, n_slave)
    loc = pts[:, :2]
    f = lambda p: np.sin(3.0 * p[0]) * np.cos(2.5 * p[1])  # noqa: E731
    field = {int(i): f(pts[i]) for i in np.unique(fc["top"])}
    rows = _mortar_rows_3d(fc["bot"], fc["top"], loc, span=1.0)
    ids = sorted(rows)
    mortar = np.array([sum(w * field[m] for m, w in rows[s]) for s in ids])
    colloc = np.array([sum(w * field[m] for m, w in _periodic_facet_weights(loc[s], fc["top"], loc)) for s in ids])
    exact = np.array([f(pts[s]) for s in ids])
    rms = lambda e: float(np.sqrt(np.mean(e**2)))  # noqa: E731
    assert rms(mortar - exact) < rms(colloc - exact)


def test_mortar_3d_gate_rejects_a_master_that_does_not_cover_the_slave():
    pts, tags, facets = _tri_faces(4, 7)
    trimmed = facets["top"][:-6]  # drop a strip of master triangles -> slave corner no longer covered
    assert _master_covers_slave_3d(facets["bot"], facets["top"], pts[:, :2])
    assert not _master_covers_slave_3d(facets["bot"], trimmed, pts[:, :2])
    res = build_periodic_prolongation(
        pts,
        [("top", "bot")],
        {"top": np.unique(trimmed), "bot": tags["bot"]},
        facets={"top": trimmed, "bot": facets["bot"]},
    )
    assert res["coupling"] == "collocated"


def test_mortar_3d_hole_in_the_master_face_raises():
    """Covered at the vertices but missing an interior triangle: the gate passes and the per-facet
    area conservation has to catch it. A bad clip yields a plausible wrong answer, so this raises."""
    pts, _tags, facets = _tri_faces(4, 9)
    holed = np.delete(facets["top"], 10, axis=0)
    with pytest.raises(ValueError, match="HOLE in the master face"):
        _mortar_rows_3d(facets["bot"], holed, pts[:, :2], span=1.0)


def test_mortar_3d_on_a_tilted_interface():
    """The frame and the segmentation must compose: a face whose normal is not a global axis is
    projected onto its own tangent plane, where clipping happens."""
    t1, t2, _n = _plane_frame([1.0, 1.0, 1.0])
    pts, tags, facets = _tri_faces(3, 6)
    tilted = pts[:, :1] * t1 + pts[:, 1:2] * t2 + pts[:, 2:3] * np.cross(t1, t2)
    res = build_periodic_prolongation(tilted, [("top", "bot")], tags, facets=facets)
    assert res["coupling"] == "mortar"
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    # Linear in the IN-PLANE coordinates only: a tie identifies the two faces, so a field with a
    # component along the interface normal genuinely differs across the gap and no tie reproduces it.
    field = 2.0 * pts[:, 0] - 1.5 * pts[:, 1] + 0.3
    assert np.allclose(P @ field[kept], field, atol=1e-9)


def test_p2_triangles_have_no_dual_basis_and_keep_collocation():
    """A real obstruction, not an oversight: the P2 triangle's vertex functions integrate to exactly
    zero, so ``A = diag(int N) Mass^-1`` is singular and this dual construction does not exist. P2
    EDGES are unaffected (``int N = 1/6``), which is why the 2-D quadratic path works."""
    bary, w = _tri_quadrature(4)
    d = w @ _tri_shape(bary, 6)
    assert abs(d[0]) < 1e-15 and abs(d[1]) < 1e-15 and abs(d[2]) < 1e-15  # the three vertex functions
    assert np.all(np.abs(d[3:]) > 1e-3)  # the midside functions are fine
    assert _tri_dual_available(3) and not _tri_dual_available(6)

    qp, qw = np.polynomial.legendre.leggauss(6)
    assert np.all(np.abs(0.5 * qw @ _edge_shape(0.5 * (qp + 1.0), 3)) > 1e-3)  # P2 edge: all non-zero


def test_mortar_3d_survives_a_high_density_ratio():
    """Extreme: a single master facet pair against a 1:12-refined slave face."""
    pts, tags, facets = _tri_faces(1, 12)
    res = build_periodic_prolongation(pts, [("top", "bot")], tags, facets=facets)
    assert res["coupling"] == "mortar"
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert len(kept) == 4 and np.allclose(P.sum(axis=1), 1.0)
    field = 1.3 * pts[:, 0] - 0.7 * pts[:, 1] + 2.0
    assert np.allclose(P @ field[kept], field, atol=1e-10)


# ------------------------------------------------- the master-side trace at arbitrary points
#
# A tie only ever needs the master field AT SLAVE NODES, so that is all `_periodic_facet_weights`
# offers. Contact needs it at **quadrature points**: the signed gap `g = g0 + n.(u_s - u_m.Phi)` is
# integrated over the slave face, so the two sides must be comparable wherever the rule samples them.
# `master_trace_weights` is that generalisation, batched -- and the two must agree where they overlap.


def test_trace_weights_reproduce_the_master_space_at_off_node_points():
    """The property that makes it a trace: exact for anything the master facets represent, evaluated
    at points that are deliberately NOT master nodes."""
    ym = np.linspace(0.0, 1.0, 5)
    pts = np.column_stack([np.ones(5), ym])
    mf = np.column_stack([np.arange(4), np.arange(1, 5)])
    q = np.array([[0.03], [0.37], [0.5], [0.99]])
    ids, w = master_trace_weights(q, mf, pts[:, 1:2])
    assert np.allclose(w.sum(axis=1), 1.0)
    got = (w * (3.0 * pts[ids, 1] - 1.0)).sum(axis=1)
    assert np.allclose(got, 3.0 * q[:, 0] - 1.0, atol=1e-12)

    p3, tris = _tri_grid(3, 1.0, 0)
    q3 = np.array([[0.13, 0.71], [0.5, 0.5], [0.92, 0.08], [0.33, 0.33]])
    ids3, w3 = master_trace_weights(q3, tris, p3[:, :2])
    assert np.allclose(w3.sum(axis=1), 1.0)
    f = lambda p: 2.0 * p[..., 0] - 1.5 * p[..., 1] + 0.3  # noqa: E731
    assert np.allclose((w3 * f(p3[ids3])).sum(axis=1), f(q3), atol=1e-12)


def test_trace_weights_agree_with_the_tie_weights_at_node_locations():
    """Where the two overlap they must be the same operator — otherwise a contact gap and a tie would
    disagree about what 'the master value here' means."""
    p3, tris = _tri_grid(3, 1.0, 0)
    loc = p3[:, :2]
    q = np.array([[0.13, 0.71], [0.5, 0.5], [0.2, 0.05]])
    ids, w = master_trace_weights(q, tris, loc)
    # Compare the operators by what they COMPUTE, not by which nodes they list: a query on a shared
    # edge has a zero barycentric, and the two may name different (equally valid) adjacent triangles.
    for i, pt in enumerate(q):
        ref = dict(_periodic_facet_weights(pt, tris, loc))
        got = {int(n): float(v) for n, v in zip(ids[i], w[i])}
        for f in (lambda p: 1.0 + 0 * p[0], lambda p: 2.0 * p[0] - 1.5 * p[1], lambda p: p[0] * p[1]):
            a = sum(v * f(loc[n]) for n, v in got.items())
            b = sum(v * f(loc[n]) for n, v in ref.items())
            assert abs(a - b) < 1e-12, f"query {pt}: trace {a} vs tie {b}"


def test_trace_weights_clamp_outside_the_face():
    """A query off the face is clamped to the nearest facet rather than extrapolated: the weights stay
    a partition of unity, so a constant field is still reproduced and nothing blows up."""
    p3, tris = _tri_grid(2, 1.0, 0)
    _ids, w = master_trace_weights(np.array([[1.4, 0.5], [-0.2, -0.2]]), tris, p3[:, :2])
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= -1e-12), "clamped weights must stay non-negative (no extrapolation)"


def test_trace_weights_handle_empty_input():
    p3, tris = _tri_grid(2, 1.0, 0)
    ids, w = master_trace_weights(np.zeros((0, 2)), tris, p3[:, :2])
    assert ids.shape == (0, 3) and w.shape == (0, 3)


# ------------------------------------------------------------------ the contact gap's geometry
#
# `g = g0 + n.(u_s - u_m.Phi)` splits into a part fixed by the geometry and a part that moves with the
# solution. `interface_gap_data` precomputes both: `g0` (the initial along-normal separation) and the
# gather that makes `u_m.Phi` a weighted sum of master DOFs — hence differentiable in the solution,
# though NOT in the mesh coordinates, since the projection is frozen at build time.


def _master_square(n=3, z=0.0, tilt=None):
    """A triangulated unit square, optionally embedded in a tilted plane. Returns (pts, tris, normal)."""
    pts, tris = _tri_grid(n, z, 0)
    if tilt is None:
        return pts, tris, np.array([0.0, 0.0, 1.0])
    t1, t2, _ = _plane_frame(tilt)
    return pts[:, 0:1] * t1 + pts[:, 1:2] * t2, tris, np.cross(t1, t2)


@pytest.mark.parametrize("offset", [0.0, 0.25, -0.1])
def test_initial_gap_is_the_signed_along_normal_separation(offset):
    """Zero for coincident (tied) faces, positive for a standoff, negative for initial penetration —
    the sign convention the contact pressure `max(0, lam + c*(-g))` depends on."""
    mp, tris, n = _master_square()
    qp = np.array([[[0.2, 0.3, offset], [0.6, 0.7, offset]], [[0.5, 0.5, offset], [0.9, 0.1, offset]]])
    ids, w, g0 = interface_gap_data(qp, tris, mp, np.broadcast_to(n, (4, 3)))
    assert ids.shape == (2, 2, 3) and w.shape == (2, 2, 3) and g0.shape == (2, 2)  # leading dims kept
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert np.allclose(g0, offset, atol=1e-12)


def test_initial_gap_on_a_tilted_interface():
    """The separation is measured along the interface's OWN normal, not a global axis — the frame is
    fitted to the master face exactly as a tie's is."""
    mp, tris, n = _master_square(tilt=[0.0, -1.0, 1.0])
    uv = np.array([[0.3, 0.4], [0.6, 0.2]])
    t1, t2, _ = _plane_frame([0.0, -1.0, 1.0])
    qp = (uv @ np.stack([t1, t2]))[None, :, :] + 0.37 * n
    _ids, _w, g0 = interface_gap_data(qp, tris, mp, np.broadcast_to(n, (2, 3)))
    assert np.allclose(g0, 0.37, atol=1e-12)


def test_initial_gap_varies_over_the_face():
    """A slave face that is not parallel to the master gives a per-point gap, not one number."""
    mp, tris, n = _master_square()
    heights = np.array([0.05, 0.2, 0.35])
    qp = np.stack([np.array([0.25, 0.25, h]) for h in heights])[None, :, :]
    _ids, _w, g0 = interface_gap_data(qp, tris, mp, np.broadcast_to(n, (3, 3)))
    assert np.allclose(g0.ravel(), heights, atol=1e-12)


def test_gap_gather_reads_the_master_field_exactly():
    """The solution-dependent half: `u_m . Phi` must reproduce anything the master facets represent,
    so a rigid relative motion produces exactly that change in gap and nothing spurious."""
    mp, tris, n = _master_square()
    qp = np.array([[[0.2, 0.3, 0.1], [0.63, 0.71, 0.1], [0.5, 0.5, 0.1]]])
    ids, w, g0 = interface_gap_data(qp, tris, mp, np.broadcast_to(n, (3, 3)))
    f = lambda p: 2.0 * p[..., 0] - 1.5 * p[..., 1] + 0.3  # noqa: E731
    u_m = (w * f(mp[ids])).sum(axis=-1)
    assert np.allclose(u_m, f(qp[..., :]), atol=1e-12)
    # A uniform master displacement must read back exactly (partition of unity through the gather),
    # so moving the master body rigidly by w0 closes the gap by exactly w0 and nothing else.
    w0 = 0.04
    assert np.allclose((w * np.full(ids.shape, w0)).sum(axis=-1), w0, atol=1e-15)
    assert np.allclose(g0, 0.1, atol=1e-12)  # and the standoff itself is what was built


def test_periodic_pair_still_ties_across_a_normal_offset():
    """No-regression at the prolongation level: the ordinary periodic case (faces separated along
    their own normal) is unchanged — a linear in-plane field is still reproduced exactly."""
    master = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    slave = np.array([[0.3, 0.2, 0.0], [0.7, 0.6, 0.0], [0.5, 0.5, 0.0]])
    pts = np.vstack([master, slave])
    tags = {"top": np.arange(4), "bot": np.arange(4, 7)}
    res = build_periodic_prolongation(pts, [("top", "bot")], tags, facets={"top": tris})
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    field = 2.0 * pts[:, 0] - 1.5 * pts[:, 1] + 0.3
    assert np.allclose(P @ field[kept], field, atol=1e-10)
