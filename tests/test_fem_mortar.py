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

import numpy as np
import pytest

from jno.utils.solver.fem_utils import (
    _edge_shape,
    _faces_span_the_same_extent,
    _facet_dual_coeffs,
    _interface_frame,
    _mortar_rows_2d,
    _periodic_facet_weights,
    build_periodic_prolongation,
)


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
