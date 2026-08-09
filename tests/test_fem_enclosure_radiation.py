"""Enclosure-radiation view factors — analytic validation (F-correctness gate).

The radiosity in an enclosure-radiation boundary condition is only as good as the geometric view
factor matrix ``F``, so these tests pin the view-factor *kernels* against closed-form results before
any radiosity / FEM coupling is built on top. The canonical analytic case is two **concentric
cylinders** (radii ``r1 < r2``): for infinitely long cylinders the surface-averaged view factors are

    F12 = 1 ,   F21 = r1/r2 ,   F22 = 1 - r1/r2     (the outer cylinder's *concave self-view*)

and the per-point kernel must satisfy **closure** (row sums -> 1 for a closed enclosure) and
**reciprocity** (A_i F_ij = A_j F_ji). The concave self-view ``F22`` is carried entirely by the
visibility matrix (the inner cylinder occludes outer-to-outer rays); the kernel only removes the
self-pair (diagonal). The axisymmetric kernel integrates the diffuse point-to-ring kernel over the
azimuthal angle and must reproduce the same factors for a tall cylinder.

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4 (view factors; concentric
cylinders; bodies of revolution).
"""

from __future__ import annotations

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np
import pytest

from jno.domain.mesh_utils import MeshUtils


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _seg_hits_disk(p, q, radius):
    """True if segment p-q passes strictly inside the disk of the given radius (occlusion test)."""
    d = q - p
    length2 = float(np.dot(d, d))
    if length2 == 0.0:
        return False
    t = float(np.clip(-np.dot(p, d) / length2, 0.0, 1.0))
    closest = p + t * d
    return float(np.hypot(*closest)) < radius - 1e-9


def _concentric_2d(r1, r2, n1, n2):
    """Concentric-circle cross-section: points, visibility (inner cylinder occludes), normals, ds."""
    th1 = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
    th2 = np.linspace(0.0, 2 * np.pi, n2, endpoint=False)
    p_in = np.c_[r1 * np.cos(th1), r1 * np.sin(th1)]
    p_out = np.c_[r2 * np.cos(th2), r2 * np.sin(th2)]
    points = np.vstack([p_in, p_out])
    normals = np.vstack([np.c_[np.cos(th1), np.sin(th1)], -np.c_[np.cos(th2), np.sin(th2)]])  # into the gap
    ds = np.concatenate([np.full(n1, 2 * np.pi * r1 / n1), np.full(n2, 2 * np.pi * r2 / n2)])
    n = n1 + n2
    vm = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or _seg_hits_disk(points[i], points[j], r1):
                vm[i, j] = 0.0
    return points, vm, normals, ds


def test_view_factor_2d_concentric_cylinders():
    r1, r2, n1, n2 = 0.20, 0.35, 240, 360
    points, vm, normals, ds = _concentric_2d(r1, r2, n1, n2)
    F = np.asarray(
        MeshUtils.get_view_factor_2d(jnp.asarray(points), jnp.asarray(vm), jnp.asarray(normals), jnp.asarray(ds))
    )
    A = ds  # 2-D "area" per node is the arc length
    rows = F.sum(axis=1)

    assert np.allclose(rows, 1.0, atol=5e-3), f"closure violated: row sums {rows.min():.4f}..{rows.max():.4f}"
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "reciprocity A_i F_ij = A_j F_ji violated"

    F12 = (A[:n1, None] * F[:n1, n1:]).sum() / A[:n1].sum()
    F21 = (A[n1:, None] * F[n1:, :n1]).sum() / A[n1:].sum()
    F22 = (A[n1:, None] * F[n1:, n1:]).sum() / A[n1:].sum()
    assert abs(F12 - 1.0) < 5e-3, f"F12 should be 1, got {F12:.4f}"
    assert abs(F21 - r1 / r2) < 5e-3, f"F21 should be r1/r2={r1 / r2:.4f}, got {F21:.4f}"
    assert abs(F22 - (1 - r1 / r2)) < 5e-3, f"concave self-view F22 should be {1 - r1 / r2:.4f}, got {F22:.4f}"


def test_view_factor_axisymmetric_tall_cylinder():
    """Axisymmetric kernel on a tall concentric cylinder reproduces F12->1, F21->r1/r2, with exact
    reciprocity. Locks the azimuthal-integral normalization (a plain mean under-counts by 2*pi)."""
    r1, r2, height, nz = 0.20, 0.35, 6.0, 60
    z = np.linspace(0.0, height, nz)
    points = np.vstack([np.c_[np.full(nz, r1), z], np.c_[np.full(nz, r2), z]])
    normals = np.vstack([np.tile([1.0, 0.0], (nz, 1)), np.tile([-1.0, 0.0], (nz, 1))])
    ds = np.concatenate([np.full(nz, height / nz), np.full(nz, height / nz)])
    vm = np.ones((2 * nz, 2 * nz))  # tall thin gap, mid-ring check ignores end occlusion

    F = np.asarray(
        MeshUtils.get_view_factor_axisymmetric(
            jnp.asarray(points), jnp.asarray(vm), jnp.asarray(normals), jnp.asarray(ds), n_phi=64
        )
    )
    A = 2 * np.pi * points[:, 0] * ds  # ring areas
    assert np.all(np.isfinite(F)) and np.allclose(np.diag(F), 0.0)
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "axisymmetric reciprocity violated"

    mid = nz // 2
    f12_mid = F[mid, nz:].sum()  # a mid inner ring sees essentially the whole outer wall
    F21 = (A[nz:, None] * F[nz:, :nz]).sum() / A[nz:].sum()
    assert abs(f12_mid - 1.0) < 3e-2, f"axisymmetric F12 (mid ring) should be ~1, got {f12_mid:.3f}"
    assert abs(F21 - r1 / r2) < 3e-2, f"axisymmetric F21 should be r1/r2={r1 / r2:.3f}, got {F21:.3f}"


def _ray_crosses_disk(p, q, radius):
    """True if the OPEN segment p-q dips inside the disk (closest approach strictly interior)."""
    d = q - p
    length2 = float(np.dot(d, d))
    if length2 == 0.0:
        return False
    t = -float(np.dot(p, d)) / length2  # unclamped: a genuine crossing needs 0 < t < 1
    if t <= 1e-9 or t >= 1.0 - 1e-9:
        return False
    closest = p + t * d
    return float(np.hypot(*closest)) < radius - 1e-9


def _concentric_2d_elements(r1, r2, n1, n2):
    """Concentric circles as boundary *elements* (edges): endpoints, element normals, and element-level
    visibility (midpoint occlusion by the inner cylinder)."""
    th1 = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
    th2 = np.linspace(0.0, 2 * np.pi, n2, endpoint=False)
    v_in = np.c_[r1 * np.cos(th1), r1 * np.sin(th1)]
    v_out = np.c_[r2 * np.cos(th2), r2 * np.sin(th2)]
    e0 = np.vstack([v_in, v_out])
    e1 = np.vstack([np.roll(v_in, -1, axis=0), np.roll(v_out, -1, axis=0)])
    mids = 0.5 * (e0 + e1)
    nrm = mids / np.linalg.norm(mids, axis=1, keepdims=True)
    nrm[n1:] *= -1.0  # inner outward, outer inward -> both point into the gap
    m = n1 + n2
    vm = np.ones((m, m))
    for i in range(m):
        for j in range(m):
            if i == j or _ray_crosses_disk(mids[i], mids[j], r1):
                vm[i, j] = 0.0
    return e0, e1, nrm, vm


def test_view_factor_2d_element_concentric_cylinders():
    """Element-based double-area kernel reproduces the analytic concentric-cylinder factors (closure,
    reciprocity, F12=1, F21=r1/r2, concave self-view F22=1-r1/r2). Occlusion is element-level
    (midpoint) and converges with resolution; the double-area quadrature sharpens the near field."""
    r1, r2, n1, n2 = 0.20, 0.35, 240, 360
    e0, e1, nrm, vm = _concentric_2d_elements(r1, r2, n1, n2)
    A = np.linalg.norm(e1 - e0, axis=1)  # element lengths
    F = np.asarray(
        MeshUtils.get_view_factor_2d_element(jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(nrm), jnp.asarray(vm), n_quad=3)
    )
    rows = F.sum(axis=1)
    F12 = (A[:n1, None] * F[:n1, n1:]).sum() / A[:n1].sum()
    F21 = (A[n1:, None] * F[n1:, :n1]).sum() / A[n1:].sum()
    F22 = (A[n1:, None] * F[n1:, n1:]).sum() / A[n1:].sum()

    assert np.allclose(rows, 1.0, atol=5e-3), f"closure: row sums {rows.min():.4f}..{rows.max():.4f}"
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "element reciprocity violated"
    assert abs(F12 - 1.0) < 5e-3, f"F12 should be 1, got {F12:.4f}"
    assert abs(F21 - r1 / r2) < 5e-3, f"F21 should be r1/r2={r1 / r2:.4f}, got {F21:.4f}"
    assert abs(F22 - (1 - r1 / r2)) < 5e-3, f"concave self-view F22 should be {1 - r1 / r2:.4f}, got {F22:.4f}"


def test_view_factor_axisymmetric_element_tall_cylinder():
    """Element-based axisymmetric kernel (meridional element quadrature x azimuthal integration) on a
    tall concentric cylinder: a mid inner element sees ~the whole outer wall (F12->1), F21->r1/r2, with
    exact reciprocity (A_i F_ij = A_j F_ji, A = ring area)."""
    r1, r2, height, nz = 0.20, 0.35, 6.0, 60
    z = np.linspace(0.0, height, nz + 1)
    z0, z1 = z[:-1], z[1:]
    e0 = np.vstack([np.c_[np.full(nz, r1), z0], np.c_[np.full(nz, r2), z0]])
    e1 = np.vstack([np.c_[np.full(nz, r1), z1], np.c_[np.full(nz, r2), z1]])
    nrm = np.vstack([np.tile([1.0, 0.0], (nz, 1)), np.tile([-1.0, 0.0], (nz, 1))])
    vm = np.ones((2 * nz, 2 * nz))  # tall thin gap; mid-element check ignores end occlusion

    F = np.asarray(
        MeshUtils.get_view_factor_axisymmetric_element(
            jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(nrm), jnp.asarray(vm), n_quad=3, n_phi=64
        )
    )
    rmid = 0.5 * (e0[:, 0] + e1[:, 0])
    length = np.linalg.norm(e1 - e0, axis=1)
    A = 2 * np.pi * rmid * length  # ring areas
    assert np.all(np.isfinite(F)) and np.allclose(np.diag(F), 0.0)
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "axisymmetric element reciprocity violated"

    mid = nz // 2
    f12_mid = F[mid, nz:].sum()
    F21 = (A[nz:, None] * F[nz:, :nz]).sum() / A[nz:].sum()
    assert abs(f12_mid - 1.0) < 3e-2, f"axisymmetric element F12 (mid) should be ~1, got {f12_mid:.3f}"
    assert abs(F21 - r1 / r2) < 3e-2, f"axisymmetric element F21 should be r1/r2={r1 / r2:.3f}, got {F21:.3f}"


def test_grey_body_radiosity_two_surface_flux():
    """Full grey-body radiosity ``q = (I-F)(I-diag(rho)F)^-1 diag(eps) sigma T^4`` (reflections
    included) reproduces the closed-form two-surface concentric-cylinder net flux

        q1 = sigma (T1^4 - T2^4) / (1/eps1 + (r1/r2)(1/eps2 - 1)),

    and conserves energy (A1 q1 + A2 q2 ~ 0). This is the radiosity the user writes in ``jno.np``;
    here it is checked in numpy against the analytic result using the committed element view factor."""
    sigma = 5.670374419e-8
    r1, r2, n1, n2 = 0.20, 0.35, 240, 360
    eps1, eps2, t1, t2 = 0.8, 0.6, 1000.0, 400.0

    e0, e1, nrm, vm = _concentric_2d_elements(r1, r2, n1, n2)
    A = np.linalg.norm(e1 - e0, axis=1)
    F = np.asarray(
        MeshUtils.get_view_factor_2d_element(jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(nrm), jnp.asarray(vm), n_quad=3)
    )

    eps = np.concatenate([np.full(n1, eps1), np.full(n2, eps2)])
    rho = 1.0 - eps
    temp = np.concatenate([np.full(n1, t1), np.full(n2, t2)])
    emissive = eps * sigma * temp**4
    radiosity = np.linalg.solve(np.eye(n1 + n2) - rho[:, None] * F, emissive)
    q = (np.eye(n1 + n2) - F) @ radiosity  # net radiative flux per element (per area)

    q1 = float((A[:n1] * q[:n1]).sum() / A[:n1].sum())
    q1_analytic = sigma * (t1**4 - t2**4) / (1.0 / eps1 + (r1 / r2) * (1.0 / eps2 - 1.0))
    assert abs(q1 - q1_analytic) / q1_analytic < 5e-3, f"radiosity flux {q1:.1f} vs analytic {q1_analytic:.1f}"

    total = float((A * q).sum())  # energy balance over the closed enclosure
    scale = float((A[:n1] * np.abs(q[:n1])).sum())
    assert abs(total) / scale < 5e-3, f"energy balance |sum A q| / scale = {abs(total) / scale:.2e}"


def test_enclosure_handle_concentric_cylinders():
    """The d.enclosure(tags) handle builds an FEM-node-aligned element view factor from a real mesh:
    two solid rings with a vacuum gap. Validates the full pipeline (boundary-edge elements, into-gap
    normals, occlusion, element kernel) against the analytic concentric-cylinder factors, and the
    F-quality gate."""
    import jno

    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import Point

    r1, r2 = 0.20, 0.25
    ring = lambda a, b: Point(0, 0).buffer(b, 48).difference(Point(0, 0).buffer(a, 48))  # noqa: E731
    d = jno.domain(ring(0.10, r1).union(ring(r2, 0.35)), mesh_size=0.08)
    radius = lambda x, y: np.hypot(x, y)  # noqa: E731
    d.tag("inner_gap", lambda x, y: np.abs(radius(x, y) - r1) < 1.5e-2)
    d.tag("outer_gap", lambda x, y: np.abs(radius(x, y) - r2) < 1.5e-2)

    gap = d.enclosure(["inner_gap", "outer_gap"])
    gap.check()  # F-quality gate (closure + reciprocity) must pass
    closure, recip = gap.quality()
    assert closure < 5e-3 and recip < 1e-12

    # node indices are valid FEM DOFs (scalar P1: DOF == mesh node)
    n_pts = np.asarray(d.mesh.points).shape[0]
    assert gap.nodes.min() >= 0 and gap.nodes.max() < n_pts

    F = np.asarray(gap.view_factor)
    A = np.asarray(gap.areas)
    mi, mo = gap.tag_mask("inner_gap"), gap.tag_mask("outer_gap")
    assert mi.sum() > 0 and mo.sum() > 0
    F12 = (A[mi, None] * F[np.ix_(mi, mo)]).sum() / A[mi].sum()
    F21 = (A[mo, None] * F[np.ix_(mo, mi)]).sum() / A[mo].sum()
    F22 = (A[mo, None] * F[np.ix_(mo, mo)]).sum() / A[mo].sum()
    assert abs(F12 - 1.0) < 1e-2, f"F12 should be 1, got {F12:.4f}"
    assert abs(F21 - r1 / r2) < 1e-2, f"F21 should be r1/r2={r1 / r2:.4f}, got {F21:.4f}"
    assert abs(F22 - (1 - r1 / r2)) < 1e-2, f"concave self-view F22 should be {1 - r1 / r2:.4f}, got {F22:.4f}"

    # --- gather (field) + consistent scatter (load) + emissivity plumbing ---
    n_dofs = n_pts
    pts = np.asarray(d.mesh.points)
    # field(u): per-element mean of endpoint values. Use a linear field u = x to check exactly.
    u_lin = jnp.asarray(pts[:, 0])
    elem_mid_x = 0.5 * (pts[gap.elements[:, 0], 0] + pts[gap.elements[:, 1], 0])
    assert np.allclose(np.asarray(gap.field(u_lin)), elem_mid_x, atol=1e-12), "field() gather is wrong"
    # load(q): total scattered load equals the surface integral of q (here q=1 -> total area).
    load1 = np.asarray(gap.load(jnp.ones(gap.size), size=n_dofs))
    assert load1.shape == (n_dofs,)
    assert abs(load1.sum() - float(np.asarray(gap.areas).sum())) < 1e-9, "load() does not conserve the integral"
    assert np.all(load1[gap.nodes] > 0) and abs(load1.sum() - load1[gap.nodes].sum()) < 1e-9, "load() leaked off-surface"
    # emissivity({tag: eps}) -> per-element vector
    eps = np.asarray(gap.emissivity({"inner_gap": 0.8, "outer_gap": 0.6}))
    assert np.allclose(eps[mi], 0.8) and np.allclose(eps[mo], 0.6)


def test_enclosure_handle_square_cavity_inward():
    """``d.enclosure(..., inward=True)`` builds the enclosure for a *meshed* cavity (an oven: the air
    inside is the mesh, the four walls are the outer boundary). Radiation then crosses the meshed
    interior, so element normals must point INTO the mesh. Validates the 2D square view factors
    (opposite walls F = sqrt(2)-1, adjacent walls F = (2-sqrt(2))/2, flat self-view 0) and that the
    default (outward) normals fail closure -- the case that motivated the flag."""
    import jno

    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import box

    L = 1.0
    d = jno.domain(box(0, 0, L, L), mesh_size=0.05)
    d.tag("hot", lambda x, y: x < 1e-6)
    d.tag("cold", lambda x, y: x > L - 1e-6)
    d.tag("bottom", lambda x, y: y < 1e-6)
    d.tag("top", lambda x, y: y > L - 1e-6)
    walls = ["hot", "cold", "top", "bottom"]

    gap = d.enclosure(walls, inward=True)
    gap.check()  # closure + reciprocity gate must pass for the closed square enclosure
    closure, recip = gap.quality()
    assert closure < 5e-2 and recip < 1e-10

    F = np.asarray(gap.view_factor)
    A = np.asarray(gap.areas)
    tags = np.asarray(gap.element_tags)

    def s2s(a, b):  # area-weighted surface-to-surface view factor (mean over receivers in a)
        ia, ib = tags == a, tags == b
        return float((A[ia, None] * F[np.ix_(ia, ib)]).sum() / A[ia].sum())

    f_opp = 0.5 * (s2s("hot", "cold") + s2s("top", "bottom"))  # opposite walls
    f_adj = 0.25 * (s2s("hot", "top") + s2s("hot", "bottom") + s2s("cold", "top") + s2s("cold", "bottom"))
    assert abs(f_opp - (np.sqrt(2) - 1)) < 1e-2, (
        f"opposite-wall F should be sqrt(2)-1={np.sqrt(2) - 1:.4f}, got {f_opp:.4f}"
    )
    assert abs(f_adj - (2 - np.sqrt(2)) / 2) < 2e-2, (
        f"adjacent-wall F should be {(2 - np.sqrt(2)) / 2:.4f}, got {f_adj:.4f}"
    )
    assert s2s("hot", "hot") < 1e-6, "a flat wall cannot see itself"
    assert abs(F.sum(axis=1) - 1.0).max() < 5e-2, "closed enclosure: row sums -> 1"

    # default (outward) normals are wrong for a meshed cavity: the walls face away from each other,
    # so no element sees another and the enclosure fails the closure gate.
    bad = d.enclosure(walls)
    assert np.asarray(bad.view_factor).max() < 1e-6, "outward normals must yield zero view factors here"
    with pytest.raises(ValueError, match="closure"):
        bad.check()


def _axisym_vf(e0, e1, nrm, dom, tags, *, n_phi=96, n_quad=3, refine, r_min_fac):
    """Axisymmetric element view factor, with/without the near-field refinement, at a chosen r_min."""
    from jno.domain.enclosure import _refine_near_pairs, _solid_polygon_visibility_3d

    mids = 0.5 * (e0 + e1)
    length = np.linalg.norm(e1 - e0, axis=1)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    vm = _solid_polygon_visibility_3d(dom, tags, mids, length, phi)
    F = np.asarray(
        MeshUtils.get_view_factor_axisymmetric_element(
            jnp.asarray(e0),
            jnp.asarray(e1),
            jnp.asarray(nrm),
            jnp.asarray(vm),
            n_quad=n_quad,
            n_phi=n_phi,
            r_min=r_min_fac * float(np.median(length)),
        )
    )
    if refine:
        F = _refine_near_pairs(F, e0, e1, nrm, dom, tags, n_phi)
    return F


def test_axisymmetric_occlusion_catches_a_wall_far_thinner_than_the_mesh():
    """A closed cavity with a 0.5 mm baffle across it, meshed with 5 mm elements: the occluder is TEN
    TIMES thinner than an element, and 400x thinner than the chords crossing the cavity.

    This is the case a fixed-stride chord test cannot do at any affordable density -- sample every
    ``L/n`` and you step clean over any wall thinner than that, and the ray passes THROUGH a solid. That
    is occlusion which silently does not happen, so it INFLATES the view factors: on the cg furnace a
    16-sample test wrongly called 12% of all visible pairs visible. It is equally fatal to scale the
    blocking tolerance to the element size (as the legacy straight-line test does): a wall thinner than
    the tolerance can then never register as a block no matter how finely it is sampled.

    Occlusion is instead sphere-traced against a signed-distance field, which advances by the distance to
    the nearest solid and therefore CANNOT skip a feature, whatever its size (Hart 1996).

    The cavity is closed, so this is checkable without a reference: rows must sum to 1. If the baffle
    leaks, the elements it should shadow see straight past it and their rows blow past 1.
    """
    from types import SimpleNamespace

    from shapely.geometry import box

    R, H, zb, Rb, thick = 0.10, 0.20, 0.10, 0.06, 0.0005  # baffle 0.5 mm thick; elements ~5 mm
    NR, NZ, NB = 20, 40, 12
    segs, nrms = [], []

    def add(p0, p1, n, k):
        t = (np.arange(k + 1) / k)[:, None]
        P = np.asarray(p0)[None, :] * (1 - t) + np.asarray(p1)[None, :] * t
        segs.append((P[:-1], P[1:]))
        nrms.append(np.tile(n, (k, 1)))

    add((1e-6, 0), (R, 0), (0, 1), NR)  # bottom disc, faces up
    add((R, 0), (R, H), (-1, 0), NZ)  # wall, faces in
    add((1e-6, H), (R, H), (0, -1), NR)  # top disc, faces down
    add((1e-6, zb + thick), (Rb, zb + thick), (0, 1), NB)  # baffle top face
    add((1e-6, zb), (Rb, zb), (0, -1), NB)  # baffle bottom face
    e0 = np.vstack([a for a, _ in segs])
    e1 = np.vstack([b for _, b in segs])
    nrm = np.vstack(nrms)
    m = e0.shape[0]
    dom = SimpleNamespace(_source_regions={"baffle": box(0.0, zb, Rb, zb + thick)})
    tags = np.array(["baffle"] * m)

    assert thick < 0.15 * float(np.median(np.linalg.norm(e1 - e0, axis=1))), "premise: baffle << element"

    F = _axisym_vf(e0, e1, nrm, dom, tags, refine=True, r_min_fac=0.0)
    rs = F.sum(axis=1)
    assert rs.max() < 1.02, f"rays are leaking through the thin baffle: max row sum {rs.max():.4f}"
    assert rs.min() > 0.98, f"closed cavity: row sums must not fall below 1, got min {rs.min():.4f}"


def test_axisymmetric_near_field_closed_cavity_rows_sum_to_one():
    """A CLOSED cylindrical cavity: every row of F must sum to exactly 1 (no radiation can escape).

    This is the corner test. Where the wall meets each end disk, two surfaces are separated by a distance
    ``d`` far smaller than their radius ``r``, and the ring kernel's azimuthal integrand becomes a peak at
    ``phi = 0`` of width ``phi_c ~ d/r``. A uniform ``n_phi`` rule with ``2*pi/n_phi >> phi_c`` samples the
    peak's crest and multiplies it by a step far wider than the peak, overshooting by ``~dphi/phi_c`` —
    driving row sums to ~1.8 here, and to ~6.7 on a real furnace mesh with sub-mm wedges. The ``r_min``
    floor only masks it (by inflating ``d`` until ``d/r ~ dphi``), at the cost of biasing every OTHER pair
    low — which is why both must be fixed together, not separately.

    Locks: the graded azimuthal rule restores closure with r_min = 0 and NO enforce_closure, and still
    reproduces the analytic coaxial-disk factor (Modest, Ch. 4 / App. D).
    """
    from types import SimpleNamespace

    R, H, NR, NZ = 0.10, 0.10, 24, 24
    rr, zz = np.linspace(1e-6, R, NR + 1), np.linspace(0.0, H, NZ + 1)
    e0 = np.vstack([np.c_[rr[:-1], np.zeros(NR)], np.c_[np.full(NZ, R), zz[:-1]], np.c_[rr[:-1], np.full(NR, H)]])
    e1 = np.vstack([np.c_[rr[1:], np.zeros(NR)], np.c_[np.full(NZ, R), zz[1:]], np.c_[rr[1:], np.full(NR, H)]])
    nrm = np.vstack([np.tile([0.0, 1.0], (NR, 1)), np.tile([-1.0, 0.0], (NZ, 1)), np.tile([0.0, -1.0], (NR, 1))])
    m = e0.shape[0]
    dom = SimpleNamespace(_source_regions={})  # closed cavity: nothing occludes
    tags = np.array(["cav"] * m)

    # WITHOUT the refinement (and no r_min to hide it), the corners overshoot badly.
    F_raw = _axisym_vf(e0, e1, nrm, dom, tags, refine=False, r_min_fac=0.0)
    assert F_raw.sum(axis=1).max() > 1.5, (
        "expected the un-refined corner overshoot (guard against a silent regression of this test's premise)"
    )

    # WITH it: closure, from the raw kernel, with no r_min and no enforce_closure.
    F = _axisym_vf(e0, e1, nrm, dom, tags, refine=True, r_min_fac=0.0)
    rs = F.sum(axis=1)
    assert rs.max() < 1.02, f"closed cavity: row sums must not exceed 1, got max {rs.max():.4f}"
    assert rs.min() > 0.98, f"closed cavity: row sums must not fall below 1, got min {rs.min():.4f}"

    # ...and the analytic coaxial parallel-disk factor still comes out right.
    mids = 0.5 * (e0 + e1)
    A = 2 * np.pi * mids[:, 0] * np.linalg.norm(e1 - e0, axis=1)
    bot, top = np.arange(NR), np.arange(NR + NZ, m)
    f_bt = float((A[bot, None] * F[np.ix_(bot, top)]).sum() / A[bot].sum())
    Rr = R / H
    S = 1 + (1 + Rr**2) / Rr**2
    ana = 0.5 * (S - np.sqrt(S**2 - 4))
    assert abs(f_bt - ana) < 5e-3, f"bottom->top disk factor {f_bt:.4f} vs analytic {ana:.4f}"


def test_axisymmetric_occlusion_is_per_azimuth_concentric_cylinders():
    """The outer cylinder's concave self-view ``F22 = 1 - r1/r2`` is carried ENTIRELY by occlusion.

    Two outer-wall elements both sit at ``r = r2``, so the straight line between them in the (r,z) meridian
    never enters the inner solid — an occlusion test that checks only that line concludes "always visible"
    and F22 comes out ~2x too big, with row sums ABOVE 1 (unphysical). The true 3-D chord between them bows
    inward to ``r2 cos(phi/2)`` and does cut the inner cylinder once ``phi > 2 arccos(r1/r2)``.
    """
    from types import SimpleNamespace

    from shapely.geometry import box

    r1, r2, H, NZ = 0.20, 0.35, 6.0, 60  # tall -> the infinite-cylinder limit
    z = np.linspace(0.0, H, NZ + 1)
    e0 = np.vstack([np.c_[np.full(NZ, r1), z[:-1]], np.c_[np.full(NZ, r2), z[:-1]]])
    e1 = np.vstack([np.c_[np.full(NZ, r1), z[1:]], np.c_[np.full(NZ, r2), z[1:]]])
    nrm = np.vstack([np.tile([1.0, 0.0], (NZ, 1)), np.tile([-1.0, 0.0], (NZ, 1))])
    dom = SimpleNamespace(_source_regions={"inner": box(0.0, 0.0, r1, H)})
    tags = np.array(["inner"] * (2 * NZ))

    F = _axisym_vf(e0, e1, nrm, dom, tags, refine=True, r_min_fac=0.0)
    mid_i, mid_o = NZ // 2, NZ + NZ // 2
    inner, outer = slice(0, NZ), slice(NZ, 2 * NZ)
    f12, f21, f22 = F[mid_i, outer].sum(), F[mid_o, inner].sum(), F[mid_o, outer].sum()

    assert abs(f12 - 1.0) < 2e-2, f"F12 should be 1, got {f12:.3f}"
    assert abs(f21 - r1 / r2) < 2e-2, f"F21 should be r1/r2={r1 / r2:.3f}, got {f21:.3f}"
    assert abs(f22 - (1 - r1 / r2)) < 2e-2, f"concave self-view F22 should be {1 - r1 / r2:.3f}, got {f22:.3f}"
    assert abs(f21 + f22 - 1.0) < 2e-2, f"row must close to 1 with no enforce_closure, got {f21 + f22:.3f}"


def test_enclosure_axisymmetric_r_min_keeps_view_factors_physical():
    """The axisymmetric ring kernel has a ``1/R^2`` near-field singularity: near-coincident / on-axis
    ring pairs blow up to F > 1 unless a near-field floor ``r_min`` is applied. ``d.enclosure`` defaults
    ``r_min`` to half the median element length, keeping the assembled view factors physical (<= 1).
    Regression for the axisymmetric branch passing ``r_min`` through to the kernel.

    Geometry: two coaxial cylinders in the meridional ``(r, z)`` half-plane (inner r1, outer r2) with a
    vacuum gap; analytic surface factors are ``F12 = 1`` and ``F21 = r1/r2`` (Modest, Ch. 4)."""
    import jno

    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import box

    r1, r2, w, H = 0.20, 0.25, 0.03, 1.6  # tall (aspect ~6) so end losses are small and F12 -> 1
    inner = box(r1 - w, 0.0, r1, H)  # inner solid, gap-facing surface at r = r1
    outer = box(r2, 0.0, r2 + w, H)  # outer solid, gap-facing surface at r = r2
    d = jno.domain(inner.union(outer), mesh_size=0.04)
    d.tag("inner_gap", lambda x, y: np.abs(x - r1) < 1e-2)
    d.tag("outer_gap", lambda x, y: np.abs(x - r2) < 1e-2)

    gap = d.enclosure(["inner_gap", "outer_gap"], axisymmetric=True)  # r_min defaulted
    F = np.asarray(gap.view_factor)
    assert F.max() <= 1.05, f"default r_min must keep axisymmetric view factors physical, got max {F.max():.3f}"

    A = np.asarray(gap.areas)
    mi, mo = gap.tag_mask("inner_gap"), gap.tag_mask("outer_gap")
    assert mi.sum() > 0 and mo.sum() > 0
    F12 = (A[mi, None] * F[np.ix_(mi, mo)]).sum() / A[mi].sum()
    F21 = (A[mo, None] * F[np.ix_(mo, mi)]).sum() / A[mo].sum()
    # finite cylinder (aspect ~6): inner surface still sees mostly the outer one, with small end losses
    assert 0.88 < F12 <= 1.0, f"axisymmetric F12 should be ~1 (small end losses), got {F12:.3f}"
    assert abs(F21 - r1 / r2) < 6e-2, f"axisymmetric F21 should be ~r1/r2={r1 / r2:.3f}, got {F21:.3f}"

    # r_min is now INERT on the axisymmetric path and must be accepted without changing anything. It
    # was a near-field 1/R^2 floor, a fudge for rings the uniform azimuthal rule could not resolve;
    # the azimuthal integral is exact now, so there is no singularity to soften and the floor would be
    # pure bias (it carried a known ~12% one). The physics assertions above are the real content, and
    # they hold with no floor at all -- which is the point.
    soft = np.asarray(d.enclosure(["inner_gap", "outer_gap"], axisymmetric=True, r_min=0.5).view_factor)
    assert np.allclose(soft, F), "r_min must be accepted and ignored on the closed-form path"


def test_coupled_conduction_radiation_concentric_cylinders():
    """End-to-end: steady conduction in two solid rings + grey-body radiation across the vacuum gap,
    coupled, matches the closed-form two-surface series solution

        Q = 2*pi*k*(T_hot - Ts1)/ln(r1/r0) = 2*pi*r1*sigma*(Ts1^4 - Ts2^4)/D = 2*pi*k*(Ts2 - T_cold)/ln(r3/r2)

    with D = 1/eps1 + (r1/r2)(1/eps2 - 1). The radiosity is written over enclosure.view_factor; the net
    flux enters as a consistent surface load (A u = b - load(q_rad)). Penalty-Dirichlet A is
    ill-conditioned, so the coupled system is solved jax-natively with a DIRECT-solve Newton (no scipy;
    a matrix-free iterative solver stalls) wrapped in jax.lax.custom_root -> differentiable end to end."""
    import jno

    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import Point

    def newton(residual, u0, steps=50, tol=1e-9):  # BYO direct-solve Newton (jno imposes no solver)
        f = lambda uu: jnp.asarray(residual(uu)).reshape(-1)

        def _solve(fn, x0):
            def body(s):
                du = jnp.linalg.solve(jax.jacfwd(fn)(s[0]), -fn(s[0]))
                return s[0] + du, jnp.linalg.norm(du), s[2] + 1

            return jax.lax.while_loop(lambda s: (s[1] > tol) & (s[2] < steps), body, (x0, jnp.array(1.0, x0.dtype), 0))[0]

        return jax.lax.custom_root(
            f, jnp.asarray(u0).reshape(-1), _solve, lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)
        )

    sigma = 5.670374419e-8
    r0, r1, r2, r3 = 0.10, 0.20, 0.25, 0.35
    k, eps1, eps2, T_hot, T_cold = 20.0, 0.8, 0.6, 1000.0, 300.0

    ring = lambda a, b: Point(0, 0).buffer(b, 16).difference(Point(0, 0).buffer(a, 16))  # noqa: E731
    d = jno.domain(ring(r0, r1).union(ring(r2, r3)), mesh_size=0.45)
    rad = lambda x, y: jnp.hypot(x, y)  # noqa: E731  (JAX-traceable: jno.fem traces tag predicates)
    d.tag("hot", lambda x, y: jnp.abs(rad(x, y) - r0) < 4e-2)
    d.tag("cold", lambda x, y: jnp.abs(rad(x, y) - r3) < 4e-2)
    d.tag("inner_gap", lambda x, y: jnp.abs(rad(x, y) - r1) < 4e-2)
    d.tag("outer_gap", lambda x, y: jnp.abs(rad(x, y) - r2) < 4e-2)

    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xh, yh, _ = d.variable("hot", split=True)
    xc, yc, _ = d.variable("cold", split=True)
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y), u(xh, yh) - T_hot, u(xc, yc) - T_cold])
    A0 = fem.operator[0]  # dense JAX array (native assembler) or BCOO -> densify either way
    A = A0.todense() if hasattr(A0, "todense") else jnp.asarray(A0)
    b = jnp.asarray(fem.operator[1]).reshape(-1)
    n = b.size

    gap = d.enclosure(["inner_gap", "outer_gap"])
    gap.check()
    F = gap.view_factor
    eps = gap.emissivity({"inner_gap": eps1, "outer_gap": eps2})
    mi, mo = gap.tag_mask("inner_gap"), gap.tag_mask("outer_gap")
    eye = jnp.eye(gap.size)
    ar = np.asarray(gap.areas)

    def q_rad(uu, eps_in=eps1):  # grey-body radiosity: q = (I - F)(I - diag(rho)F)^-1 diag(eps) sigma T^4
        e = jnp.where(jnp.asarray(mi), eps_in, eps)  # eps_in lets us differentiate w.r.t. inner emissivity
        Ts = gap.field(uu)
        J = jnp.linalg.solve(eye - (1.0 - e)[:, None] * F, e * sigma * Ts**4)
        return J - F @ J

    def coupled(eps_in):  # the whole coupled solve, as a differentiable function of inner emissivity
        return newton(lambda uu: A @ uu - b + gap.load(q_rad(uu, eps_in), size=n), jnp.linalg.solve(A, b))

    T = np.asarray(coupled(eps1))
    assert np.all(np.isfinite(T)) and T.min() > T_cold - 1 and T.max() < T_hot + 1

    Tsf = np.asarray(gap.field(jnp.asarray(T)))
    Ts1 = float((Tsf[mi] * ar[mi]).sum() / ar[mi].sum())
    Ts2 = float((Tsf[mo] * ar[mo]).sum() / ar[mo].sum())
    Q_fem = float((np.asarray(q_rad(jnp.asarray(T)))[mi] * ar[mi]).sum())

    D = 1 / eps1 + (r1 / r2) * (1 / eps2 - 1)
    from scipy.optimize import fsolve

    ts1_a, ts2_a = fsolve(
        lambda x: [
            2 * np.pi * k * (T_hot - x[0]) / np.log(r1 / r0) - 2 * np.pi * r1 * sigma * (x[0] ** 4 - x[1] ** 4) / D,
            2 * np.pi * r1 * sigma * (x[0] ** 4 - x[1] ** 4) / D - 2 * np.pi * k * (x[1] - T_cold) / np.log(r3 / r2),
        ],
        [800.0, 500.0],
    )
    Q_a = 2 * np.pi * r1 * sigma * (ts1_a**4 - ts2_a**4) / D
    assert T_hot > Ts1 > Ts2 > T_cold, "temperatures must be monotone hot>Ts1>Ts2>cold"
    assert abs(Ts1 - ts1_a) / ts1_a < 5e-3, f"Ts1 {Ts1:.1f} vs analytic {ts1_a:.1f}"
    assert abs(Ts2 - ts2_a) / ts2_a < 5e-3, f"Ts2 {Ts2:.1f} vs analytic {ts2_a:.1f}"
    assert abs(Q_fem - Q_a) / abs(Q_a) < 1e-2, f"Q {Q_fem:.0f} vs analytic {Q_a:.0f}"

    # Differentiable end to end: jax.grad of the inner-surface temperature w.r.t. its emissivity flows
    # through the whole coupled radiation+conduction solve (inverse-through-radiation), matching FD.
    def mean_inner_T(eps_in):
        return (gap.field(coupled(eps_in))[jnp.asarray(mi)] * jnp.asarray(ar)[jnp.asarray(mi)]).sum() / jnp.asarray(ar)[
            jnp.asarray(mi)
        ].sum()

    g = float(jax.grad(mean_inner_T)(eps1))
    fd = float((mean_inner_T(eps1 + 1e-3) - mean_inner_T(eps1 - 1e-3)) / 2e-3)
    assert np.isfinite(g) and abs(g - fd) / (abs(fd) + 1e-8) < 1.5e-2, f"grad {g:.3f} vs finite-diff {fd:.3f}"


def test_radiation_coupling_term_in_jno_fem_matches_analytic():
    """A user-written radiosity wrapped in ``jno.Coupling`` and passed IN the ``jno.fem([...])`` list folds
    the nonlocal grey-body load into the residual (promoting the linear conduction form to nonlinear); the
    coupled conduction+radiation system then reproduces the closed-form concentric-cylinder series solution
    -- i.e. the term assembles the same physics as the hand-rolled ``A u - b + load(q_rad(u))`` above. The
    enclosure supplies only the geometry (``field``/``view_factor``/``emissivity``/``load``); the radiosity
    is the user's, on top of it. A ``jno.np.parameter``
    (conductivity, here in the form via the operator's runtime args) stays differentiable through it, so it
    trains through ``jno.core``. (The matrix-free default solver stalls on this penalty-Dirichlet case, so
    we drive the operator's residual with a direct-solve Newton -- jno imposes no solver.)"""
    from shapely.geometry import Point

    import jno

    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from scipy.optimize import fsolve

    def newton(residual, u0, steps=60, tol=1e-9):  # BYO direct-solve Newton (custom_root -> differentiable)
        f = lambda uu: jnp.asarray(residual(uu)).reshape(-1)  # noqa: E731

        def _solve(fn, x0):
            def body(s):
                du = jnp.linalg.solve(jax.jacfwd(fn)(s[0]), -fn(s[0]))
                return s[0] + du, jnp.linalg.norm(du), s[2] + 1

            return jax.lax.while_loop(lambda s: (s[1] > tol) & (s[2] < steps), body, (x0, jnp.array(1.0, x0.dtype), 0))[0]

        return jax.lax.custom_root(
            f, jnp.asarray(u0).reshape(-1), _solve, lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)
        )

    sigma = 5.670374419e-8
    r0, r1, r2, r3 = 0.10, 0.20, 0.25, 0.35
    k0, eps1, eps2, T_hot, T_cold = 20.0, 0.8, 0.6, 1000.0, 300.0
    ring = lambda a, b: Point(0, 0).buffer(b, 16).difference(Point(0, 0).buffer(a, 16))  # noqa: E731
    d = jno.domain(ring(r0, r1).union(ring(r2, r3)), mesh_size=0.45)
    rad = lambda x, y: jnp.hypot(x, y)  # noqa: E731
    d.tag("hot", lambda x, y: jnp.abs(rad(x, y) - r0) < 4e-2)
    d.tag("cold", lambda x, y: jnp.abs(rad(x, y) - r3) < 4e-2)
    d.tag("inner_gap", lambda x, y: jnp.abs(rad(x, y) - r1) < 4e-2)
    d.tag("outer_gap", lambda x, y: jnp.abs(rad(x, y) - r2) < 4e-2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xh, yh, _ = d.variable("hot", split=True)
    xc, yc, _ = d.variable("cold", split=True)
    gap = d.enclosure(["inner_gap", "outer_gap"])
    gap.check()
    mi, mo = gap.tag_mask("inner_gap"), gap.tag_mask("outer_gap")
    ar = np.asarray(gap.areas)

    # the user writes the grey-body radiosity on top of the enclosure geometry, then wraps it in a Coupling
    F = jnp.asarray(gap.view_factor)
    eps = gap.emissivity({"inner_gap": eps1, "outer_gap": eps2})
    rho, eye, s_row = 1.0 - eps, jnp.eye(gap.size), F.sum(axis=1)

    def radiation(w):  # net grey-body surface load (n_dofs,) -- pure JAX, differentiable in w
        Tk = gap.field(w)
        J = jnp.linalg.solve(eye - rho[:, None] * F, eps * sigma * Tk**4)
        return gap.load(s_row * J - F @ J)

    # conductivity as a runtime parameter; radiation as a Coupling term in the jno.fem list
    kp = jno.np.parameter((1,), name="kcond")
    kp.initialize(jax.nn.initializers.constant(k0))
    fem = jno.fem(
        [
            kp * (ui.x * vi.x + ui.y * vi.y),
            radiation,  # the bare residual function IS the coupling -- no wrapper needed
            u(xh, yh) - T_hot,
            u(xc, yc) - T_cold,
        ]
    )
    assert fem._mode == "nonlinear", "the radiation coupling must promote the linear conduction form to nonlinear"
    op = fem.operator
    nd = int(fem.dofs)

    def coupled(kval):  # the coupled solve as a function of the conductivity parameter (via the operator's args)
        res = lambda w: jnp.asarray(op.residual(w, {"kcond": jnp.atleast_1d(kval)})).reshape(-1)  # noqa: E731
        return newton(res, jnp.full((nd,), 0.5 * (T_hot + T_cold)))

    T = np.asarray(coupled(k0))
    assert np.all(np.isfinite(T)) and T.min() > T_cold - 1 and T.max() < T_hot + 1
    Tsf = np.asarray(gap.field(jnp.asarray(T)))
    Ts1 = float((Tsf[mi] * ar[mi]).sum() / ar[mi].sum())
    Ts2 = float((Tsf[mo] * ar[mo]).sum() / ar[mo].sum())
    D = 1 / eps1 + (r1 / r2) * (1 / eps2 - 1)
    ts1_a, ts2_a = fsolve(
        lambda x: [
            2 * np.pi * k0 * (T_hot - x[0]) / np.log(r1 / r0) - 2 * np.pi * r1 * sigma * (x[0] ** 4 - x[1] ** 4) / D,
            2 * np.pi * r1 * sigma * (x[0] ** 4 - x[1] ** 4) / D - 2 * np.pi * k0 * (x[1] - T_cold) / np.log(r3 / r2),
        ],
        [800.0, 500.0],
    )
    assert T_hot > Ts1 > Ts2 > T_cold
    assert abs(Ts1 - ts1_a) / ts1_a < 1.5e-2 and abs(Ts2 - ts2_a) / ts2_a < 1.5e-2, (
        f"coupling-term Ts1 {Ts1:.1f}/{ts1_a:.1f}, Ts2 {Ts2:.1f}/{ts2_a:.1f}"
    )

    # the conductivity parameter is differentiable THROUGH the coupled (conduction+radiation) solve
    qoi = lambda kv: (gap.field(coupled(kv))[jnp.asarray(mi)]).mean()  # noqa: E731
    g = float(jax.grad(qoi)(k0))
    fd = float((qoi(k0 + 1e-2) - qoi(k0 - 1e-2)) / 2e-2)
    assert np.isfinite(g) and abs(g) > 1e-6 and abs(g - fd) / (abs(fd) + 1e-8) < 3e-2, (
        f"parameter grad through the radiation coupling {g:.4f} vs FD {fd:.4f}"
    )

    # a jitted residual is a callable *object*, not a plain function: the bare shorthand cannot see it, so
    # jno.fem must reject it with a clear "wrap it in jno.Coupling" message rather than fail obscurely later.
    with pytest.raises(TypeError, match="jno.Coupling"):
        jno.fem([kp * (ui.x * vi.x + ui.y * vi.y), jax.jit(radiation), u(xh, yh) - T_hot, u(xc, yc) - T_cold])
    # and wrapping the jitted residual explicitly is accepted (same coupling, just an object callable)
    femj = jno.fem(
        [kp * (ui.x * vi.x + ui.y * vi.y), jno.Coupling(jax.jit(radiation)), u(xh, yh) - T_hot, u(xc, yc) - T_cold]
    )
    assert femj._mode == "nonlinear"


def test_radiation_coupling_parameter_flows_and_is_differentiable():
    """A ``jno.np.parameter`` that lives only inside a coupling (here the emissivity) is invisible to the
    weak-form trace walk, so it is declared with ``jno.Coupling(fn, params=[eps])`` and read from the
    threaded ``{name: value}`` dict. This test checks the two things ``jno.core`` rides on: the parameter
    appears in the solve's runtime args (so the ``fem.solve()`` FunctionCall lists it as a trainable input),
    and the coupled solve is differentiable in it (nonzero, FD-consistent) -- i.e. it would calibrate."""
    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import Point

    import jno

    sigma = 5.670374419e-8
    r0, r1, r2, r3 = 0.10, 0.20, 0.25, 0.35
    k0, eps0, T_hot, T_cold = 20.0, 0.7, 1000.0, 300.0
    ring = lambda a, b: Point(0, 0).buffer(b, 16).difference(Point(0, 0).buffer(a, 16))  # noqa: E731
    d = jno.domain(ring(r0, r1).union(ring(r2, r3)), mesh_size=0.45)
    rad = lambda x, y: jnp.hypot(x, y)  # noqa: E731
    d.tag("hot", lambda x, y: jnp.abs(rad(x, y) - r0) < 4e-2)
    d.tag("cold", lambda x, y: jnp.abs(rad(x, y) - r3) < 4e-2)
    d.tag("inner_gap", lambda x, y: jnp.abs(rad(x, y) - r1) < 4e-2)
    d.tag("outer_gap", lambda x, y: jnp.abs(rad(x, y) - r2) < 4e-2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xh, yh, _ = d.variable("hot", split=True)
    xc, yc, _ = d.variable("cold", split=True)
    gap = d.enclosure(["inner_gap", "outer_gap"])
    gap.check()
    mi = gap.tag_mask("inner_gap")
    F = jnp.asarray(gap.view_factor)
    eye, s_row = jnp.eye(gap.size), F.sum(axis=1)

    # emissivity as a coupling-only trainable parameter, read from the threaded args dict
    eps = jno.np.parameter((1,), name="eps")
    eps.initialize(jax.nn.initializers.constant(eps0))

    def radiation(w, p):  # uniform grey emissivity p["eps"]; net radiosity load
        e = p["eps"].reshape(())
        J = jnp.linalg.solve(eye - (1.0 - e) * F, e * sigma * gap.field(w) ** 4)
        return gap.load(s_row * J - F @ J)

    fem = jno.fem(
        [
            k0 * (ui.x * vi.x + ui.y * vi.y),
            jno.Coupling(radiation, params=[eps]),
            u(xh, yh) - T_hot,
            u(xc, yc) - T_cold,
        ]
    )
    assert fem._mode == "nonlinear"
    # the coupling-only parameter reached the solve's runtime args -> it is a trainable input of fem.solve()
    assert "eps" in fem.operator.runtime_parameter_exprs, "coupling param did not flow into the solve args"

    op, nd = fem.operator, int(fem.dofs)

    def coupled(eps_val):  # direct-solve Newton (penalty-Dirichlet stalls the matrix-free default)
        res = lambda w: jnp.asarray(op.residual(w, {"eps": jnp.atleast_1d(eps_val)})).reshape(-1)  # noqa: E731

        def _solve(fn, x0):
            def body(s):
                du = jnp.linalg.solve(jax.jacfwd(fn)(s[0]), -fn(s[0]))
                return s[0] + du, jnp.linalg.norm(du), s[2] + 1

            return jax.lax.while_loop(lambda s: (s[1] > 1e-9) & (s[2] < 60), body, (x0, jnp.array(1.0), 0))[0]

        return jax.lax.custom_root(
            res,
            jnp.full((nd,), 0.5 * (T_hot + T_cold)),
            _solve,
            lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y),
        )

    qoi = lambda e: (gap.field(coupled(e))[jnp.asarray(mi)]).mean()  # noqa: E731
    g = float(jax.grad(qoi)(eps0))
    fd = float((qoi(eps0 + 1e-3) - qoi(eps0 - 1e-3)) / 2e-3)
    # more emissive inner wall radiates more across the gap -> cooler inner surface: grad must be real, < 0
    assert np.isfinite(g) and g < -1e-6 and abs(g - fd) / (abs(fd) + 1e-8) < 3e-2, (
        f"coupling-parameter grad through the solve {g:.3f} vs FD {fd:.3f}"
    )


def _newton_direct(res_fn, x0, steps=60, tol=1e-9):
    """BYO direct-solve Newton (matrix-free default stalls on penalty-Dirichlet)."""
    f = lambda x: jnp.asarray(res_fn(x)).reshape(-1)  # noqa: E731

    def _solve(fn, y0):
        def body(s):
            du = jnp.linalg.solve(jax.jacfwd(fn)(s[0]), -fn(s[0]))
            return s[0] + du, jnp.linalg.norm(du), s[2] + 1

        return jax.lax.while_loop(lambda s: (s[1] > tol) & (s[2] < steps), body, (y0, jnp.array(1.0), 0))[0]

    return jax.lax.custom_root(
        f, jnp.asarray(x0).reshape(-1), _solve, lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)
    )


def test_radiation_coupling_targets_one_field_in_a_multifield_system():
    """A coupling with ``field_key=`` acts on *one* field's DOF block. Build a two-scalar-field system --
    field T (conduction + a gap-radiation coupling) and an independent Laplace field w -- and check the
    coupling lands only in T's block: the multifield T-block reproduces the standalone coupled-radiation
    solve, while the w-block reproduces its own radiation-free Laplace solve (the coupling did not leak)."""
    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import Point

    import jno

    sigma = 5.670374419e-8
    r0, r1, r2, r3 = 0.10, 0.20, 0.25, 0.35
    k0, eps0, T_hot, T_cold, w_hot, w_cold = 20.0, 0.7, 1000.0, 300.0, 5.0, 1.0
    ring = lambda a, b: Point(0, 0).buffer(b, 16).difference(Point(0, 0).buffer(a, 16))  # noqa: E731
    d = jno.domain(ring(r0, r1).union(ring(r2, r3)), mesh_size=0.45)
    rad = lambda x, y: jnp.hypot(x, y)  # noqa: E731
    d.tag("hot", lambda x, y: jnp.abs(rad(x, y) - r0) < 4e-2)
    d.tag("cold", lambda x, y: jnp.abs(rad(x, y) - r3) < 4e-2)
    d.tag("inner_gap", lambda x, y: jnp.abs(rad(x, y) - r1) < 4e-2)
    d.tag("outer_gap", lambda x, y: jnp.abs(rad(x, y) - r2) < 4e-2)
    xi, yi, _ = d.variable("interior", split=True)
    xh, yh, _ = d.variable("hot", split=True)
    xc, yc, _ = d.variable("cold", split=True)
    gap = d.enclosure(["inner_gap", "outer_gap"])
    gap.check()
    F = jnp.asarray(gap.view_factor)
    eye, s_row = jnp.eye(gap.size), F.sum(axis=1)

    def radiation(wT):  # net grey-body load on the T field's nodes (same code single- or multi-field)
        Tk = gap.field(wT)
        J = jnp.linalg.solve(eye - (1.0 - eps0) * F, eps0 * sigma * Tk**4)
        return gap.load(s_row * J - F @ J)

    # --- single-field references -----------------------------------------------------------------
    T, sT = d.fem_symbols(names=("T", "sT"))
    Ti, sTi = T.bind(x=xi, y=yi), sT.bind(x=xi, y=yi)
    femT = jno.fem([k0 * (Ti.x * sTi.x + Ti.y * sTi.y), radiation, T(xh, yh) - T_hot, T(xc, yc) - T_cold])
    T_ref = np.asarray(_newton_direct(lambda u: femT.operator.residual(u, {}), jnp.full((int(femT.dofs),), 650.0)))

    w, sw = d.fem_symbols(names=("w", "sw"))
    wi, swi = w.bind(x=xi, y=yi), sw.bind(x=xi, y=yi)
    femW = jno.fem([wi.x * swi.x + wi.y * swi.y, w(xh, yh) - w_hot, w(xc, yc) - w_cold])
    w_ref = np.asarray(femW.solve())  # linear, radiation-free

    # --- the coupled two-field system: radiation targets the T block via field_key ---------------
    Tkey = getattr(T, "field_key", None)
    fem = jno.fem(
        [
            k0 * (Ti.x * sTi.x + Ti.y * sTi.y),
            jno.Coupling(radiation, field_key=Tkey),  # only T's block
            wi.x * swi.x + wi.y * swi.y,
            T(xh, yh) - T_hot,
            T(xc, yc) - T_cold,
            w(xh, yh) - w_hot,
            w(xc, yc) - w_cold,
        ]
    )
    assert fem._mode == "nonlinear"
    fk = list(getattr(d, "_fem_native_field_keys"))
    offs = list(fem.offsets)
    iT, iw = fk.index(Tkey), fk.index(getattr(w, "field_key", None))
    nd = int(fem.dofs)
    x0 = jnp.zeros((nd,)).at[offs[iT] : offs[iT + 1]].set(650.0).at[offs[iw] : offs[iw + 1]].set(3.0)
    sol = np.asarray(_newton_direct(lambda u: fem.operator.residual(u, {}), x0))
    T_blk = sol[offs[iT] : offs[iT + 1]]
    w_blk = sol[offs[iw] : offs[iw + 1]]

    # the coupling acted on T (matches the standalone coupled solve) and NOT on w (matches plain Laplace)
    assert np.allclose(T_blk, T_ref, rtol=2e-3, atol=1.0), f"T block off coupled ref: max {np.abs(T_blk - T_ref).max():.2f}"
    assert np.allclose(w_blk, w_ref, rtol=2e-3, atol=1e-3), (
        f"w block perturbed by coupling: max {np.abs(w_blk - w_ref).max():.3e}"
    )
    # and the radiation genuinely did something to T (so the match above is not a no-op)
    assert np.abs(T_blk - T_blk.mean()).max() > 50.0

    # a field_key that names no field fails loud rather than silently mis-placing the load
    with pytest.raises(ValueError, match="not among the fields"):
        jno.fem(
            [
                k0 * (Ti.x * sTi.x + Ti.y * sTi.y),
                jno.Coupling(radiation, field_key="nope"),
                wi.x * swi.x + wi.y * swi.y,
                T(xh, yh) - T_hot,
                T(xc, yc) - T_cold,
                w(xh, yh) - w_hot,
                w(xc, yc) - w_cold,
            ]
        )


def test_radiation_coupling_in_a_transient_solve_reaches_the_steady_coupled_state():
    """A coupling enters each implicit time step (the transient block, linear here, is promoted to a
    nonlinear backward-Euler block whose residual carries the coupling). The strongest check that this is
    correct: a transient radiation-coupled heat solve, marched by the default fem.solve() integrator, must
    relax to the *steady* radiation-coupled solution (at u_t -> 0 the step residual is exactly the steady
    coupled residual). Validates that transient coupling runs end-to-end and is physically consistent."""
    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import Point

    import jno

    sigma = 5.670374419e-8
    r0, r1, r2, r3 = 0.10, 0.20, 0.25, 0.35
    k0, eps0, T_hot, T_cold = 20.0, 0.7, 1000.0, 300.0
    ring = lambda a, b: Point(0, 0).buffer(b, 16).difference(Point(0, 0).buffer(a, 16))  # noqa: E731
    d = jno.domain(ring(r0, r1).union(ring(r2, r3)), mesh_size=0.45, time=(0.0, 0.05, 26))
    rad = lambda x, y: jnp.hypot(x, y)  # noqa: E731
    d.tag("hot", lambda x, y: jnp.abs(rad(x, y) - r0) < 4e-2)
    d.tag("cold", lambda x, y: jnp.abs(rad(x, y) - r3) < 4e-2)
    d.tag("inner_gap", lambda x, y: jnp.abs(rad(x, y) - r1) < 4e-2)
    d.tag("outer_gap", lambda x, y: jnp.abs(rad(x, y) - r2) < 4e-2)
    xi, yi, ti = d.variable("interior", split=True)
    xh, yh, _ = d.variable("hot", split=True)
    xc, yc, _ = d.variable("cold", split=True)
    ci = d.variable("initial", split=True)
    gap = d.enclosure(["inner_gap", "outer_gap"])
    gap.check()
    F = jnp.asarray(gap.view_factor)
    eye, s_row = jnp.eye(gap.size), F.sum(axis=1)

    def radiation(wT):  # net grey-body load on the temperature nodes (identical steady or transient)
        Tk = gap.field(wT)
        J = jnp.linalg.solve(eye - (1.0 - eps0) * F, eps0 * sigma * Tk**4)
        return gap.load(s_row * J - F @ J)

    # steady coupled reference (no time derivative -> steady form), direct-solve Newton
    Tr, sTr = d.fem_symbols(names=("Tr", "sTr"))
    Tri, sTri = Tr.bind(x=xi, y=yi), sTr.bind(x=xi, y=yi)
    femS = jno.fem([k0 * (Tri.x * sTri.x + Tri.y * sTri.y), radiation, Tr(xh, yh) - T_hot, Tr(xc, yc) - T_cold])
    T_steady = np.asarray(_newton_direct(lambda u: femS.operator.residual(u, {}), jnp.full((int(femS.dofs),), 650.0)))

    # transient coupled solve: mass term u_t*v + same conduction + same radiation coupling; IC at T_cold
    T, sT = d.fem_symbols(names=("T", "sT"))
    Ti, sTi = T.bind(x=xi, y=yi, t=ti), sT.bind(x=xi, y=yi, t=ti)
    femT = jno.fem(
        [
            Ti.t * sTi + k0 * (Ti.x * sTi.x + Ti.y * sTi.y),
            radiation,
            T(xh, yh) - T_hot,
            T(xc, yc) - T_cold,
            T(ci[0], ci[1]) - T_cold,
        ]
    )
    assert femT._mode == "transient", "the coupled block must stay a transient block"
    # non-parametric transient solve() is a FunctionCall trace node; .fn() evaluates the trajectory
    traj = np.asarray(femT.solve().fn())  # (n_save, n_dofs) from the default backward-Euler integrator
    T_final = traj[-1]

    # the marched solution left the cold initial state and relaxed onto the steady coupled solution
    assert T_final.max() > T_cold + 100.0, "transient never heated up"
    rel = np.abs(T_final - T_steady).max() / (np.abs(T_steady).max() + 1e-9)
    assert rel < 2e-2, f"transient end state did not reach the steady coupled solution: rel {rel:.3f}"


def _axisym_disc_enclosure(R, n, z=0.0):
    """A bare radial (disc) surface as enclosure elements: r from 0 to R at height z."""
    from jno.domain.enclosure import Enclosure

    r = np.linspace(0.0, R, n + 1)
    e0 = np.c_[r[:-1], np.full(n, z)]
    e1 = np.c_[r[1:], np.full(n, z)]
    elements = np.c_[np.arange(n), np.arange(1, n + 1)]
    areas = 2 * np.pi * (0.5 * (e0[:, 0] + e1[:, 0])) * np.linalg.norm(e1 - e0, axis=1)
    normals = np.tile([0.0, 1.0], (n, 1))
    return Enclosure(
        domain=None,
        tags=["disc"],
        F=np.zeros((n, n)),
        elements=elements,
        element_tags=np.array(["disc"] * n, dtype=object),
        areas=areas,
        normals=normals,
        midpoints=0.5 * (e0 + e1),
        axisymmetric=True,
        endpoints=(e0, e1),
    ), r


def test_enclosure_load_axisymmetric_ring_measure_is_exact_for_a_linear_field():
    """``Enclosure.load`` must scatter ``∫_Γ q v (2πr) ds``, not the 2-D ``q·area/2`` half-and-half split.

    Oracle: for constant ``q`` on a disc of radius ``R``, the consistent load tested against the linear
    nodal field ``v(r) = r`` must reproduce ``∫_0^R q·r·2πr dr = 2πqR³/3`` EXACTLY (the ring weights
    integrate a linear test function exactly, element by element). The equal split cannot — it puts too
    much weight on the inner node of every radial element.
    """
    R, n, q0 = 0.7, 5, 3.0
    gap, r = _axisym_disc_enclosure(R, n)
    q = jnp.full(n, q0)

    load = np.asarray(gap.load(q, size=n + 1))
    exact = 2 * np.pi * q0 * R**3 / 3.0
    assert abs(float(load @ r) - exact) < 1e-12 * exact, f"ring load {load @ r:.9f} vs exact {exact:.9f}"

    # total load is still the ring area x flux (the split redistributes, it does not create or destroy)
    assert abs(load.sum() - q0 * np.pi * R**2) < 1e-12 * q0 * np.pi * R**2

    # ...and the 2-D half-and-half split, which is what the code did before, is measurably wrong here
    half = np.zeros(n + 1)
    A = np.asarray(gap.areas)
    np.add.at(half, np.arange(n), q0 * A * 0.5)
    np.add.at(half, np.arange(1, n + 1), q0 * A * 0.5)
    assert abs(float(half @ r) - exact) > 1e-3 * exact, "premise broken: the equal split should NOT be exact"


def test_enclosure_load_axisymmetric_weights_reduce_to_halves_on_a_cylindrical_wall():
    """A wall at constant radius has Δr = 0, so the ring weights must collapse back to the even split —
    the radius-weighting is a correction for RADIAL extent only, not a global rescale."""
    from jno.domain.enclosure import _consistent_node_weights

    r_wall, H, n = 0.4, 1.0, 6
    z = np.linspace(0.0, H, n + 1)
    e0 = np.c_[np.full(n, r_wall), z[:-1]]
    e1 = np.c_[np.full(n, r_wall), z[1:]]
    areas = 2 * np.pi * r_wall * np.linalg.norm(e1 - e0, axis=1)

    w = _consistent_node_weights(areas, (e0, e1), axisymmetric=True)
    assert np.allclose(w[:, 0], w[:, 1]), "constant-radius elements must split evenly"
    assert np.allclose(w.sum(axis=1), areas), "weights must sum to the ring area"


def test_enclosure_load_2d_is_unchanged_by_the_ring_measure_work():
    """Regression: the 2-D path must keep the plain half-and-half split (edge length, no 2πr)."""
    from jno.domain.enclosure import _consistent_node_weights

    areas = np.array([0.3, 1.7, 0.05])
    w = _consistent_node_weights(areas, None, axisymmetric=False)
    assert np.allclose(w, np.stack([areas * 0.5, areas * 0.5], axis=1))


def test_axisymmetric_boundary_mode_closure_when_nothing_occludes():
    """A CLOSED convex cavity built the plain way (``d.enclosure(tags)``, no ``medium_tags``) must close.

    The near-field refinement used to be gated to interface mode outright, because it re-derived its
    occluders from ``domain._source_regions`` keyed by the element tags — a lookup that comes back empty
    in boundary mode, silently recomputing every refined pair as fully visible. The occluder model is now
    passed in explicitly, so boundary mode can run the refinement exactly where "fully visible" is the
    truth: an unobstructed enclosure. Without it, row sums here sit ~0.87 (the r_min floor destroying
    energy) or ~2.3 (no floor, the corner overshoot creating it).
    """
    import jno

    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import box

    R, H = 0.10, 0.10
    d = jno.domain(box(0.0, 0.0, R, H), mesh_size=0.012)
    d.tag("bottom", lambda x, y: jnp.abs(y) < 1e-9)
    d.tag("side", lambda x, y: jnp.abs(x - R) < 1e-9)
    d.tag("top", lambda x, y: jnp.abs(y - H) < 1e-9)

    # occlude=False is the caller asserting "nothing blocks any ray" — true for this convex cavity, and
    # what lets the refinement run in boundary mode (there are no solid polygons to trace against here,
    # and the meridian visibility test is blind to axisymmetric self-occlusion, so it cannot be inferred).
    gap = d.enclosure(["bottom", "side", "top"], axisymmetric=True, inward=True, occlude=False)
    rows = np.asarray(gap.view_factor).sum(axis=1)
    assert rows.max() < 1.02, f"closed cavity: row sums must not exceed 1, got max {rows.max():.4f}"
    assert rows.min() > 0.98, f"closed cavity: row sums must not fall below 1, got min {rows.min():.4f}"

    # the analytic coaxial-disk factor still comes out of the same F (Modest Ch. 4 / App. D)
    F, A, tags = np.asarray(gap.view_factor), np.asarray(gap.areas), np.asarray(gap.element_tags)
    bot, top = tags == "bottom", tags == "top"
    f_bt = float((A[bot, None] * F[np.ix_(bot, top)]).sum() / A[bot].sum())
    Rr = R / H
    S = 1 + (1 + Rr**2) / Rr**2
    assert abs(f_bt - 0.5 * (S - np.sqrt(S**2 - 4))) < 1e-2, f"bottom->top disk factor {f_bt:.4f}"


def test_refine_near_pairs_requires_its_occluder_model():
    """Passing the occluders explicitly is what stops the refinement from fabricating lines of sight.

    Concentric cylinders: the outer wall's self-view is occlusion-limited to ``1 - r1/r2``. Refining with
    the real solid keeps it; refining with the old empty-lookup fallback (a domain with no
    ``_source_regions``) inflates it and pushes the row sum well past 1.
    """
    from types import SimpleNamespace

    from shapely.geometry import box

    from jno.domain.enclosure import _refine_near_pairs, _solid_polygon_visibility_3d

    r1, r2, H, NZ, n_phi = 0.8, 1.0, 12.0, 48, 64
    z = np.linspace(0.0, H, NZ + 1)
    e0 = np.vstack([np.c_[np.full(NZ, r1), z[:-1]], np.c_[np.full(NZ, r2), z[:-1]]])
    e1 = np.vstack([np.c_[np.full(NZ, r1), z[1:]], np.c_[np.full(NZ, r2), z[1:]]])
    nrm = np.vstack([np.tile([1.0, 0.0], (NZ, 1)), np.tile([-1.0, 0.0], (NZ, 1))])
    mids, length = 0.5 * (e0 + e1), np.linalg.norm(e1 - e0, axis=1)
    m = e0.shape[0]

    solid = box(0.0, 0.0, r1, H)
    dom = SimpleNamespace(_source_regions={"inner": solid})
    tags = np.array(["inner"] * m, dtype=object)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    vm = _solid_polygon_visibility_3d(dom, tags, mids, length, phi)
    F0 = np.asarray(
        MeshUtils.get_view_factor_axisymmetric_element(
            jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(nrm), jnp.asarray(vm), n_quad=3, n_phi=n_phi, r_min=0.0
        )
    )
    mid_o, outer = NZ + NZ // 2, slice(NZ, 2 * NZ)

    good = _refine_near_pairs(F0, e0, e1, nrm, dom, tags, n_phi, occluders=([solid], solid))
    assert abs(good[mid_o, outer].sum() - (1 - r1 / r2)) < 3e-2, (
        f"with its occluders, F22 should be {1 - r1 / r2:.3f}, got {good[mid_o, outer].sum():.3f}"
    )

    # the old behaviour: no occluders found -> refined pairs invented as fully visible
    blind = _refine_near_pairs(F0, e0, e1, nrm, SimpleNamespace(_source_regions={}), tags, n_phi)
    assert blind[mid_o, outer].sum() > (1 - r1 / r2) + 0.2, "premise broken: the blind fallback should inflate F22"
    assert blind[mid_o].sum() > 1.05, "premise broken: the blind fallback should break the physical row bound"


def _chords_cross_solid(vis, mids, phi, solids, n_sample=200, frac=0.02):
    """Independent audit of a visibility array: of the pairs it calls VISIBLE, how many have a 3-D
    chord that actually passes through solid material?

    Shares no code with the occlusion test — it samples the chord, maps it to the meridian via
    ``rho(t) = |(1-t)P_i + t P_j|_xy``, and asks shapely whether those points are inside a solid.
    """
    import shapely
    from shapely.ops import unary_union

    uni = unary_union(list(solids)).buffer(-2e-4)  # erode: endpoints legitimately sit ON a surface
    rng = np.random.default_rng(0)
    m = mids.shape[0]
    t = np.linspace(0.0, 1.0, 240)[:, None]
    leaked = checked = 0
    for _ in range(n_sample):
        i, j, k = (int(rng.integers(m)), int(rng.integers(m)), int(rng.integers(len(phi))))
        if not vis[i, j, k]:
            continue
        checked += 1
        Pi = np.array([mids[i, 0], 0.0, mids[i, 1]])
        Pj = np.array([mids[j, 0] * np.cos(phi[k]), mids[j, 0] * np.sin(phi[k]), mids[j, 1]])
        Q = Pi[None, :] * (1 - t) + Pj[None, :] * t
        if shapely.contains_xy(uni, np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2]).mean() > frac:
            leaked += 1
    return leaked, checked


def test_visibility_occluders_must_be_passed_not_inferred_from_tags():
    """``_solid_polygon_visibility_3d`` infers its occluders from the SET of element tags unless they
    are passed. A solid that owns no radiating element — or whose elements were tagged under another
    name — is then silently transparent, and the resulting F is plausible but wrong.

    Here a baffle splits a cavity in two. Tag the elements only by the outer shell (as a caller
    lumping several parts under one name would), and the baffle stops blocking anything: chords sail
    straight through it. Passing ``occluders=`` restores it.
    """
    from types import SimpleNamespace

    from shapely.geometry import box

    from jno.domain.enclosure import _solid_polygon_visibility_3d

    R, H, RB, ZB, TB = 0.10, 0.20, 0.045, 0.10, 0.006
    baffle = box(RB, ZB - TB / 2, R, ZB + TB / 2)
    shell = box(R, 0.0, R + 0.01, H)  # a wall that owns the tag the elements carry
    dom = SimpleNamespace(_source_regions={"baffle": baffle, "shell": shell})

    # elements on the cavity wall, above and below the baffle
    z = np.concatenate([np.linspace(0.01, ZB - TB, 8), np.linspace(ZB + TB, H - 0.01, 8)])
    mids = np.c_[np.full(z.size, R - 1e-6), z]
    length = np.full(z.size, 0.01)
    phi = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    tags = np.array(["shell"] * z.size, dtype=object)  # baffle owns no element -> never an occluder

    inferred = _solid_polygon_visibility_3d(dom, tags, mids, length, phi)
    passed = _solid_polygon_visibility_3d(dom, tags, mids, length, phi, occluders=[baffle, shell])

    lo_hi = np.ix_(np.arange(8), np.arange(8, 16), [0])  # below-baffle -> above-baffle, same meridian
    assert inferred[lo_hi].all(), (
        "premise broken: with the baffle missing from the tag-derived occluder set, every "
        "across-the-baffle pair should come back visible"
    )
    assert not passed[lo_hi].any(), (
        "with occluders passed explicitly the baffle must block every across-the-baffle chord at phi=0"
    )

    # and the audit agrees: the inferred set leaks, the explicit one does not
    leak_i, n_i = _chords_cross_solid(inferred, mids, phi, [baffle, shell])
    leak_p, n_p = _chords_cross_solid(passed, mids, phi, [baffle, shell])
    assert leak_i > 0, f"expected leaks from the inferred occluder set (checked {n_i})"
    assert leak_p == 0, f"explicit occluders must not leak, got {leak_p} of {n_p}"


def test_build_enclosure_occludes_with_every_solid_not_just_tagged_ones():
    """End-to-end: ``d.enclosure(..., medium_tags=[...])`` must treat EVERY non-medium region as
    opaque, including one that carries no radiating surface of its own."""
    import jno

    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import box

    from jno.domain.enclosure import _solid_polygon_visibility_3d  # noqa: F401  (documents the path)

    R, H = 0.05, 0.12
    inner = box(0.0, 0.0, R, H)  # the transparent medium
    baffle = box(0.02, 0.055, R, 0.065)  # an opaque shelf with no radiating tag of its own
    outer = box(R, 0.0, R + 0.01, H)
    d = jno.domain({"gas": inner.difference(baffle), "baffle": baffle, "wall": outer}, mesh_size=0.01)
    gap = d.enclosure(["wall"], medium_tags=["gas"], axisymmetric=True, n_phi=16)

    # the baffle is not a 'wall' element owner, so a tag-derived occluder set would omit it entirely
    mids = np.asarray(gap.midpoints)
    below = mids[:, 1] < 0.05
    above = mids[:, 1] > 0.07
    assert below.any() and above.any(), "need elements on both sides of the baffle"
    F = np.asarray(gap.view_factor)
    across = F[np.ix_(np.flatnonzero(below), np.flatnonzero(above))]
    assert across.max() < 0.5 * F.max(), (
        f"the baffle must attenuate across-shelf exchange; got max {across.max():.4f} vs overall {F.max():.4f}"
    )


def _monte_carlo_row(e0, e1, nrm, cavity, src, n_rays=60_000, step=1e-4, seed=5):
    """Brute-force the view-factor row of one element by tracing diffuse rays in true 3-D.

    Shares no code with the view-factor machinery: it samples the source ring uniformly in AREA
    (density proportional to r), fires cosine-weighted directions, marches each ray until it leaves
    the cavity polygon, and counts where it lands. Those hit fractions ARE F_ij by definition
    (Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4).
    """
    import shapely

    rng = np.random.default_rng(seed)
    (ra, za), (rb, zb) = e0[src], e1[src]
    dr = rb - ra
    u = rng.random(n_rays)
    s = u if abs(dr) < 1e-14 else (-ra + np.sqrt(ra**2 + 2 * dr * u * (ra + 0.5 * dr))) / dr
    P = np.c_[ra + s * dr, np.zeros(n_rays), za + s * (zb - za)]

    nv = np.array([nrm[src, 0], 0.0, nrm[src, 1]])
    nv /= np.linalg.norm(nv)
    t1 = np.cross(nv, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(nv, t1)
    u1, u2 = rng.random(n_rays), rng.random(n_rays)
    ct, st, ps = np.sqrt(u1), np.sqrt(1 - u1), 2 * np.pi * u2
    D = (st * np.cos(ps))[:, None] * t1 + (st * np.sin(ps))[:, None] * t2 + ct[:, None] * nv
    P = P + 1e-9 * D

    span = float(np.hypot(np.ptp(e0[:, 0]) * 2, np.ptp(e0[:, 1]))) * 2.5
    t_hit = np.full(n_rays, np.nan)
    alive = np.ones(n_rays, bool)
    t = np.zeros(n_rays)
    for _ in range(int(span / step)):
        if not alive.any():
            break
        tt = t + step
        Q = P + tt[:, None] * D
        left = alive & ~shapely.contains_xy(cavity, np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2])
        t_hit[left] = tt[left]
        alive &= ~left
        t = np.where(alive, tt, t)
    lo = np.where(np.isnan(t_hit), 0.0, t_hit - step)
    hi = np.where(np.isnan(t_hit), 0.0, t_hit)
    ok = ~np.isnan(t_hit)
    for _ in range(28):
        md = 0.5 * (lo + hi)
        Q = P + md[:, None] * D
        ins = shapely.contains_xy(cavity, np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2])
        lo, hi = np.where(ins, md, lo), np.where(ins, hi, md)
    Q = P + (0.5 * (lo + hi))[:, None] * D
    X = np.c_[np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2]]

    seg = e1 - e0
    L2 = np.maximum((seg**2).sum(1), 1e-30)
    tt = np.clip(((X[:, None, :] - e0[None]) * seg[None]).sum(-1) / L2[None], 0, 1)
    d = np.linalg.norm(X[:, None, :] - (e0[None] + tt[:, :, None] * seg[None]), axis=-1)
    land = np.where(ok, np.argmin(d, axis=1), -1)
    good = land >= 0
    return np.bincount(land[good], minlength=e0.shape[0]) / good.sum(), int(good.sum())


@pytest.mark.slow
def test_axisymmetric_view_factors_match_a_monte_carlo_ray_trace():
    """Independent oracle for a case with NO closed form: a cavity with an annular baffle.

    Monte-Carlo ray tracing shares none of the view-factor code — not the azimuthal quadrature, not
    the sphere-traced occlusion, not the near-field refinement. Agreement to the MC noise floor is
    therefore a real check on the assembled ``F``, not a restatement of it. Locks both the occluded
    zeros (the baffle's shadow) and the concave self-view.
    """
    from types import SimpleNamespace

    from shapely.geometry import Polygon, box

    from jno.domain.enclosure import _refine_near_pairs, _solid_polygon_visibility_3d

    R, HH, RB, ZB, TB = 1.0, 2.0, 0.45, 1.0, 0.06
    baffle = box(RB, ZB - TB / 2, R, ZB + TB / 2)
    cavity = Polygon(
        [(0, 0), (R, 0), (R, ZB - TB / 2), (RB, ZB - TB / 2), (RB, ZB + TB / 2), (R, ZB + TB / 2), (R, HH), (0, HH)]
    )

    def wall(p0, p1, n, normal):
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        P = np.asarray(p0)[None, :] + t * (np.asarray(p1) - np.asarray(p0))[None, :]
        return P[:-1], P[1:], np.tile(np.asarray(normal, float), (n, 1))

    parts = [
        wall((1e-6, 0), (R, 0), 10, (0, 1)),
        wall((R, 0), (R, ZB - TB / 2), 10, (-1, 0)),
        wall((R, ZB - TB / 2), (RB, ZB - TB / 2), 6, (0, -1)),
        wall((RB, ZB - TB / 2), (RB, ZB + TB / 2), 2, (-1, 0)),
        wall((RB, ZB + TB / 2), (R, ZB + TB / 2), 6, (0, 1)),
        wall((R, ZB + TB / 2), (R, HH), 10, (-1, 0)),
        wall((R, HH), (1e-6, HH), 10, (0, -1)),
    ]
    e0 = np.vstack([p[0] for p in parts])
    e1 = np.vstack([p[1] for p in parts])
    nrm = np.vstack([p[2] for p in parts])
    mids, length, m = 0.5 * (e0 + e1), np.linalg.norm(e1 - e0, axis=1), e0.shape[0]

    n_phi = 64
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    dom = SimpleNamespace(_source_regions={"baffle": baffle})
    tags = np.array(["baffle"] * m, dtype=object)
    vis = _solid_polygon_visibility_3d(dom, tags, mids, length, phi, occluders=[baffle])
    F = np.asarray(
        MeshUtils.get_view_factor_axisymmetric_element(
            jnp.asarray(e0),
            jnp.asarray(e1),
            jnp.asarray(nrm),
            jnp.asarray(vis.astype(float)),
            n_quad=3,
            n_phi=n_phi,
            r_min=0.0,
        )
    )
    F = _refine_near_pairs(F, e0, e1, nrm, dom, tags, n_phi, occluders=([baffle], baffle))

    rows = F.sum(axis=1)
    assert rows.min() > 0.97 and rows.max() < 1.03, f"closure: row sums {rows.min():.4f}..{rows.max():.4f}"

    for src in (4, len(e0) - 5):  # a floor element and a ceiling element
        mc, used = _monte_carlo_row(e0, e1, nrm, cavity, src)
        big = F[src] > 0.01
        err = np.abs(mc - F[src])
        noise = np.sqrt(0.05 / used)  # 1-sigma on a bin holding ~5% of the rays
        assert err[big].max() < 12 * noise, (
            f"element {src}: max |F_MC - F_jNO| = {err[big].max():.5f} over F>0.01 "
            f"(MC 1-sigma ~ {noise:.5f}); MC row sum {mc.sum():.4f} vs {F[src].sum():.4f}"
        )
        # the baffle's shadow must be shadow in BOTH: nothing the trace reaches may have F == 0
        assert mc[F[src] < 1e-12].sum() < 5e-3, (
            f"element {src}: rays landed on elements the view factor calls unreachable "
            f"({mc[F[src] < 1e-12].sum():.4f} of the energy)"
        )


def test_axisymmetric_kernel_row_chunking_matches_the_unblocked_build():
    """The ring kernel builds its ``(M, M, n_phi)`` intermediate in receiver row-blocks so a fine mesh
    does not need it all at once (816 MB at M=1785, n_phi=32 — an OOM on an 8 GB GPU). The blocking
    must be pure bookkeeping: every element sees the same arithmetic either way.

    Not *bit*-identical, though: the azimuthal ``sum(..., axis=-1)`` runs over a differently-shaped
    array once it is blocked, and XLA reassociates the reduction accordingly. Measured at 3 ULP
    (max relative 4.1e-16), which is rounding, not a change of result — so the tolerance here is a
    few ULP rather than exact equality. Checked by forcing several block sizes through the public
    entry point and comparing to a run whose block covers the whole array.
    """
    from unittest import mock

    r1, r2, H, NZ, n_phi = 0.25, 0.5, 1.0, 14, 24
    z = np.linspace(0.0, H, NZ + 1)
    e0 = np.vstack([np.c_[np.full(NZ, r1), z[:-1]], np.c_[np.full(NZ, r2), z[:-1]]])
    e1 = np.vstack([np.c_[np.full(NZ, r1), z[1:]], np.c_[np.full(NZ, r2), z[1:]]])
    nrm = np.vstack([np.tile([1.0, 0.0], (NZ, 1)), np.tile([-1.0, 0.0], (NZ, 1))])
    m = e0.shape[0]

    def run(vm):
        return np.asarray(
            MeshUtils.get_view_factor_axisymmetric_element(
                jnp.asarray(e0),
                jnp.asarray(e1),
                jnp.asarray(nrm),
                jnp.asarray(vm),
                n_quad=3,
                n_phi=n_phi,
                r_min=0.0,
            )
        )

    for vm in (np.ones((m, m)), np.ones((m, m, n_phi))):  # both the 2-D and per-azimuth paths
        ref = run(vm)
        for cap in (2**24, 4096, 512):  # forces 1, several, many blocks
            with mock.patch.object(MeshUtils, "_kernel_block_doubles", cap, create=True):
                got = run(vm)
            # A few ULP of the reference magnitude — reduction reassociation, nothing more.
            assert np.allclose(got, ref, rtol=8e-16, atol=0.0), (
                f"chunking changed the result beyond rounding (cap={cap}, vm.ndim={vm.ndim}, "
                f"max rel {np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-300):.2e})"
            )
        assert np.all(np.isfinite(ref))
