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

    # The r_min kwarg must be threaded to the kernel: a large near-field floor softens the 1/R^2 kernel
    # and measurably shrinks every view factor (vs the default floor) — proving it is wired through.
    soft = np.asarray(d.enclosure(["inner_gap", "outer_gap"], axisymmetric=True, r_min=0.5).view_factor)
    assert soft.max() < 0.8 * F.max(), f"explicit r_min must soften F (got soft {soft.max():.3f} vs {F.max():.3f})"


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
