"""Solving on quadrilateral and hexahedral meshes: rates, exactness, and what still refuses.

Three levels of evidence, weakest to strongest:

1. **It runs and matches the simplex mesh.** Necessary, and almost worthless on its own — a wrong
   Jacobian still produces a plausible-looking field.
2. **Convergence rates.** Q1 must recover O(h²); a rate is what catches a quadrature degree that is
   too low or a geometry map that is subtly wrong, because both leave the answer *converging to the
   wrong thing* or converging too slowly rather than obviously broken.
3. **The patch test on a DISTORTED mesh.** The decisive one. A linear field must be reproduced
   exactly, and the mesh is deliberately non-parallelogram so that the Jacobian genuinely varies
   within each cell. An assembler that formed one Jacobian per cell — which is all a simplex ever
   needs — passes every test above this line and fails this one.

Also pinned: what is *not* supported refuses by name. Surface integrals need per-quadrature-point
facet geometry (a hex face is bilinear), and order > 1 needs a non-barycentric node placement.
Both raise rather than approximating, because the shape mismatch they used to produce said nothing.
"""

from __future__ import annotations

import jax
import meshio
import numpy as np
import pytest

import jno
from jno.domain.geometries import Geometries

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _poisson_2d(ctor):
    """-Δu = 2π² sin(πx) sin(πy) on the unit square, u = 0 on ∂Ω. Returns the discrete L2 error."""
    d = jno.domain(constructor=ctor, compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = 2 * PI**2 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])
    sol = np.asarray(fem.solve()).ravel()
    pts = np.asarray(d._fem_native_dof_points)
    exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    return float(np.sqrt(np.mean((sol - exact) ** 2)))


def _poisson_3d(ctor):
    d = jno.domain(constructor=ctor, compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    f = 3 * PI**2 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi) * jno.np.sin(PI * zi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - f * vi, u(xb, yb, zb) - 0.0])
    sol = np.asarray(fem.solve()).ravel()
    pts = np.asarray(d._fem_native_dof_points)
    exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1]) * np.sin(PI * pts[:, 2])
    return float(np.sqrt(np.mean((sol - exact) ** 2)))


def _rates(errs):
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


# ------------------------------------------------------------------------------ convergence rates


def test_q1_on_quads_converges_at_second_order():
    errs = [_poisson_2d(Geometries.equi_distant_rect(nx=n, ny=n, cell="quad")) for n in (4, 8, 16, 32)]
    rates = _rates(errs)
    assert all(r > 1.7 for r in rates), f"Q1 quad rates {rates} (expected ~2)"
    assert rates[-1] > 1.85, f"rate has not reached 2: {rates}"
    assert errs[-1] < 1e-3


def test_q1_on_hexes_converges_at_second_order():
    errs = [_poisson_3d(Geometries.equi_distant_box(nx=n, ny=n, nz=n, cell="hex")) for n in (3, 6, 12)]
    rates = _rates(errs)
    assert all(r > 1.6 for r in rates), f"Q1 hex rates {rates} (expected ~2)"
    assert errs[-1] < 5e-3


def test_quads_match_the_triangulation_of_the_same_grid():
    """Same nodes, same problem: the two discretizations must agree to their own accuracy. This
    catches a quad element that converges to a *different* answer than the triangles do."""
    n = 16
    e_tri = _poisson_2d(Geometries.equi_distant_rect(nx=n, ny=n))
    e_quad = _poisson_2d(Geometries.equi_distant_rect(nx=n, ny=n, cell="quad"))
    assert e_quad < 2 * e_tri, f"quad error {e_quad:.3e} vs triangle {e_tri:.3e}"


# ----------------------------------------------------------------------------------- the patch test


def _distorted_quad_mesh(nx=4, ny=4, amp=0.18, seed=0):
    """A structured quad grid with its INTERIOR nodes pushed off the lattice.

    The boundary is left straight so the exact linear field is imposed exactly; the interior cells
    become genuine non-parallelograms, whose Jacobian varies within the cell.
    """
    mesh, dim, ds = Geometries.equi_distant_rect(nx=nx, ny=ny, cell="quad")(None)
    pts = np.asarray(mesh.points).copy()
    rng = np.random.default_rng(seed)
    inside = (pts[:, 0] > 1e-12) & (pts[:, 0] < 1 - 1e-12) & (pts[:, 1] > 1e-12) & (pts[:, 1] < 1 - 1e-12)
    pts[inside, :2] += (amp / nx) * rng.uniform(-1.0, 1.0, size=(int(inside.sum()), 2))
    moved = meshio.Mesh(points=pts, cells=mesh.cells, cell_sets=mesh.cell_sets)
    return lambda geo: (moved, dim, ds)


def _patch_error(ctor):
    """-Δu = 0 with u = 1 + 2x + 3y prescribed on ∂Ω. The FE solution must be that field exactly."""
    d = jno.domain(constructor=ctor, compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - (1.0 + 2.0 * xb + 3.0 * yb)])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()
    pts = np.asarray(d._fem_native_dof_points)
    return float(np.abs(sol - (1.0 + 2.0 * pts[:, 0] + 3.0 * pts[:, 1])).max())


def test_patch_test_on_a_distorted_quad_mesh():
    """THE test for the per-quadrature-point Jacobian.

    On these cells ``det J`` is not constant, so an assembler that formed one Jacobian per cell
    would integrate the stiffness of a *different* element and break linear exactness. Reproducing
    a linear field to solver tolerance is what proves the geometry map is right.
    """
    assert _patch_error(_distorted_quad_mesh(amp=0.0)) < 1e-9  # the undistorted control
    assert _patch_error(_distorted_quad_mesh(amp=0.18)) < 1e-9


def test_the_distortion_is_real():
    """Guard on the test itself: if the perturbation ever stopped producing non-affine cells, the
    patch test above would still pass and would silently stop testing anything."""
    from jno.utils.solver.fem_lagrange import lagrange_on, vtk_to_basix_vertex_perm

    mesh, _, _ = _distorted_quad_mesh(amp=0.18)(None)
    pts = np.asarray(mesh.points)[:, :2]
    quads = {c.type: np.asarray(c.data) for c in mesh.cells}["quad"]
    spec = lagrange_on("quad", 1)
    verts = pts[quads][:, vtk_to_basix_vertex_perm("quad")]
    det = np.linalg.det(np.einsum("cad,qan->cqdn", verts, spec.ref_grads[..., 0, :]))
    spread = (det.max(axis=1) - det.min(axis=1)) / det.mean(axis=1)
    assert spread.max() > 0.05, "the distorted mesh has no genuinely non-affine cell"
    assert np.all(det > 0), "the distortion inverted a cell"


# ------------------------------------------------------------------------------------- the refusals


@pytest.mark.parametrize("region,exact", [("right", 1.0), ("bottom", 2.0), ("boundary", 6.0)])
def test_surface_terms_on_quads_integrate_the_right_measure(region, exact):
    """The load vector of ``1.0 * v(region)`` is ∫_region φ_i ds, so summing it gives the region's
    MEASURE — the shape functions are a partition of unity. That tests the facet quadrature and the
    facet area element directly, with no dependence on a sign convention or a solve.

    A quad's facet is a straight edge: restricted to one edge the bilinear map is LINEAR, so the
    tangent is constant and one normal per facet is exact. Only the tabulated basis and the cell
    Jacobian had to become cell-aware.
    """
    ctor = Geometries.equi_distant_rect(x_range=(0.0, 2.0), y_range=(0.0, 1.0), nx=6, ny=6, cell="quad")
    d = jno.domain(constructor=ctor, compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    c = d.variable(region, split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, 1.0 * v(c[0], c[1])])
    np.testing.assert_allclose(float(np.abs(np.asarray(fem.b)).sum()), exact, rtol=1e-10)


def test_a_surface_term_on_a_hex_mesh_still_refuses():
    """A hexahedron's facet is a bilinear SURFACE, not a straight edge: its normal and area element
    vary across the facet and need Nanson's formula per quadrature point, where the assembler still
    carries one frozen normal per facet. That is the remaining half of the facet work."""
    ctor = Geometries.equi_distant_box(nx=2, ny=2, nz=2, cell="hex")
    d = jno.domain(constructor=ctor, compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    cr = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    with pytest.raises(NotImplementedError, match="Nanson"):
        jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, 1.0 * v(cr[0], cr[1], cr[2])]).solve()


def test_higher_order_on_a_quad_mesh_refuses_by_name():
    """P2+ node promotion places nodes by barycentric weights, which only describe a simplex."""
    d = jno.domain(constructor=Geometries.equi_distant_rect(nx=3, ny=3, cell="quad"), compute_mesh_connectivity=False)
    with pytest.raises(NotImplementedError, match="BARYCENTRIC"):
        u, v = d.fem_symbols(order=2)
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0]).solve()


# ---------------------------------------------------------------------------------------- extremes


def test_dirichlet_pins_exactly_the_boundary():
    """A regression on the facet table used to find boundary DOFs: reading a quad's 4-node cell as
    a triangle pinned 24 of 25 nodes and left one CORNER free, which still 'solved'."""
    d = jno.domain(constructor=Geometries.equi_distant_rect(nx=4, ny=4, cell="quad"), compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    A = np.asarray(fem.A.todense() if hasattr(fem.A, "todense") else fem.A)
    pinned = np.array([np.count_nonzero(A[i]) == 1 and np.isclose(A[i, i], 1.0) for i in range(len(A))])
    pts = np.asarray(d._fem_native_dof_points)
    on_boundary = (np.abs(pts) < 1e-12) | (np.abs(pts - 1.0) < 1e-12)
    np.testing.assert_array_equal(pinned, on_boundary.any(axis=1))


def test_a_single_cell_solve():
    """One quad, every node on the boundary: the solution is exactly the prescribed field."""
    assert _patch_error(_distorted_quad_mesh(nx=1, ny=1, amp=0.0)) < 1e-9


def test_a_high_aspect_ratio_mesh_still_converges():
    """1000:1 cells. The Jacobian is badly scaled, so this is where a push-forward that inverts J
    carelessly loses accuracy."""

    def ctor(n):
        return Geometries.equi_distant_rect(x_range=(0.0, 1000.0), y_range=(0.0, 1.0), nx=n, ny=n, cell="quad")

    d = jno.domain(constructor=ctor(8), compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - (2.0 + 0.001 * xb + 3.0 * yb)])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()
    pts = np.asarray(d._fem_native_dof_points)
    exact = 2.0 + 0.001 * pts[:, 0] + 3.0 * pts[:, 1]
    assert np.abs(sol - exact).max() < 1e-8


# ------------------------------------------------------------------- Shape.quad() (recombination)


def test_shape_quad_recombines_arbitrary_geometry():
    """gmsh meshes triangles and recombines them, which is not restricted to boxes: a DISK comes
    back as pure quadrilaterals, with no triangle left behind."""
    for shape in (jno.Shape.rect(0, 0, 1, 1, size=0.25), jno.Shape.disk(0, 0, 1, size=0.3)):
        blocks = {c.type: len(c.data) for c in shape.quad().build()[0].cells}
        assert "triangle" not in blocks, f"recombination left triangles: {blocks}"
        assert blocks.get("quad", 0) > 0
        assert blocks.get("line", 0) > 0  # a quad's facet is still a 2-node edge


def test_shape_quad_solves_and_converges():
    def err(h):
        d = jno.Shape.rect(0, 0, 1, 1, size=h).quad().domain()
        u, v = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        f = 2 * PI**2 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
        sol = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0]).solve()).ravel()
        p = np.asarray(d._fem_native_dof_points)
        return float(np.sqrt(np.mean((sol - np.sin(PI * p[:, 0]) * np.sin(PI * p[:, 1])) ** 2)))

    errs = [err(h) for h in (0.2, 0.1, 0.05)]
    rates = _rates(errs)
    assert all(r > 1.6 for r in rates), f"recombined-quad rates {rates}"


def test_patch_test_on_a_recombined_disk():
    """The strongest geometric case available: unstructured quads on a curved boundary, where cell
    shapes are irregular by construction. A linear field is still reproduced to machine precision."""
    d = jno.Shape.disk(0, 0, 1, size=0.15).quad().domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - (1.0 + 2.0 * xb + 3.0 * yb)])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()
    p = np.asarray(d._fem_native_dof_points)
    assert np.abs(sol - (1.0 + 2.0 * p[:, 0] + 3.0 * p[:, 1])).max() < 1e-10


def test_shape_quad_refuses_3d_and_curved():
    """gmsh cannot hex-mesh general 3-D geometry, and a curved quad needs a 9-node block neither the
    emitter nor the element path has. Both refuse by name instead of silently meshing simplices."""
    with pytest.raises(NotImplementedError, match="2-D only"):
        jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).quad()
    with pytest.raises(NotImplementedError, match="curved"):
        jno.Shape.rect(0, 0, 1, 1, size=0.5).quad().curved().build()


def test_shape_quad_survives_the_other_modifiers():
    """`quad()` is a meshing property like `sized()`, so it must propagate through the plan
    operators rather than being dropped by the next transformation."""
    s = (jno.Shape.rect(0, 0, 2, 1, size=0.4).quad() - jno.Shape.disk(1.0, 0.5, 0.25)).sized(0.25)
    blocks = {c.type: len(c.data) for c in s.build()[0].cells}
    assert "triangle" not in blocks and blocks.get("quad", 0) > 0


# ---------------------------------------------------------------------- the cell choice is per-shape


def test_tri_is_the_explicit_opposite_of_quad():
    """`.tri()` cancels a `.quad()` a shape inherited, and simplices remain the default."""
    plain = {c.type for c in jno.Shape.rect(0, 0, 1, 1, size=0.4).build()[0].cells}
    quad = {c.type for c in jno.Shape.rect(0, 0, 1, 1, size=0.4).quad().build()[0].cells}
    back = {c.type for c in jno.Shape.rect(0, 0, 1, 1, size=0.4).quad().tri().build()[0].cells}
    assert "triangle" in plain and "quad" not in plain
    assert "quad" in quad and "triangle" not in quad
    assert back == plain, ".tri() must undo .quad()"


def test_a_mixed_cell_plan_refuses_rather_than_picking_one():
    """Two different cell choices in one plan is a MIXED mesh. gmsh could build it — recombination
    is per-surface and the regions conform along a shared edge in 2-D — but the assembler carries
    one element table, so it would build and fail to assemble. Refuse at the mesher, and say what to
    do instead (independent meshes + a mortar tie)."""
    mixed = jno.Shape.regions(
        left=jno.Shape.rect(0, 0, 1, 1, size=0.3).quad(),
        right=jno.Shape.rect(1, 0, 2, 1, size=0.3).tri(),
    )
    assert mixed.cell_choices() == frozenset({"quad", "simplex"})
    with pytest.raises(NotImplementedError, match="more than one cell type"):
        mixed.build()


def test_one_cell_choice_through_a_plan_is_not_mixed():
    """A single choice, however deep in the plan, must still mesh."""
    s = (jno.Shape.rect(0, 0, 2, 1, size=0.4).quad() - jno.Shape.disk(1.0, 0.5, 0.25)).sized(0.3)
    assert s.cell_choices() == frozenset({"quad"})
    assert "quad" in {c.type for c in s.build()[0].cells}


# ------------------------------------------------------------------------ the measured feature set


def _quad_domain(**kw):
    return jno.domain(
        constructor=Geometries.equi_distant_rect(nx=6, ny=6, cell="quad"), compute_mesh_connectivity=False, **kw
    )


def test_nonlinear_newton_on_quads():
    d = _quad_domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    sol = np.asarray(jno.fem([(1.0 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0]).solve())
    assert np.isfinite(sol).all() and sol.max() > 0


def test_vector_elasticity_on_quads():
    """Mechanics is the reason tensor-product cells exist, so a vector field is the load-bearing case."""
    d = _quad_domain()
    u, v = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    eu, ev = jno.np.symgrad(u, [xi, yi]), jno.np.symgrad(v, [xi, yi])
    frm = 2.0 * jno.np.inner(eu, ev, n_contract=2) + 1.0 * jno.np.trace(eu) * jno.np.trace(ev)
    sol = np.asarray(jno.fem([frm - 1.0 * v.bind(x=xi, y=yi)[1], u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0]).solve())
    assert np.isfinite(sol).all() and np.abs(sol).max() > 0


def test_transient_march_on_quads():
    d = _quad_domain(time=(0.0, 0.2, 4))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial")
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    traj = np.asarray(jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(*ci) - 1.0]).solve().eval())
    assert traj.shape[0] == 4 and np.isfinite(traj).all()


def test_bounds_and_periodic_ties_on_quads():
    d = _quad_domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    capped = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - 5.0 * vi, u(xb, yb) - 0.0, u.bounds(None, 0.05)]).solve())
    assert capped.max() <= 0.05 + 1e-8, "u.bounds did not hold on a quad mesh"

    cl, cr, cb, ct = (d.variable(t) for t in ("left", "right", "bottom", "top"))
    f = jno.np.sin(2 * PI * xi) * jno.np.sin(2 * PI * yi)
    tied = np.asarray(
        jno.fem([ui.x * vi.x + ui.y * vi.y + 1.0 * ui * vi - f * vi, u(*cl) - u(*cr), u(*cb) - u(*ct)]).solve()
    )
    assert np.isfinite(tied).all()


def test_adaptivity_refuses_on_quads_by_name():
    """mmg adapts simplices and the recovery estimator differentiates P1 shape functions; neither
    generalises. This raised a bare `KeyError: 'triangle'` before, which named nothing."""
    d = _quad_domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    with pytest.raises(NotImplementedError, match="simplicial mesh"):
        fem.solve(adapt=jno.solve.remesh(max_iters=1))
