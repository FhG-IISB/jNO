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


def test_a_surface_term_on_a_quad_mesh_refuses_by_name():
    """Facet geometry is still per-facet, which is only right for a straight simplex facet. Before
    the guard this surfaced as a raw broadcasting error between two basis sizes."""
    d = jno.domain(constructor=Geometries.equi_distant_rect(nx=3, ny=3, cell="quad"), compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xl, yl, _ = d.variable("left", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="boundary/surface term on a quad mesh"):
        jno.fem([ui.x * vi.x + ui.y * vi.y, 1.0 * v(xr, yr), u(xl, yl) - 0.0]).solve()


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
