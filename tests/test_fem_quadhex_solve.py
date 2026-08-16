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


# ------------------------------------------------------------------ surface terms on hexahedra


def _warped_hex_mesh(warp, n=3, seed=0):
    """A structured hex mesh with every non-corner node displaced, so its faces are genuinely
    non-planar. Returns ``(constructor, min_detJ)`` — the caller checks the mesh is not tangled."""
    import basix
    import meshio

    from jno.utils.solver.fem_lagrange import vtk_to_basix_vertex_perm

    m, _, ds = Geometries.equi_distant_box(nx=n, ny=n, nz=n, cell="hex")(None)
    pts = np.asarray(m.points).copy()
    if warp:
        rng = np.random.default_rng(seed)
        mv = ((pts > 1e-12) & (pts < 1 - 1e-12)).sum(axis=1) >= 1
        pts[mv] += (warp / n) * rng.uniform(-1.0, 1.0, size=(int(mv.sum()), 3))
    hx = basix.create_element(basix.ElementFamily.P, basix.CellType.hexahedron, 1)
    qp, _ = basix.make_quadrature(basix.CellType.hexahedron, 6)
    dN = np.stack([np.asarray(hx.tabulate(1, qp))[i][..., 0] for i in (1, 2, 3)], axis=-1)
    cells = {c.type: np.asarray(c.data) for c in m.cells}["hexahedron"]
    verts = pts[cells[:, vtk_to_basix_vertex_perm("hexahedron")]]
    det = np.linalg.det(np.einsum("cad,qan->cqdn", verts, dN))
    moved = meshio.Mesh(points=pts, cells=m.cells, cell_sets=m.cell_sets)
    return (lambda geo: (moved, 3, ds)), float(det.min())


@pytest.mark.parametrize("region,exact", [("right", 1.0), ("front", 1.0), ("boundary", 6.0)])
def test_surface_terms_on_hexes_integrate_the_right_measure(region, exact):
    """The area element, on its own. A unit boundary source's load vector sums to the region's AREA
    (partition of unity) — no dependence on the normal's direction or on a sign convention."""
    d = jno.domain(constructor=Geometries.equi_distant_box(nx=3, ny=3, nz=3, cell="hex"), compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    c = d.variable(region, split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, 1.0 * v(c[0], c[1], c[2])])
    np.testing.assert_allclose(float(np.abs(np.asarray(fem.b)).sum()), exact, rtol=1e-10)


@pytest.mark.parametrize("warp", [0.0, 0.1, 0.2, 0.3])
def test_divergence_theorem_on_warped_hexes(warp):
    """THE test for the per-quadrature-point (Nanson) normal.

    ``∮ x·n dS = 3·Vol`` is sensitive to the normal's direction AND its outward orientation at every
    quadrature point — a sign flip on a single face breaks it. Both sides come from the assembler on
    the same mesh, so the oracle and the quantity share one geometry: computing the volume any other
    way (splitting each face into two triangles, say) describes a *different solid* once faces are
    non-planar, and measured a 6 % gap that would swamp what is being tested.

    A single frozen normal per facet cannot pass this on a warped mesh — that is exactly what the
    refusal this replaces was protecting against.
    """
    ctor, min_det = _warped_hex_mesh(warp)
    assert min_det > 0, "the test mesh is tangled; a tangled mesh has no well-defined volume"
    d = jno.domain(constructor=ctor, compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    lap = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z
    cb = d.variable("boundary", split=True)
    nb = d.variable("boundary", normals=True, split=True)
    vol = float(np.asarray(jno.fem([lap - 1.0 * vi], quad_degree=6).b).sum())
    xn = cb[0] * nb[-3] + cb[1] * nb[-2] + cb[2] * nb[-1]
    flux = float(np.asarray(jno.fem([lap, xn * v(cb[0], cb[1], cb[2])], quad_degree=6).b).sum())
    np.testing.assert_allclose(abs(flux), 3.0 * vol, rtol=1e-10)


def test_the_warped_hex_faces_are_really_non_planar():
    """Guard on the test above: if the perturbation stopped producing non-planar faces, the
    divergence-theorem test would pass with a single per-facet normal and prove nothing."""
    from jno.domain.mesh_utils import MeshUtils

    ctor, min_det = _warped_hex_mesh(0.3)
    mesh, _, _ = ctor(None)
    pts = np.asarray(mesh.points)
    cells = {c.type: np.asarray(c.data) for c in mesh.cells}["hexahedron"]
    fac, _ = MeshUtils._boundary_facets_unsorted(cells, "hexahedron")
    V = pts[fac]
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    warp = np.abs(np.einsum("ij,ij->i", V[:, 3] - V[:, 0], n))
    assert min_det > 0 and warp.max() > 0.02, f"faces are nearly planar (max warp {warp.max():.3e})"


# -------------------------------------------------------------------------- higher order (Q2 / Q3)


def _poisson_order(n, order, cell):
    """MMS Poisson at a given element order; returns (nodal RMS error, dof count)."""
    kw = {"cell": cell} if cell else {}
    d = jno.domain(constructor=Geometries.equi_distant_rect(nx=n, ny=n, **kw), compute_mesh_connectivity=False)
    u, v = d.fem_symbols(order=order)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = 2 * PI**2 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()
    p = np.asarray(d._fem_native_dof_points)
    return float(np.sqrt(np.mean((sol - np.sin(PI * p[:, 0]) * np.sin(PI * p[:, 1])) ** 2))), len(p)


@pytest.mark.parametrize("order", [2, 3])
def test_higher_order_quads_converge_faster_than_q1(order):
    """Element ORDER is the lever that raises the convergence rate — cell shape is not (measured:
    Q1 and P1 agree to within 1 % at every aspect ratio). Q1's rate on this problem is ~1.9, so a
    working Q2/Q3 must clearly exceed it."""
    errs = [_poisson_order(n, order, "quad")[0] for n in (4, 8, 16)]
    rates = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    assert all(r > 2.8 for r in rates), f"Q{order} rates {rates} — no better than Q1"
    assert errs[-1] < 1e-5


@pytest.mark.parametrize("order", [2, 3])
def test_higher_order_quads_match_the_simplex_dof_for_dof(order):
    """A Q{k} quad mesh and a P{k} triangulation of the same grid have the SAME node count, so this
    is a fair comparison rather than a favourable one. Reported as an inequality that would still
    pass if the quads were merely competitive, not just when they win."""
    e_quad, n_quad = _poisson_order(16, order, "quad")
    e_tri, n_tri = _poisson_order(16, order, None)
    assert n_quad == n_tri, f"the comparison is not DOF-for-DOF: {n_quad} vs {n_tri}"
    assert e_quad <= e_tri, f"Q{order} ({e_quad:.2e}) worse than P{order} ({e_tri:.2e}) at equal dofs"


def test_q2_hexes_converge():
    """The promotion is cell-generic, so hexes get Q2 from the same change."""

    def err(n):
        d = jno.domain(
            constructor=Geometries.equi_distant_box(nx=n, ny=n, nz=n, cell="hex"), compute_mesh_connectivity=False
        )
        u, v = d.fem_symbols(order=2)
        xi, yi, zi, _ = d.variable("interior", split=True)
        xb, yb, zb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
        f = 3 * PI**2 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi) * jno.np.sin(PI * zi)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - f * vi, u(xb, yb, zb) - 0.0])
        sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()
        p = np.asarray(d._fem_native_dof_points)
        ex = np.sin(PI * p[:, 0]) * np.sin(PI * p[:, 1]) * np.sin(PI * p[:, 2])
        return float(np.sqrt(np.mean((sol - ex) ** 2)))

    e2, e4 = err(2), err(4)
    assert float(np.log2(e2 / e4)) > 2.8 and e4 < 1e-3


def test_the_promoted_quad_mesh_has_the_interior_node():
    """A P2 triangle has NO cell-interior node; a Q2 quad has one and a Q3 quad four. They are the
    genuinely new topological case, and the coordinate dedup must neither merge nor duplicate them.
    The node COUNT is the direct check: a Q{k} mesh of an n x n grid has exactly (k*n + 1)^2 nodes.
    """
    from jno.utils.solver.fem_native import _get_mesh

    n = 4
    d = jno.domain(constructor=Geometries.equi_distant_rect(nx=n, ny=n, cell="quad"), compute_mesh_connectivity=False)
    for order, per_cell in ((2, 9), (3, 16)):
        pts_p1, cells_p1, pts_f, cells_f = _get_mesh(d, 2, order)
        assert cells_f.shape == (n * n, per_cell)
        assert len(pts_f) == (order * n + 1) ** 2, "the dedup merged or duplicated nodes"
        # exactly one node of each cell lies strictly inside it (Q2); four for Q3
        v = pts_f[cells_f]
        lo, hi = v.min(axis=1), v.max(axis=1)
        strictly_inside = ((v > lo[:, None, :] + 1e-12) & (v < hi[:, None, :] - 1e-12)).all(axis=2)
        assert strictly_inside.sum(axis=1).min() == (order - 1) ** 2


@pytest.mark.parametrize("order", [1, 2, 3])
def test_surface_terms_still_exact_at_higher_order(order):
    """The facet tables tabulate the PARENT basis at the facet quadrature points, so they should be
    degree-generic — a Q2 quad's edge carries three of its nine DOFs automatically. Checked by the
    boundary measure, which is exact regardless of order."""
    d = jno.domain(
        constructor=Geometries.equi_distant_rect(x_range=(0.0, 2.0), nx=4, ny=4, cell="quad"),
        compute_mesh_connectivity=False,
    )
    u, v = d.fem_symbols(order=order)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, 1.0 * v(xb, yb)])
    np.testing.assert_allclose(float(np.abs(np.asarray(fem.b)).sum()), 6.0, rtol=1e-10)


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


def test_shape_quad_refuses_an_unstructured_3d_plan_and_a_curved_one():
    """gmsh cannot hex-mesh general 3-D geometry, and a curved quad needs a 9-node block neither the
    emitter nor the element path has. Both refuse by name instead of silently meshing simplices.

    The 3-D refusal fires at BUILD time, not inside ``.quad()``: a lattice can be hex-meshed, so the
    answer depends on whether ``.structured()`` is anywhere in the plan, and requiring it to come
    first would make the chain order-dependent."""
    with pytest.raises(NotImplementedError, match=r"\.structured\(\)"):
        jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).quad().build()
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


# ------------------------------------------------------------------- facet tables by cell, not dim


def test_boundary_facets_of_a_quad_mesh_use_the_quad_table():
    """`_boundary_facets` picks its local-facet table from the CELL. Called without one, a quad mesh
    took the triangle branch: on this 4x4 grid, whose true boundary is 16 edges over 16 nodes, that
    returned 48 facets over 24 nodes — a quarter of them interior.

    The periodic reduction was calling it that way. It went unnoticed because a structured
    conforming mesh matches its periodic nodes by coordinate and never consults this table, so the
    path was unexercised rather than proven harmless. This test consults it directly.
    """
    from jno._fem import _boundary_facets
    from jno.utils.solver.fem_lagrange import vtk_to_basix_vertex_perm
    from jno.utils.solver.fem_native import mesh_cell_type

    d = jno.domain(constructor=Geometries.equi_distant_rect(nx=4, ny=4, cell="quad"), compute_mesh_connectivity=False)
    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["quad"])[:, vtk_to_basix_vertex_perm("quad")]
    assert mesh_cell_type(d, 2) == "quad"

    bf = _boundary_facets(pts, cells, 2, 1, "quad")
    nodes = np.unique(bf)
    on_boundary = ((np.abs(pts) < 1e-12) | (np.abs(pts - 1.0) < 1e-12)).any(axis=1)
    assert len(bf) == 16 and len(nodes) == 16
    assert on_boundary[nodes].all(), "an interior node was reported as a boundary facet node"

    # and the simplex branch really is wrong here — the guard is load-bearing, not decorative
    wrong = _boundary_facets(pts, cells, 2, 1)
    assert not on_boundary[np.unique(wrong)].all()


def _hex_domain(**kw):
    return jno.domain(
        constructor=Geometries.equi_distant_box(nx=3, ny=3, nz=3, cell="hex"), compute_mesh_connectivity=False, **kw
    )


def test_hex_neumann_problem_matches_the_tet_mesh():
    """A flux BC on one face with the opposite face pinned — the solution is linear in x either way,
    so the two cell types must agree to discretization error."""

    def solve(cell):
        kw = {"cell": cell} if cell else {}
        d = jno.domain(constructor=Geometries.equi_distant_box(nx=3, ny=3, nz=3, **kw), compute_mesh_connectivity=False)
        u, v = d.fem_symbols()
        xi, yi, zi, _ = d.variable("interior", split=True)
        cl, cr = d.variable("left", split=True), d.variable("right", split=True)
        ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
        fem = jno.fem(
            [
                ui.x * vi.x + ui.y * vi.y + ui.z * vi.z,
                1.0 * v(cr[0], cr[1], cr[2]),
                u(cl[0], cl[1], cl[2]) - 0.0,
            ]
        )
        sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()
        return sol, np.asarray(d._fem_native_dof_points)

    s_hex, p_hex = solve("hex")
    s_tet, p_tet = solve(None)
    # same structured nodes either way, so compare pointwise after matching coordinates
    order_h = np.lexsort(p_hex.T)
    order_t = np.lexsort(p_tet.T)
    np.testing.assert_allclose(p_hex[order_h], p_tet[order_t], atol=1e-12)
    np.testing.assert_allclose(s_hex[order_h], s_tet[order_t], rtol=1e-8, atol=1e-8)


def _periodic_hex(dirs, n=4, cell="hex"):
    """-Delta u + u = sin2pix sin2piy sin2piz, periodic in `dirs`. Exact u = f / (12 pi^2 + 1)."""
    kw = {"cell": cell} if cell else {}
    d = jno.domain(constructor=Geometries.equi_distant_box(nx=n, ny=n, nz=n, **kw), compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    f = jno.np.sin(2 * PI * xi) * jno.np.sin(2 * PI * yi) * jno.np.sin(2 * PI * zi)
    terms = [ui.x * vi.x + ui.y * vi.y + ui.z * vi.z + 1.0 * ui * vi - f * vi]
    for axis, (lo, hi) in zip("xyz", (("left", "right"), ("bottom", "top"), ("front", "back"))):
        if axis in dirs:
            terms.append(u(*d.variable(lo)) - u(*d.variable(hi)))
    sol = np.asarray(jno.fem(terms).solve()).ravel()
    return sol, np.asarray(d._fem_native_dof_points)


@pytest.mark.parametrize("dirs", ["x", "xy", "xyz"])
def test_periodic_ties_on_a_hex_mesh(dirs):
    """Whole-domain periodicity on hexahedra, in one, two and all three directions.

    This used to refuse -- not because a hexahedron cannot support it (a structured hex mesh matches
    its opposite faces node-for-node, needing no interpolation at all) but because the availability
    CHECK for a mortar dual basis raised on a 4-node facet instead of answering False, aborting the
    build before node matching was ever tried.

    Periodicity is asserted as an identity, not a tolerance: matched nodes collapse onto one DOF, so
    the two faces must agree EXACTLY.
    """
    sol, p = _periodic_hex(dirs)
    for axis in range(3):
        if "xyz"[axis] not in dirs:
            continue
        lo = np.where(np.abs(p[:, axis]) < 1e-12)[0]
        assert len(lo), "no nodes found on the periodic face"
        for i in lo:
            q = p[i].copy()
            q[axis] = 1.0
            j = int(np.argmin(np.linalg.norm(p - q, axis=1)))
            if np.linalg.norm(p[j] - q) < 1e-12:
                assert sol[i] == sol[j], "a periodic pair does not share its value exactly"


def test_periodic_hexes_solve_the_right_problem():
    """Enforcing periodicity is not enough -- u = 0 is periodic too. Checked against the analytic
    solution, and against the tetrahedral mesh of the same box."""
    sol, p = _periodic_hex("xyz")
    ex = np.sin(2 * PI * p[:, 0]) * np.sin(2 * PI * p[:, 1]) * np.sin(2 * PI * p[:, 2]) / (12 * PI**2 + 1)
    err_hex = float(np.sqrt(np.mean((sol - ex) ** 2)))
    sol_t, pt = _periodic_hex("xyz", cell=None)
    ex_t = np.sin(2 * PI * pt[:, 0]) * np.sin(2 * PI * pt[:, 1]) * np.sin(2 * PI * pt[:, 2]) / (12 * PI**2 + 1)
    err_tet = float(np.sqrt(np.mean((sol_t - ex_t) ** 2)))
    assert err_hex < 5e-3 and err_hex < 3 * err_tet, f"hex {err_hex:.2e} vs tet {err_tet:.2e}"


def test_a_nonmatching_hex_facet_tie_still_refuses():
    """The genuinely unsupported case survives: interpolating ACROSS a quadrilateral facet needs
    shape functions the barycentric weights do not have. Only the availability check changed."""
    from jno.utils.solver.fem_utils import _tri_dual_available, _tri_shape

    assert _tri_dual_available(3) is True  # a P1 triangle facet
    assert _tri_dual_available(4) is False  # a hex's quad facet: unavailable, not an error
    with pytest.raises(NotImplementedError, match="HEXAHEDRAL facet"):
        _tri_shape(np.zeros((1, 3)), 4)


def test_taylor_hood_q2_q1_stokes_on_quads():
    """Mixed element ORDERS on one quad mesh — Q2 velocity over Q1 pressure.

    This works for the reason the promotion was written: it keeps the original vertices at ids
    0..nv-1, so a degree-1 field on a degree-k mesh is exactly the leading vertex block. That held
    for P2/P1 on simplices and carries over unchanged, because basix puts vertices first on a
    quadrilateral too. Checked against the triangulation of the same grid, which has the same
    number of DOFs.
    """
    inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace

    def stokes(cell):
        kw = {"cell": cell} if cell else {}
        d = jno.domain(
            constructor=Geometries.equi_distant_rect(x_range=(0.0, 4.0), y_range=(0.0, 1.0), nx=12, ny=4, **kw),
            compute_mesh_connectivity=False,
        )
        u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"), order=2)
        p, q = d.fem_symbols(names=("p", "q"), order=1)
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        gu, gw = grad(u, [xi, yi]), grad(w, [xi, yi])
        pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
        fem = jno.fem(
            [
                1.0 * inner_(gu, gw, n_contract=2) - pp * trace(gw),
                -qq * trace(gu),
                u(xb, yb)[0] - yb * (1 - yb),
                u(xb, yb)[1] - 0.0,
                p.pin(),
            ]
        )
        return np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))

    s_quad, s_tri = stokes("quad"), stokes(None)
    assert s_quad.size == s_tri.size, "the mixed-order spaces differ in size between the two cells"
    assert np.isfinite(s_quad).all()
    np.testing.assert_allclose(np.abs(s_quad).max(), np.abs(s_tri).max(), rtol=1e-6)
