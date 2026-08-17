"""Every boundary DOF must reach the boundary facet table — including a hexahedron's edge nodes.

``_boundary_node_ids`` resolves an essential condition through the boundary FACET table, so a DOF
missing from that table is simply never constrained. It does not raise; the solve returns a
plausible wrong answer.

Order-2 hexes did exactly that. ``_ref_interior_facet_dofs`` walked a facet's edges as *consecutive
vertex pairs*, documented as "``fv`` arrives in perimeter order" — but basix lists a hexahedron
face's vertices in **tensor** order, so the consecutive pairs are ``[edge, DIAGONAL, edge,
DIAGONAL]`` on every one of the six faces. Two of the four real edges were never walked, their
edge-interior DOFs never entered the table, and the same assumption handed the face-interior test a
diagonal as its second in-plane basis vector.

Measured before the fix: **2 of 98** boundary DOFs uncovered on an ``n=2`` hex cube, 3 of 218 at
``n=3`` — always on one edge, tetrahedra unaffected.

What made it expensive to find is that it hides. The missing constraint is invisible whenever the
prescribed value happens to be consistent with the natural condition at that edge, so ``g = 1``,
``z``, ``x*y`` and ``x*x - y*y`` all passed while ``g = x`` failed by 1.3e-01 — i.e. the order-2
space appeared to reproduce *quadratics* but not *linears*, which is impossible for a real space and
sent the diagnosis down the wrong path.

The oracle here is coverage of the facet table itself, not a solution error: coverage is exact and
mesh-independent, whereas a solution error only reveals the nodes where the missing constraint
happened to matter.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno
from jno._fem import _boundary_facets, _facet_perimeter_order
from jno.domain.geometries import Geometries

meshio = pytest.importorskip("meshio")


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly is float64; these tests opt in per-test (the session default is x64-off)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _box(cell, n, order):
    """Build a unit-cube problem and return (dof points, assembly cells, meshio cell type)."""
    m, _, _ = Geometries.equi_distant_box(x_range=(0, 1), y_range=(0, 1), z_range=(0, 1), nx=n, ny=n, nz=n, cell=cell)(None)
    d = jno.domain(lambda g: (m, 3, 1.0 / n), compute_mesh_connectivity=True)
    u, v = d.fem_symbols(names=("u", "v"), order=order)
    c = list(d.variable("interior", split=True)[:3])
    fem = jno.fem([jno.np.inner(jno.np.grad(u, c), jno.np.grad(v, c), n_contract=1)])
    pts = np.asarray(fem.field_points[0])
    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    return pts, cells, ("hexahedron" if cell == "hex" else None)


def _on_unit_cube_boundary(pts):
    return (np.abs(pts).min(axis=1) < 1e-9) | (np.abs(pts - 1.0).min(axis=1) < 1e-9)


def test_perimeter_order_walks_only_real_edges():
    """The permutation must turn a basix quad facet into a cycle of genuine edges.

    Checked on the hexahedron's own reference geometry rather than restating ``(0, 1, 3, 2)``: two
    reference vertices span a real edge exactly when they differ in ONE coordinate.
    """
    basix = pytest.importorskip("basix")
    from jno.utils.solver.fem_lagrange import basix_cell

    assert _facet_perimeter_order(3) == (0, 1, 2), "a triangular facet is already cyclic"
    perm = _facet_perimeter_order(4)
    assert sorted(perm) == [0, 1, 2, 3], "must be a permutation of the facet's vertices"

    cell, tdim = basix_cell("hexahedron")
    geo = np.asarray(basix.geometry(cell))
    for face in basix.topology(cell)[tdim - 1]:
        ring = [face[i] for i in perm]
        for k in range(4):
            a, b = geo[ring[k]], geo[ring[(k + 1) % 4]]
            assert int(np.sum(np.abs(a - b) > 1e-12)) == 1, f"{ring} takes a diagonal, not an edge"
        # the two in-plane basis vectors must be independent (a diagonal makes them skew)
        o = geo[ring[0]]
        assert np.linalg.matrix_rank(np.stack([geo[ring[1]] - o, geo[ring[-1]] - o], axis=1)) == 2


@pytest.mark.parametrize("cell", ["hex", "tetra"])
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_boundary_facet_table_covers_every_boundary_dof(cell, n, order):
    """No DOF on the geometric boundary may be absent from the boundary facet table.

    This is the direct oracle for the bug: it failed at 2/98 for ``hex, n=2, order=2`` and 3/218 at
    ``n=3``, while every tetrahedral case passed.
    """
    pts, cells, ctype = _box(cell, n, order)
    covered = np.unique(np.asarray(_boundary_facets(pts, cells, 3, order, ctype)).reshape(-1))
    on_boundary = np.flatnonzero(_on_unit_cube_boundary(pts))
    missed = np.setdiff1d(on_boundary, covered)
    assert missed.size == 0, (
        f"{missed.size} of {on_boundary.size} boundary DOFs are absent from the facet table "
        f"({cell}, n={n}, order={order}); an essential condition on them would be silently dropped. "
        f"First few at {[tuple(np.round(pts[k], 4)) for k in missed[:4]]}"
    )


def _solve_laplace_with_dirichlet(cell, n, which):
    """Solve -Laplace(u) = 0 with u = g on the whole boundary, for a HARMONIC g of degree <= 2.

    The exact solution is then g itself, so any deviation is a failure to impose the condition (or
    to represent g), not discretisation error.
    """
    m, _, _ = Geometries.equi_distant_box(x_range=(0, 1), y_range=(0, 1), z_range=(0, 1), nx=n, ny=n, nz=n, cell=cell)(None)
    d = jno.domain(lambda g: (m, 3, 1.0 / n), compute_mesh_connectivity=True)
    u, v = d.fem_symbols(names=("u", "v"), order=2)
    x, y, z, _ = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    gs = {
        "one": (lambda c: 0 * c[0] + 1.0, lambda P: np.ones(len(P))),
        "x": (lambda c: c[0], lambda P: P[:, 0]),
        "z": (lambda c: c[2], lambda P: P[:, 2]),
        "x+2y+3z": (lambda c: c[0] + 2 * c[1] + 3 * c[2], lambda P: P[:, 0] + 2 * P[:, 1] + 3 * P[:, 2]),
        "x*y": (lambda c: c[0] * c[1], lambda P: P[:, 0] * P[:, 1]),
        "x*x-y*y": (lambda c: c[0] * c[0] - c[1] * c[1], lambda P: P[:, 0] ** 2 - P[:, 1] ** 2),
    }
    gfn, exact = gs[which]
    fem = jno.fem(
        [
            jno.np.inner(jno.np.grad(u, [x, y, z]), jno.np.grad(v, [x, y, z]), n_contract=1),
            u(b[0], b[1], b[2]) - gfn(b),
        ]
    )
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.field_points[0])
    err = np.abs(sol[: len(pts)] - exact(pts))
    return float(err[_on_unit_cube_boundary(pts)].max())


@pytest.mark.parametrize("which", ["x", "x+2y+3z"])
def test_hex_dirichlet_imposes_a_field_varying_normal_to_the_faces(which):
    """The cases the bug actually broke: g varying normal to the two faces meeting at an edge.

    Before the fix these came back at 1.3e-01 and 4.0e-01 on the Q2 edge nodes.
    """
    assert _solve_laplace_with_dirichlet("hex", 2, which) < 1e-7


@pytest.mark.parametrize("which", ["one", "z", "x*y", "x*x-y*y"])
def test_hex_dirichlet_cases_that_already_passed_still_pass(which):
    """Guards against a 'fix' that merely moves which nodes are unconstrained.

    These passed even with the broken facet table, because the missing constraint happened to agree
    with the natural condition at that edge. They must keep passing.
    """
    assert _solve_laplace_with_dirichlet("hex", 2, which) < 1e-7


@pytest.mark.parametrize("which", ["x", "x+2y+3z", "x*y"])
def test_tetra_dirichlet_is_unaffected(which):
    """The simplex path is deliberately untouched — its DOF ordering is a contract other code reads."""
    assert _solve_laplace_with_dirichlet("tetra", 2, which) < 1e-7
