"""Lagrange elements on tensor-product cells, and the vertex order that connects them to a mesh.

Two things are pinned here.

**The generic builder.** ``lagrange_on`` replaces three near-identical per-cell builders that
differed only in the cell they named and how many derivative blocks they stacked. The first test
asserts it reproduces all three *bit-for-bit*, so adding quadrilaterals and hexahedra costs the
existing elements nothing.

**The vertex order.** basix and meshio/VTK number a reference cell differently, and only for the
tensor-product cells: VTK walks a quadrilateral around its perimeter, basix does not. Getting this
wrong is silent — the basis still tabulates, the assembly still runs, and the cell is quietly a
bow-tie whose Jacobian changes sign inside it. So the permutation is not merely compared against a
hardcoded list; it is checked by the property it exists to guarantee, that the isoparametric map
built from it reproduces the cell's own geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from jno.utils.solver.fem_lagrange import (
    basix_cell,
    lagrange_interval,
    lagrange_on,
    lagrange_tet,
    lagrange_triangle,
    vtk_to_basix_vertex_perm,
)

VTK_QUAD = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
VTK_HEX = np.array([[x, y, z] for z in (0.0, 1.0) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))], dtype=float)


# ------------------------------------------------------------------- the generic builder is exact


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "builder,cell", [(lagrange_interval, "line"), (lagrange_triangle, "triangle"), (lagrange_tet, "tetra")]
)
def test_generic_builder_reproduces_the_per_cell_ones(builder, cell, degree):
    """Collapsing three builders into one must be a refactor, not a change of numbers."""
    a, b = builder(degree), lagrange_on(cell, degree)
    assert a.n_dof == b.n_dof
    np.testing.assert_array_equal(a.ref_values, b.ref_values)
    np.testing.assert_array_equal(a.ref_grads, b.ref_grads)
    np.testing.assert_array_equal(a.quad_points, b.quad_points)
    np.testing.assert_array_equal(a.quad_weights, b.quad_weights)
    np.testing.assert_array_equal(a.ref_hess, b.ref_hess)
    assert tuple(map(tuple, a.local_edges)) == tuple(map(tuple, b.local_edges))


# --------------------------------------------------------------------------- the new element specs


@pytest.mark.parametrize("cell,degree,ndof,nedges", [("quad", 1, 4, 4), ("quad", 2, 9, 4), ("hexahedron", 1, 8, 12)])
def test_tensor_product_element_shapes(cell, degree, ndof, nedges):
    spec = lagrange_on(cell, degree)
    assert spec.n_dof == ndof
    assert len(spec.local_edges) == nedges
    tdim = 2 if cell == "quad" else 3
    assert spec.ref_values.shape == (len(spec.quad_weights), ndof, 1)
    assert spec.ref_grads.shape == (len(spec.quad_weights), ndof, 1, tdim)
    assert spec.ref_hess.shape == (len(spec.quad_weights), ndof, 1, tdim, tdim)


@pytest.mark.parametrize("cell", ["quad", "hexahedron"])
@pytest.mark.parametrize("degree", [1, 2])
def test_partition_of_unity_and_gradient_consistency(cell, degree):
    """Σφ = 1 and Σ∇φ = 0 at every quadrature point — the identity a nodal basis must satisfy, and
    the cheapest detector of a wrong element or a wrong tabulation block."""
    spec = lagrange_on(cell, degree)
    np.testing.assert_allclose(spec.ref_values[..., 0].sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(spec.ref_grads[..., 0, :].sum(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize("cell", ["quad", "hexahedron"])
def test_reference_cell_has_unit_measure(cell):
    """The reference quad and hex have measure 1, where the triangle has 1/2 and the tet 1/6 —
    the factor every hardcoded `/2` or `/6` divisor in simplex code silently assumes."""
    spec = lagrange_on(cell, 1)
    np.testing.assert_allclose(spec.quad_weights.sum(), 1.0, atol=1e-12)


def test_quadrature_integrates_what_it_claims():
    """The default rule (2*degree+1) must integrate a polynomial of that degree exactly on the
    REFERENCE cell, where the map is the identity and no rational 1/detJ is involved."""
    spec = lagrange_on("quad", 2)
    qp, qw = spec.quad_points, spec.quad_weights
    # ∫₀¹∫₀¹ x²y² = 1/9
    np.testing.assert_allclose(np.sum(qw * qp[:, 0] ** 2 * qp[:, 1] ** 2), 1.0 / 9.0, rtol=1e-12)
    hexs = lagrange_on("hexahedron", 1)
    q, w = hexs.quad_points, hexs.quad_weights
    np.testing.assert_allclose(np.sum(w * q[:, 0] * q[:, 1] * q[:, 2]), 1.0 / 8.0, rtol=1e-12)


def test_an_unsupported_cell_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="prism"):
        lagrange_on("prism", 1)
    with pytest.raises(NotImplementedError, match="pyramid"):
        basix_cell("pyramid")


# ---------------------------------------------------------------------------- the vertex ordering


def test_simplex_vertex_order_needs_no_permutation():
    """basix and VTK agree on an interval, triangle and tetrahedron — the permutation must be the
    identity there, or every existing element would be silently reordered."""
    for cell in ("line", "triangle", "tetra"):
        perm = vtk_to_basix_vertex_perm(cell)
        np.testing.assert_array_equal(perm, np.arange(len(perm)))


@pytest.mark.parametrize("cell,vtk", [("quad", VTK_QUAD), ("hexahedron", VTK_HEX)])
def test_the_permutation_makes_the_geometry_map_the_identity(cell, vtk):
    """THE test for the permutation, stated as the property it protects.

    Placing a cell's own vertices through its own Q1 basis must reproduce the reference cell:
    ``x(ξ) = Σ_a x_a N_a(ξ)`` has to return ξ itself. With the vertices in the wrong order the map
    is a bow-tie, and this is what detects that — no reference to any hardcoded index list.
    """
    spec = lagrange_on(cell, 1)
    perm = vtk_to_basix_vertex_perm(cell)
    verts_basix = vtk[perm]  # basix DOF order
    xq = spec.ref_values[..., 0] @ verts_basix  # (n_quad, dim)
    np.testing.assert_allclose(xq, spec.quad_points, atol=1e-12)


@pytest.mark.parametrize("cell,vtk", [("quad", VTK_QUAD), ("hexahedron", VTK_HEX)])
def test_the_jacobian_is_the_identity_and_never_changes_sign(cell, vtk):
    """The same statement in the derivative: J = ∂x/∂ξ = I, so det J = 1 > 0 at every quadrature
    point. A mis-ordered cell gives a det that is negative somewhere inside it."""
    spec = lagrange_on(cell, 1)
    verts = vtk[vtk_to_basix_vertex_perm(cell)]
    J = np.einsum("ad,qan->qdn", verts, spec.ref_grads[..., 0, :])
    dim = verts.shape[1]
    np.testing.assert_allclose(J, np.broadcast_to(np.eye(dim), J.shape), atol=1e-12)
    assert np.all(np.linalg.det(J) > 0)


@pytest.mark.parametrize("cell,vtk", [("quad", VTK_QUAD), ("hexahedron", VTK_HEX)])
def test_an_unpermuted_cell_would_have_been_silently_wrong(cell, vtk):
    """Proof that the permutation is load-bearing rather than decorative: feeding VTK order
    straight to the basix basis produces a map whose Jacobian determinant changes sign inside the
    cell — a bow-tie that no assembly would flag."""
    spec = lagrange_on(cell, 1)
    J = np.einsum("ad,qan->qdn", vtk, spec.ref_grads[..., 0, :])
    det = np.linalg.det(J)
    assert det.min() < 0 < det.max(), "expected the un-permuted map to fold over"


def test_a_distorted_quad_keeps_a_positive_jacobian():
    """A genuinely non-affine (non-parallelogram) quad: det J varies over the cell but stays
    positive. This is the case that separates a tensor-product element from a simplex one, where
    det J is a single number per cell."""
    spec = lagrange_on("quad", 1)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.7, 1.4], [0.0, 1.0]])  # VTK order, ccw
    verts = corners[vtk_to_basix_vertex_perm("quad")]
    det = np.linalg.det(np.einsum("ad,qan->qdn", verts, spec.ref_grads[..., 0, :]))
    assert np.all(det > 0)
    assert det.max() - det.min() > 1e-3, "a non-parallelogram quad must have a varying Jacobian"
    # The cell's area is ∫ det J, which the quadrature gives exactly for a bilinear map. Compared
    # against the shoelace area of the same four corners — an independent formula, not a constant.
    x, y = corners[:, 0], corners[:, 1]
    shoelace = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    np.testing.assert_allclose(np.sum(spec.quad_weights * det), shoelace, rtol=1e-12)
