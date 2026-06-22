"""Reference tabulation + per-cell push-forward for the non-nodal element zoo, and
Lagrange element factories + boundary facet connectivity.

``jno/utils/solver/fem_elements.py`` wraps basix's reference tabulation and pairs it
with the per-cell push-forward. These tests pin the Raviart–Thomas (H(div)) and the
Nédélec (H(curl)) slices against basix's own ``push_forward`` oracle and the edge-DOF
normalization (``∫_cell div Φ_e dx = ±1`` for RT, ``∫_cell curl Φ_e dx = ±1`` for N1E
by Stokes), and check the edge-orientation sign flips the basis.

``jno/utils/solver/fem_lagrange.py`` adds Lagrange factories + :func:`identity_pushforward`.
``jno/utils/solver/fem_facets.py`` adds boundary facet connectivity + normals.
"""

from __future__ import annotations

import numpy as np
import pytest

basix = pytest.importorskip("basix", reason="basix required for element tabulation")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jno.utils.solver.fem_elements import (  # noqa: E402
    nedelec_triangle,
    piola_contravariant,
    piola_contravariant_grad,
    piola_covariant,
    piola_covariant_grad,
    raviart_thomas_triangle,
)


@pytest.fixture(autouse=True)
def _x64():
    """The push-forward is compared to basix's float64 oracle, so opt into x64 per-test
    (session default is x64-off; save/restore keeps the flag from leaking)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _phys_tri():
    p0, p1, p2 = np.array([0.3, 0.1]), np.array([1.7, 0.4]), np.array([0.6, 1.9])
    J = np.column_stack([p1 - p0, p2 - p0])
    return J, float(np.linalg.det(J))


def test_rt_spec_shapes_and_constant_divergence():
    spec = raviart_thomas_triangle(degree=1, quad_degree=2)
    assert spec.n_dof == 3 and spec.value_size == 2
    assert spec.ref_values.shape == (len(spec.quad_weights), 3, 2)
    # RT0 divergence is constant over the cell (identical at every quad point)
    assert np.allclose(spec.ref_div, spec.ref_div[0][None, :])


def test_contravariant_piola_matches_basix_pushforward():
    spec = raviart_thomas_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    nq = spec.ref_values.shape[0]
    val, _ = piola_contravariant(
        jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_div), jnp.asarray(J), detJ, jnp.ones(spec.n_dof)
    )
    e = basix.create_element(basix.ElementFamily.RT, basix.CellType.triangle, 1)
    Jb = np.broadcast_to(J, (nq, 2, 2)).copy()
    Kb = np.broadcast_to(np.linalg.inv(J), (nq, 2, 2)).copy()
    oracle = e.push_forward(spec.ref_values, Jb, np.full((nq,), detJ), Kb)
    np.testing.assert_allclose(np.asarray(val), oracle, atol=1e-12)


def test_rt_flux_normalization_integral_of_div_is_pm_one():
    spec = raviart_thomas_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    _, div = piola_contravariant(
        jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_div), jnp.asarray(J), detJ, jnp.ones(spec.n_dof)
    )
    # div is constant per basis; ∫_cell = div * area_phys, area_phys = detJ * (ref-tri area 1/2)
    int_div = np.asarray(div)[0] * (detJ * 0.5)
    np.testing.assert_allclose(np.abs(int_div), np.ones(spec.n_dof), atol=1e-12)


def test_rt_physical_gradient_trace_recovers_divergence():
    spec = raviart_thomas_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    signs = jnp.ones(spec.n_dof)
    grad = piola_contravariant_grad(jnp.asarray(spec.ref_grads), jnp.asarray(J), detJ, signs)
    _, div = piola_contravariant(jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_div), jnp.asarray(J), detJ, signs)
    assert grad.shape == (spec.ref_values.shape[0], spec.n_dof, 2, 2)
    # divergence is the trace over (component i, spatial direction l) of the physical gradient
    div_from_grad = grad[:, :, 0, 0] + grad[:, :, 1, 1]
    np.testing.assert_allclose(np.asarray(div_from_grad), np.asarray(div), atol=1e-12)


def test_orientation_sign_flips_basis():
    spec = raviart_thomas_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    args = (jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_div), jnp.asarray(J), detJ)
    base, _ = piola_contravariant(*args, jnp.ones(spec.n_dof))
    flipped, _ = piola_contravariant(*args, jnp.array([-1.0, 1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(flipped[:, 0, :]), -np.asarray(base[:, 0, :]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(flipped[:, 1:, :]), np.asarray(base[:, 1:, :]), atol=1e-12)


def test_n1e_spec_shapes_and_constant_curl():
    spec = nedelec_triangle(degree=1, quad_degree=2)
    assert spec.n_dof == 3 and spec.value_size == 2
    assert spec.ref_values.shape == (len(spec.quad_weights), 3, 2)
    assert spec.ref_div is None and spec.ref_curl is not None
    # N1E0 curl is constant over the cell (identical at every quad point)
    assert np.allclose(spec.ref_curl, spec.ref_curl[0][None, :])


def test_covariant_piola_matches_basix_pushforward():
    spec = nedelec_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    nq = spec.ref_values.shape[0]
    val, _ = piola_covariant(
        jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_curl), jnp.asarray(J), detJ, jnp.ones(spec.n_dof)
    )
    e = basix.create_element(basix.ElementFamily.N1E, basix.CellType.triangle, 1)
    Jb = np.broadcast_to(J, (nq, 2, 2)).copy()
    Kb = np.broadcast_to(np.linalg.inv(J), (nq, 2, 2)).copy()
    oracle = e.push_forward(spec.ref_values, Jb, np.full((nq,), detJ), Kb)
    np.testing.assert_allclose(np.asarray(val), oracle, atol=1e-12)


def test_n1e_curl_normalization_integral_of_curl_is_pm_one():
    # Stokes: ∫_cell curl Φ_e dx = ∮_∂cell Φ_e·t ds = ±1 (the tangential edge DOF). curl is
    # constant per basis, so ∫_cell = curl * area_phys, area_phys = detJ * (ref-tri area 1/2).
    spec = nedelec_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    _, curl = piola_covariant(
        jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_curl), jnp.asarray(J), detJ, jnp.ones(spec.n_dof)
    )
    int_curl = np.asarray(curl)[0] * (detJ * 0.5)
    np.testing.assert_allclose(np.abs(int_curl), np.ones(spec.n_dof), atol=1e-12)


def test_n1e_physical_gradient_offdiag_recovers_curl():
    spec = nedelec_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    signs = jnp.ones(spec.n_dof)
    grad = piola_covariant_grad(jnp.asarray(spec.ref_grads), jnp.asarray(J), detJ, signs)
    _, curl = piola_covariant(jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_curl), jnp.asarray(J), detJ, signs)
    assert grad.shape == (spec.ref_values.shape[0], spec.n_dof, 2, 2)
    # 2-D curl is the antisymmetric off-diagonal of the physical gradient (∂_x u_y - ∂_y u_x)
    curl_from_grad = grad[:, :, 1, 0] - grad[:, :, 0, 1]
    np.testing.assert_allclose(np.asarray(curl_from_grad), np.asarray(curl), atol=1e-12)


def test_n1e_orientation_sign_flips_basis():
    spec = nedelec_triangle(degree=1, quad_degree=2)
    J, detJ = _phys_tri()
    args = (jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_curl), jnp.asarray(J), detJ)
    base, _ = piola_covariant(*args, jnp.ones(spec.n_dof))
    flipped, _ = piola_covariant(*args, jnp.array([-1.0, 1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(flipped[:, 0, :]), -np.asarray(base[:, 0, :]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(flipped[:, 1:, :]), np.asarray(base[:, 1:, :]), atol=1e-12)


# =============================================================================
# Lagrange element factories + identity push-forward  (fem_lagrange.py)
# =============================================================================

from jno.utils.solver.fem_lagrange import (  # noqa: E402
    identity_pushforward,
    lagrange_tet,
    lagrange_triangle,
)


def test_lagrange_p1_spec_shapes():
    spec = lagrange_triangle(1)
    nq = len(spec.quad_weights)
    assert spec.n_dof == 3
    assert spec.value_size == 1
    assert spec.ref_values.shape == (nq, 3, 1)
    assert spec.ref_grads.shape == (nq, 3, 1, 2)
    assert spec.ref_div is None and spec.ref_curl is None


def test_lagrange_p2_spec_shapes():
    spec = lagrange_triangle(2)
    nq = len(spec.quad_weights)
    assert spec.n_dof == 6
    assert spec.ref_values.shape == (nq, 6, 1)
    assert spec.ref_grads.shape == (nq, 6, 1, 2)


def test_lagrange_tet_p1_spec_shapes():
    spec = lagrange_tet(1)
    nq = len(spec.quad_weights)
    assert spec.n_dof == 4
    assert spec.ref_values.shape == (nq, 4, 1)
    assert spec.ref_grads.shape == (nq, 4, 1, 3)


def test_lagrange_tet_p2_spec_shapes():
    spec = lagrange_tet(2)
    assert spec.n_dof == 10


def test_lagrange_partition_of_unity():
    # Sum of all Lagrange basis functions = 1 at every quad point
    for degree in (1, 2):
        spec = lagrange_triangle(degree)
        phi = spec.ref_values[..., 0]  # (n_quad, n_dof)
        np.testing.assert_allclose(
            phi.sum(axis=1), np.ones(phi.shape[0]), atol=1e-13, err_msg=f"P{degree} partition of unity"
        )


def test_lagrange_gradient_sums_to_zero():
    # Sum of gradients of all basis functions = 0 (follows from partition of unity)
    for degree in (1, 2):
        spec = lagrange_triangle(degree)
        dN = spec.ref_grads[..., 0, :]  # (n_quad, n_dof, tdim)
        np.testing.assert_allclose(dN.sum(axis=1), 0.0, atol=1e-13, err_msg=f"P{degree} grad sum")


def test_identity_pushforward_phi_unchanged():
    spec = lagrange_triangle(1)
    J, detJ = _phys_tri()
    phi, _ = identity_pushforward(jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_grads), jnp.asarray(J), detJ)
    # For Lagrange, shape values are coordinate-invariant
    np.testing.assert_allclose(np.asarray(phi), spec.ref_values[..., 0], atol=1e-14)


def test_identity_pushforward_gradient_chain_rule():
    # Physical gradient dphi/dx must equal dphi/dxi @ J^{-1} (chain rule)
    for degree in (1, 2):
        spec = lagrange_triangle(degree)
        J, detJ = _phys_tri()
        _, dphi_phys = identity_pushforward(jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_grads), jnp.asarray(J), detJ)
        K = np.linalg.inv(J)
        dphi_ref = spec.ref_grads[..., 0, :]  # (nq, n_dof, 2)
        expected = np.einsum("qnd,dD->qnD", dphi_ref, K)  # (nq, n_dof, 2)
        np.testing.assert_allclose(np.asarray(dphi_phys), expected, atol=1e-13, err_msg=f"P{degree}")


def test_identity_pushforward_partition_of_unity_gradients_physical():
    # Partition of unity in physical space: sum of physical gradients = 0
    for degree in (1, 2):
        spec = lagrange_triangle(degree)
        J, detJ = _phys_tri()
        _, dphi_phys = identity_pushforward(jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_grads), jnp.asarray(J), detJ)
        np.testing.assert_allclose(np.asarray(dphi_phys).sum(axis=1), 0.0, atol=1e-13, err_msg=f"P{degree}")


# =============================================================================
# Boundary facet connectivity + normals  (fem_facets.py)
# =============================================================================

from jno.utils.solver.fem_facets import (  # noqa: E402
    build_facet_connectivity,
    compute_face_normals,
    facet_quad_data,
)


def _unit_square_mesh():
    """Two P1 triangles on the unit square, split along the diagonal (0,0)-(1,1)."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    return points, cells


def test_facet_connectivity_counts():
    pts, cells = _unit_square_mesh()
    conn = build_facet_connectivity(cells, "triangle")
    assert conn.n_bfaces == 4  # 4 boundary edges, 1 interior diagonal
    assert conn.parent_cell.shape == (4,)
    assert conn.local_face.shape == (4,)
    assert conn.face_nodes.shape == (4, 2)


def test_facet_connectivity_boundary_node_sets():
    pts, cells = _unit_square_mesh()
    conn = build_facet_connectivity(cells, "triangle")
    # All 4 mesh vertices should appear in some boundary face node set
    boundary_verts = set(conn.face_nodes.flatten().tolist())
    assert boundary_verts == {0, 1, 2, 3}


def test_facet_connectivity_interior_diagonal_absent():
    pts, cells = _unit_square_mesh()
    conn = build_facet_connectivity(cells, "triangle")
    # The shared edge is (0,2) — should not appear as a boundary face
    face_sets = [frozenset(row.tolist()) for row in conn.face_nodes]
    assert frozenset({0, 2}) not in face_sets


def test_face_normals_unit_length():
    pts, cells = _unit_square_mesh()
    conn = build_facet_connectivity(cells, "triangle")
    norms = compute_face_normals(pts, conn, cells, "triangle")
    assert norms.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(norms, axis=1), np.ones(4), atol=1e-14)


def test_face_normals_outward_orientation():
    # On the unit square the outward normals must point away from the interior (0.5, 0.5)
    pts, cells = _unit_square_mesh()
    conn = build_facet_connectivity(cells, "triangle")
    norms = compute_face_normals(pts, conn, cells, "triangle")
    interior = np.array([0.5, 0.5])
    for i in range(conn.n_bfaces):
        mid = pts[conn.face_nodes[i]].mean(axis=0)
        assert np.dot(norms[i], mid - interior) > 0, f"face {i} normal points inward"


def test_facet_quad_data_2d():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    gp = np.array([0.0, 0.5, 1.0])
    xq, jac = facet_quad_data(pts, np.array([0, 1]), gp)
    assert xq.shape == (3, 2)
    np.testing.assert_allclose(xq[:, 0], [0.0, 1.0, 2.0], atol=1e-14)
    np.testing.assert_allclose(jac, 2.0, atol=1e-14)


def test_build_facet_connectivity_tet():
    # Two tets sharing a face: 3+3-1 = 5 boundary faces
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    conn = build_facet_connectivity(cells, "tetrahedron")
    assert conn.n_bfaces == 6  # 4 + 4 - 2 (shared face counted once from each, interior)
    assert conn.face_nodes.shape[1] == 3
