"""Reconstruct a 3-D Nédélec (H(curl)) field and its curl at tet centroids.

``n1e_field_at_tet_centroids`` reads a solved N1E edge-DOF vector back as a physical vector field at
each tet centroid — the tetrahedral counterpart of ``n1e_field_at_centroids`` (triangles). It is how a
magnetic-vector-potential solve is post-processed into ``A`` and ``B = curl A`` for plotting/quantities.

Validated against a field that lies exactly in the lowest-order N1E space: ``u* = (-y/2, x/2, 0)``,
which has constant ``curl u* = (0, 0, 1)``. Projecting ``u*`` to edge DOFs and reconstructing must
recover both the value (pointwise, since ``u*`` is affine and in the space) and the constant curl.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _project_and_reconstruct(mesh_size):
    from jno.utils.solver.fem_nonnodal import n1e_field_at_tet_centroids
    from jno.utils.solver.fem_topology import BASIX_TET_EDGES, build_edge_topology

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    ustar = jno.np.vector(-0.5 * y, 0.5 * x, 0.0 * z)  # curl = (0, 0, 1)

    M = np.asarray(jnp.asarray(jno.fem([inner(ui, vi)]).operator[0].todense()))
    b = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - inner(ustar, vi)]).b)).reshape(-1)
    A = jnp.asarray(np.linalg.solve(M, b))  # L²-projection of u* onto the N1E space

    pts = np.asarray(d.mesh.points)
    cells = np.asarray(d.mesh.cells_dict["tetra"])
    top = build_edge_topology(cells, BASIX_TET_EDGES)
    val, crl = n1e_field_at_tet_centroids(pts, cells, top, A, curl=True)
    cent = pts[cells].mean(axis=1)
    exact = np.stack([-0.5 * cent[:, 1], 0.5 * cent[:, 0], 0.0 * cent[:, 2]], axis=1)
    return np.asarray(val), np.asarray(crl), exact


def test_tet_value_reconstruction_is_exact_for_affine_field():
    """``u*`` is affine and lies in the lowest-order N1E space, so its projection is itself and the
    reconstructed centroid values match ``u*(centroid)`` to solver tolerance."""
    val, _, exact = _project_and_reconstruct(0.34)
    np.testing.assert_allclose(val, exact, atol=1e-6)


def test_tet_curl_reconstruction_recovers_constant_curl():
    """The curl recovered from the antisymmetric parts of the physical gradient equals the analytic
    constant ``curl u* = (0, 0, 1)`` at every centroid — validating the edge-orientation signs and the
    covariant push-forward together (this is the ``B = curl A`` post-processing path)."""
    _, crl, _ = _project_and_reconstruct(0.34)
    np.testing.assert_allclose(crl, np.tile([0.0, 0.0, 1.0], (crl.shape[0], 1)), atol=1e-5)


def test_reconstruction_is_complex_linear():
    """A complex edge vector reconstructs to a complex field (the map is linear), so eddy-current
    ``A = A_r + i A_i`` post-processes in one call."""
    from jno.utils.solver.fem_nonnodal import n1e_field_at_tet_centroids
    from jno.utils.solver.fem_topology import BASIX_TET_EDGES, build_edge_topology

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    ustar = jno.np.vector(-0.5 * y, 0.5 * x, 0.0 * z)
    M = np.asarray(jnp.asarray(jno.fem([inner(ui, vi)]).operator[0].todense()))
    b = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - inner(ustar, vi)]).b)).reshape(-1)
    A = jnp.asarray(np.linalg.solve(M, b)) * (1.0 + 2.0j)  # scale into the complex plane
    pts, cells = np.asarray(d.mesh.points), np.asarray(d.mesh.cells_dict["tetra"])
    top = build_edge_topology(cells, BASIX_TET_EDGES)
    val = np.asarray(n1e_field_at_tet_centroids(pts, cells, top, A))
    assert np.iscomplexobj(val) and np.max(np.abs(val.imag)) > 0.0
