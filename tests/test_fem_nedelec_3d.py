"""3D Nédélec (N1E) edge elements — H(curl) mass + curl-curl on a tetrahedral mesh.

The first 3D non-nodal element in ``jno.fem``: lowest-order first-kind Nédélec on tets (6 edge DOFs,
``value_size=3``), the correct H(curl)-conforming discretisation for Maxwell / curl-curl problems. It
reuses the 2D edge-DOF machinery — the covariant Piola ``Φ_phys = J^{-T} Φ_ref`` is dimension-agnostic,
and the (vector) curl is recovered from the covariant physical gradient (its antisymmetric parts), so no
separate 3D curl push-forward is needed. The ``.curl(x, y, z)`` DSL view assembles it directly.

Reference: J.-C. Nédélec, *Mixed finite elements in* R³, Numer. Math. 35 (1980) 315–341 (first-kind
H(curl) edge elements).
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner
_dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_nedelec_tet_element_pushforward():
    """Element level (no assembler): the covariant push-forward gives a symmetric-PD H(curl) mass, and the
    physical curl taken from the covariant gradient (antisymmetric parts) equals the contravariant push of
    the reference curl ``(1/detJ) J·curl_ref`` — the identity that lets ``.curl()`` skip a dedicated 3D curl
    map."""
    from jno.utils.solver.fem_elements import nedelec_tet, piola_covariant, piola_covariant_grad

    spec = nedelec_tet(degree=1, quad_degree=3)
    assert spec.n_dof == 6 and spec.value_size == 3
    rv, rg, qw = jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_grads), jnp.asarray(spec.quad_weights)

    verts = np.array([[0.0, 0, 0], [2.0, 0, 0], [0.3, 1.5, 0], [0.1, 0.2, 1.7]])  # a generic tet
    J = jnp.asarray(np.stack([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]], axis=1))
    detJ = jnp.linalg.det(J)
    signs = jnp.ones(6)
    vals, curl2 = piola_covariant(rv, None, J, detJ, signs)  # 3D N1E: no scalar curl
    assert curl2 is None
    grad = piola_covariant_grad(rg, J, detJ, signs)  # (nq, 6, 3, 3)

    M = jnp.einsum("q,qai,qbi->ab", qw * jnp.abs(detJ), vals, vals)  # H(curl) mass
    assert float(jnp.abs(M - M.T).max()) < 1e-12
    assert float(jnp.linalg.eigvalsh(M).min()) > 0.0

    def _curl(g):  # antisymmetric parts of a (..., 3, 3) gradient tensor -> (..., 3)
        return jnp.stack([g[..., 2, 1] - g[..., 1, 2], g[..., 0, 2] - g[..., 2, 0], g[..., 1, 0] - g[..., 0, 1]], -1)

    curl_from_grad = _curl(grad)
    ref_curl = _curl(rg)  # reference curl vector
    curl_piola = jnp.einsum("ij,qnj->qni", J, ref_curl) / detJ  # contravariant push
    assert float(jnp.abs(curl_from_grad - curl_piola).max()) < 1e-12


def _n1e_cube(mesh_size=0.5):
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    coords = d.variable("interior", split=True)
    xi, yi, zi = coords[0], coords[1], coords[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    return d, (xi, yi, zi), (ui, vi), (cu, cv)


def test_nedelec_tet_mass_and_curlcurl_symmetric_pd():
    """On a tet cube the coercive H(curl) operator ``∫ u·v + ∫ curl u · curl v`` assembles to a symmetric,
    positive-definite matrix of size ``n_edges`` — the whole ``value_size=3`` covariant edge path (mass +
    vector curl-curl through ``.curl(x,y,z)``)."""
    d, _, (ui, vi), (cu, cv) = _n1e_cube(0.5)
    A = _dense(jno.fem([inner(ui, vi) + inner(cu, cv)]).A)
    n_edges = A.shape[0]
    assert A.shape == (n_edges, n_edges)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(A).min()) > 0.0  # PD (the +mass block; pure curl-curl is singular)


def test_nedelec_tet_curl_curl_exact_bilinear():
    """The definitive edge-element check on a **multi-tet** mesh (shared edges of differing local
    orientation — a single tet would be a false green): a field in the N1E0 space, ``u* = ½(−y, x, 0)`` with
    constant ``curl u* = (0,0,1)``, has ``∫|curl u*|² = vol(unit cube) = 1``. Projecting ``u*`` for its edge
    DOFs and evaluating the pure curl-curl form ``uᵀ K u`` must reproduce ``1`` exactly — this validates the
    curl-curl assembly *and* the global tet-edge orientation together."""
    d, (xi, yi, zi), (ui, vi), (cu, cv) = _n1e_cube(0.5)
    _solve = lambda A, b: np.linalg.solve(_dense(A), np.asarray(jnp.asarray(b)).reshape(-1))  # noqa: E731
    M = _dense(jno.fem([inner(ui, vi)]).A)
    b = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - (-0.5 * yi * vi[0] + 0.5 * xi * vi[1] + 0.0 * vi[2])]).b))
    u_dof = np.linalg.solve(M, b.reshape(-1))
    K = _dense(jno.fem([inner(cu, cv)]).A)  # pure curl-curl (no mass)
    np.testing.assert_allclose(float(u_dof @ K @ u_dof), 1.0, atol=1e-9)


def test_3d_nonnodal_only_nedelec_supported():
    """On a 3D tet mesh only N1E is wired; the 2D-only families (here Morley) must raise a clear
    NotImplementedError rather than silently mis-map their 2D edge/vertex machinery onto a tet."""
    d = jno.domain(constructor=jno.domain.cube(mesh_size=0.6))
    u, phi = d.fem_symbols(space="Morley")
    coords = d.variable("interior", split=True)
    xi, yi, zi = coords[0], coords[1], coords[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    with pytest.raises(NotImplementedError, match="3D"):
        jno.fem([inner(jno.np.hessian(ui, [xi, yi, zi]), jno.np.hessian(vi, [xi, yi, zi]), n_contract=2) - vi]).A
