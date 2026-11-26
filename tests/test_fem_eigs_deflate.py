"""``eigs(deflate=Y)``: LOBPCG kept M-orthogonal to a known subspace (Knyazev, SIAM J. Sci. Comput. 23(2),
2001, Sec. 4).

Oracles, all against ``scipy.linalg.eigh`` on the dense pencil: deflating the first ``m`` eigenvectors returns
eigenpairs ``m+1 ... m+k``; a free plate's rigid-body modes written as formulas skip its three zero eigenvalues;
a Neumann Laplacian with ``[1]`` starts at its first non-zero eigenvalue; and a ``Y`` that is NOT an invariant
subspace gives the constrained (Courant-Fischer) eigenvalues, the pencil restricted to ``Y``'s M-orthogonal
complement -- which only comes out right if the residual is projected as well as the iterate.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import scipy.linalg as sla

import jno

R = jno.shape.rect


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _neumann_laplacian(size=0.12):
    d = R(0.0, 0.0, 2.0, 1.0, size=size).domain()
    x, y, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y]), [ui * vi]


def _dense(fem, mass):
    K = np.asarray(fem.operator[0].todense())
    M = np.asarray(jno.fem(list(mass)).operator[0].todense())
    return K, M


def test_deflating_known_eigenvectors_returns_the_next_ones():
    fem, mass = _neumann_laplacian()
    K, M = _dense(fem, mass)
    lam, V = sla.eigh(K, M)
    m, k = 3, 4
    got, X = jno.solve.eigs(k=k, precond=jno.precond.jacobi(), deflate=V[:, :m])(fem.operator[0], jno.fem(mass).operator[0])
    np.testing.assert_allclose(np.asarray(got), lam[m : m + k], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(X).T @ M @ V[:, :m], 0.0, atol=1e-8)  # M-orthogonal to what was deflated


def test_a_neumann_laplacian_with_the_constant_deflated_starts_at_its_first_nonzero_eigenvalue():
    fem, mass = _neumann_laplacian()
    K, M = _dense(fem, mass)
    lam = sla.eigh(K, M, eigvals_only=True)
    assert abs(lam[0]) < 1e-8 * lam[-1]  # the constant: a zero eigenvalue
    got, _X = fem.eigs(mass=mass, k=3, precond=jno.precond.jacobi(), deflate=[1])
    np.testing.assert_allclose(np.asarray(got), lam[1:4], rtol=1e-6)


def test_a_free_plate_skips_its_rigid_body_modes():
    """Plane elasticity with no support: three zero eigenvalues (two translations, one rotation). Written as
    formulas and deflated, the solve returns the first elastic modes."""
    d = R(0.0, 0.0, 3.0, 1.0, size=0.2).domain()
    x, y, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols(value_shape=(2,))
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    eu, ev = jno.np.symgrad(ui, [x, y]), jno.np.symgrad(vi, [x, y])
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    fem = jno.fem([2.0 * ddot(eu, ev) + jno.np.trace(eu) * jno.np.trace(ev)])
    mass = [jno.np.inner(ui, vi, n_contract=1)]
    K, M = _dense(fem, mass)
    lam = sla.eigh(K, M, eigvals_only=True)
    assert np.all(np.abs(lam[:3]) < 1e-8 * lam[-1]) and lam[3] > 1e6 * np.abs(lam[:3]).max()
    got, _X = fem.eigs(mass=mass, k=4, precond=jno.precond.jacobi(), deflate=[(1, 0), (0, 1), (-y, x)])
    np.testing.assert_allclose(np.asarray(got), lam[3:7], rtol=1e-6)


def test_a_non_invariant_space_gives_the_constrained_eigenvalues():
    """``Y`` = the coordinate x: not an eigenvector. The answer is the pencil on ``{v : Y^T M v = 0}``."""
    fem, mass = _neumann_laplacian()
    K, M = _dense(fem, mass)
    pts = np.asarray(fem.field_points[0])
    Y = np.stack([np.ones(len(pts)), pts[:, 0]], axis=1)
    C = sla.null_space(Y.T @ M)  # a basis of the M-orthogonal complement
    want = sla.eigh(C.T @ K @ C, C.T @ M @ C, eigvals_only=True)[:4]
    got, X = jno.solve.eigs(k=4, precond=jno.precond.jacobi(), deflate=Y)(fem.operator[0], jno.fem(mass).operator[0])
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6)
    np.testing.assert_allclose(Y.T @ M @ np.asarray(X), 0.0, atol=1e-8)


def test_refusals():
    fem, mass = _neumann_laplacian(0.3)
    with pytest.raises(ValueError, match="precond"):
        fem.eigs(mass=mass, k=2, deflate=[1])
    with pytest.raises(NotImplementedError, match="LOBPCG"):
        fem.eigs(mass=mass, k=2, sigma=3.0, deflate=[1])
    with pytest.raises(TypeError, match="fem.eigs"):
        jno.solve.eigs(k=2, precond=jno.precond.jacobi(), deflate=[1])
