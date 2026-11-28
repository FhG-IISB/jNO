"""``jno.precond.fdm_schwarz()`` and ``jno.precond.pmg()``: degree-robust preconditioners for ``matrix_free=True``.

The pieces are checked against dense constructions on small problems: the 1-D fast-diagonalisation eigenproblem,
the separable coefficients of an axis-aligned cell, the Q1 Galerkin coarse operator ``P̂ᵀ A P̂`` (Dirichlet rows,
a variable coefficient, a vector field with coupled components), the Schwarz operator's symmetry and definiteness.
Then the solves: the same answer as Jacobi in far fewer iterations, steady linear and Newton-Krylov, with the
coarse operator solved densely or by AMG.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jno.numpy import cos, sin

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _scalar(order=4, n=3, *, coef="variable", sigma=0.5, rough=True, matrix_free=True, nonlinear=False, load=30.0):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=n).quad().domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols(order=order)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    kappa = (1.0 + x * x) if coef == "variable" else 2.0
    if nonlinear:
        kappa = kappa * (1.0 + ui * ui)
    f = 1.0 + (sin(23.0 * x + 7.0 * y) * cos(19.0 * y) if rough else 0.0)
    form = kappa * (ui.x * vi.x + ui.y * vi.y) + sigma * ui * vi - load * f * vi
    return jno.fem([form, u(xb, yb) - 0.0], quadrature="gll", matrix_free=matrix_free)


def _dense(A):
    n = A.shape[0]
    return np.asarray(jax.vmap(A.mv)(jnp.eye(n)).T)


def test_the_1d_fast_diagonalisation_solves_its_extended_problem():
    from jno._spectral.bases import Legendre
    from jno.utils.solver.sem_precond import fdm_1d

    for p in (4, 6, 9):
        S, lam, nodes = fdm_1d(p)
        ax = Legendre(p, 0.0, 1.0)
        w, D = np.asarray(ax.weights), np.asarray(ax.diff_matrix(1))
        K = D.T @ (w[:, None] * D)
        Kt, Mt = K.copy(), np.diag(w)
        Kt[0, 0] += K[-1, -1]
        Kt[-1, -1] += K[0, 0]
        Mt[0, 0] += w[-1]
        Mt[-1, -1] += w[0]
        np.testing.assert_allclose(S.T @ Mt @ S, np.eye(p), atol=1e-12)
        np.testing.assert_allclose(Kt @ S, Mt @ S * lam[None, :], atol=1e-9 * np.abs(Kt).max())
        assert np.all(lam > 0)  # the overlap node's Dirichlet makes the pure stiffness invertible


def test_an_axis_aligned_cell_with_constant_coefficients_has_the_exact_separable_numbers():
    """c_a = κ |det J| (∂ξ_a/∂x_a)² = κ (unit square cells: h^(d-2) = 1 in 2-D), c_0 = σ |det J| = σ h²."""
    n = 4
    fem = _scalar(order=4, n=n, coef="constant", sigma=0.5)
    A, _b = fem.operator
    data = A.sem_fn()
    c = np.asarray(data.coeffs)  # (cells, vec, 1 + tdim)
    np.testing.assert_allclose(c[:, 0, 1:], 2.0, rtol=1e-12)
    np.testing.assert_allclose(c[:, 0, 0], 0.5 / n**2, rtol=1e-12)


@pytest.mark.parametrize("order", [3, 5])
def test_the_coarse_operator_is_the_galerkin_product(order):
    fem = _scalar(order=order, n=3)
    A, _b = fem.operator
    data = A.sem_fn()
    P = np.asarray(data.st.P.todense())
    Ph = np.asarray(data.mask)[:, None] * P
    Ac = Ph.T @ _dense(A) @ Ph
    np.testing.assert_allclose(np.asarray(data.coarse_dense()), Ac, atol=1e-12 * np.abs(Ac).max())
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-13  # the vertex functions are a partition of unity


def test_the_coarse_operator_keeps_the_coupling_between_components():
    J = jno.np
    d = jno.shape.rect(0, 0, 1, 1).structured(n=2).quad().domain(compute_mesh_connectivity=False)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown.vector(2, order=3)
    v = u.test()
    gu, gv = J.jacobian(u, [x, y]), J.jacobian(v, [x, y])
    vi = v.bind(x=x, y=y)
    form = J.inner(gu, gv, n_contract=2) + 3.0 * J.trace(gu) * J.trace(gv) - vi[0]
    fem = jno.fem([form, u(xb, yb) - 0.0], quadrature="gll", matrix_free=True)
    A, _b = fem.operator
    data = A.sem_fn()
    P = np.asarray(data.st.P.todense())
    Ph = np.asarray(data.mask)[:, None] * P
    Ac = Ph.T @ _dense(A) @ Ph
    np.testing.assert_allclose(np.asarray(data.coarse_dense()), Ac, atol=1e-12 * np.abs(Ac).max())


def test_the_schwarz_operator_is_symmetric_positive_and_better_conditioned_than_jacobi():
    fem = _scalar(order=5, n=3)
    A, _b = fem.operator
    data = A.sem_fn()
    Ad = _dense(A)
    Sd = np.asarray(jax.vmap(data.schwarz)(jnp.eye(Ad.shape[0])).T)
    np.testing.assert_allclose(Sd, Sd.T, atol=1e-12 * np.abs(Sd).max())
    free = np.asarray(data.mask) > 0
    ev = np.linalg.eigvals((Sd @ Ad)[np.ix_(free, free)]).real
    evj = np.linalg.eigvals((Ad / np.diag(Ad)[:, None])[np.ix_(free, free)]).real
    assert ev.min() > 0
    assert ev.max() / ev.min() < 0.5 * evj.max() / evj.min()


def _iters(fem):
    return int(fem.stats["linear_iterations"]["total"])


@pytest.mark.parametrize(
    "spec", [jno.precond.fdm_schwarz(coarse=False), jno.precond.fdm_schwarz(), jno.precond.pmg()], ids=repr
)
def test_a_steady_solve_reaches_jacobis_answer_in_fewer_iterations(spec):
    fem = _scalar(order=5, n=6)
    ref = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=jno.precond.jacobi())).reshape(-1)
    n_ref = _iters(fem)
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=spec)).reshape(-1)
    assert np.abs(got - ref).max() < 1e-8 * np.abs(ref).max()
    assert _iters(fem) < 0.6 * n_ref, (_iters(fem), n_ref)


def test_the_amg_coarse_solve_gives_the_same_answer(monkeypatch):
    from jno.utils.solver import sem_precond

    fem = _scalar(order=4, n=8)
    a = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=jno.precond.pmg())).reshape(-1)
    monkeypatch.setattr(sem_precond, "DENSE_COARSE_MAX", 10)  # 81 vertices: an AMG hierarchy now
    b = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=jno.precond.pmg())).reshape(-1)
    assert np.abs(a - b).max() < 1e-8 * np.abs(a).max()


@pytest.mark.parametrize("dense_max", [None, 10])
def test_newton_krylov_takes_the_preconditioners_at_each_iterate(monkeypatch, dense_max):
    from jno.utils.solver import sem_precond

    if dense_max is not None:  # the AMG coarse solve, its hierarchy from the reference state
        monkeypatch.setattr(sem_precond, "DENSE_COARSE_MAX", dense_max)
    fem = _scalar(order=4, n=4, nonlinear=True)
    ref = np.asarray(fem.solve()).reshape(-1)
    n_ref = _iters(fem)
    for spec in (jno.precond.fdm_schwarz(), jno.precond.pmg()):
        got = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.bicgstab(), precond=spec)).reshape(-1)
        assert fem.stats["nonlinear"]["driver"] == "newton_krylov"
        assert np.abs(got - ref).max() < 1e-7 * np.abs(ref).max()
        assert _iters(fem) < 0.6 * n_ref, (spec, _iters(fem), n_ref)


def test_an_operator_that_is_not_matrix_free_spectral_elements_is_refused():
    fem = _scalar(order=4, n=2, matrix_free=False)
    with pytest.raises(ValueError, match="matrix_free"):
        fem.solve(linear=jno.solve.cg(), precond=jno.precond.fdm_schwarz())
    with pytest.raises(ValueError, match="Q_k"):
        _scalar(order=2, n=2).solve(linear=jno.solve.cg(), precond=jno.precond.pmg())
    with pytest.raises(ValueError, match="smoother"):
        jno.precond.pmg(smoother="gauss-seidel")


def test_the_amg_coarse_cycle_contracts_at_a_state_far_from_the_one_it_was_built_at(monkeypatch):
    """The AMG hierarchy is built once, from the coarse operator at the zero state (coefficient 1 + u² = 1); a Newton
    iterate's is several times stiffer. With the finest level's smoothing window frozen at the reference state the
    polynomial amplified there (a GPU Newton solve, Q3 at 551k DOFs, diverged to NaN). The window is measured on each
    build's own coarse operator: the cycle's error propagator ``I - M A_c`` contracts."""
    from jno.utils.solver import sem_precond
    from jno.utils.solver.sem_precond import coarse_levels, coarse_solver

    monkeypatch.setattr(sem_precond, "DENSE_COARSE_MAX", 10)
    fem = _scalar(order=4, n=14, nonlinear=True)
    op = fem._op
    n = int(op.size)
    levels = coarse_levels(op.jacobian_sem_ref())
    assert levels is not None and "Ainv" not in levels[0]  # a real hierarchy (225 vertices)
    stiff = op.jacobian_sem(jnp.full((n,), 3.0))  # 1 + u² = 10 in the interior
    Ac = np.asarray(stiff.coarse_dense())
    cycle = coarse_solver(stiff, levels)
    M = np.asarray(jax.vmap(cycle)(jnp.eye(Ac.shape[0])).T)
    rho = np.abs(np.linalg.eigvals(np.eye(Ac.shape[0]) - M @ Ac)).max()
    assert rho < 1.0, rho


def test_the_amg_hierarchy_is_built_once_per_problem(monkeypatch):
    from jno.utils.solver import sem_precond

    monkeypatch.setattr(sem_precond, "DENSE_COARSE_MAX", 10)
    builds = []
    real = sem_precond.coarse_levels
    monkeypatch.setattr(sem_precond, "coarse_levels", lambda d: builds.append(1) or real(d))
    fem = _scalar(order=4, n=6)
    for _ in range(3):  # a new spec each call, as `fem.solve(precond=jno.precond.pmg())` makes one
        fem.solve(linear=jno.solve.cg(tol=1e-10), precond=jno.precond.pmg())
    assert len(builds) == 1
