"""Tests for the residual-based PINN optimizers in ``jno.optimizers`` (lm, dngd, nystrom_ngd,
dsgnar) and the shared machinery under them (``_residuals``, ``_residual_callback``, ``_randnla``).

Exactness checks run on a LINEAR residual, where a Gauss-Newton step is the exact least-squares
solution and every method's answer is known in closed form; the PINN checks reuse the Poisson2D
problem of ``tests/test_engd_optimizer.py``.
"""

from __future__ import annotations

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno.optimizers._residual_callback import ResidualStepCallback, armijo, grid_line_search, half_sq_norm
from jno.optimizers._residuals import jacobian, linearize, second_directional
from tests.test_engd_optimizer import _build_poisson2d


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------------------------
# Shared residual core
# ---------------------------------------------------------------------------------------------


class _Probe(ResidualStepCallback):
    """Records r̂, Ĵ and the solver's gradient at the first step; takes no step itself."""

    who = "probe"

    def on_solve_begin(self, **kwargs):
        self.kwargs = kwargs
        super().on_solve_begin(**kwargs)

    def direction(self, r_of, flat_p, state):
        r = r_of(flat_p)
        J = jacobian(r_of, flat_p, r.shape[0])
        return jnp.zeros_like(flat_p), state, {"r": r, "J": J}

    def on_before_update(self, *, grads, trainable, context, rng, epoch, **kwargs):
        super().on_before_update(grads=grads, trainable=trainable, context=context, rng=rng, epoch=epoch)
        if epoch == 0:
            self.grads0 = jax.flatten_util.ravel_pytree(grads[self._lid])[0]
            self.trainable0 = trainable
        return None


def test_residual_jacobian_reproduces_engd_gram_and_loss_gradient():
    """Ĵᵀ Ĵ is ENGD's Gram matrix, and Ĵᵀ r̂ is the gradient of the mean of the two .mse losses
    (times the number of terms): r̂ and the loss jNO descends are the same objective."""
    import equinox as eqx

    from jno.trace_compiler import TraceCompiler

    net, losses, gram_terms, _ = _build_poisson2d(seed=0)
    net.optimizer(optax.sgd(0.0))
    probe = _Probe(gram_terms)
    crux = jno.core(losses)
    crux.solve(1, callbacks=[probe])

    # reference: the per-point Jacobians of the NetworkGradient nodes themselves (an independent path)
    kw = probe.kwargs
    full = eqx.combine(kw["trainable"], kw["frozen"], kw["static"])
    P = probe.P
    G = jnp.zeros((P, P))
    for expr, w in gram_terms:
        Jk = TraceCompiler.compile_multi_expression([expr], kw["all_ops"])(
            full, kw["context"], batchsize=None, key=kw["rng"], min_consecutive=1
        )[0]
        Jk = Jk.reshape(-1, Jk.shape[-1])
        G = G + (w / Jk.shape[0]) * (Jk.T @ Jk)
    J, r = probe.last_info["J"], probe.last_info["r"]
    assert J.shape == (probe.m, P) and probe.m == 30 * 30 + 4 * 30
    np.testing.assert_allclose(np.asarray(J.T @ J), np.asarray(G), rtol=1e-10, atol=1e-12 * float(jnp.abs(G).max()))
    # loss = mean over the 2 terms of mean(r_k^2) = ||r̂||^2 / 2  ->  grad = Jᵀ r̂
    np.testing.assert_allclose(np.asarray(J.T @ r), np.asarray(probe.grads0), rtol=1e-9, atol=1e-13)


RESIDUAL_SENTINELS = [
    jno.optimizers.engd,
    jno.optimizers.lm,
    jno.optimizers.dngd,
    jno.optimizers.nystrom_ngd,
    jno.optimizers.dsgnar,
]


@pytest.mark.parametrize("make", RESIDUAL_SENTINELS, ids=lambda f: f.__name__)
def test_residual_sentinels_are_not_wrapped_by_bounded(make):
    with pytest.raises(TypeError, match="wraps an optax optimiser"):
        jno.optimizers.bounded(make(), lower=0.0, upper=1.0)


def test_linearize_and_second_directional_match_dense_derivatives():
    A = jax.random.normal(jax.random.PRNGKey(0), (7, 4))

    def r_of(q):
        return jnp.sin(A @ q) + q[0] * jnp.arange(7.0)

    q = jnp.array([0.3, -0.2, 0.1, 0.5])
    lin = linearize(r_of, q)
    J = jacobian(r_of, q, 7)
    v, u = jnp.arange(4.0), jnp.arange(7.0)
    np.testing.assert_allclose(lin.jvp(v), J @ v, rtol=1e-12)
    np.testing.assert_allclose(lin.vjp(u), J.T @ u, rtol=1e-12)
    # d²/dt² sin(A(q+tv)) = -sin(A q) * (A v)²; the linear term has no curvature
    np.testing.assert_allclose(second_directional(r_of, q, v), -jnp.sin(A @ q) * (A @ v) ** 2, rtol=1e-12)


def test_step_length_rules():
    obj = lambda a: (a - 0.25) ** 2  # noqa: E731
    a, f = grid_line_search(obj)
    assert float(a) == 0.25 and float(f) == 0.0
    # Armijo along f(t) = (1 - t)^2 from t=1: f(1) = 0 <= 1 + c1 * 1 * (-2) is accepted at once
    t, ft = armijo(lambda t: (1.0 - t) ** 2, jnp.asarray(1.0), jnp.asarray(-2.0))
    assert float(t) == 1.0 and float(ft) == 0.0
    # ascent direction: no t gives sufficient decrease -> no step
    t, ft = armijo(lambda t: (1.0 + t) ** 2, jnp.asarray(1.0), jnp.asarray(-2.0), max_steps=5)
    assert float(t) == 0.0 and float(ft) == 1.0
    assert float(half_sq_norm(jnp.array([3.0, 4.0]))) == 12.5


# ---------------------------------------------------------------------------------------------
# Pure-step helpers: drive a method's direction() on a hand-written residual
# ---------------------------------------------------------------------------------------------


def _iterate(direction, state, r_of, theta, steps):
    """Apply ``theta <- theta - d`` for ``steps`` steps of a pure direction function."""
    step = jax.jit(lambda th, st: direction(r_of, th, st))
    info = {}
    for _ in range(steps):
        d, state, info = step(theta, state)
        theta = theta - d
    return theta, state, info


def _linear_problem(m, P, seed=0, cond=1e3):
    """r(θ) = A θ - b with a prescribed condition number; returns (r_of, A, b)."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    U, _ = jnp.linalg.qr(jax.random.normal(k1, (m, min(m, P))))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (P, min(m, P))))
    s = jnp.logspace(0, -np.log10(cond), min(m, P))
    A = (U * s) @ V.T
    b = jax.random.normal(k3, (m,))
    return (lambda q: A @ q - b), A, b


def _rosenbrock(q):
    return jnp.array([10.0 * (q[1] - q[0] ** 2), 1.0 - q[0]])


# ---------------------------------------------------------------------------------------------
# lm
# ---------------------------------------------------------------------------------------------

from jno.optimizers.lm import _lm_direction, _lm_init  # noqa: E402


@pytest.mark.parametrize("m,P", [(30, 8), (8, 30)], ids=["overdetermined", "underdetermined"])
def test_lm_reaches_the_least_squares_solution_of_a_linear_residual(m, P):
    r_of, A, b = _linear_problem(m, P)
    theta, state, info = _iterate(_lm_direction, _lm_init(jnp.float64), r_of, jnp.zeros(P), 40)
    if m > P:
        expected = jnp.linalg.lstsq(A, b)[0]
        np.testing.assert_allclose(theta, expected, rtol=1e-8, atol=1e-10)
    else:  # consistent system: the residual vanishes (from 0, LM stays in range(Aᵀ): min-norm)
        np.testing.assert_allclose(A @ theta, b, atol=1e-9)
        np.testing.assert_allclose(theta, jnp.linalg.pinv(A) @ b, rtol=1e-7, atol=1e-9)
    assert float(state.damping) < 1e-6  # the damping falls as the linear model keeps predicting well


def test_lm_never_increases_the_loss_and_solves_rosenbrock():
    step = jax.jit(lambda th, st: _lm_direction(_rosenbrock, th, st))
    theta, state = jnp.array([-1.2, 1.0]), _lm_init(jnp.float64)
    losses = []
    for _ in range(60):
        d, state, info = step(theta, state)
        theta = theta - d
        losses.append(float(half_sq_norm(_rosenbrock(theta))))
    assert all(b <= a * (1 + 1e-14) for a, b in zip(losses, losses[1:]))
    np.testing.assert_allclose(theta, [1.0, 1.0], atol=1e-10)


def test_lm_rejected_step_raises_the_damping_and_keeps_the_parameters():
    # a tiny initial damping on Rosenbrock from far away: the full GN step overshoots, is rejected,
    # and the retry at a larger damping is what gets accepted
    theta = jnp.array([-1.2, 1.0])
    d, state, info = jax.jit(lambda th, st: _lm_direction(_rosenbrock, th, st, max_tries=1))(
        theta, _lm_init(jnp.float64, damping=1e-12)
    )
    if not bool(info["accepted"]):
        assert float(jnp.abs(d).max()) == 0.0
        assert float(state.damping) > 1e-12
    d2, _, info2 = jax.jit(lambda th, st: _lm_direction(_rosenbrock, th, st, max_tries=20))(
        theta, _lm_init(jnp.float64, damping=1e-12)
    )
    assert bool(info2["accepted"])
    assert float(half_sq_norm(_rosenbrock(theta - d2))) < float(half_sq_norm(_rosenbrock(theta)))


def test_lm_rejects_bad_arguments():
    with pytest.raises(ValueError, match="damping"):
        jno.optimizers.lm(damping=0.0)
    with pytest.raises(ValueError, match="max_tries"):
        jno.optimizers.lm(max_tries=0)


# ---------------------------------------------------------------------------------------------
# Through jno.core: Poisson2D (the ENGD convergence test, same bar)
# ---------------------------------------------------------------------------------------------


def _poisson_error(opt, epochs, **build):
    net, losses, _, eval_error = _build_poisson2d(seed=0, **build)
    net.optimizer(opt)
    crux = jno.core(losses)
    crux.solve(epochs)
    return eval_error(crux), crux


def test_lm_converges_on_poisson2d():
    err, _ = _poisson_error(jno.optimizers.lm(), 200)
    assert np.isfinite(err) and err < 1e-3, f"lm: rel-L2 {err:.3e}"


@pytest.mark.parametrize(
    "make",
    [jno.optimizers.lm, jno.optimizers.dngd, jno.optimizers.nystrom_ngd],
    ids=lambda f: f.__name__,
)
def test_residual_sentinel_resolve_is_idempotent(make):
    """A second solve() on the same model rebuilds the callback and continues from the weights."""
    net, losses, _, eval_error = _build_poisson2d(seed=0)
    net.optimizer(make())
    crux = jno.core(losses)
    crux.solve(20)
    e1 = eval_error(crux)
    crux.solve(20)
    e2 = eval_error(crux)
    assert np.isfinite(e2) and e2 <= e1 * 1.01


# ---------------------------------------------------------------------------------------------
# dngd
# ---------------------------------------------------------------------------------------------

from jno.optimizers.dngd import _dngd_direction  # noqa: E402
from jno.utils.explainability import engd_direction  # noqa: E402


def test_dngd_step_is_the_engd_step_written_in_residual_space():
    """-Jᵀ(JJᵀ+λI)⁻¹r = (JᵀJ+λI)⁻¹Jᵀr: the m x m solve and ENGD's P x P damped solve agree."""
    r_of, A, b = _linear_problem(12, 40, cond=1e2)
    theta = jax.random.normal(jax.random.PRNGKey(3), (40,))
    lam = 1e-6
    d, _, _ = _dngd_direction(r_of, theta, (), damping=lam, geodesic=False, line_search=False)
    G, g = A.T @ A, A.T @ r_of(theta)
    d_engd = engd_direction(G, g, jnp.zeros(40), damping=lam)
    np.testing.assert_allclose(d, d_engd, rtol=1e-8, atol=1e-12)


def test_dngd_solves_a_consistent_linear_residual_in_one_step():
    r_of, A, b = _linear_problem(10, 50, cond=1e2)
    theta0 = jnp.zeros(50)
    d, _, info = _dngd_direction(r_of, theta0, (), damping=1e-14)
    theta = theta0 - d
    np.testing.assert_allclose(A @ theta, b, atol=1e-9)
    np.testing.assert_allclose(theta, jnp.linalg.pinv(A) @ b, rtol=1e-7, atol=1e-9)
    assert float(info["eta"]) == 1.0
    assert bool(info["geodesic"])  # f_vv = 0 for a linear residual: a = 0 always passes the ratio test


def test_dngd_geodesic_acceleration_cancels_the_second_order_term():
    """For r(θ) = Aθ + q(θ) with q quadratic, the accelerated step leaves an O(|v|³) residual where
    the plain Gauss-Newton step leaves O(|v|²)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    m, P = 6, 20
    A = jax.random.normal(k1, (m, P))
    B = 0.05 * jax.random.normal(k2, (m, P, P))
    c = 1e-2 * jnp.arange(1.0, m + 1)

    def r_of(q):
        return A @ q + 0.5 * jnp.einsum("ipq,p,q->i", B, q, q) - c

    theta = jnp.zeros(P)
    d_plain, _, _ = _dngd_direction(r_of, theta, (), damping=1e-14, geodesic=False, line_search=False)
    d_acc, _, info = _dngd_direction(r_of, theta, (), damping=1e-14, geodesic=True, line_search=False)
    assert bool(info["geodesic"])
    r_plain = float(jnp.linalg.norm(r_of(theta - d_plain)))
    r_acc = float(jnp.linalg.norm(r_of(theta - d_acc)))
    assert r_acc < 0.05 * r_plain, (r_plain, r_acc)


def test_dngd_converges_on_poisson2d_with_more_parameters_than_rows():
    # hidden 64 x 2 layers: P = 4417 > m = 1020, the regime the residual-space solve is for
    err, _ = _poisson_error(jno.optimizers.dngd(), 200, hidden_dims=64, num_layers=2)
    assert np.isfinite(err) and err < 1e-3, f"dngd: rel-L2 {err:.3e}"


# ---------------------------------------------------------------------------------------------
# _randnla: Nyström approximation, preconditioner, PCG
# ---------------------------------------------------------------------------------------------

from jno.optimizers._randnla import nystrom_approx, nystrom_precond, pcg  # noqa: E402


def _psd(n, rank, seed=0, decay=1.0):
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (n, rank)))
    lam = 10.0 ** (-decay * jnp.arange(rank))
    return (Q * lam) @ Q.T


@pytest.mark.parametrize("sketch", ["gaussian", "sampling"])
def test_nystrom_is_exact_on_a_low_rank_psd_matrix(sketch):
    A = _psd(40, 6)
    ny = nystrom_approx(lambda v: A @ v, 40, 10, jax.random.PRNGKey(1), jnp.float64, sketch=sketch)
    np.testing.assert_allclose((ny.U * ny.lam) @ ny.U.T, A, atol=1e-10)
    assert bool(jnp.all(ny.lam >= 0)) and bool(jnp.all(jnp.diff(ny.lam) <= 1e-14))


def test_nystrom_survives_an_indefinite_operator():
    A = _psd(30, 5) - 0.3 * jnp.eye(30)
    ny = nystrom_approx(lambda v: A @ v, 30, 8, jax.random.PRNGKey(2), jnp.float64)
    assert bool(jnp.all(jnp.isfinite(ny.U))) and bool(jnp.all(ny.lam >= 0))


def test_pcg_with_a_nystrom_preconditioner_converges_in_a_few_iterations():
    """A = (rank-8 PSD, eigenvalues 1..1e-7) + 1e-8 I: cond ~1e8 with 9 distinct eigenvalues, so plain
    CG needs about 9 iterations; with the rank-12 preconditioner the spectrum collapses to one value."""
    n, mu = 60, 1e-8
    A = _psd(n, 8) + mu * jnp.eye(n)
    b = jax.random.normal(jax.random.PRNGKey(4), (n,))
    x_true = jnp.linalg.solve(A, b)
    ny = nystrom_approx(lambda v: A @ v - mu * v, n, 12, jax.random.PRNGKey(5), jnp.float64)
    sol = pcg(lambda v: A @ v, b, nystrom_precond(ny, mu, 1.0 / (ny.lam[-1] + mu)), tol=1e-12, maxiter=50)
    assert int(sol.iters) <= 5
    np.testing.assert_allclose(sol.x, x_true, rtol=1e-6)
    plain = pcg(lambda v: A @ v, b, lambda v: v, tol=1e-12, maxiter=50)
    assert int(plain.iters) >= 8 > int(sol.iters)


def test_pcg_stops_at_negative_curvature():
    A = jnp.diag(jnp.array([2.0, 1.0, -1.0]))
    sol = pcg(lambda v: A @ v, jnp.ones(3), lambda v: v, maxiter=10)
    assert bool(sol.negative_curvature) and bool(jnp.all(jnp.isfinite(sol.x)))


# ---------------------------------------------------------------------------------------------
# nystrom_ngd, dngd(solver="nystrom_pcg")
# ---------------------------------------------------------------------------------------------

from jno.optimizers.dngd import _dngd_direction_pcg  # noqa: E402
from jno.optimizers.nystrom_ngd import _NysState, _nystrom_ngd_direction  # noqa: E402


def test_nystrom_ngd_with_full_rank_is_the_damped_gauss_newton_step():
    r_of, A, b = _linear_problem(30, 12, cond=1e3)
    theta = jax.random.normal(jax.random.PRNGKey(6), (12,))
    d, _, info = _nystrom_ngd_direction(r_of, theta, _NysState(jax.random.PRNGKey(0)), rank=12, damping=1e-9)
    g = A.T @ r_of(theta)
    d_ref = jnp.linalg.solve(A.T @ A + 1e-9 * jnp.eye(12), g)
    assert float(info["t"]) == 1.0 and int(info["cg_iters"]) <= 3
    np.testing.assert_allclose(d, d_ref, rtol=1e-8)


def test_nystrom_ngd_solves_a_consistent_linear_residual():
    """The automatic damping max(10 eps lambda_1, 1e-4 L^2) vanishes with the loss, as for a PINN with
    an exact solution; on a consistent system the iteration then converges to it."""
    _, A, _ = _linear_problem(30, 12, cond=1e3)
    x_true = jnp.linspace(-1.0, 1.0, 12)

    def r_of(q):
        return A @ (q - x_true)

    theta, _, _ = _iterate(
        lambda r, q, s: _nystrom_ngd_direction(r, q, s, rank=12), _NysState(jax.random.PRNGKey(0)), r_of, jnp.zeros(12), 15
    )
    np.testing.assert_allclose(theta, x_true, atol=1e-8)


def test_dngd_matrix_free_step_matches_the_dense_step():
    r_of, A, b = _linear_problem(15, 60, cond=1e2)
    theta = jax.random.normal(jax.random.PRNGKey(8), (60,))
    kw = dict(damping=1e-8, geodesic=False, line_search=False)
    d_dense, _, _ = _dngd_direction(r_of, theta, (), **kw)
    d_pcg, _, info = _dngd_direction_pcg(r_of, theta, jax.random.PRNGKey(0), rank=15, **kw)
    np.testing.assert_allclose(d_pcg, d_dense, rtol=1e-8)
    assert int(info["cg_iters"]) <= 3


def test_nystrom_ngd_rank_rule():
    from jno.optimizers.nystrom_ngd import _next_rank

    grow = {"lambda_min_sketch": 1.0, "damping": 1e-3, "n_above_10mu": 100}
    assert _next_rank(100, grow, 20, 500) == 120
    assert _next_rank(490, grow, 20, 500) == 500
    shrink = {"lambda_min_sketch": 1e-5, "damping": 1e-3, "n_above_10mu": 41}
    assert _next_rank(100, shrink, 20, 500) == 60  # 41 eigenvalues above 10 mu -> next multiple of 20 above
    assert _next_rank(100, {**shrink, "n_above_10mu": 0}, 20, 500) == 20


def test_nystrom_ngd_converges_on_poisson2d():
    err, _ = _poisson_error(jno.optimizers.nystrom_ngd(rank=100), 200)
    assert np.isfinite(err) and err < 1e-3, f"nystrom_ngd: rel-L2 {err:.3e}"


def test_dngd_matrix_free_converges_on_poisson2d():
    err, _ = _poisson_error(
        jno.optimizers.dngd(solver="nystrom_pcg", rank=500, cg_maxiter=200), 200, hidden_dims=64, num_layers=2
    )
    assert np.isfinite(err) and err < 1e-3, f"dngd(nystrom_pcg): rel-L2 {err:.3e}"


# ---------------------------------------------------------------------------------------------
# dsgnar
# ---------------------------------------------------------------------------------------------

import importlib  # noqa: E402

_ds = importlib.import_module("jno.optimizers.dsgnar")  # the module; jno.optimizers.dsgnar is the factory


def _sketch(m, P, s, seed=0):
    return _ds._draw_sketch(jax.random.PRNGKey(seed), m, P, s, 2, jnp.float64)


def test_dsgnar_parameter_sketch_is_an_isometry_and_the_row_sketch_is_linear():
    m, P, s = 50, 37, 12
    sk = _sketch(m, P, s)
    pt = jax.random.normal(jax.random.PRNGKey(1), (s,))
    np.testing.assert_allclose(jnp.linalg.norm(_ds._lift(sk, pt, P)), jnp.linalg.norm(pt), rtol=1e-12)
    a, b = jax.random.normal(jax.random.PRNGKey(2), (2, m))
    np.testing.assert_allclose(
        _ds._count_sketch(sk, a + 2 * b, s), _ds._count_sketch(sk, a, s) + 2 * _ds._count_sketch(sk, b, s), atol=1e-12
    )
    # each row lands in K=2 buckets with weight 1/sqrt(2): ||C e_j||^2 = 1 unless both hashes collide
    col = _ds._count_sketch(sk, jax.nn.one_hot(3, m), s)
    assert float(jnp.sum(col**2)) in (pytest.approx(1.0), pytest.approx(2.0), pytest.approx(0.0))


def test_dsgnar_damping_for_a_radius_meets_the_radius():
    sig = jnp.array([3.0, 1.0, 1e-3])
    b = jnp.array([1.0, -2.0, 0.5])
    for delta in (1e-2, 0.3, 1.0):
        lam = _ds._lm_lambda_for_radius(sig, b, delta)
        np.testing.assert_allclose(jnp.linalg.norm(sig / (sig**2 + lam) * b), delta, rtol=1e-8)
    # the undamped step is shorter than a huge radius: no damping
    assert float(_ds._lm_lambda_for_radius(sig, b, 1e6)) == 0.0


def test_dsgnar_stage_switch_follows_the_trend_of_log_damping():
    W = _ds._WINDOW
    rising = jnp.linspace(-10.0, -5.0, W)
    falling = rising[::-1]
    assert bool(_ds._stage_switch(rising, jnp.asarray(W)))
    assert not bool(_ds._stage_switch(falling, jnp.asarray(W)))
    assert not bool(_ds._stage_switch(rising, jnp.asarray(W - 1)))  # window not yet full


def test_dsgnar_never_increases_the_loss_and_solves_a_consistent_linear_residual():
    m, P = 40, 12
    _, A, _ = _linear_problem(m, P, cond=1e2)
    x_true = jnp.linspace(-1.0, 1.0, P)

    def r_of(q):
        return A @ (q - x_true)

    sk = _sketch(m, P, P)
    state = _ds._DSGNARState(sk, jnp.asarray(1.0), jnp.asarray(False), jnp.zeros(_ds._WINDOW), jnp.asarray(0))
    step = jax.jit(lambda th, st: _ds._dsgnar_direction(r_of, th, st, s=P))
    theta, losses = jnp.zeros(P), []
    for _ in range(40):
        d, state, info = step(theta, state)
        theta = theta - d
        losses.append(float(half_sq_norm(r_of(theta))))
    assert all(b <= a for a, b in zip(losses, losses[1:]))
    np.testing.assert_allclose(theta, x_true, atol=1e-7)


def test_dsgnar_makes_progress_on_poisson2d():
    """A characterisation, not the paper's accuracy: at this size (129 parameters, a 43 x 43 sketch
    of 1020 residual rows) the sketched gradient's sign is close to random, so many steps are
    rejected and the path is sensitive to rounding: run alone the error reached 3.4e-2, after the
    rest of this file 1.1e-1 (lm: 4.9e-5 on the same problem). The bar only asks for a 3x fall from
    a standard initialisation; from the 0.1-scaled one of the other tests it accepts no step at all."""
    err, _ = _poisson_error(jno.optimizers.dsgnar(), 200, init_scale=1.0)
    assert np.isfinite(err) and err < 0.3, f"dsgnar: rel-L2 {err:.3e}"


# ---------------------------------------------------------------------------------------------
# nncg (an optax transform)
# ---------------------------------------------------------------------------------------------


def _optax_minimise(opt, loss, x, steps):
    @jax.jit
    def step(x, state):
        value, grad = jax.value_and_grad(loss)(x)
        updates, state = opt.update(grad, state, x, value=value, value_fn=loss)
        x = optax.apply_updates(x, updates)
        return x, state, loss(x)

    state = opt.init(x)
    values = []
    for _ in range(steps):
        x, state, v = step(x, state)
        values.append(float(v))
    return x, values


def test_nncg_handles_the_indefinite_hessian_of_rosenbrock():
    def loss(q):
        return 100.0 * (q[1] - q[0] ** 2) ** 2 + (1.0 - q[0]) ** 2

    x, values = _optax_minimise(jno.optimizers.nncg(1e-3, rank=2, precond_every=1), loss, jnp.array([-1.2, 1.0]), 200)
    assert all(b <= a for a, b in zip(values, values[1:]))  # Armijo: never uphill
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-6)


def test_nncg_needs_value_fn():
    opt = jno.optimizers.nncg()
    x = jnp.zeros(3)
    with pytest.raises(ValueError, match="value_fn"):
        opt.update(x, opt.init(x), x)


def test_nncg_fine_tunes_poisson2d():
    """A characterisation, matching the paper's use as a fine-tuner: 200 NNCG steps take the 0.1-scaled
    start to rel-L2 7.9e-3 on GPU (measured; ENGD/LM reach < 1e-6 here), below Adam's 0.32 after 1000.
    On CPU the same run ends between 4e-2 and 1e-1 (measured): the CG solves are inexact, so rounding
    differences between the backends change the path. Hence a bound per backend."""
    err, _ = _poisson_error(jno.optimizers.nncg(cg_maxiter=100), 200)
    bound = 2e-2 if jax.default_backend() == "gpu" else 0.15
    assert np.isfinite(err) and err < bound, f"nncg on {jax.default_backend()}: rel-L2 {err:.3e}"


# ---------------------------------------------------------------------------------------------
# score (an optax transform)
# ---------------------------------------------------------------------------------------------


def _quadratic(seed=0, n=6, cond=1e3):
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (n, n)))
    A = (Q * jnp.logspace(0, np.log10(cond), n)) @ Q.T
    x_star = jnp.arange(1.0, n + 1)
    return (lambda q: 0.5 * (q - x_star) @ A @ (q - x_star)), x_star


def test_score_minimises_an_ill_conditioned_quadratic():
    loss, x_star = _quadratic()
    x, values = _optax_minimise(jno.optimizers.score(), loss, jnp.zeros(6), 60)
    assert all(b <= a * (1 + 1e-12) + 1e-300 for a, b in zip(values, values[1:]))
    np.testing.assert_allclose(x, x_star, rtol=1e-8)


def test_score_minimises_rosenbrock():
    def loss(q):
        return 100.0 * (q[1] - q[0] ** 2) ** 2 + (1.0 - q[0]) ** 2

    x, values = _optax_minimise(jno.optimizers.score(), loss, jnp.array([-1.2, 1.0]), 200)
    assert all(b <= a * (1 + 1e-12) for a, b in zip(values, values[1:]))
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-6)


def test_score_rejects_bad_constants():
    with pytest.raises(ValueError, match="c1 < c2"):
        jno.optimizers.score(c1=0.9, c2=0.5)
    with pytest.raises(ValueError, match="mu_min"):
        jno.optimizers.score(mu_min=2.0, mu_max=1.0)


def test_score_matches_ssbroyden_on_poisson2d():
    """A quasi-Newton method needs more steps than the Gauss-Newton ones: 500 from the 0.1-scaled
    start (measured rel-L2 8.8e-4 for score, 4.5e-4 for ssbroyden on the GPU; 1.2e-3 and 3.2e-4 on
    the CPU: the paths differ by round-off, so the comparison is to an order of magnitude). The
    paper's advantage over SSBroyden was shown after Adam pre-training and 150 000 refinement steps,
    not at this size."""
    err_score, _ = _poisson_error(jno.optimizers.score(), 500)
    err_ssb, _ = _poisson_error(jno.optimizers.ssbroyden(), 500)
    assert np.isfinite(err_score) and err_score < 2e-3, f"score: rel-L2 {err_score:.3e}"
    assert err_score < 10.0 * err_ssb, (err_score, err_ssb)


# ---------------------------------------------------------------------------------------------
# ss_esoap (an optax transform)
# ---------------------------------------------------------------------------------------------

from jno.optimizers.soap import Preconditioner  # noqa: E402
from jno.optimizers.ss_esoap import _leaf_step  # noqa: E402

_ESOAP_KW = dict(b1=0.9, b2=0.95, eps=1e-8, tau_min=1e-2, trigger=0.2, check_every=1, warmup=0)


def test_ss_esoap_fits_a_matrix_least_squares_problem():
    A = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
    B = jax.random.normal(jax.random.PRNGKey(1), (10, 3))

    def loss(p):
        return 0.5 * jnp.sum((A @ p["W"] + p["b"] - B) ** 2)

    X = jnp.concatenate([A, jnp.ones((10, 1))], 1)
    best = float(0.5 * jnp.sum((X @ jnp.linalg.lstsq(X, B)[0] - B) ** 2))
    opt = jno.optimizers.ss_esoap(1e-2)
    p = {"W": jnp.zeros((4, 3)), "b": jnp.zeros(3)}
    state = opt.init(p)
    step = jax.jit(lambda q, st: opt.update(jax.grad(loss)(q), st, q))
    for _ in range(3000):
        u, state = step(p, state)
        p = optax.apply_updates(p, u)
    assert float(loss(p)) < best * (1 + 1e-3), (float(loss(p)), best)


def _leaf_inputs(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    g, w, w_prev, g_prev = (jax.random.normal(kk, (4, 3)) for kk in k)
    L = jnp.eye(4) + 0.5 * jnp.ones((4, 4))  # far from diagonal in the identity basis
    R = jnp.eye(3)
    gg, q = Preconditioner([L, R]), Preconditioner([jnp.eye(4), jnp.eye(3)])
    return g, w, w_prev, g_prev, gg, q


def test_ss_esoap_secant_scale_only_ever_enlarges_the_update_uniformly():
    g, w, w_prev, _, gg, q = _leaf_inputs()
    g_prev = g - 3.0 * (w - w_prev)  # c = <y, s> = 3 ||s||^2 > 0
    m = v = jnp.zeros((4, 3))
    t = jnp.asarray(5.0)
    u_scaled = _leaf_step(g, w, w_prev, g_prev, m, v, gg, q, t, **_ESOAP_KW)[0]
    u_plain = _leaf_step(g, w, w_prev, g_prev, m, v, gg, q, t, **{**_ESOAP_KW, "tau_min": 1.0})[0]
    ratio = u_scaled / u_plain
    np.testing.assert_allclose(ratio, ratio[0, 0] * jnp.ones_like(ratio), rtol=1e-10)
    assert 1.0 <= float(ratio[0, 0]) <= 10.0 + 1e-12  # tau^{-1/2} with tau in [tau_min = 1e-2, 1]


def test_ss_esoap_refreshes_a_drifted_basis_and_downscales_the_variance():
    g, w, w_prev, g_prev, gg, q = _leaf_inputs()
    m, v = jnp.zeros((4, 3)), jnp.ones((4, 3))
    _, _, v_new, gg_new, q_new = _leaf_step(g, w, w_prev, g_prev, m, v, gg, q, jnp.asarray(2.0), **_ESOAP_KW)
    QL = q_new.matrices[0]
    B = QL.T @ gg_new.matrices[0] @ QL
    assert float(jnp.linalg.norm(B - jnp.diag(jnp.diag(B)))) < 1e-10 * float(jnp.linalg.norm(B))  # eigenbasis now
    # v <- gamma v (gamma <= 0.75) before the usual update: v_new <= 0.95 * 0.75 + 0.05 * g~^2
    gt = q_new.matrices[0].T @ g @ q_new.matrices[1]
    assert bool(jnp.all(v_new <= 0.95 * 0.75 + 0.05 * gt**2 + 1e-12))
    # with the trigger out of reach the basis stays
    _, _, v_keep, _, q_keep = _leaf_step(
        g, w, w_prev, g_prev, m, v, gg, q, jnp.asarray(2.0), **{**_ESOAP_KW, "trigger": 2.0}
    )
    np.testing.assert_array_equal(q_keep.matrices[0], jnp.eye(4))
    np.testing.assert_allclose(v_keep, 0.95 + 0.05 * g**2)


def test_ss_esoap_trains_poisson2d_far_better_than_adam():
    """3000 steps at lr 1e-3 from the 0.1-scaled start: rel-L2 7.9e-3 measured, Adam 0.29, SOAP 1.4."""
    err, _ = _poisson_error(jno.optimizers.ss_esoap(1e-3), 3000)
    assert np.isfinite(err) and err < 3e-2, f"ss_esoap: rel-L2 {err:.3e}"


# ---------------------------------------------------------------------------------------------
# engd builds its Gram matrix from the chunked residual Jacobian
# ---------------------------------------------------------------------------------------------


def test_engd_runs_with_thousands_of_parameters():
    """P = 4417 (two hidden layers of 64) on the 900-point Laplacian residual: the Gram matrix used to
    come from one jacfwd over every parameter at once, whose intermediates (P copies of the residual's)
    did not fit the GPU; built 64 columns at a time it does."""
    err, crux = _poisson_error(jno.optimizers.engd(), 3, hidden_dims=64, num_layers=2)
    assert np.isfinite(err)
