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
    from jno.utils.explainability import make_engd_fn

    net, losses, gram_terms, _ = _build_poisson2d(seed=0)
    net.optimizer(optax.sgd(0.0))
    probe = _Probe(gram_terms)
    crux = jno.core(losses)
    crux.solve(1, callbacks=[probe])

    kw = probe.kwargs
    gram_fn, _ = make_engd_fn(gram_terms, kw["all_ops"], None, kw["frozen"], kw["static"], probe._lid, kw["trainable"])
    P = probe.P
    _, G = gram_fn(kw["trainable"], kw["context"], kw["rng"], jnp.zeros(P), jnp.zeros(P))
    J, r = probe.last_info["J"], probe.last_info["r"]
    assert J.shape == (probe.m, P) and probe.m == 30 * 30 + 4 * 30
    np.testing.assert_allclose(np.asarray(J.T @ J), np.asarray(G), rtol=1e-10, atol=1e-12 * float(jnp.abs(G).max()))
    # loss = mean over the 2 terms of mean(r_k^2) = ||r̂||^2 / 2  ->  grad = Jᵀ r̂
    np.testing.assert_allclose(np.asarray(J.T @ r), np.asarray(probe.grads0), rtol=1e-9, atol=1e-13)


RESIDUAL_SENTINELS = [jno.optimizers.engd, jno.optimizers.lm, jno.optimizers.dngd]


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


@pytest.mark.parametrize("make", [jno.optimizers.lm, jno.optimizers.dngd], ids=lambda f: f.__name__)
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
