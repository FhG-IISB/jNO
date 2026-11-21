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


RESIDUAL_SENTINELS = [jno.optimizers.engd]


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
