"""``jno.optimizers.bounded(opt, lower=, upper=)`` -- any optax optimiser, kept inside a box.

    minimise  sum_i (x_i - a_i)**2  over  lower <= x <= upper

has the solution ``clip(a, lower, upper)``. With the budget ``sum x <= V`` added (through
``.auglag()``) the KKT conditions give ``x_i = clip(a_i - lam/2, lower, upper)``; for a = (1, 2, 3, 4),
the box [0, 2] and V = 6 that is ``lam = 1``, ``x = (0.5, 1.5, 2, 2)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno.utils.adaptive.callbacks import Callback

A = np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


class _Range(Callback):
    """The smallest and largest entry of the design after every step."""

    def __init__(self):
        self.lo, self.hi = np.inf, -np.inf

    def on_epoch_end(self, **kw):
        v = np.concatenate([np.asarray(leaf).reshape(-1) for leaf in jax.tree_util.tree_leaves(kw["trainable"])])
        self.lo, self.hi = min(self.lo, v.min()), max(self.hi, v.max())
        return False


def _problem(opt, extra=()):
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    x = jno.np.parameter((A.size,), name="x")
    x.dtype(jnp.float64)
    x.initialize(jax.nn.initializers.constant(0.0))
    x.optimizer(opt)
    f0 = jno.fn(lambda v: jnp.sum((v - jnp.asarray(A)) ** 2), [x], name="f0")
    s = jno.fn(lambda v: jnp.sum(v), [x], name="sum_x")
    return d, x, f0, s


@pytest.mark.parametrize("make", [lambda: optax.adam(5e-2), lambda: optax.sgd(0.2)], ids=["adam", "sgd"])
def test_it_converges_to_the_clipped_optimum_and_never_leaves_the_box(make):
    d, x, f0, _ = _problem(jno.optimizers.bounded(make(), lower=0.0, upper=2.5))
    rng = _Range()
    crux = jno.core([f0], domain=d)
    crux.solve(800, callbacks=[rng])
    np.testing.assert_allclose(np.asarray(crux.eval([x])).reshape(-1), np.clip(A, 0.0, 2.5), atol=1e-6)
    assert rng.lo >= 0.0 and rng.hi <= 2.5


def test_an_optimiser_factory_with_its_learning_rate_from_the_model():
    """``bounded(optax.sgd)``: the rate is ``Model.scale``, which jno.core applies inside the chain.

    One step at rate 0.4 from x = 0 lands at 0.8 a, so the last entry (3.2) must be clipped.
    """
    d, x, f0, _ = _problem(jno.optimizers.bounded(optax.sgd, lower=0.0, upper=2.5))
    x.model.scale(0.4)
    crux = jno.core([f0], domain=d)
    crux.solve(1)
    np.testing.assert_allclose(np.asarray(crux.eval([x])).reshape(-1), [0.8, 1.6, 2.4, 2.5], atol=1e-12)


def test_per_entry_bounds():
    hi = np.array([0.5, 5.0, 5.0, 1.0])
    d, x, f0, _ = _problem(jno.optimizers.bounded(optax.adam(5e-2), upper=hi))
    crux = jno.core([f0], domain=d)
    crux.solve(800)
    np.testing.assert_allclose(np.asarray(crux.eval([x])).reshape(-1), np.minimum(A, hi), atol=1e-6)


def _budget(opt, steps):
    d, x, f0, s = _problem(jno.optimizers.bounded(opt, lower=0.0, upper=2.0))
    c = (s <= 6.0).auglag(every=100)
    crux = jno.core([f0, c], domain=d)
    crux.solve(steps)
    return np.asarray(crux.eval([x])).reshape(-1), float(np.asarray(crux.eval([c.multiplier])).reshape(-1)[0])


def test_with_an_augmented_lagrangian_budget():
    """Adam with a decaying rate, as a long first-order run needs.

    The multiplier update used to average max(c, -lam/mu), which Adam's jitter biases upwards:
    mu grew to 1e7, lam fell to 0 and the design stopped short of the budget.
    """
    x, lam = _budget(optax.adam(optax.exponential_decay(5e-2, 1000, 0.1)), 3000)
    np.testing.assert_allclose(x, [0.5, 1.5, 2.0, 2.0], atol=1e-6)
    assert lam == pytest.approx(1.0, rel=1e-6)


def test_a_settled_multiplier_does_not_raise_mu():
    """At a fixed rate Adam reaches the optimum to 1e-11; a violation that small cannot shrink
    fourfold per window, and growing mu on it made the loss too stiff for Adam (lam thrown to 0)."""
    x, lam = _budget(optax.adam(5e-2), 1400)
    np.testing.assert_allclose(x, [0.5, 1.5, 2.0, 2.0], atol=1e-6)
    assert lam == pytest.approx(1.0, rel=1e-6)


class TestGuards:
    def test_mma_is_not_wrapped(self):
        with pytest.raises(TypeError, match="lower=/upper="):
            jno.optimizers.bounded(jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0), lower=0.0)

    def test_crossed_bounds_are_refused(self):
        with pytest.raises(ValueError, match="lower must not exceed upper"):
            jno.optimizers.bounded(optax.adam(1e-2), lower=1.0, upper=0.0)

    def test_no_bounds_are_refused(self):
        with pytest.raises(ValueError, match="neither"):
            jno.optimizers.bounded(optax.adam(1e-2))
