"""``expr.start``: the value of ``expr`` at the start of the optimisation, a constant.

Relative targets read like the maths: ``jno.le(abs(L / L.start - 1), 0.01)`` holds ``L`` within 1 % of where it
started, and ``f / f.start`` is an objective that starts at 1. ``jno.core`` evaluates every ``.start`` of its
entries (and of what ``eval`` is asked for) once, together, at the first ``solve`` or ``eval``, before any step;
the value then stays, whatever the design does.

Oracle: ``f = sum p^2`` at ``p = (1, 2, 3)`` is 14.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _param(values, name="p"):
    v = jnp.asarray(values, dtype=jnp.float64)
    p = jno.np.parameter(v.shape, name=name).initialize(lambda *_a, **_k: v)
    p.dtype(jnp.float64)
    return p


def _value(crux, node):
    return float(np.asarray(crux.eval([node])).reshape(-1)[0])


def test_start_is_one_node_per_expression():
    p = _param([1.0, 2.0, 3.0])
    f = jno.np.sum(p * p)
    assert f.start is f.start


def test_start_is_the_value_at_the_first_evaluation_and_stays_after_steps():
    p = _param([1.0, 2.0, 3.0])
    f = jno.np.sum(p * p)
    p.optimizer(optax.sgd(0.05))
    crux = jno.core([f / f.start])
    assert _value(crux, f / f.start) == pytest.approx(1.0, rel=1e-14)
    crux.solve(5)
    now = _value(crux, f)
    assert now < 14.0  # the design moved
    assert _value(crux, f.start) == pytest.approx(14.0, rel=1e-14)
    assert _value(crux, f / f.start) == pytest.approx(now / 14.0, rel=1e-12)


def test_the_gradient_of_a_ratio_to_the_start_is_scaled_by_the_start_only():
    """d(f / f0)/dp = 2 p / 14: one SGD step of rate lr from p0 lands at p0 (1 - 2 lr / 14)."""
    p0 = np.array([1.0, 2.0, 3.0])
    p = _param(p0)
    f = jno.np.sum(p * p)
    lr = 0.1
    p.optimizer(optax.sgd(lr))
    crux = jno.core([f / f.start])
    crux.solve(1)
    np.testing.assert_allclose(np.asarray(crux.eval([p])).reshape(-1), p0 * (1 - 2 * lr / 14.0), rtol=1e-12)


def test_a_band_around_the_start_holds_under_a_constrained_optimiser():
    """min sum p (pulls every p down) s.t. |f / f0 - 1| <= 0.01: GCMMA ends with f on the band's lower edge."""
    p = _param([1.0, 2.0, 3.0])
    f = jno.np.sum(p * p)
    p.optimizer(jno.optimizers.gcmma(move=0.05, lower=0.0, upper=4.0))
    crux = jno.core([jno.np.sum(p), jno.le(f / f.start, 1.01), jno.ge(f / f.start, 0.99)])
    crux.solve(60)
    assert _value(crux, f) == pytest.approx(0.99 * 14.0, rel=1e-5)


def test_start_outside_a_core_says_where_it_is_set():
    p = _param([1.0])
    node = jno.np.sum(p).start
    with pytest.raises(ValueError, match="jno.core"):
        node.fn()


def test_a_start_in_a_constraint_s_bound_is_set_before_the_first_step():
    """``jno.ge(v, v.start - 0.5)``: the bound is no child of the constraint's expression, and was never set"""
    p = jno.np.parameter((1,), name="p").initialize(lambda *_a, **_k: jnp.array([2.0]))
    p.dtype(jnp.float64)
    p.optimizer(jno.optimizers.mma(move=0.2, lower=-5.0, upper=5.0))
    v = p[0]
    crux = jno.core([v, jno.ge(v, v.start - 0.5)], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    crux.solve(20)
    np.testing.assert_allclose(np.asarray(crux.eval([p])).reshape(-1), [1.5], atol=1e-3)
