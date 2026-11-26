"""``jno.le(v, b, margin=m)`` / ``jno.ge``: a bound on EVERY entry of an array-valued ``v``.

The array goes to the optimiser as one row, its Kreisselmeier-Steinhauser smooth maximum
``KS(v) = log(sum_i exp(rho v_i)) / rho`` with ``rho = log(n) / m``, which lies in ``[max v, max v + m]``:
a design that satisfies the row satisfies every entry, and one whose largest entry is ``m`` below the bound
always satisfies it. ``jno.ge`` takes the smooth minimum ``-KS(-v)`` the same way.

Without ``margin`` an array-valued row is refused: it used to be reduced by its MEAN, so "every cell <= 1"
silently meant "the mean <= 1".

Oracles: the closed form of KS; and the optimum of ``min -sum p  s.t.  p_i <= c_i for every i`` held through the
KS row, where all ``p_i - c_i`` are equal at the optimum, so ``n exp(rho (p_i - c_i)) = 1`` gives
``p_i = c_i - m`` exactly (and ``c_i + m`` for the ``jno.ge`` mirror).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

N, MARGIN = 6, 0.05
CAPS = np.random.default_rng(3).uniform(0.5, 1.5, N)


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
    return np.asarray(crux.eval([node])).reshape(-1)


def _ks(v, margin):
    rho = np.log(v.size) / margin
    return np.log(np.sum(np.exp(rho * (v - v.max())))) / rho + v.max()


def test_the_row_is_the_smooth_maximum_and_bounds_the_largest_entry_within_the_margin():
    v = np.random.default_rng(0).normal(size=40)
    p = _param(v)
    c = jno.le(p, 0.3, margin=MARGIN)
    crux = jno.core([jno.np.sum(p), c])
    row = float(_value(crux, c.residual)[0])
    np.testing.assert_allclose(row, _ks(v, MARGIN) - 0.3, rtol=1e-12)
    assert v.max() - 0.3 <= row <= v.max() - 0.3 + MARGIN


def test_ge_takes_the_smooth_minimum():
    v = np.random.default_rng(1).normal(size=40)
    p = _param(v)
    c = jno.ge(p, -0.2, margin=MARGIN)
    crux = jno.core([jno.np.sum(p), c])
    row = float(_value(crux, c.residual)[0])  # g <= 0 convention: bound - smooth min
    np.testing.assert_allclose(row, -0.2 + _ks(-v, MARGIN), rtol=1e-12)
    assert -0.2 - v.min() <= row <= -0.2 - v.min() + MARGIN


def test_a_margin_on_one_value_is_that_value():
    p = _param([0.7])
    c = jno.le(p, 1.0, margin=MARGIN)
    crux = jno.core([jno.np.sum(p), c])
    np.testing.assert_allclose(_value(crux, c.residual), [0.7 - 1.0], rtol=1e-14)


def test_the_margin_must_be_positive():
    p = _param(np.zeros(3))
    for m in (0.0, -1.0):
        with pytest.raises(ValueError, match="margin"):
            jno.le(p, 1.0, margin=m)


def test_an_array_valued_row_without_a_margin_is_refused_not_averaged():
    p = _param(np.zeros(N))
    p.optimizer(jno.optimizers.gcmma(move=0.2, lower=-1.0, upper=3.0))
    crux = jno.core([-jno.np.sum(p), jno.le(p, 1.0)])
    with pytest.raises(ValueError, match="margin="):
        crux.solve(1)


def test_a_scalar_row_needs_no_margin():
    p = _param(np.zeros(N))
    p.optimizer(jno.optimizers.gcmma(move=0.2, lower=-1.0, upper=3.0))
    crux = jno.core([-jno.np.sum(p), jno.le(jno.np.sum(p), 1.0)])
    crux.solve(30)
    np.testing.assert_allclose(_value(crux, p).sum(), 1.0, atol=1e-6)


@pytest.mark.parametrize("sense", ["le", "ge"])
def test_the_optimum_of_a_bound_on_every_entry_sits_one_margin_inside(sense):
    """min -sum p s.t. p_i <= c_i (every i): p_i = c_i - margin; the mirror min sum p s.t. p_i >= c_i: c_i + margin.
    GCMMA keeps every iterate feasible, so no entry passes its cap at any step."""
    p = _param(np.zeros(N) if sense == "le" else np.full(N, 2.0))  # a feasible start
    p.optimizer(jno.optimizers.gcmma(move=0.2, lower=-1.0, upper=3.0))
    caps = jnp.asarray(CAPS)
    if sense == "le":
        crux = jno.core([-jno.np.sum(p), jno.le(p - caps, 0.0, margin=MARGIN)])
    else:
        crux = jno.core([jno.np.sum(p), jno.ge(p - caps, 0.0, margin=MARGIN)])
    crux.solve(80)
    got = _value(crux, p)
    want = CAPS - MARGIN if sense == "le" else CAPS + MARGIN
    np.testing.assert_allclose(got, want, atol=2e-4)
    if sense == "le":
        assert (got <= CAPS).all()
    else:
        assert (got >= CAPS).all()
