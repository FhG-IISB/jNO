"""MMA's history (asymptotes, the two previous iterates, the step count) carries over between ``jno.core.solve``
calls, as an optax state does.

Before, ``jno.core.solve`` built a new MMA callback on every call, so a run split into several ``solve`` calls
restarted the asymptotes at their initial distance at every call."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rosenbrock():
    """min (1 - p0)^2 + 10 (p1 - p0^2)^2 subject to p0 + p1 <= 1.2, from (-1, 1.5)"""
    p = jno.np.parameter((2,), name="p").initialize(lambda *_a, **_k: jnp.array([-1.0, 1.5]))
    p.dtype(jnp.float64)
    p.optimizer(jno.optimizers.mma(move=0.2, lower=-2.0, upper=2.0))
    f = (1.0 - p[0]) ** 2 + 10.0 * (p[1] - p[0] ** 2) ** 2
    return jno.core([f, jno.le(p[0] + p[1], 1.2)], domain=_DUMMY), p


def _p(crux, p):
    return np.asarray(crux.eval([p])).reshape(-1)


def test_a_second_solve_continues_mma_where_the_first_stopped():
    one, p1 = _rosenbrock()
    one.solve(8)
    two, p2 = _rosenbrock()
    two.solve(4)
    two.solve(4)
    np.testing.assert_allclose(_p(two, p2), _p(one, p1), rtol=1e-12)
    assert two._mma_cb._k == 8
    # the old behaviour, a restart at the second call, takes another path: the comparison above can fail
    three, p3 = _rosenbrock()
    three.solve(4)
    three._mma_cb = None
    three.solve(4)
    assert np.abs(_p(three, p3) - _p(one, p1)).max() > 1e-6


def _two_blocks(b_frozen):
    """the Rosenbrock problem in `a` and an independent (b - 0.5)^2 in `b`: the subproblem separates, so `a` takes
    the same steps whether or not `b` is a design variable (to the dual solve's tolerance: 3e-6 apart after 8 steps,
    while a restarted history is 1.0 apart)"""
    a = jno.np.parameter((2,), name="a").initialize(lambda *_a, **_k: jnp.array([-1.0, 1.5]))
    a.dtype(jnp.float64)
    a.optimizer(jno.optimizers.mma(move=0.2, lower=-2.0, upper=2.0))
    f = (1.0 - a[0]) ** 2 + 10.0 * (a[1] - a[0] ** 2) ** 2
    if b_frozen is None:
        return jno.core([f, jno.le(a[0] + a[1], 1.2)], domain=_DUMMY), a, None
    b = jno.np.parameter((1,), name="b").initialize(lambda *_a, **_k: jnp.array([-1.0]))
    b.dtype(jnp.float64)
    b.optimizer(jno.optimizers.mma(move=0.2, lower=-2.0, upper=2.0))
    if b_frozen:
        b.freeze()
    crux = jno.core([f + (b[0] - 0.5) ** 2, jno.le(a[0] + a[1], 1.2)], domain=_DUMMY)
    return crux, a, b


def test_a_block_unfrozen_between_solves_keeps_the_history_of_the_others():
    """`b` frozen for 4 steps, then trained: `a` continues its 8-step path (its history is not restarted when the
    design vector grows) and `b` takes the steps of a fresh start"""
    ref, a_ref, _ = _two_blocks(None)
    ref.solve(8)
    fresh, _, b_fresh = _two_blocks(False)
    fresh.solve(4)

    crux, a, b = _two_blocks(True)
    crux.solve(4)
    np.testing.assert_allclose(_p(crux, b), [-1.0])  # frozen: not moved, not a design variable
    assert len(crux._mma_cb._blocks) == 1  # only `a` is a design variable while `b` is frozen
    b.unfreeze()
    crux.solve(4)
    assert crux._mma_cb._k == 8
    np.testing.assert_allclose(_p(crux, a), _p(ref, a_ref), atol=1e-5)
    np.testing.assert_allclose(_p(crux, b), _p(fresh, b_fresh), atol=1e-5)


def test_a_block_frozen_between_solves_keeps_the_history_of_the_others():
    """the reverse: both trained for 4 steps, then `b` frozen -- `a` goes on as in an 8-step run"""
    ref, a_ref, _ = _two_blocks(None)
    ref.solve(8)
    crux, a, b = _two_blocks(False)
    crux.solve(4)
    b_at = _p(crux, b)
    b.freeze()
    crux.solve(4)
    assert crux._mma_cb._k == 8
    np.testing.assert_allclose(_p(crux, a), _p(ref, a_ref), atol=1e-5)
    np.testing.assert_allclose(_p(crux, b), b_at)
