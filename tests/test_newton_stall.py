"""``newton(stall=N)`` ends a Newton solve whose residual has stopped falling, instead of letting it run to
``max_steps``, and raises saying so.

A globalised Newton that has stalled -- at a local minimum of ``||r||`` that is not a root, or where the
line search can only take vanishing steps -- spends every remaining step at the same residual. With a
tangent that is expensive to factorize, that is most of a run's time spent on an answer that is already
known: "not converged". ``stall=N`` stops once ``N`` steps in a row have not brought the residual 1%
below the best it has reached, and the error names the stall (not the step cap) and its residual.

The oracle: ``r(u) = u^2 + 1`` has NO real root; ``||r||`` is smallest (1) at ``u = 0``, so a
line-searched Newton wanders around 0 with ``||r||`` near 1 forever. ``r(u) = u^3 - 1`` has a root at 1,
reached without a stall, so ``stall`` must not change that solve.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


from jno.utils.solver.newton_krylov import LAST_NEWTON_STATS, newton_direct, newton_krylov  # noqa: E402

_dense = lambda A, b: jnp.linalg.solve(A, b)  # noqa: E731


def _rootless(u):
    return jnp.asarray(u) ** 2 + 1.0


def _rootless_jac(u):
    return jnp.diag(2.0 * jnp.asarray(u).reshape(-1))


def _cubic(u):
    return jnp.asarray(u) ** 3 - 1.0


def _cubic_jac(u):
    return jnp.diag(3.0 * jnp.asarray(u).reshape(-1) ** 2)


def _direct(res, jac, u0, **kw):
    return newton_direct(res, jac, u0, linear_solve=_dense, line_search=True, **kw)


@pytest.mark.parametrize("reuse", [False, True])
def test_direct_stops_on_a_stall_long_before_the_step_cap(reuse):
    with pytest.raises(
        RuntimeError, match=r"newton_direct stalled: the residual has not fallen 1% below its best in the last 5 steps"
    ):
        _direct(_rootless, _rootless_jac, jnp.array([0.5]), max_steps=1000, stall=5, reuse=reuse)
    assert LAST_NEWTON_STATS["steps"] < 100 and not LAST_NEWTON_STATS["converged"]


def test_krylov_stops_on_a_stall_long_before_the_step_cap():
    with pytest.raises(RuntimeError, match=r"newton_krylov stalled: .* in the last 5 steps"):
        newton_krylov(_rootless, jnp.array([0.5]), max_steps=1000, line_search=True, stall=5)


def test_without_stall_the_same_solve_runs_to_the_step_cap():
    with pytest.raises(RuntimeError, match="did not converge in max_steps=200"):
        _direct(_rootless, _rootless_jac, jnp.array([0.5]), max_steps=200)
    assert LAST_NEWTON_STATS["steps"] == 200


@pytest.mark.parametrize("reuse", [False, True])
def test_stall_leaves_a_converging_solve_alone(reuse):
    u0 = jnp.array([50.0])
    plain = _direct(_cubic, _cubic_jac, u0, max_steps=100, reuse=reuse)
    steps = LAST_NEWTON_STATS["steps"]
    watched = _direct(_cubic, _cubic_jac, u0, max_steps=100, reuse=reuse, stall=3)
    assert LAST_NEWTON_STATS["steps"] == steps
    np.testing.assert_array_equal(np.asarray(watched), np.asarray(plain))
    np.testing.assert_allclose(np.asarray(watched), [1.0], rtol=1e-3)  # the bound is relative to ||r(50)||


def test_the_newton_slot_takes_stall_and_refuses_a_bad_one():
    import jno

    assert jno.solve.newton(stall=10) is not None
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="stall"):
            jno.solve.newton(stall=bad)
