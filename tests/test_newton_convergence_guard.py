"""A Newton driver that exits on its step cap must RAISE, not return the last iterate.

Both drivers iterate ``while (||r|| > atol + rtol*||r(u0)||) & (k < max_steps)``. Leaving on the
second clause used to return silently, so a stalled solve was indistinguishable from a converged one
-- the caller got a plausible-looking field and no signal. The steady-linear default has always
refused to do that; these pin the same contract on the nonlinear path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jno.utils.solver.newton_krylov import newton_direct, newton_krylov  # noqa: E402


def _scalar_residual(u):
    """Root at u = 1, but slow enough from a far start that a tight step cap cannot reach it."""
    return jnp.asarray(u) ** 3 - 1.0


def _jac(u):
    return jnp.diag(3.0 * jnp.asarray(u).reshape(-1) ** 2)


def test_krylov_raises_when_it_hits_the_step_cap():
    with pytest.raises(RuntimeError, match="did not converge in max_steps=1"):
        newton_krylov(_scalar_residual, jnp.array([50.0]), max_steps=1)


def _assert_meets_contract(u, u0, *, rtol=1e-8, atol=1e-8):
    """The driver promises ``||r(u)|| <= atol + rtol*||r(u0)||`` -- a bound RELATIVE to the starting
    residual, not an absolute accuracy on the root. From u0 = 50 that bound is ~1.2e-3, so the root
    itself is only good to ~1e-4; asserting more would test a promise nobody made."""
    r = float(jnp.linalg.norm(_scalar_residual(u)))
    assert r <= atol + rtol * float(jnp.linalg.norm(_scalar_residual(u0)))


def test_krylov_converges_without_the_cap():
    u0 = jnp.array([50.0])
    u = newton_krylov(_scalar_residual, u0, max_steps=100)
    _assert_meets_contract(u, u0)
    np.testing.assert_allclose(np.asarray(u), [1.0], rtol=1e-3)


def test_direct_raises_when_it_hits_the_step_cap():
    with pytest.raises(RuntimeError, match="did not converge in max_steps=1"):
        newton_direct(
            _scalar_residual, _jac, jnp.array([50.0]), max_steps=1, linear_solve=lambda A, b: jnp.linalg.solve(A, b)
        )


def test_direct_converges_without_the_cap():
    u0 = jnp.array([50.0])
    u = newton_direct(_scalar_residual, _jac, u0, max_steps=100, linear_solve=lambda A, b: jnp.linalg.solve(A, b))
    _assert_meets_contract(u, u0)
    np.testing.assert_allclose(np.asarray(u), [1.0], rtol=1e-3)


def test_guard_is_a_noop_under_jit():
    """Under a transform the residual cannot concretise, so the check must step aside rather than
    raise ConcretizationTypeError -- the same trade the two linear checks already make."""
    f = jax.jit(lambda x0: newton_krylov(_scalar_residual, x0, max_steps=1))
    out = f(jnp.array([50.0]))  # stalls, but must not raise: there is nothing concrete to test
    assert jnp.isfinite(out).all()


def test_gradient_still_flows_through_a_converged_solve():
    """The guard adds residual evaluations around custom_root; implicit diff must be untouched."""

    def root_of(a):
        return newton_krylov(lambda u: u**3 - a, jnp.array([2.0]), max_steps=100)[0]

    # d/da (a^(1/3)) = 1/3 a^(-2/3); at a = 8 that is 1/12
    np.testing.assert_allclose(float(jax.grad(root_of)(8.0)), 1.0 / 12.0, rtol=1e-6)
