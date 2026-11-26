"""``newton_direct`` traces the residual and the Jacobian ONCE each.

The driver uses the residual at five places (the start, the loop body, custom_root's residual, and the
Jacobian at every step and at the root). Each place used to re-trace or replay the whole form; on a
many-region FEM form that replay was most of the solve's trace. Oracle for the gradient: the implicit
function theorem on F(u; p) = A u + u^3 - p b, du/dp = J^{-1} b, against central finite differences.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.newton_krylov import newton_direct


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


N = 30
A = np.diag(np.full(N, 2.5)) + np.diag(np.full(N - 1, -1.0), 1) + np.diag(np.full(N - 1, -1.0), -1)  # float64: the
B = np.linspace(0.5, 1.5, N)  # x64 fixture runs after import, so jnp arrays made here would be float32


def _solve(p, calls):
    def residual(u):
        calls["r"] += 1
        return A @ u + u**3 - p * B

    def jacobian(u):
        calls["J"] += 1
        return jsp.BCOO.fromdense(A + jnp.diag(3 * u**2), nse=3 * N - 2)

    return newton_direct(residual, jacobian, jnp.zeros(N), rtol=1e-12, atol=0.0, max_steps=50)


def test_residual_and_jacobian_are_traced_once():
    calls = {"r": 0, "J": 0}
    jax.jit(lambda p: _solve(p, calls))(1.0)
    assert calls == {"r": 1, "J": 1}


def test_the_gradient_through_the_root_is_the_implicit_one():
    calls = {"r": 0, "J": 0}
    u = _solve(1.0, calls)
    np.testing.assert_allclose(A @ u + u**3, B, atol=1e-10)
    g = jax.grad(lambda p: jnp.sum(_solve(p, calls) ** 2))(1.0)
    h = 1e-5
    fd = (jnp.sum(_solve(1.0 + h, calls) ** 2) - jnp.sum(_solve(1.0 - h, calls) ** 2)) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-6)
