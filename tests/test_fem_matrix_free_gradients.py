"""``jax.grad`` through ``jno.fem(..., matrix_free=True).solve()``: by a source, by a coefficient of the operator,
on the default solve and on a Krylov slot, linear and nonlinear. Each against a central difference.

The default linear solve's BiCGStab is a ``while_loop``, which reverse mode cannot differentiate; it raised.
It now runs in ``custom_linear_solve``, whose gradient is the transposed solve.
"""

import jax
import jax.numpy as jnp
import jno.numpy as jnn
import numpy as np
import pytest

import jno

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _fem(source, kappa, *, nonlinear=False):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=2).quad().domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols(order=4)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = kappa * (1.0 + ui * ui) if nonlinear else kappa
    form = k * (ui.x * vi.x + ui.y * vi.y) - source * jnn.sin(PI * x) * jnn.cos(PI * y) * vi
    return jno.fem([form, u(xb, yb) - 0.25 * xb], quadrature="gll", matrix_free=True)


def _check(loss, s0=1.3, h=1e-3):
    # the solves stop at a relative residual of 1e-8, so the difference quotient carries ~1e-5 of noise
    g = float(jax.grad(loss)(s0))
    fd = (float(loss(s0 + h)) - float(loss(s0 - h))) / (2 * h)
    assert abs(g - fd) < 1e-4 * abs(fd), (g, fd)


@pytest.mark.parametrize("by", ["source", "coefficient"])
@pytest.mark.parametrize("slot", [None, "cg"])
def test_a_linear_matrix_free_solve_differentiates(by, slot):
    def loss(s):
        fem = _fem(s, 1.0) if by == "source" else _fem(1.0, s)
        assert getattr(fem.operator[0], "matrix_free", False)
        sol = fem.solve() if slot is None else fem.solve(linear=jno.solve.cg(tol=1e-12), precond=jno.precond.jacobi())
        return jnp.sum(jnp.asarray(sol).reshape(-1) ** 2)

    _check(loss)


@pytest.mark.parametrize("by", ["source", "coefficient"])
def test_a_nonlinear_matrix_free_solve_differentiates(by):
    def loss(s):
        fem = _fem(s, 1.0, nonlinear=True) if by == "source" else _fem(1.0, s, nonlinear=True)
        return jnp.sum(jnp.asarray(fem.solve()).reshape(-1) ** 2)

    _check(loss)
