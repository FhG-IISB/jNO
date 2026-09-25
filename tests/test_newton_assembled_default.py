"""The default Newton runs on the ASSEMBLED tangent (``jno.solve.newton(direct=None)``).

Oracles: the matrix-free Newton (``direct=False``, the previous default) and the sparse-direct one
(``direct=True``) must reach the same root; gradients must match central finite differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.newton_krylov import newton_default, newton_krylov


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _nonlinear(size=0.1, param=False):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y = d.variable("interior", split=True)[:2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    k = jno.np.parameter((1,), name="k") if param else 1.0
    return jno.fem([(1 + k * u * u) * (ui.x * vi.x + ui.y * vi.y) - 10.0 * vi, u(cb[0], cb[1]) - 0.0])


def _leaf(sol):
    return np.asarray(jax.tree_util.tree_leaves(sol)[0]).reshape(-1)


def _count_jacobian(fem, monkeypatch):
    op = fem.operator
    calls = {"n": 0}
    orig = op.jacobian

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(op, "jacobian", counting)
    return calls


def test_the_default_uses_the_assembled_tangent_and_reaches_the_same_root(monkeypatch):
    fem = _nonlinear()
    calls = _count_jacobian(fem, monkeypatch)
    u_default = _leaf(fem.solve())
    assert calls["n"] > 0, "the default Newton did not assemble the tangent"
    u_free = _leaf(fem.solve(nonlinear=jno.solve.newton(direct=False)))
    u_lu = _leaf(fem.solve(nonlinear=jno.solve.newton(direct=True)))
    assert np.abs(u_default).max() > 1e-2
    np.testing.assert_allclose(u_default, u_free, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(u_default, u_lu, rtol=1e-7, atol=1e-10)


def test_explicit_matrix_free_never_assembles(monkeypatch):
    fem = _nonlinear()
    calls = _count_jacobian(fem, monkeypatch)
    fem.solve(nonlinear=jno.solve.newton(direct=False))
    assert calls["n"] == 0


def test_a_preconditioner_that_needs_the_assembled_matrix_now_composes():
    """Used to raise ``LinearOperator.diag(): a matvec-only operator has no assembled diagonal``."""
    fem = _nonlinear()
    u = _leaf(fem.solve(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi()))
    np.testing.assert_allclose(u, _leaf(fem.solve(nonlinear=jno.solve.newton(direct=False))), rtol=1e-7, atol=1e-10)


def test_without_an_assembled_tangent_the_default_is_matrix_free():
    n = 30
    A = 2.0 * jnp.eye(n) - jnp.eye(n, k=1) - jnp.eye(n, k=-1)
    b = jnp.ones(n)
    f = lambda u: A @ u + 0.5 * u**3 - b  # noqa: E731
    np.testing.assert_allclose(
        np.asarray(newton_default(f, jnp.zeros(n))), np.asarray(newton_krylov(f, jnp.zeros(n))), rtol=1e-12, atol=1e-14
    )


def test_gradient_through_the_assembled_default_matches_finite_differences():
    """The inverse-problem case: a parameter inside the nonlinear coefficient, differentiated through the
    real assembled jNO tangent."""
    fem = _nonlinear(size=0.15, param=True)
    op = fem.operator
    (name,) = op.runtime_parameter_exprs
    u0 = jnp.zeros(int(op.size))

    def loss(kv, driver):
        args = {name: jnp.asarray([kv])}
        if driver == "assembled":
            u = newton_default(lambda u: op.residual(u, args), u0, jacobian=lambda u: op.jacobian(u, args))
        else:
            u = newton_krylov(lambda u: op.residual(u, args), u0)
        return jnp.sum(u**2)

    k0, e = 0.8, 1e-5
    g = float(jax.grad(lambda kv: loss(kv, "assembled"))(k0))
    fd = float((loss(k0 + e, "assembled") - loss(k0 - e, "assembled")) / (2 * e))
    np.testing.assert_allclose(g, fd, rtol=1e-6)
    np.testing.assert_allclose(g, float(jax.grad(lambda kv: loss(kv, "free"))(k0)), rtol=1e-7)


def test_a_nonlinear_transient_march_defaults_to_the_assembled_tangent():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.15).domain(time=(0.0, 0.05, 6))
    x, y, t = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    fem = jno.fem([ui.t * vi + (1 + u * u) * (ui.x * vi.x + ui.y * vi.y), u(cb[0], cb[1]) - 0.0, ic])
    assert fem.operator.jacobian is not None
    traj = np.asarray(fem.solve().fn())
    ref = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=False)).fn())
    assert np.abs(traj).max() > 1e-2
    np.testing.assert_allclose(traj, ref, rtol=1e-7, atol=1e-10)
