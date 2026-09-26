"""jno.precond.fsai(): factored sparse approximate inverse (Kolotilina & Yeremin 1993).

Oracles: the default solve of the same problem (the answer must not change), the definition of the
preconditioner (symmetric positive definite), and its effect (a much smaller CG residual than Jacobi after
the same number of iterations).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.05, reaction=None, advection=0.0):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y = d.variable("interior", split=True)[:2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    form = ui.x * vi.x + ui.y * vi.y - 10.0 * vi
    if reaction is not None:
        form = form + reaction(ui) * vi
    if advection:
        form = form + advection * ui.x * vi
    return jno.fem([form, u(cb[0], cb[1]) - 0.0])


def _leaf(sol):
    return np.asarray(jax.tree_util.tree_leaves(sol)[0]).reshape(-1)


def _cg_residual(op, b, M, its=25):
    x, _ = jax.scipy.sparse.linalg.cg(op.mv, b, M=M, maxiter=its, tol=1e-30)
    return float(jnp.linalg.norm(b - op.mv(x)) / jnp.linalg.norm(b))


def test_it_is_spd_and_far_stronger_than_jacobi_per_iteration():
    fem = _poisson()
    A, b = fem._op
    b = jnp.asarray(b).reshape(-1)
    op = LinearOperator(A)
    ctx = PrecondContext(op, fem)
    M1 = jno.precond.fsai().materialize(ctx)
    M2 = jno.precond.fsai(power=2).materialize(ctx)
    v, w = (jax.random.normal(jax.random.PRNGKey(k), b.shape) for k in (0, 1))
    for M in (M1, M2):
        assert abs(v @ M(w) - w @ M(v)) <= 1e-12 * abs(v @ M(w))
        assert float(v @ M(v)) > 0
    r_jac = _cg_residual(op, b, jno.precond.jacobi().materialize(ctx))
    r1, r2 = _cg_residual(op, b, M1), _cg_residual(op, b, M2)
    assert r1 < 0.05 * r_jac and r2 < r1, (r_jac, r1, r2)


def test_the_answer_is_unchanged_and_the_compiled_path_reuses_the_pattern():
    fem = _poisson()
    ref = _leaf(fem.solve())
    spec = jno.precond.fsai()
    a = _leaf(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=spec))
    assert spec.traceable  # built on the first solve: later ones may compile
    b = _leaf(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=spec))
    np.testing.assert_allclose(a, ref, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-13)


def test_a_newton_tangent_arriving_traced_is_factored_from_its_current_values():
    """-Δu + u^3 = f: an SPD tangent K + 3u^2 M whose values change every Newton step."""
    fem = _poisson(reaction=lambda u: u**3)
    ref = _leaf(fem.solve())
    got = _leaf(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=jno.precond.fsai()))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-10)


def test_a_transient_march():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(time=(0.0, 0.1, 6))
    x, y, t = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(cb[0], cb[1]) - 0.0, ic])
    ref = np.asarray(fem.solve().fn())
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-12), precond=jno.precond.fsai()).fn())
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_a_non_symmetric_operator_is_refused():
    fem = _poisson(advection=5.0)
    with pytest.raises(ValueError, match="not symmetric"):
        fem.solve(linear=jno.solve.bicgstab(), precond=jno.precond.fsai())


def test_bad_power_is_refused():
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="power"):
            jno.precond.fsai(power=bad)
