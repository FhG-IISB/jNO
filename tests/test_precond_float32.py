"""jno.precond.*(float32=True): preconditioners built and applied in single precision inside a double solve.

Oracle: the default (double) solve of the same problem -- a preconditioner changes how fast the solve
converges, never what it converges to -- plus the precision the spec actually sees.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.precond import _Spec
from jno.utils.solver.solver_api import LinearOperator, PrecondApplier, PrecondContext


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.04):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y = d.variable("interior", split=True)[:2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - 10.0 * vi, u(cb[0], cb[1]) - 0.0])


def _leaf(sol):
    return np.asarray(jax.tree_util.tree_leaves(sol)[0]).reshape(-1)


@pytest.mark.parametrize("make", ["jacobi", "chebyshev", "fsai", "amg", "nystrom"])
def test_the_answer_is_the_double_precision_one(make):
    fem = _poisson()
    ref = _leaf(fem.solve())
    got = _leaf(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=getattr(jno.precond, make)(float32=True)))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-10)


def test_the_spec_is_built_in_float32_and_applied_back_in_float64():
    seen = {}

    class _Probe(_Spec):
        def materialize(self, ctx):
            seen["dtype"] = ctx.A.bcoo.dtype
            return PrecondApplier(lambda v: (seen.setdefault("v", v.dtype), v)[1])

    fem = _poisson(0.2)
    A = fem._op[0]
    spec = _Probe()
    spec.float32 = True
    M = spec.materialize(PrecondContext(LinearOperator(A), fem))
    out = M(jnp.ones(A.shape[0]))
    assert seen["dtype"] == jnp.float32 and seen["v"] == jnp.float32 and out.dtype == jnp.float64
    assert M.low_precision


def test_precision_is_part_of_the_spec_identity():
    a, b = jno.precond.jacobi(), jno.precond.jacobi(float32=True)
    assert a != b and hash(a) != hash(b) and a == jno.precond.jacobi()
    with pytest.raises(TypeError, match="float32"):
        jno.precond.jacobi(float32=1)


def test_an_explicitly_built_amg_hierarchy_is_float32():
    A = _poisson()._op[0]
    spec = jno.precond.amg(float32=True).build(A)
    assert all(lv["A"].dtype == jnp.float32 for lv in spec._levels[:-1])
    assert spec._levels[-1]["Ainv"].dtype == jnp.float32


def test_flexible_cg_matches_cg_for_an_exact_preconditioner():
    from jno.utils.solver.krylov import flexible_cg

    A = _poisson(0.1)._op[0]
    op = LinearOperator(A)
    b = jnp.ones(A.shape[0])
    d = op.diag()
    x_f = flexible_cg(op.mv, b, M=lambda v: v / d, tol=1e-12, maxiter=2000)
    x_c = jax.scipy.sparse.linalg.cg(op.mv, b, M=lambda v: v / d, tol=1e-12, maxiter=2000)[0]
    np.testing.assert_allclose(np.asarray(x_f), np.asarray(x_c), rtol=1e-9, atol=1e-12)


def test_a_transient_march_with_a_float32_preconditioner():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(time=(0.0, 0.1, 6))
    x, y, t = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(cb[0], cb[1]) - 0.0, ic])
    ref = np.asarray(fem.solve().fn())
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-12), precond=jno.precond.amg(float32=True)).fn())
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)
