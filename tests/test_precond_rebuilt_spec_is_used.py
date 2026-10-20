"""A compiled solve is keyed on WHICH preconditioner it holds, not only on what it prints.

The parametric solve (``fem.solve(p=...)``) compiles once and caches the program under the composed solver's
``cache_key``, which was ``repr(precond)``. Every built ``amg()`` prints the same, so a spec rebuilt between
optimisation steps hit the program compiled around the FIRST hierarchy and was silently ignored -- only the
speed suffers, never the answer, which is why nothing noticed.

Oracle: the CG iteration count, which a poor hierarchy (built for J + 100 I) visibly changes.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.utils.solver.krylov as kr

pytest.importorskip("pyamg")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_a_rebuilt_spec_is_the_one_a_parametric_solve_uses(monkeypatch):
    # Break: keying the compiled solve on repr(precond) alone (the second spec is then never used).
    counted = []
    real = kr.cg_counted

    def counting(*a, **kw):
        out = real(*a, **kw)
        jax.debug.callback(lambda kk: counted.append(int(kk)), out[1])
        return out

    monkeypatch.setattr(kr, "cg_counted", counting)
    d = jno.shape.rect(0, 0, 1, 1, size=0.05).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = jno.np.parameter((1,), name="k")
    fem = jno.fem([(1.0 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - k * vb, u(xb, yb) - 0.0])
    J = fem._op.jacobian(jnp.zeros(fem.dofs), {"k": jnp.asarray([1.0])})

    def iterations(spec):
        counted.clear()
        jax.block_until_ready(jnp.asarray(fem.solve(k=1.0, linear=jno.solve.cg(), precond=spec)))
        jax.effects_barrier()
        return sum(counted)

    good = iterations(jno.precond.amg().build(J))
    shifted = jsp.BCOO.fromdense(J.todense() + 100.0 * jnp.eye(fem.dofs))  # same structure, the wrong values
    poor = iterations(jno.precond.amg().build(shifted))  # so a weaker preconditioner for J: more iterations
    assert poor != good, f"the second spec was not used: {poor} iterations both times"
    assert np.isfinite(poor)


def _parametric_problem():
    d = jno.shape.rect(0, 0, 1, 1, size=0.05).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = jno.np.parameter((1,), name="k")
    return jno.fem([(1.0 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - k * vb, u(xb, yb) - 0.0])


def test_a_rebuilt_spec_of_the_same_shapes_reuses_the_compiled_solve(monkeypatch):
    # Break: baking the hierarchy into the program as constants (one compile, ~7 s at 100k DOFs, and one
    # retained executable + hierarchy per rebuild -- the device ran out of memory on the fourth).
    import jno.utils.solver.newton_krylov as nk

    traces = [0]
    real = nk.newton_direct

    def counting(*a, **kw):
        traces[0] += 1
        return real(*a, **kw)

    monkeypatch.setattr(nk, "newton_direct", counting)
    fem = _parametric_problem()
    sols = []
    for kv in (1.0, 1.5, 2.0):  # an optimisation loop: rebuild at the current state, solve at the new value
        J = fem._op.jacobian(jnp.zeros(fem.dofs), {"k": jnp.asarray([kv])})
        sols.append(np.asarray(fem.solve(k=kv, linear=jno.solve.cg(), precond=jno.precond.amg().build(J))))
    assert traces[0] == 1, f"{traces[0]} traces for three rebuilt specs of the same shapes"
    assert len(fem._op.__dict__["_eager_solve_cache"]) == 1
    ref = np.asarray(fem.solve(k=2.0))  # the answer is the preconditioner-free one
    np.testing.assert_allclose(sols[-1], ref, rtol=0, atol=1e-6 * np.abs(ref).max())


def test_a_linear_solve_with_a_rebuilt_spec_reuses_its_program(monkeypatch):
    # The linear path keys its compiled program the same way (`_composed_compiled_stated`).
    calls = [0]
    real = kr.cg_counted

    def counting(*a, **kw):
        calls[0] += 1
        return real(*a, **kw)

    monkeypatch.setattr(kr, "cg_counted", counting)
    d = jno.shape.rect(0, 0, 1, 1, size=0.05).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([ub.x * vb.x + ub.y * vb.y - vb, u(xb, yb) - 0.0])
    A = fem.A
    sols = [np.asarray(fem.solve(linear=jno.solve.cg(), precond=jno.precond.amg().build(A)))]
    first = calls[0]  # one trace (the solve and its transpose for the adjoint are both traced once)
    sols += [np.asarray(fem.solve(linear=jno.solve.cg(), precond=jno.precond.amg().build(A))) for _ in range(2)]
    assert calls[0] == first, f"the Krylov body was traced again ({calls[0] - first}x) for rebuilt specs"
    np.testing.assert_allclose(sols[2], sols[0], rtol=0, atol=1e-10 * np.abs(sols[0]).max())
