"""A nonlinear march merges its step tangent ``J + M/dt`` by a host-side plan, not by an in-trace sort.

Inside the march every index array is a tracer, so ``M/dt + J`` used to reach the linear solve as a raw
concatenation of both operands, and the CSR conversion ARGSORTED it on every Newton iteration: 6.15M
triplets and ~300 MiB of sort scratch on a 24^3 periodic P1/P1 Navier-Stokes march. The pattern is fixed
by mesh and constraints, so it is planned once, eagerly, and the merged operator is flagged sorted and
unique.

Oracles: the planned march equals the unplanned one; the unplanned path is kept wherever the pattern can
move, and wherever the operands do not have the planned sizes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _nonlinear_heat(n=6, steps=4):
    """``u_t = div((1 + u^2) grad u) + 1`` on the unit square, u = 0 on the wall: a nonlinear march."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=n).domain(time=(0.0, 0.05 * steps, steps + 1))
    e = 1e-9
    d.tag("wall", lambda x, y: (x < e) | (x > 1 - e) | (y < e) | (y > 1 - e))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem([ub.t * vb + (1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - vb, u(xw, yw) - 0.0, u(*ci) - ic])


def _march(fem):
    # An explicit Newton with a fresh tangent per iteration: the plan must not change the answer, and the
    # default driver's tangent carry -- which needs the plan -- would make the two marches stop at different
    # points under the same tolerance, hiding what this compares.
    return np.asarray(fem.solve(time=jno.solve.bdf2(), nonlinear=jno.solve.newton(rtol=1e-12, atol=1e-14)).fn())


def test_the_planned_march_equals_the_unplanned_one(monkeypatch):
    planned = _march(_nonlinear_heat())
    import jno.utils.solver.solver_api as sa

    monkeypatch.setattr(sa, "_plan_step_tangent_merge", lambda block, state=None: None)
    unplanned = _march(_nonlinear_heat())
    assert np.abs(planned).max() > 1e-2
    assert np.abs(planned - unplanned).max() <= 1e-10 * np.abs(unplanned).max()


def test_the_step_tangent_reaches_the_csr_conversion_already_sorted(monkeypatch):
    """Force the CSR matvec (the GPU's usual choice) and record every operator the march converts. The
    unplanned tangent is the concatenation of J and M -- ``nse(J) + nse(M)`` triplets, unsorted, so the
    conversion argsorts it inside the Newton loop. With the plan no operator of that size reaches it, and
    the merged tangent arrives flagged sorted. (The mass matrix itself also passes through, for the
    residual's ``M (u_new - u)``; its indices are constants, so its conversion folds at compile time.)"""
    import jno.utils.solver.matvec_format as mf
    import jno.utils.solver.solver_api as sa

    seen = []
    orig = mf.csr_parts

    def spy(A):
        seen.append((int(A.nse), bool(getattr(A, "indices_sorted", False))))
        return orig(A)

    monkeypatch.setattr(mf, "csr_parts", spy)
    monkeypatch.setattr(mf, "_FORMAT", "csr")
    monkeypatch.setenv("JNO_COMPILE_CACHE", "0")

    fem = _nonlinear_heat(n=5)
    blk = fem.operator[0] if isinstance(fem.operator, tuple) else fem.operator
    concat = int(blk.jacobian(blk.state0, 0.0, None).nse) + int(blk.mass(0.0, None).nse)
    plan = sa._plan_step_tangent_merge(blk)
    assert plan is not None
    merged = int(plan[0][2])
    assert merged < concat  # the merge removes the overlap of J and M

    jax.clear_caches()
    _march(fem)
    assert not [n for n, _ in seen if n == concat], seen
    assert any(n == merged and srt for n, srt in seen), seen

    seen.clear()
    monkeypatch.setattr(sa, "_plan_step_tangent_merge", lambda block, state=None: None)
    jax.clear_caches()
    _march(_nonlinear_heat(n=5, steps=3))
    assert [n for n, srt in seen if n == concat and not srt], "without the plan the concatenated tangent is converted"


def test_no_plan_where_the_pattern_can_move():
    from jno.utils.solver.solver_api import _plan_step_tangent_merge

    fem = _nonlinear_heat(n=4)
    block = fem.operator[0] if isinstance(fem.operator, tuple) else fem.operator
    assert _plan_step_tangent_merge(block) is not None
    block.metadata = {**(block.metadata or {}), "pattern_moves": True}
    assert _plan_step_tangent_merge(block) is None


def test_a_plan_for_other_operand_sizes_is_not_applied():
    import jax.experimental.sparse as jsp

    from jno.utils.solver.solver_api import _add_step_operator

    rng = np.random.default_rng(1)
    idx = np.array([[0, 0], [0, 1], [1, 1], [2, 2]], np.int32)
    J = jsp.BCOO((jnp.asarray(rng.standard_normal(4)), jnp.asarray(idx)), shape=(3, 3))
    M = jsp.BCOO((jnp.asarray(rng.standard_normal(3)), jnp.asarray(idx[[0, 2, 3]])), shape=(3, 3))
    ref = np.asarray(J.todense()) + 0.5 * np.asarray(M.todense())
    stale = ((jnp.asarray(idx), jnp.zeros(99, jnp.int32), 4), (5, 3, (3, 3)))  # sizes do not match
    out = _add_step_operator(J, M, 0.5, plan=stale)
    assert np.abs(np.asarray(out.todense()) - ref).max() < 1e-14
