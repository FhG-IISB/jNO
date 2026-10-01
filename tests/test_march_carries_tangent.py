"""A nonlinear march keeps its Newton tangent while it contracts -- within a step and from step to step.

Assembling the step tangent was the dominant cost of a march with a long element integrand: a stabilised
3-D Navier-Stokes step (55k DOFs, RTX 3070) spent ~220 of ~300 ms per Newton iteration on it, twice per
step. The march's default Newton now keeps the tangent (``newton(reuse=True)``'s contraction rule: keep it
while ``||r_new|| < ||r_old|| / 2``, refresh otherwise, reject a step that does not reduce the residual),
and an eager march carries it in the scan from one step to the next on the step-merge plan's fixed pattern.
Measured on that flow: one tangent for an 8-step march, 0.80 -> 0.30 s per step, the same trajectory.

Oracles: the carried march lands on the same trajectory as one that assembles a fresh tangent at every
Newton iteration, to the Newton tolerance; it assembles a handful of tangents for the whole march, not one
per step; and ``newton(reuse=True)`` works on the assembled-tangent iterative default, still refused for
the matrix-free mode.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
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


E = 1e-9


def _nonlinear_heat(steps=12, n=8):
    """``u_t = div((1 + u²) grad u) + 1``, u = 0 on the wall: a nonlinear march with an assembled tangent."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=n).domain(time=(0.0, 0.02 * steps, steps + 1))
    d.tag("wall", lambda x, y: (x < E) | (x > 1 - E) | (y < E) | (y > 1 - E))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = 2.0 * jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem([ub.t * vb + (1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - vb, u(xw, yw) - 0.0, u(*ci) - ic])


def _count_tangents(fem):
    """Wrap the time block's tangent so every EXECUTED assembly is counted (a host callback, test only)."""
    block = fem._op
    count = {"n": 0}
    real = block.jacobian

    def counted(u, t, args=None):
        jax.debug.callback(lambda: count.__setitem__("n", count["n"] + 1))
        return real(u, t, args)

    block.jacobian = counted
    return count


@pytest.mark.parametrize("scheme", ["theta", "bdf2"])
def test_the_carried_march_matches_a_fresh_tangent_march(scheme):
    kw = {"time": jno.solve.bdf2()} if scheme == "bdf2" else {}
    carried = np.asarray(_nonlinear_heat().solve(**kw).fn())
    fresh = np.asarray(_nonlinear_heat().solve(nonlinear=jno.solve.newton(rtol=1e-12, atol=1e-12), **kw).fn())
    assert np.abs(fresh).max() > 0.5
    # Both stop on the residual test, at different points under it: agreement to the Newton tolerance.
    assert np.abs(carried - fresh).max() <= 1e-7 * np.abs(fresh).max()


def test_a_march_assembles_a_handful_of_tangents_not_one_per_step():
    fem = _nonlinear_heat(steps=12)
    count = _count_tangents(fem)
    fem.solve(time=jno.solve.bdf2()).fn()
    carried = count["n"]
    fem2 = _nonlinear_heat(steps=12)
    count2 = _count_tangents(fem2)
    fem2.solve(time=jno.solve.bdf2(), nonlinear=jno.solve.newton()).fn()  # an explicit driver: no reuse
    every_iteration = count2["n"]
    assert every_iteration >= 12, every_iteration  # at least one per step without the carry
    assert carried <= every_iteration // 3, (carried, every_iteration)


def test_newton_reuse_works_on_the_assembled_iterative_default():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=8).domain()
    d.tag("wall", lambda x, y: (x < E) | (x > 1 - E) | (y < E) | (y > 1 - E))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, *_ = d.variable("interior", split=True)
    xw, yw, *_ = d.variable("wall", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([(1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - 10 * vb, u(xw, yw) - 0.0])
    ref = np.asarray(fem.solve(nonlinear=jno.solve.newton(rtol=1e-12, atol=1e-12)))
    got = np.asarray(fem.solve(nonlinear=jno.solve.newton(reuse=True, rtol=1e-12, atol=1e-12)))
    assert np.abs(ref).max() > 0.1
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()
    with pytest.raises(ValueError, match="matrix-free"):
        jno.solve.newton(direct=False, reuse=True)
