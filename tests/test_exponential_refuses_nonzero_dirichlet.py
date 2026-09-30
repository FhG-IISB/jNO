"""``jno.solve.exponential`` holds every zero-mass DOF at 0, so a NON-zero wall value must be refused.

Before the guard it was silently wrong on a 2-D heat block: with ``u = 1`` on the wall the boundary came
back 0, and with ``u = t`` the whole field stayed 0. A homogeneous wall is exactly what the integrator is
built for and must keep working (oracle: the backward-Euler march at a fine step).
"""

from __future__ import annotations

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


def _heat(wall, n_steps=21):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=8).domain(time=(0.0, 0.1, n_steps))
    e = 1e-9
    d.tag("wall", lambda x, y: (x < e) | (x > 1 - e) | (y < e) | (y > 1 - e))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, tw = d.variable("wall", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem([ub.t * vb + ub.x * vb.x + ub.y * vb.y, u(xw, yw) - wall(tw), u(*ci) - ic])


def test_a_constant_nonzero_wall_is_refused():
    with pytest.raises(NotImplementedError, match="Dirichlet value on this block is NOT zero"):
        _heat(lambda t: 1.0 + 0.0 * t).solve(time=jno.solve.exponential()).fn()


def test_a_time_varying_wall_is_refused():
    with pytest.raises(NotImplementedError, match="Dirichlet value on this block is NOT zero"):
        _heat(lambda t: t).solve(time=jno.solve.exponential()).fn()


def test_a_homogeneous_wall_still_integrates_exactly():
    exp = np.asarray(_heat(lambda t: 0.0 * t).solve(time=jno.solve.exponential(mass="consistent")).fn())
    fine = np.asarray(_heat(lambda t: 0.0 * t, n_steps=801).solve(time=jno.solve.theta(0.5)).fn())
    # Same spatial operator (consistent mass: the default lumped one is a different semi-discretisation);
    # exponential is exact in time, CN at 800 steps is within ~1e-6 of it.
    assert np.abs(exp[-1] - fine[-1]).max() < 1e-4 * np.abs(fine[-1]).max()


def test_an_algebraic_row_that_reaches_the_interior_is_refused():
    """A pressure-like constraint (zero mass, coupled to interior unknowns) is not a Dirichlet row: the
    exponential integrator would hold it at 0, which is not its solution. A small Stokes-type transient."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, 0.1, 3))
    e = 1e-9
    d.tag("wall", lambda x, y: (x < e) | (x > 1 - e) | (y < e) | (y > 1 - e))
    d.point_region("pin", (0.5, 0.5))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    xp, yp, _ = d.variable("pin", split=True)
    ci = d.variable("initial", split=True)
    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    ub, vv = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    fem = jno.fem(
        [
            inner(ub.t, vv, 1) + inner(gu, gv, 2) - pp * trace(gv) - vv[0],
            -qq * trace(gu),
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
            p(xp, yp) - 0.0,
            u(*ci)[0] - 0.0,
            u(*ci)[1] - 0.0,
        ]
    )
    with pytest.raises(NotImplementedError, match="couple to other unknowns"):
        fem.solve(time=jno.solve.exponential(mass="consistent")).fn()
