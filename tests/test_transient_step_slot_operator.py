"""The composed transient linear slot must solve the step operator the step actually forms.

Oracle: the same march without any slot (the block's own default step solve).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import compose_transient_step_solvers


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(n=11):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.15).domain(time=(0.0, 0.1, n))
    x, y, t = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(cb[0], cb[1]) - 0.0, ic])


def _march(scheme, block, fem, linear):
    save = jnp.linspace(block.t0, block.t1, 5)
    lin, nl = compose_transient_step_solvers(None, linear, None, fem, block, scheme) if linear else (None, None)
    return np.asarray(scheme.integrate(block, {}, save, linear_solve=lin, nonlinear_solve=nl))


def test_a_mass_fn_block_is_not_given_a_prebuilt_operator_of_the_constant_mass():
    fem = _heat()
    block = dataclasses.replace(fem.operator, mass_fn=lambda t, args=None: 2.0 * fem.operator.M)
    th = jno.solve.theta(1.0)
    ref = _march(th, block, fem, None)
    got = _march(th, block, fem, jno.solve.bicgstab(tol=1e-12))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-10)


def test_an_adaptive_march_solves_its_own_step_operator_with_a_slot():
    """A traced step size used to get the operator prebuilt for the default step: a different system."""
    fem = _heat()
    sch = jno.solve.theta(0.5).adaptive(rtol=1e-5)
    ref = _march(sch, fem.operator, fem, None)
    got = _march(sch, fem.operator, fem, jno.solve.bicgstab(tol=1e-12))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_the_nonlinear_step_tangent_carries_theta(theta):
    """G = M(w-u)/dt + theta R(w) + ... has Jacobian M/dt + theta J. Built as M/dt + J, a Crank-Nicolson
    march's Newton ran on the wrong tangent and custom_root's GRADIENT came out 7.7% wrong."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_fem_sdirk import _heat

    fem = _heat(6, T=0.05, nonlinear=True)
    blk = fem.operator
    save = jnp.linspace(blk.t0, blk.t1, 3)
    sch = jno.solve.theta(theta)

    def loss(s):
        b = dataclasses.replace(blk, state0=s * blk.state0)
        return jnp.sum(sch.integrate(b, {}, save, linear_solve=None, nonlinear_solve=None) ** 2)

    g = float(jax.grad(loss)(1.3))
    fd = float((loss(1.3 + 1e-6) - loss(1.3 - 1e-6)) / 2e-6)
    np.testing.assert_allclose(g, fd, rtol=1e-6)
