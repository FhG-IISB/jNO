"""A nonlocal ``Coupling`` on a NONLINEAR transient form enters the step's tangent, not only its residual.

The transient branch of ``_wrap_couplings`` added the coupling to the time block's residual but left the block's
assembled Jacobian as the local tangent alone. The per-step Newton then ran on a tangent without the coupling:
started from a steady solution (residual ~3e-4) on an enclosure-radiation problem, it grew the residual every
iteration to NaN in the first step. A coupling that declares its ``_support`` must add its block to the step
tangent (as on a steady form); one that does not must leave the step matrix-free.

Oracles: the block's Jacobian equals forward-mode AD of its residual; a stiff coupling's march converges.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno._fem import Coupling


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


E = 1e-9
K = 40.0  # a stiff nodal sink K u^3 (per dof); without its tangent the step Newton cannot converge


def _sink(u):
    return K * u**3


def _form(support):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, 0.5, 3))
    d.tag("left", lambda x, y: x < E)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    terms = [ub.t * vb + (1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y), u(xl, yl) - 1.0, u(*ci) - 1.0]
    c = Coupling(_sink, name="sink")
    if support:
        c._support = np.arange(25)
    terms.append(c)
    return jno.fem(terms)


def test_the_step_tangent_carries_a_supported_coupling():
    block = _form(support=True)._op
    assert block.jacobian is not None
    u = jnp.linspace(0.5, 1.5, 25)
    J = np.asarray(block.jacobian(u, 0.1, None).todense())
    J_ad = np.asarray(jax.jacfwd(lambda w: block.residual(w, 0.1, None))(u))
    assert np.abs(J - J_ad).max() <= 1e-10 * np.abs(J_ad).max()


def test_an_unsupported_coupling_leaves_the_step_matrix_free():
    assert _form(support=False)._op.jacobian is None


def test_a_stiff_coupling_marches():
    fem = _form(support=True)
    ys = np.asarray(fem.solve().fn())
    assert np.all(np.isfinite(ys)) and ys.shape == (3, 25)
    free = np.asarray(fem.points)[:, 0] > E
    assert np.all(ys[-1, free] < 1.0)  # the sink cools every free dof; the wall stays at 1
