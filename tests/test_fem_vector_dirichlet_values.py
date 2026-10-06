"""A vector Dirichlet value imposes EVERY component, not its first one broadcast to all.

``u(wall) - (1.0, -0.5)`` on a 2-vector field used to impose ``(1.0, 1.0)`` -- silently, steady and
transient alike -- because the value table was cut to its first column. The docs only ever used
``(0, 0)``, where the two agree, which is how it went unnoticed.

Oracles (all exact for P1 and backward Euler, so the tolerance is the solver's):
* a constant wall value ``c`` for vector Laplace gives ``u = c`` everywhere;
* a linear wall value ``(x, 2y + 5)`` is harmonic, so the interior reproduces it;
* a wall driven as ``(t, -2t)`` with source ``(1, -2)`` gives ``u = (t, -2t)`` everywhere.
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


def _vector_laplace(wall_value, time=None):
    shape = jno.shape.rect(0, 0, 1, 1).structured(n=6)
    d = shape.domain(time=time) if time else shape.domain()
    e = 1e-9
    d.tag("wall", lambda x, y: (x < e) | (x > 1 - e) | (y < e) | (y > 1 - e))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"))
    V = d.variable("interior", split=True)
    W = d.variable("wall", split=True)
    ub, vb = u.bind(x=V[0], y=V[1], t=V[2]), v.bind(x=V[0], y=V[1], t=V[2])
    grad, inner = jno.np.grad, jno.np.inner
    form = inner(grad(u, [V[0], V[1]]), grad(v, [V[0], V[1]]), n_contract=2)
    terms = []
    if time:
        ci = d.variable("initial", split=True)
        form = form + inner(ub.t, vb, 1) - (1.0 * vb[0] - 2.0 * vb[1])
        terms += [u(*ci)[0] - 0.0, u(*ci)[1] - 0.0]
    terms = [form, u(W[0], W[1]) - wall_value(W)] + terms
    return jno.fem(terms), u


def _nodal(fem, u, sol):
    blk = fem.blocks[fem.block_index(u)] if fem.blocks is not None else slice(0, None)
    return np.asarray(sol)[blk].reshape(-1, 2), np.asarray(fem.points)


def test_a_constant_vector_wall_value_holds_every_component():
    fem, u = _vector_laplace(lambda W: (1.0, -0.5))
    U, _ = _nodal(fem, u, fem.solve())
    assert np.abs(U - np.array([1.0, -0.5])).max() < 1e-10, U[:3]


def test_a_varying_vector_wall_value_is_reproduced_inside():
    fem, u = _vector_laplace(lambda W: jno.np.stack([W[0], 2.0 * W[1] + 5.0], axis=-1))
    U, P = _nodal(fem, u, fem.solve())
    exact = np.stack([P[:, 0], 2.0 * P[:, 1] + 5.0], axis=1)
    assert np.abs(U - exact).max() < 1e-6  # the default iterative solve (rtol 1e-8) on values up to 7


def test_a_time_varying_vector_wall_value_drives_every_component():
    fem, u = _vector_laplace(lambda W: jno.np.stack([W[2], -2.0 * W[2]], axis=-1), time=(0.0, 0.5, 6))
    traj = np.asarray(fem.solve(time=jno.solve.theta(1.0)).fn())
    for k, t in enumerate(np.linspace(0.0, 0.5, 6)):
        U, _ = _nodal(fem, u, traj[k])
        assert np.abs(U - np.array([t, -2.0 * t])).max() < 1e-8, (k, U[:2])


def test_a_vector_value_on_one_component_is_refused():
    shape = jno.shape.rect(0, 0, 1, 1).structured(n=4)
    d = shape.domain()
    d.tag("wall", lambda x, y: y < 1e-9)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"))
    V = d.variable("interior", split=True)
    W = d.variable("wall", split=True)
    grad, inner = jno.np.grad, jno.np.inner
    form = inner(grad(u, [V[0], V[1]]), grad(v, [V[0], V[1]]), n_contract=2)
    with pytest.raises(ValueError, match="components per point"):
        jno.fem([form, u(W[0], W[1])[0] - (1.0, 2.0), u(W[0], W[1])[1] - 0.0]).solve()
