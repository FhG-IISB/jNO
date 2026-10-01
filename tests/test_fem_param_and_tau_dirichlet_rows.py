"""A parameter-valued wall and a τ-ramped wall on ONE load-path march are both imposed.

The parametric residual wrote its non-constant held values in an ``if/elif``: parameter- (or net-) valued
rows first, τ-dependent rows only when there were none. With both present the ramped face was simply left
free -- measured at 0.25 on every step where 0.5 -> 1.0 -> 1.5 was prescribed -- and nothing said so.

Oracles: each wall dof holds its prescribed value at every step; the march equals its twin with the
parameter replaced by the same constant (a path known to be right); the gradient in the wall parameter
matches central finite differences.
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


G_LEFT = 0.25


def _ramped(left):
    """``(1 + u²) ∇u·∇v + η (u - u⁻) v = 0`` on the unit square over τ ∈ [0, 1]: ``u = left`` on x = 0 and the
    displacement-controlled ``u = 0.5 + τ`` on x = 1. The ``η`` term reads the previous load step, so the
    form marches the τ grid."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(tau=(0.0, 1.0, 3))
    e = 1e-9
    d.tag("left", lambda x, y: x < e)
    d.tag("right", lambda x, y: x > 1 - e)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, tr = d.variable("right", split=True)
    ub, vb = u.bind(x=xi, y=yi, tau=ti), v.bind(x=xi, y=yi, tau=ti)
    grad_dot = ub.x * vb.x + ub.y * vb.y
    terms = [
        (1 + ub * ub) * grad_dot + 1e-3 * (ub - ub.i(-1)) * vb,
        u(xl, yl) - left,
        u(xr, yr) - (0.5 + tr),
    ]
    return jno.fem(terms)


def _march_at():
    fem = _ramped(jno.np.parameter((1,), name="g"))
    return fem, fem.solve()


def test_both_walls_hold_their_values_on_every_step():
    fem, node = _march_at()
    ys = np.asarray(node.fn(jnp.array([G_LEFT])))
    x = np.asarray(fem.points)[:, 0]
    left, right = x < 1e-9, x > 1 - 1e-9
    assert ys.shape[0] == 3
    for k, tau in enumerate([0.0, 0.5, 1.0]):
        assert np.abs(ys[k, left] - G_LEFT).max() < 1e-12, f"step {k}: the parameter-valued wall moved"
        assert np.abs(ys[k, right] - (0.5 + tau)).max() < 1e-12, f"step {k}: the τ-ramped wall was dropped"


def test_the_march_equals_its_constant_wall_twin():
    _, node = _march_at()
    ys = np.asarray(node.fn(jnp.array([G_LEFT])))
    twin = np.asarray(_ramped(G_LEFT).solve())  # no parameter: the march returns the array itself
    assert np.abs(twin).max() > 1.0
    assert np.abs(ys - twin).max() <= 1e-9 * np.abs(twin).max()


def test_the_wall_parameter_still_differentiates():
    _, node = _march_at()

    def total(g):
        return jnp.sum(node.fn(g)[-1])

    g0 = jnp.array([G_LEFT])
    ad = float(jax.grad(total)(g0)[0])
    h = 1e-5
    fd = float((total(g0 + h) - total(g0 - h)) / (2 * h))
    assert abs(ad) > 1e-3
    assert abs(ad - fd) <= 1e-5 * max(1.0, abs(fd))
