"""``jno.solve.exponential`` with trainables: a coefficient, a network.

It refused every runtime-parametric operator; an autonomous one (the coefficient trainable, no time in the
operator) now integrates, and its gradient matches FD on the lumped and the consistent path. The lumped path's
Lanczos returned a NaN gradient at breakdown, and inside jno.core the structural refusals were skipped (the
march is traced there); they now run once at ``fem.solve``.

Oracle: the gradient jno.core takes (one SGD step) against central FD of the same loss. The helpers here are
shared with tests/test_fem_transient_trainable_ic.py.
"""

import jax
import numpy as np
import pytest

import jno
from tests.test_fem_trainables_gaps import (
    PI,
    _affine,
    _affine_net,
    _central_fd,
    _const,
    _crux_gradient,
    _evaluate,
    _param,
    _read_leaf,
    _read_param,
    _rel,
    grad,
    inner,
    sin,
)

T = (0.0, 0.05, 6)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(k=1.0, ic=None, *, cubic=False, space=None, tt=False):
    """``u_t = k lap u (- u^3)`` on the unit square, walls at 0; ``ic(x, y)`` is the initial value."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.3).domain(time=T)
    u, v = d.fem_symbols() if space is None else d.fem_symbols(space=space)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    X = [xi, yi]
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    weak = (ui.tt if tt else ui.t) * vi + (k(xi) if callable(k) and not hasattr(k, "initialize") else k) * inner(
        grad(u, X), grad(v, X), 1
    )
    if cubic:
        weak = weak + ui**3 * vi
    u0 = (ic or (lambda x, y: sin(PI * x) * sin(PI * y)))(ci[0], ci[1])
    return [weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0]


def _scalar_gradient(build, scheme):
    u_obs = _evaluate(jno.fem(build(1.0)).solve(time=scheme()))
    a = _param("a", 1.3)
    loss = (jno.fem(build(a)).solve(time=scheme()) - u_obs).mse
    g, loss0 = _crux_gradient(loss, a, _read_param(a), lr=1.0)
    return g, _central_fd(loss, a, _const, 1.3, 1e-4), loss0


def _net_gradient(build, scheme):
    u_obs = _evaluate(jno.fem(build(lambda x: 1.0 + 0.5 * x)).solve(time=scheme()))
    net = _affine_net(1.2, 0.1)
    loss = (jno.fem(build(lambda x: net(x))).solve(time=scheme()) - u_obs).mse
    g, loss0 = _crux_gradient(loss, net, _read_leaf(net, "b"), lr=1.0)
    return g, _central_fd(loss, net, lambda b: _affine(1.2, b), 0.1, 1e-4), loss0


_EXP = {
    "lumped": lambda: jno.solve.exponential(),
    "consistent": lambda: jno.solve.exponential(mass="consistent"),
}


@pytest.mark.parametrize("mass", list(_EXP))
def test_exponential_differentiates_a_trainable_coefficient(mass):
    g, fd, loss0 = _scalar_gradient(lambda a: _heat(k=a), _EXP[mass])
    assert loss0 > 0 and abs(g) > 0
    assert np.isfinite(g), "the Lanczos breakdown must not turn the gradient into NaN"
    assert _rel(g, fd) < 1e-6, f"crux {g:.10e} vs FD {fd:.10e}"


def test_exponential_differentiates_a_network_coefficient():
    g, fd, loss0 = _net_gradient(lambda k: _heat(k=k), _EXP["lumped"])
    assert loss0 > 0 and abs(g) > 0
    assert _rel(g, fd) < 1e-6, f"crux {g:.10e} vs FD {fd:.10e}"


def test_exponential_refuses_a_time_dependent_parametric_operator():
    d = jno.shape.rect(0, 0, 1, 1, size=0.3).domain(time=T)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    a = _param("a", 1.3)
    fem = jno.fem([ui.t * vi + a * (1.0 + ti) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])
    with pytest.raises(NotImplementedError, match="time-INDEPENDENT"):
        fem.solve(time=jno.solve.exponential())


def test_exponential_refusals_hold_inside_jno_core():
    """The structural refusals read concrete values. Inside jno.core the march is traced, and a lumped
    Hermite march returned NaN there instead of the refusal it gets eagerly; they now run at fem.solve."""
    with pytest.raises(NotImplementedError, match="lumping gives a zero or negative mass"):
        jno.fem(_heat(k=_param("a", 1.3), space="Hermite")).solve(time=jno.solve.exponential())
