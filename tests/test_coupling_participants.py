"""What a coupling reads from each kind of problem (jno._coupling.participant).

u = 1 + 2x on the unit square (u = 1 on the left, u = 3 on the right, insulated top and bottom) is linear, so
every discretisation holds it exactly: on the right edge a participant must read the value 3 and the outward
flux du/dn = 2 at every interface node, from its own discrete flux (the finite-volume boundary rule, the
finite-difference boundary stencil, the finite-element consistent reaction).
"""

import jax
import numpy as np
import pytest

import jno
from jno._coupling.participant import participant


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _strong(make):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    cons = [-(ui.xx + ui.yy), u(xl, yl) - 1.0, u(xr, yr) - 3.0]
    for tag in ("top", "bottom"):
        xt, yt, _ = d.variable(tag, split=True)
        cons.append(u.bind(x=xt, y=yt).d(d.variable(tag, normals=True)) - 0.0)
    return make(cons), ur, ur.d(nr)


def _fem():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    ui, vi, ur = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xr, y=yr)
    return jno.fem([ui.x * vi.x + ui.y * vi.y, u(xl, yl) - 1.0, u(xr, yr) - 3.0]), ur, ur.d(nr)


CASES = {
    "fdm": lambda: _strong(jno.fdm),
    "fvm-cell": lambda: _strong(jno.fvm),
    "fvm-vertex": lambda: _strong(lambda c: jno.fvm(c, centering="vertex")),
    "fem": _fem,
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_participant_reads_value_and_flux_of_a_linear_solution(case):
    prob, value, flux = CASES[case]()
    p = participant(prob)
    state = p.solve(prob, {})
    iface = p.interface("right")
    assert p.owns(value) and not p.owns(jno.shape.rect(0, 0, 1, 1).domain().unknown())
    np.testing.assert_allclose(np.asarray(p.sample(prob, state, value, "right", iface)), 3.0, atol=1e-10)
    # each problem solves with its own default: the iterative ones (fdm, fem) stop at a relative residual near
    # 1e-8, which the flux, a gradient over h = 0.1, carries as a few 1e-7
    np.testing.assert_allclose(np.asarray(p.sample(prob, state, flux, "right", iface)), 2.0, atol=1e-6)


def _steady_gas_2d():
    """A steady conducting gas in a box: a reservoir at rest (p = 1, T = 1) on the left, slip walls elsewhere, the
    right one held at T = 2; the temperature is an algebraic unknown T = p/rho. And a state at rest to start from."""
    import jno.jnp_ops as jnn

    gamma = 1.4
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(8, 8)).quad().domain()
    x, y, _ = d.variable("interior", split=True)
    q, T = d.unknown.vector(4), d.unknown()
    qi, Ti = q.bind(x=x, y=y), T.bind(x=x, y=y)
    rho, mx, my, En = qi[0], qi[1], qi[2], qi[3]
    u, v = mx / rho, my / rho
    p = (gamma - 1.0) * (En - 0.5 * (mx * u + my * v))
    fv = jno.fv(flux="roe")
    Fx = jnn.stack([mx, mx * u + p, my * u, u * (En + p)], axis=-1)
    Fy = jnn.stack([my, mx * v, my * v + p, v * (En + p)], axis=-1)
    cons = [Fx.d(x, scheme=fv) + Fy.d(y, scheme=fv) - jnn.stack([0, 0, 0, 0.1 * (Ti.xx + Ti.yy)], axis=-1)]
    cons.append(Ti - p / rho)
    xl, yl, _ = d.variable("left", split=True)
    rest = jnn.stack([1.0 + 0.0 * xl, 0.0 * xl, 0.0 * xl, 1.0 / (gamma - 1.0) + 0.0 * xl], axis=-1)
    cons += [q(xl, yl) - rest, T(xl, yl) - 1.0]
    for tag in ("right", "top", "bottom"):
        xe, ye, _ = d.variable(tag, split=True)
        ne = d.variable(tag, normals=True)
        qe = q.bind(x=xe, y=ye)
        cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
    xr, yr, _ = d.variable("right", split=True)
    cons.append(T(xr, yr) - 2.0)
    gas = jno.fvm(cons)
    n = gas.centroids.shape[0]
    return gas, np.concatenate([np.ones(n), np.zeros(n), np.zeros(n), np.full(n, 1.0 / (gamma - 1.0)), np.ones(n)])


def test_a_steady_gas_participant_solves_from_its_start():
    # Name the break: a steady participant's solve is compiled with its start traced, and the stencil of its
    # Jacobian was checked at zeros, where a gas has no density ("could not assemble"). Its concrete start is
    # recorded for that check now: the participant's solve is the eager solve from the same start.
    gas, at_rest = _steady_gas_2d()
    coupled = np.asarray(participant(gas).solve(gas, {}, x0=at_rest))
    alone, _ = _steady_gas_2d()
    np.testing.assert_allclose(coupled, np.asarray(alone.solve(x0=at_rest)).reshape(-1), atol=1e-12)
