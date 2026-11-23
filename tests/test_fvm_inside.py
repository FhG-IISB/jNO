"""``q.bind(...).inside``: the value in the cell next to a boundary face, so a condition can give the face its outer
state as a function of the inside one (``qb - G(qb.inside)``): an open outflow, an outlet at a back pressure, an
inflow from a plenum, a valve that closes into a wall."""

import jax
import numpy as np
import pytest

import jno
import jno.jnp_ops as jnn

GAMMA = 1.4
CP = GAMMA / (GAMMA - 1.0)  # R = 1
P0, T0 = 2.0, 1.0  # a plenum
RHO0 = P0 / T0
PR_CRIT = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _advection(right_inside):
    d = jno.shape.rect(0.0, 0.0, 1.0, 0.05).structured(n=(20, 1)).quad().domain(time=(0.0, 0.5, 11))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    cons = [ui.t + ui.d(x, scheme=jno.fv(flux="rusanov")), u(xl, yl) - 1.0, u(xi, yi) - jnn.exp(-50.0 * (xi - 0.5) ** 2)]
    if right_inside:
        ur = u.bind(x=xr, y=yr)
        cons.append(ur - ur.inside)  # the face takes the inside value: an open outflow, written out
    return np.asarray(jno.fvm(cons).solve(time=jno.solve.erk("ssprk3", cfl=0.5)))


def test_a_face_given_its_inside_value_is_the_open_outflow():
    # Name the break: if `.inside` read the face's own value, `ur - ur.inside` would be 0 = 0 and the march
    # would fail to build; if it read another cell, the outflow would differ from the open one.
    np.testing.assert_allclose(_advection(True), _advection(False), atol=1e-13)


def _pressure(q):
    return (GAMMA - 1.0) * (q[3] - 0.5 * (q[1] * q[1] + q[2] * q[2]) / q[0])


def _plenum_tube(p_back):
    """A tube fed at its left end from the plenum (p0, T0) through an injector that closes into a wall when the
    pressure inside reaches p0, held at p_back at its right end; both ends are outer states of the inside one."""
    L, n = 0.5, 20  # short: the outlet reflects sound fully, the injector damps it once per round trip
    d = jno.shape.rect(0.0, 0.0, L, L / n).structured(n=(n, 1)).quad().domain(time=(0.0, 40.0, 5))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    q = d.unknown.vector(4)  # (rho, m_x, m_y, E)
    qi = q.bind(x=x, y=y, t=t)
    rho, mx, my, En = qi[0], qi[1], qi[2], qi[3]
    u, v = mx / rho, my / rho
    p = _pressure(qi)
    fv = jno.fv(flux="hll")
    Fx = jnn.stack([mx, mx * u + p, my * u, u * (En + p)], axis=-1)
    Fy = jnn.stack([my, mx * v, my * v + p, v * (En + p)], axis=-1)
    cons = [qi.t + Fx.d(x, scheme=fv) + Fy.d(y, scheme=fv)]
    for tag in ("top", "bottom"):
        xe, ye, _ = d.variable(tag, split=True)
        qe = q.bind(x=xe, y=ye)
        cons.append(qe[2] - 0.0)
    # the injector: the plenum gas expanded isentropically to the inside pressure (sonic below the critical one),
    # entering along +x; closed (a mirror) once the inside pressure reaches the plenum's
    xl, yl, _ = d.variable("left", split=True)
    ql = q.bind(x=xl, y=yl)
    qin = ql.inside
    pin = _pressure(qin)
    pr = jnn.minimum(jnn.maximum(pin / P0, PR_CRIT), 1.0)
    rho_j = RHO0 * pr ** (1.0 / GAMMA)
    vel = jnn.sqrt(jnn.maximum(2.0 * CP * T0 * (1.0 - pr ** ((GAMMA - 1.0) / GAMMA)), 0.0))
    p_j = P0 * pr
    jet = jnn.stack([rho_j, rho_j * vel, 0.0 * vel, p_j / (GAMMA - 1.0) + 0.5 * rho_j * vel * vel], axis=-1)
    wall = jnn.stack([qin[0], -qin[1], qin[2], qin[3]], axis=-1)
    cons.append(ql - jnn.where(pin >= P0, wall, jet))
    # the outlet: the inside state at the back pressure
    xr, yr, _ = d.variable("right", split=True)
    qr = q.bind(x=xr, y=yr)
    qo = qr.inside
    out = jnn.stack([qo[0], qo[1], qo[2], p_back / (GAMMA - 1.0) + 0.5 * (qo[1] * qo[1] + qo[2] * qo[2]) / qo[0]], axis=-1)
    cons.append(qr - out)
    rest = jnn.stack([p_back + 0.0 * xi, 0.0 * xi, 0.0 * xi, p_back / (GAMMA - 1.0) + 0.0 * xi], axis=-1)
    cons.append(q(xi, yi) - rest)  # at rest at the back pressure, T = 1
    traj = np.asarray(jno.fvm(cons).solve(time=jno.solve.erk("ssprk3", cfl=0.5)))
    return traj.reshape(traj.shape[0], 4, -1)


def test_a_plenum_feeds_a_tube_at_the_isentropic_state():
    # Subsonic: the steady tube holds the plenum gas expanded isentropically to the back pressure, the flow the
    # injector model gives at that pressure. A slip wall behind a source in the first cell held 63 % of it.
    g = _plenum_tube(1.5)
    pr = 1.5 / P0
    rho = RHO0 * pr ** (1.0 / GAMMA)
    vel = np.sqrt(2.0 * CP * T0 * (1.0 - pr ** ((GAMMA - 1.0) / GAMMA)))
    assert np.abs(g[-1] - g[-2]).max() < 1e-8  # steady
    # a uniform steady state of the discrete equations: exact to the march's convergence
    np.testing.assert_allclose(g[-1, 0], rho, rtol=1e-8)
    np.testing.assert_allclose(g[-1, 1], rho * vel, rtol=1e-8)
    np.testing.assert_allclose(_pressure(g[-1]), 1.5, rtol=1e-8)


def test_a_closed_injector_holds_the_tube_at_rest():
    g = _plenum_tube(2.5)  # back pressure above the plenum's: the injector stays shut
    np.testing.assert_allclose(g[-1, 1], 0.0, atol=1e-8)
    np.testing.assert_allclose(_pressure(g[-1]), 2.5, rtol=1e-8)


def test_inside_belongs_to_a_finite_volume_problem():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(8, 8)).domain()
    x, y, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    with pytest.raises(ValueError, match="inside"):
        jno.fdm([ui.xx + ui.yy, ur - 0.5 * ur.inside, u(xl, yl) - 1.0])
