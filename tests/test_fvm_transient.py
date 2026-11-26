"""``jno.fvm`` transient problems and the explicit Runge-Kutta scheme ``jno.solve.erk``."""

import jax
import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _heat(scheme, nu=0.05, T=0.5):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).domain(time=(0.0, T, 100))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    p = jno.fvm([ui.t - nu * (ui.xx + ui.yy), u(xb, yb) - 0.0, u(xi, yi) - jnn.sin(PI * xi) * jnn.sin(PI * yi)])
    traj = np.asarray(p.solve(time=scheme))
    c = p.centroids
    exact = np.exp(-2 * nu * PI**2 * T) * np.sin(PI * c[:, 0]) * np.sin(PI * c[:, 1])
    return traj, float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact))


@pytest.mark.parametrize(
    "scheme",
    [None, jno.solve.theta(0.5), jno.solve.erk("ssprk3"), jno.solve.erk("ssprk3", cfl=0.8)],
    ids=["backward-euler", "crank-nicolson", "ssprk3", "ssprk3-cfl"],
)
def test_heat_decay(scheme):
    traj, err = _heat(scheme)
    assert traj.shape[0] == 100
    assert err < 5e-3


def _line(h, T, n):
    d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=h).domain(time=(0.0, T, n))
    x, t = d.variable("interior", split=True)[:2]
    xl, xr = d.variable("left", split=True)[0], d.variable("right", split=True)[0]
    xi = d.variable("initial", split=True)[0]
    u = d.unknown()
    return d, x, t, xl, xr, xi, u, u.bind(x=x, t=t)


def test_a_tvd_scheme_keeps_the_total_variation_of_a_pulse():
    """u_t + u_x = 0, a square pulse, minmod + SSPRK3 at CFL 0.5: the total variation never grows and the
    values stay within [0, 1]."""
    _, x, t, xl, xr, xi, u, ui = _line(0.01, 0.3, 61)
    s = jno.fv(limiter="minmod")
    pulse = (jnn.tanh(1e4 * (xi - 0.1)) - jnn.tanh(1e4 * (xi - 0.3))) / 2
    p = jno.fvm([ui.t + ui.d(x, scheme=s), u(xl) - 0.0, u(xi) - pulse])
    traj = np.asarray(p.solve(time=jno.solve.erk("ssprk3")))
    tv = np.abs(np.diff(traj, axis=1)).sum(axis=1)
    assert np.all(np.diff(tv) <= 1e-10)
    assert traj.min() > -1e-12 and traj.max() < 1 + 1e-12
    centre = (p.centroids[:, 0] * traj[-1]).sum() / traj[-1].sum()
    assert centre == pytest.approx(0.2 + 0.3, abs=0.01)  # carried at speed 1


def test_an_inviscid_burgers_shock_moves_at_the_rankine_hugoniot_speed():
    """u_t + (u²/2)_x = 0 from u = 1 | 0 at x = 0.2: a shock at speed (1 + 0)/2."""
    T = 0.6
    _, x, t, xl, xr, xi, u, ui = _line(0.005, T, 121)
    s = jno.fv(limiter="minmod")
    step = (1.0 - jnn.tanh(1e4 * (xi - 0.2))) / 2
    p = jno.fvm([ui.t + (ui * ui / 2).d(x, scheme=s), u(xl) - 1.0, u(xi) - step])
    traj = np.asarray(p.solve(time=jno.solve.erk("ssprk3", cfl=0.5)))
    c = p.centroids[:, 0]
    front = c[np.argmin(np.abs(traj[-1] - 0.5))]
    assert front == pytest.approx(0.2 + 0.5 * T, abs=0.01)


def test_ssprk54_is_fourth_order():
    """u_t = -u in every cell (a reaction only): the error at t = 1 falls by 16 per halving of the step."""
    errs = []
    for n in (6, 11):
        d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.25).domain(time=(0.0, 1.0, n))
        x, t = d.variable("interior", split=True)[:2]
        xi = d.variable("initial", split=True)[0]
        u = d.unknown()
        ui = u.bind(x=x, t=t)
        traj = np.asarray(jno.fvm([ui.t + ui, u(xi) - 1.0]).solve(time=jno.solve.erk("ssprk54")))
        errs.append(abs(traj[-1, 0] - np.exp(-1.0)))
    assert errs[0] / errs[1] > 12


def test_the_wave_equation_marches_u_and_its_velocity():
    """u_tt = u_xx, u = sin(πx) at rest: u = cos(πt) sin(πx) (Crank-Nicolson on [u; v])."""
    _, x, t, xl, xr, xi, u, ui = _line(0.02, 1.0, 201)
    p = jno.fvm([ui.tt - ui.xx, u(xl) - 0.0, u(xr) - 0.0, u(xi) - jnn.sin(PI * xi)])
    traj = np.asarray(p.solve(time=jno.solve.theta(0.5)))
    c = p.centroids[:, 0]
    assert np.abs(traj[-1] - np.cos(PI) * np.sin(PI * c)).max() < 2e-2


def test_erk_marches_an_fdm_problem_with_algebraic_rows():
    """jno.fdm's Dirichlet rows have zero mass: erk solves them at every stage."""
    nu, T = 0.05, 0.2
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(time=(0.0, T, 200))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    cons = [ui.t - nu * (ui.xx + ui.yy), u(xb, yb) - 0.0, u(xi, yi) - jnn.sin(PI * xi) * jnn.sin(PI * yi)]
    a = np.asarray(jno.fdm(cons).solve(time=jno.solve.erk("ssprk3")))
    b = np.asarray(jno.fdm(cons).solve(time=jno.solve.theta(0.5)))
    assert np.abs(a[-1] - b[-1]).max() < 1e-4


def test_erk_refuses_a_bad_tableau():
    with pytest.raises(ValueError, match="lower"):
        jno.solve.erk(tableau=([[0.0, 1.0], [0.0, 0.0]], [0.5, 0.5], [0.0, 1.0], 2))
    with pytest.raises(ValueError, match="method"):
        jno.solve.erk("dopri5")


def test_periodic_advection_returns_after_one_period():
    """``u(left) - u(right)`` joins the two ends: a sine carried at speed 1 comes back after t = 1, and the
    total is conserved to round-off."""
    _, x, t, xl, xr, xi, u, ui = _line(0.02, 1.0, 101)
    p = jno.fvm([ui.t + ui.d(x, scheme=jno.fv()), u(xl) - u(xr), u(xi) - jnn.sin(2 * PI * xi)])
    traj = np.asarray(p.solve(time=jno.solve.erk("ssprk3", cfl=0.5)))
    c = p.centroids[:, 0]
    assert np.abs(traj[-1] - np.sin(2 * PI * c)).max() < 0.08
    assert abs((traj[-1] - traj[0]) @ p.volumes) < 1e-13


@pytest.mark.slow
def test_taylor_green_vortex_decays_at_its_rate():
    """Periodic in x and y (ties for U and p on the same faces join them once), a pressure gauge, and
    Crank-Nicolson: the kinetic energy decays as exp(-4 nu t)."""
    nu, T = 0.1, 0.5
    L = 2 * PI
    d = jno.shape.rect(0.0, 0.0, L, L).structured(n=24).quad().domain(time=(0.0, T, 26))
    d.point_region("ppin", (0.0, 0.0))
    x, y, t = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xp, yp, _ = d.variable("ppin", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    U, P = d.unknown.vector(2), d.unknown()
    Ui, pi = U.bind(x=x, y=y, t=t), P.bind(x=x, y=y, t=t)
    u, v = Ui[0], Ui[1]
    s = jno.fv()
    mom = Ui.t + (u * Ui).d(x, scheme=s) + (v * Ui).d(y, scheme=s) - nu * (Ui.xx + Ui.yy) + jnn.stack([pi.x, pi.y], axis=-1)
    u0 = jnn.stack([-jnn.cos(xi) * jnn.sin(yi), jnn.sin(xi) * jnn.cos(yi)], axis=-1)
    cons = [
        mom,
        Ui[0].x + Ui[1].y,
        U(xl, yl) - U(xr, yr),
        U(xb, yb) - U(xt, yt),
        P(xl, yl) - P(xr, yr),
        P(xb, yb) - P(xt, yt),
    ]
    cons += [P(xp, yp) - 0.0, U(xi, yi) - u0]
    prob = jno.fvm(cons)
    traj = np.asarray(prob.solve(time=jno.solve.theta(0.5)))
    energy = (traj[:, 0] ** 2 + traj[:, 1] ** 2) @ prob.volumes
    assert energy[-1] / energy[0] == pytest.approx(np.exp(-4 * nu * T), rel=2e-2)


def _conducting_gas():
    """A gas at rest in a closed box (slip walls): the Euler equations with conduction of an algebraic
    temperature ``T = p/rho`` in the energy row, held at T = 1 on the left wall; a hot spot in the middle."""
    gamma, k = 1.4, 0.05
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(8, 8)).quad().domain(time=(0.0, 0.04, 3))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    q, T = d.unknown.vector(4), d.unknown()
    qi, Ti = q.bind(x=x, y=y, t=t), T.bind(x=x, y=y, t=t)
    rho, mx, my, En = qi[0], qi[1], qi[2], qi[3]
    u, v = mx / rho, my / rho
    p = (gamma - 1.0) * (En - 0.5 * (mx * u + my * v))
    fv = jno.fv(flux="hll")
    Fx = jnn.stack([mx, mx * u + p, my * u, u * (En + p)], axis=-1)
    Fy = jnn.stack([my, mx * v, my * v + p, v * (En + p)], axis=-1)
    cons = [qi.t + Fx.d(x, scheme=fv) + Fy.d(y, scheme=fv) - jnn.stack([0, 0, 0, k * (Ti.xx + Ti.yy)], axis=-1)]
    cons.append(Ti - p / rho)
    for tag in ("left", "right", "top", "bottom"):
        xe, ye, _ = d.variable(tag, split=True)
        ne = d.variable(tag, normals=True)
        qe = q.bind(x=xe, y=ye)
        cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
    xl, yl, _ = d.variable("left", split=True)
    cons.append(T(xl, yl) - 1.0)
    T0 = 1.0 + 0.5 * jnn.exp(-20.0 * ((xi - 0.5) ** 2 + (yi - 0.5) ** 2))
    cons += [q(xi, yi) - jnn.stack([1.0 / T0, 0.0 * xi, 0.0 * xi, 1.0 / (gamma - 1.0) + 0.0 * xi], axis=-1)]
    cons += [T(xi, yi) - T0]
    return jno.fvm(cons)


def test_a_gas_steps_implicitly_inside_a_compiled_program():
    # Name the break: inside jax.jit (a coupling window) the state is traced, and the sparsity of the Jacobian
    # was probed at zeros plus 1 % noise: negative densities and pressures, where the flux has complex wave
    # speeds and the assembled tangent missed the jvp by 1-10 % at every stencil width, so the march raised
    # "could not assemble". Slip walls close the box: the backward Euler steps keep the mass to round-off.
    gas = _conducting_gas()
    x0 = jax.numpy.asarray(gas._initial_values(gas._params_scope())).reshape(-1)
    traj = np.asarray(jax.jit(lambda s: gas.solve(x0=s))(x0)).reshape(3, 5, -1)
    assert np.isfinite(traj).all()
    mass = traj[:, 0] @ np.asarray(gas.volumes)
    np.testing.assert_allclose(mass, mass[0], rtol=1e-13)
    assert np.abs(traj[-1, 1:3]).max() > 1e-6  # the hot spot has started to push the gas: the steps did march
