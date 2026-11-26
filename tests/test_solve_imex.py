"""IMEX Runge-Kutta for stiff local sources (jno.solve.imex): face fluxes explicit, cell-local terms implicit.

A one-step Arrhenius reaction, Y' = -ω, T' = Q ω, ω = A Y exp(-Ta/T), in a box held uniform (diffusion present
but idle, insulated walls): every cell is the same stiff ODE, whose ignition delay and end state scipy's Radau
gives to 1e-10. Near ignition the reaction rate reaches 4e5 against a step of 5e-3, two thousand times past an
explicit step's limit; the implicit local solve does not care.

With a temperature-dependent heat capacity the temperature is an algebraic unknown of the energy,
e = (c0 + c1 T) T: a local equation without a time derivative, solved in the same per-cell Newton.
"""

import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import jno
import jno.jnp_ops as jnn

A_PRE, TA, Q, T0, D = 1e7, 20.0, 5.0, 1.0, 1e-3
T_END, N_TIME = 1.0, 201  # dt = 5e-3
C0, C1 = 1.0, 0.1


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _box(extra=None):
    d = jno.shape.rect(0.0, 0.0, 1.0, 0.25).structured(n=(4, 1)).quad().domain(time=(0.0, T_END, N_TIME))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    return d, x, y, t, xi, yi


def _ignition(time):
    d, x, y, t, xi, yi = _box()
    Yf, T = d.unknown(), d.unknown()
    Yi, Ti = Yf.bind(x=x, y=y, t=t), T.bind(x=x, y=y, t=t)
    w = A_PRE * Yi * jnn.exp(-TA / Ti)
    prob = jno.fvm(
        [
            Yi.t - D * (Yi.xx + Yi.yy) + w,
            Ti.t - D * (Ti.xx + Ti.yy) - Q * w,
            Yf(xi, yi) - (1.0 + 0.0 * xi),
            T(xi, yi) - (T0 + 0.0 * xi),
        ]
    )
    return np.asarray(prob.solve(time=time))  # (n_time, 2, cells)


def _reference(times, cv=False):
    def rhs(_, z):
        if cv:
            Yv, e = z
            T = (-C0 + np.sqrt(C0**2 + 4 * C1 * e)) / (2 * C1)
        else:
            Yv, T = z
        w = A_PRE * Yv * np.exp(-TA / T)
        return [-w, Q * w]

    z0 = [1.0, (C0 + C1 * T0) * T0] if cv else [1.0, T0]
    sol = solve_ivp(rhs, (0.0, T_END), z0, method="Radau", rtol=1e-10, atol=1e-12, t_eval=times, dense_output=True)
    return sol


def _delay(times, T, level):
    k = int(np.argmax(T >= level))
    return float(np.interp(level, T[k - 1 : k + 1], times[k - 1 : k + 1]))


@pytest.mark.parametrize("method", ["ars222", "ars443"])
def test_imex_ignites_when_the_stiff_ode_does(method):
    times = np.linspace(0.0, T_END, N_TIME)
    traj = _ignition(jno.solve.imex(method))
    T = traj[:, 1, :]
    assert np.abs(T - T[:, :1]).max() < 1e-12  # the box stays uniform
    ref = _reference(times)
    level = T0 + 0.5 * Q
    got, want = _delay(times, T[:, 0], level), _delay(times, ref.y[1], level)
    assert abs(got - want) < 0.01 * want  # the delay to 1 %, at dt = 5e-3 across a front 1e-3 wide in time
    np.testing.assert_allclose(T[-1], T0 + Q, atol=1e-6)  # burnt out: the adiabatic temperature
    np.testing.assert_allclose(traj[-1, 0, :], 0.0, atol=1e-6)


def test_an_explicit_march_cannot_take_that_step():
    with pytest.raises(Exception):
        traj = _ignition(jno.solve.erk("ssprk3"))
        assert np.isfinite(traj).all() and np.abs(traj[-1, 1] - (T0 + Q)).max() < 1e-3


def test_an_algebraic_temperature_is_solved_with_the_local_terms():
    times = np.linspace(0.0, T_END, N_TIME)
    d, x, y, t, xi, yi = _box()
    Yf, e, T = d.unknown(), d.unknown(), d.unknown()
    Yi, ei, Ti = Yf.bind(x=x, y=y, t=t), e.bind(x=x, y=y, t=t), T.bind(x=x, y=y, t=t)
    w = A_PRE * Yi * jnn.exp(-TA / Ti)
    prob = jno.fvm(
        [
            Yi.t - D * (Yi.xx + Yi.yy) + w,
            ei.t - D * (ei.xx + ei.yy) - Q * w,
            ei - (C0 + C1 * Ti) * Ti,  # e(T): no time derivative, the temperature follows the energy
            Yf(xi, yi) - (1.0 + 0.0 * xi),
            e(xi, yi) - ((C0 + C1 * T0) * T0 + 0.0 * xi),
            T(xi, yi) - (T0 + 0.0 * xi),
        ]
    )
    traj = np.asarray(prob.solve(time=jno.solve.imex("ars222")))
    ref = _reference(times, cv=True)
    e_ref = ref.y[1]
    T_ref = (-C0 + np.sqrt(C0**2 + 4 * C1 * e_ref)) / (2 * C1)
    level = 0.5 * (T_ref[0] + T_ref[-1])
    got, want = _delay(times, traj[:, 2, 0], level), _delay(times, T_ref, level)
    assert abs(got - want) < 0.01 * want
    # the algebraic row holds at every saved step
    ev, Tv = traj[:, 1, :], traj[:, 2, :]
    np.testing.assert_allclose(ev, (C0 + C1 * Tv) * Tv, atol=1e-9)


def test_an_algebraic_unknown_beside_a_vector_equation_is_one_more_equation():
    # A vector `q.t` and `jnn.stack` keep no coordinate binding of their own, so the vector equation's region read
    # as unknown and the algebraic row s = q0 + q1 beside it was taken for a pin: "one PDE equation per unknown".
    d, x, y, t, xi, yi = _box()
    q, s = d.unknown.vector(2), d.unknown()
    qi, si = q.bind(x=x, y=y, t=t), s.bind(x=x, y=y, t=t)
    prob = jno.fvm(
        [
            qi.t + jnn.stack([qi[0], 2.0 * qi[1]], axis=-1),
            si - (qi[0] + qi[1]),
            q(xi, yi) - jnn.stack([1.0 + 0.0 * xi, 1.0 + 0.0 * xi], axis=-1),
            s(xi, yi) - (2.0 + 0.0 * xi),
        ]
    )
    traj = np.asarray(prob.solve(time=jno.solve.imex("ars222"))).reshape(N_TIME, 3, -1)
    times = np.linspace(0.0, T_END, N_TIME)[:, None]
    np.testing.assert_allclose(traj[:, 2], traj[:, 0] + traj[:, 1], atol=1e-10)  # the algebraic row, to Newton's tol
    # ARS(2,2,2) at h = 5e-3 on rates 1 and 2: a second-order error of ~1e-5 relative
    np.testing.assert_allclose(traj[:, 0], np.exp(-times) + 0.0 * traj[:, 0], rtol=1e-4)
    np.testing.assert_allclose(traj[:, 1], np.exp(-2.0 * times) + 0.0 * traj[:, 1], rtol=1e-4)


def test_a_flux_through_an_algebraic_unknown_gets_its_stable_step():
    # e conducted through an algebraic T = e/c: the flux depends on e only through T, so a stable-step estimate
    # that leaves the algebraic rows out saw no flux at all, took one step where the diffusion needed several, and
    # the march blew up. Filled in from the local algebraic rows, it is the scalar u.t = (k/c) Δu, step for step.
    k, c = 0.1, 2.5
    d = jno.shape.rect(0.0, 0.0, 0.5, 0.5).sized(0.05).domain(time=(0.0, 0.05, 11))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    xr, yr, _ = d.variable("right", split=True)
    e, T = d.unknown(), d.unknown()
    ei, Ti = e.bind(x=x, y=y, t=t), T.bind(x=x, y=y, t=t)
    pair = jno.fvm(
        [
            ei.t - k * (Ti.xx + Ti.yy),
            Ti - ei / c,
            T(xr, yr) - 1.0,
            e(xi, yi) - (2.0 * c + 0.0 * xi),
            T(xi, yi) - (2.0 + 0.0 * xi),
        ]
    )
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    scalar = jno.fvm([ui.t - (k / c) * (ui.xx + ui.yy), u(xr, yr) - 1.0, u(xi, yi) - (2.0 + 0.0 * xi)])
    got = np.asarray(pair.solve(time=jno.solve.imex("ars222", cfl=0.5))).reshape(11, 2, -1)[:, 1]
    want = np.asarray(scalar.solve(time=jno.solve.imex("ars222", cfl=0.5))).reshape(11, -1)
    np.testing.assert_allclose(got, want, atol=1e-10)  # the same steps of the same ODE: round-off and Newton tol


FLAME = dict(a=100.0, ta=10.0, q=3.0, t0=1.0, tc=1.2, width=0.02)


def _flame_rate(Yv, Tv):
    f = FLAME  # Arrhenius, switched off smoothly below the ignition temperature (the cold-boundary difficulty)
    return f["a"] * Yv * np.exp(-f["ta"] / Tv) * 0.5 * (1.0 + np.tanh((Tv - f["tc"]) / f["width"]))


def _flame_speed():
    """The travelling-wave speed of Y'' + s Y' - w(Y) = 0 (T = Tb - Q Y, unit Lewis number), in the phase plane: with
    P = Y' a function of Y, dP/dY = -s + w(Y)/P, P = 0 on the burnt side Y = 0; above the ignition cutoff w = 0 and
    P = s (1 - Y) (Williams, Combustion Theory, ch. 5), here from where the switch is e^-10. Shot in Q = P² (regular
    where P vanishes), dQ/dY = -2 s sqrt(Q) + 2 w, down to Y = 0; s is the root of Q(0)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    f = FLAME
    tb = f["t0"] + f["q"]
    y_cut = (tb - (f["tc"] - 5.0 * f["width"])) / f["q"]  # the switch is e^-10 there: w ~ 1e-9 of its peak

    def q_at_zero(s):
        r = solve_ivp(
            lambda Y, Qv: [-2.0 * s * np.sqrt(max(Qv[0], 0.0)) + 2.0 * _flame_rate(Y, tb - f["q"] * Y)],
            (y_cut, 0.0),
            [(s * (1.0 - y_cut)) ** 2],
            method="LSODA",
            rtol=1e-12,
            atol=1e-15,
        )
        return float(r.y[0, -1])

    return brentq(q_at_zero, 0.5, 8.0, xtol=1e-12)


@pytest.mark.slow
def test_a_premixed_flame_runs_at_the_travelling_wave_speed():
    # Y_t = Y_xx - w, T_t = T_xx + Q w, w = A Y exp(-Ta/T) above an ignition temperature, a burnt region igniting a
    # fresh one: the front settles to the travelling wave. The march takes the diffusion explicitly (its CFL limit
    # decides the substeps) and the reaction implicitly per cell.
    f = FLAME
    s_ref = _flame_speed()
    n, x1 = 400, 20.0
    d = jno.shape.rect(0.0, 0.0, x1, x1 / n).structured(n=(n, 1)).quad().domain(time=(0.0, 8.0, 81))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    Yf, T = d.unknown(), d.unknown()
    Yi, Ti = Yf.bind(x=x, y=y, t=t), T.bind(x=x, y=y, t=t)
    w = f["a"] * Yi * jnn.exp(-f["ta"] / Ti) * 0.5 * (1.0 + jnn.tanh((Ti - f["tc"]) / f["width"]))
    burnt = xi < 2.0
    prob = jno.fvm(
        [
            Yi.t - (Yi.xx + Yi.yy) + w,
            Ti.t - (Ti.xx + Ti.yy) - f["q"] * w,
            Yf(xi, yi) - jnn.where(burnt, 0.0, 1.0),
            T(xi, yi) - jnn.where(burnt, f["t0"] + f["q"], f["t0"]),
        ]
    )
    traj = np.asarray(prob.solve(time=jno.solve.imex("ars222", cfl=0.5)))
    xc = np.asarray(prob.centroids)[:, 0]
    order = np.argsort(xc)
    times = np.linspace(0.0, 8.0, 81)
    front = np.array([np.interp(0.5, traj[k, 0, order], xc[order]) for k in range(len(times))])
    late = times >= 4.0  # after the start-up transient
    s_num = np.polyfit(times[late], front[late], 1)[0]
    # h = 0.05 against a flame some 0.5 thick: the speed to 2 %
    assert abs(s_num - s_ref) < 0.02 * s_ref, (s_num, s_ref)
