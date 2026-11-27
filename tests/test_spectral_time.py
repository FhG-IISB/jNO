"""Transient spectral problems: the method of lines on jNO's time schemes, and ETDRK4 on a periodic box.

Oracles: exact decaying modes (heat, Taylor-Green vorticity), the measured order of ETDRK4 in dt, and the
condensed streamfunction of ``Δψ = -ω`` against its closed form.
"""

import jax
import jno.numpy as jnn
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


def _box(n_t, T, L=2 * np.pi):
    return jno.shape.rect(0.0, 0.0, L, L).domain(time=(0.0, T, n_t))


def _ties(d, *unknowns):
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    out = []
    for U in unknowns:
        out += [U(xl, yl) - U(xr, yr), U(xb, yb) - U(xt, yt)]
    return out


def test_heat_mode_decays_exactly_with_etdrk4():
    nu, T = 0.3, 1.0
    d = _box(11, T)
    U = d.unknown(space="spectral", order=(16, 16))
    x, y, _ = d.variable("interior", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    u = U.bind(x=x, y=y, t=d.variable("interior", split=True)[2])
    traj = jno.spectral([u.t - nu * (u.xx + u.yy), U(x0, y0) - jnn.sin(2 * x0) * jnn.cos(y0)] + _ties(d, U)).solve(
        time=jno.solve.etdrk4()
    )
    P = U.basis.nodes
    exact = np.exp(-5 * nu * T) * np.sin(2 * P[:, 0]) * np.cos(P[:, 1])
    assert np.abs(np.asarray(traj[-1]) - exact).max() < 1e-13


def test_heat_mode_with_backward_euler_march():
    nu, T, n = 0.1, 0.2, 41
    d = _box(n, T)
    U = d.unknown(space="spectral", order=(8, 8))
    x, y, t = d.variable("interior", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    u = U.bind(x=x, y=y, t=t)
    traj = jno.spectral([u.t - nu * (u.xx + u.yy), U(x0, y0) - jnn.cos(x0)] + _ties(d, U)).solve(time=jno.solve.theta(0.5))
    P = U.basis.nodes
    exact = np.exp(-nu * T) * np.cos(P[:, 0])
    assert np.abs(np.asarray(traj[-1]) - exact).max() < 1e-6  # Crank-Nicolson, dt = 5e-3


def _taylor_green(n_space, n_t, T, nu, dealias=None):
    d = _box(n_t, T)
    W = d.unknown(space="spectral", order=(n_space, n_space), name="w")
    Psi = d.unknown(space="spectral", order=(n_space, n_space), name="psi")
    x, y, t = d.variable("interior", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    sch = jno.sp(dealias=dealias) if dealias else None
    w = W.bind(x=x, y=y, t=t, scheme=sch)
    psi = Psi.bind(x=x, y=y, scheme=sch)
    cons = [
        w.t + psi.y * w.x - psi.x * w.y - nu * (w.xx + w.yy),
        psi.xx + psi.yy + w,
        psi.mean,
        W(x0, y0) - 2 * jnn.sin(x0) * jnn.sin(y0),
    ] + _ties(d, W, Psi)
    return d, W, Psi, jno.spectral(cons)


def test_taylor_green_vortex_decays_exactly():
    """ω = 2 sin x sin y is a steady Euler solution (u·∇ω = 0) decaying as e^{-2νt}: the nonlinear term cancels
    only if the streamfunction is condensed right, so this checks the whole ETDRK4 path."""
    nu, T = 0.05, 1.0
    d, W, Psi, prob = _taylor_green(16, 21, T, nu, dealias=1.5)
    w, psi = prob.solve(time=jno.solve.etdrk4())
    P = W.basis.nodes
    decay = np.exp(-2 * nu * T)
    assert np.abs(np.asarray(w[-1]) - 2 * decay * np.sin(P[:, 0]) * np.sin(P[:, 1])).max() < 1e-12
    assert np.abs(np.asarray(psi[-1]) - decay * np.sin(P[:, 0]) * np.sin(P[:, 1])).max() < 1e-12


def test_etdrk4_is_fourth_order_on_a_nonlinear_problem():
    """Perturbed Taylor-Green: errors against a fine-step run fall with slope ≈ 4 in dt."""
    nu, T = 0.05, 0.8

    def run(n_t):
        d = _box(n_t, T)
        W = d.unknown(space="spectral", order=(16, 16))
        Psi = d.unknown(space="spectral", order=(16, 16))
        x, y, t = d.variable("interior", split=True)
        x0, y0, _ = d.variable("initial", split=True)
        w, psi = W.bind(x=x, y=y, t=t, scheme=jno.sp(dealias=1.5)), Psi.bind(x=x, y=y, scheme=jno.sp(dealias=1.5))
        w0 = 2 * jnn.sin(x0) * jnn.sin(y0) + 0.3 * jnn.cos(2 * x0 + y0)
        cons = [
            w.t + psi.y * w.x - psi.x * w.y - nu * (w.xx + w.yy),
            psi.xx + psi.yy + w,
            psi.mean,
            W(x0, y0) - w0,
        ] + _ties(d, W, Psi)
        return np.asarray(jno.spectral(cons).solve(time=jno.solve.etdrk4(), save_ts=[T])[0][-1])

    ref = run(321)
    errs = [np.abs(run(n) - ref).max() for n in (11, 21, 41)]
    slopes = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(slopes > 3.6), (errs, slopes)


def test_condensed_streamfunction_and_gauge():
    nu, T = 0.0, 0.1
    d, W, Psi, prob = _taylor_green(16, 3, T, nu)
    w, psi = prob.solve(time=jno.solve.etdrk4(), save_ts=[0.0])
    assert abs(float(np.mean(np.asarray(psi[0])))) < 1e-14


def test_etdrk4_refuses_walls():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).domain(time=(0.0, 1.0, 5))
    U = d.unknown(space="spectral", order=(8, 8))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    u = U.bind(x=x, y=y, t=t)
    prob = jno.spectral([u.t - (u.xx + u.yy), U(xb, yb) - 0.0, U(x0, y0) - 0.0])
    with pytest.raises(NotImplementedError, match="periodic box"):
        prob.solve(time=jno.solve.etdrk4())


def test_channel_heat_march_with_walls_is_preconditioned():
    """Fourier x Chebyshev with Dirichlet walls, ``precond=jno.precond.spectral()``: the march's steps run
    Newton-GMRES on the Fourier-mode block inverse of M + s∂R (built once); Crank-Nicolson reaches the exact decay
    of sin(x)cos(πy/2). (Opt-in: jNO's default step solve was measured faster at 16x17 .. 64x65.)"""
    import time as _time

    from jno._spectral import march as _march

    nu, T, n = 0.2, 0.2, 41
    d = jno.shape.rect(0.0, -1.0, 2 * np.pi, 1.0).domain(time=(0.0, T, n))
    U = d.unknown(space="spectral", order=(16, 25))
    x, y, t = d.variable("interior", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    u = U.bind(x=x, y=y, t=t)
    prob = jno.spectral(
        [
            u.t - nu * (u.xx + u.yy),
            U(x0, y0) - jnn.sin(x0) * jnn.cos(np.pi * y0 / 2),
            U(xb, yb) - 0.0,
            U(xt, yt) - 0.0,
            U(xl, yl) - U(xr, yr),
        ]
    )
    built = []
    orig = _march._preconditioned_newton

    def spy(*a, **k):
        out = orig(*a, **k)
        built.append(out is not None)
        return out

    _march._preconditioned_newton = spy
    try:
        t_start = _time.perf_counter()
        traj = prob.solve(time=jno.solve.theta(0.5), precond=jno.precond.spectral())
        _ = _time.perf_counter() - t_start
    finally:
        _march._preconditioned_newton = orig
    assert built == [True]
    P = U.basis.nodes
    exact = np.exp(-nu * (1 + np.pi**2 / 4) * T) * np.sin(P[:, 0]) * np.cos(np.pi * P[:, 1] / 2)
    assert np.abs(np.asarray(traj[-1]) - exact).max() < 1e-5  # CN at dt = 5e-3
