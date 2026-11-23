"""Pseudo-transient continuation (jno.solve.ptc): a steady solve along the transient path.

Steady viscous Burgers, (u²/2)' = ε u'' on [-1, 1] with u(∓1) = ±1, has a family of near-solutions
-tanh((x - x0)/2ε): the shock position is held only by exponentially small terms, so Newton from rest lands
wherever its first step throws it. The state the TIME march from rest reaches is the symmetric one, x0 = 0, and
pseudo-time follows that march. Where the steady problem has one solution, it ends where Newton does.
"""

import jax
import numpy as np
import pytest

import jno

EPS = 0.01


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _burgers(time=None):
    d = (
        jno.shape.rect(-1.0, 0.0, 1.0, 0.05)
        .structured(n=(200, 1))
        .quad()
        .domain(**({} if time is None else {"time": time}))
    )
    x, y, t = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t) if time is not None else u.bind(x=x, y=y)
    cons = [(0.5 * ui * ui).x - EPS * (ui.xx + ui.yy), u(xl, yl) - 1.0, u(xr, yr) + 1.0]
    if time is not None:
        xi, yi, _ = d.variable("initial", split=True)
        cons = [ui.t + cons[0], *cons[1:], u(xi, yi) - 0.0 * xi]
    return jno.fvm(cons)


def test_ptc_reaches_the_state_the_time_march_reaches():
    prob = _burgers()
    x = np.asarray(prob.centroids)[:, 0]
    sol = np.asarray(prob.solve(nonlinear=jno.solve.ptc(), x0=np.zeros(prob._N)))
    # the long-time state of the march from rest (t = 20, many shock-crossing times)
    marched = np.asarray(_burgers(time=(0.0, 20.0, 201)).solve())[-1]
    np.testing.assert_allclose(sol, marched, atol=1e-3)
    # and that state is the symmetric shock, to the discretisation (h = ε)
    np.testing.assert_allclose(sol, -np.tanh(x / (2 * EPS)), atol=0.05)
    assert abs(float(np.interp(0.0, x, sol))) < 1e-6  # symmetric: the shock sits at x = 0


def test_ptc_ends_where_newton_does_on_a_well_posed_nonlinear_problem():
    # -div((1 + u²) grad u) = 1 has one solution; Newton finds it from rest, and pseudo-time must end there too
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    prob = jno.fvm([-((1.0 + ui * ui) * ui.x).x - ((1.0 + ui * ui) * ui.y).y - 1.0, u(xb, yb) - 0.0])
    ref = np.asarray(prob.solve())
    assert np.abs(ref).max() > 0.05
    np.testing.assert_allclose(np.asarray(prob.solve(nonlinear=jno.solve.ptc())), ref, atol=1e-9)


def _oblique_shock(mach, theta, gamma):
    """The weak oblique shock of a ramp: its angle beta (the theta-beta-M relation, by bisection) and the pressure
    ratio across it."""

    def turn(beta):
        s2 = (mach * np.sin(beta)) ** 2
        return np.arctan(2.0 / np.tan(beta) * (s2 - 1.0) / (mach**2 * (gamma + np.cos(2.0 * beta)) + 2.0)) - theta

    lo, hi = np.arcsin(1.0 / mach) + 1e-9, np.deg2rad(64.0)  # between the Mach angle and the detachment angle
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if turn(mid) < 0.0 else (lo, mid)
    beta = 0.5 * (lo + hi)
    return beta, 1.0 + 2.0 * gamma / (gamma + 1.0) * ((mach * np.sin(beta)) ** 2 - 1.0)


def _wedge(h=0.06, mach=3.0, theta_deg=15.0, gamma=1.4, flux="hll", limiter="minmod", scale=1.0, smooth=False):
    """Mach 3 over a 15-degree ramp, steady: the freestream (p and rho ``scale`` times 1 and gamma, sound speed 1) on
    the inflow and the far field, slip on the plate and the ramp, an open supersonic exit. Returns the problem and the
    freestream state."""
    import jno.jnp_ops as jnn

    th, x0, L, H = np.deg2rad(theta_deg), 0.3, 1.2, 1.0
    d = jno.shape.polygon([(0.0, 0.0), (x0, 0.0), (L, (L - x0) * np.tan(th)), (L, H), (0.0, H)], size=h).domain()
    x, y, _ = d.variable("interior", split=True)
    q = d.unknown.vector(4)
    qi = q.bind(x=x, y=y)
    rho, mx, my, E = qi[0], qi[1], qi[2], qi[3]
    u, v = mx / rho, my / rho
    p = (gamma - 1.0) * (E - 0.5 * (mx * u + my * v))
    s = jno.fv(flux=flux, limiter=limiter, smooth=smooth)
    Fx = jnn.stack([mx, mx * u + p, my * u, u * (E + p)], axis=-1)
    Fy = jnn.stack([my, mx * v, my * v + p, v * (E + p)], axis=-1)
    free = tuple(scale * f for f in (gamma, gamma * mach, 0.0, 1.0 / (gamma - 1.0) + 0.5 * gamma * mach**2))
    cons = [Fx.d(x, scheme=s) + Fy.d(y, scheme=s)]
    for e in ("e4", "e3"):  # the inflow and the far field above: the freestream
        xe, ye, _ = d.variable(e, split=True)
        cons.append(q(xe, ye) - jnn.stack([free[k] + 0.0 * xe for k in range(4)], axis=-1))
    for e in ("e0", "e1"):  # slip on the plate and the ramp
        xe, ye, _ = d.variable(e, split=True)
        ne = d.variable(e, normals=True)
        qe = q.bind(x=xe, y=ye)
        cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
    return jno.fvm(cons), free, (rho, p)


@pytest.mark.parametrize("scale", [1.0, 1e-2])
def test_ptc_reaches_the_steady_supersonic_wedge(scale):
    # Name the break: from the freestream, the pseudo-time march let the density go negative while the shock formed
    # (rho_min -0.7 at step 29, git-bug d7f66c4) and never recovered; kept positive, it still ate the pressure of one
    # cell away (a damping factor per conserved variable advances them at different rates) and stalled at |R| 2.1. Expected: the weak oblique shock of the
    # theta-beta-M relation, p2/p1 = 2.82 at Mach 3 and 15 degrees, on the plateau next to the ramp. h = 0.06 smears
    # the shock over a few cells, not the plateau: 3 %. Smooth settings (van Albada, a smooth HLL): with minmod the
    # end phase rides its kinks, and on the GPU (reductions in a varying order) 1 run in 3 stalled at |R| 2e-3. The Euler equations do not see a common scale of
    # density and pressure, so a freestream at 1 % of them gives the same ratio; there the stencil check's probe used
    # to add an absolute 0.01 to every entry, a negative density (a NaN residual: "could not assemble").
    gamma, theta = 1.4, np.deg2rad(15.0)
    prob, free, (rho, p) = _wedge(scale=scale, limiter="vanalbada", smooth=True)
    n = prob._N
    start = np.concatenate([np.full(n, f) for f in free])
    sol = np.asarray(prob.solve(nonlinear=jno.solve.ptc(positive=[rho, p]), x0=start)).reshape(4, n)
    assert np.isfinite(sol).all() and sol[0].min() > 0.0
    pres = (gamma - 1.0) * (sol[3] - 0.5 * (sol[1] ** 2 + sol[2] ** 2) / sol[0])
    c = np.asarray(prob.centroids)
    beside_ramp = (c[:, 0] > 0.7) & (c[:, 1] < (c[:, 0] - 0.3) * np.tan(theta) + 0.05)
    _, ratio = _oblique_shock(3.0, theta, gamma)
    assert abs(ratio - 2.822) < 2e-3  # the closed form itself
    np.testing.assert_allclose(np.median(pres[beside_ramp]) / scale, ratio, rtol=0.03)


def test_ptc_keeps_a_declared_positive_quantity_positive():
    # R(u) = 1 - 1/u: one root, u = 1, increasing and concave. From u = 50 a pseudo-time step lands below zero, where
    # R is finite (it tends to 1) and has no root, so the march runs off to minus infinity (measured: -inf after 80
    # steps): the residual never grows by the 100x the step check refuses. Declared positive (a step may take at most
    # 80 % of it away), the march stays on the side of the root and ends on it.
    import jax.numpy as jnp

    def R(u):
        return 1.0 - 1.0 / u

    u0 = jnp.array([50.0])
    J = jax.jacfwd(R)
    solve = jnp.linalg.solve
    plain = np.asarray(jno.solve.ptc(max_steps=80)(R, u0, jacobian=J, linear_solve=solve))
    assert not np.allclose(plain, 1.0, atol=1e-3)  # the failure the positivity guards against
    kept = np.asarray(jno.solve.ptc(max_steps=80)(R, u0, jacobian=J, linear_solve=solve, positive=lambda u: u))
    np.testing.assert_allclose(kept, 1.0, atol=1e-10)


def test_a_gradient_flows_through_ptc():
    # Name the break: ptc's pseudo-time march is a lax.while_loop, which reverse mode cannot differentiate, so a
    # gradient through a steady solve by ptc raised. With the implicit function theorem at the root (custom_root,
    # one solve with J^T), d/ds of a functional of -div((1 + u^2) grad u) = s f matches central finite differences;
    # and it equals Newton's gradient of the same problem (the same root, so the same derivative).
    import jax.numpy as jnp

    import jno.jnp_ops as jnn

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.12).domain()

    def functional(s, nonlinear):
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        f = 10.0 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
        prob = jno.fvm([-((1.0 + ui * ui) * ui.x).x - ((1.0 + ui * ui) * ui.y).y - s * f, u(xb, yb) - 0.0])
        sol = jnp.asarray(prob.solve(nonlinear=nonlinear))
        return jnp.sum(sol**3)

    s0, h = 1.3, 1e-5
    g = float(jax.grad(functional)(s0, jno.solve.ptc()))
    fd = (float(functional(s0 + h, jno.solve.ptc())) - float(functional(s0 - h, jno.solve.ptc()))) / (2 * h)
    assert abs(g) > 1e-3
    assert abs(g - fd) <= 1e-6 * abs(fd)  # central differences: O(h^2) of a smooth functional
    g_newton = float(jax.grad(functional)(s0, None))
    assert abs(g - g_newton) <= 1e-8 * abs(g_newton)
