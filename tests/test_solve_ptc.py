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
