"""A scalar multiplying a weak term must not change the answer.

`split_weak_additive_terms` used to treat a product as one atomic sub-term. The transient routes
classify each sub-term by its temporal order, so `c * (u_t-term + spatial-term)` reported ONE order
for the whole product: the spatial part was stripped into the MASS matrix and the stiffness came out
empty. The march then returned a finite, plausible trajectory with no restoring force.

`-(a + b)` is `Literal(-1) * (a + b)`, so a negated group was the same trap.

The oracle is the bare spelling: scaling an equation by 1 cannot move its solution, and distributing
a scalar over a sum is exact, so every spelling below must agree to round-off.
"""

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


inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
RHO, MU, LX, LY, DT, N = 7000.0, 6e-3, 1.2e-3, 250e-6, 8.3333e-6, 20
TRACTION = 1733.0


def _solve(spelling):
    """Transient Stokes driven by a tangential surface traction; `spelling` picks how the momentum
    term is written. The traction term is never scaled, so a scaled EQUATION legitimately changes the
    solution by 1/scale -- which is why the scaled cases below are compared against that prediction."""
    d = jno.Shape.rect(0.0, 0.0, LX, LY, size=4e-5).domain(time=(0.0, DT * N, N + 1))
    d.tag("top", lambda x, y: y > LY - 1e-9)
    d.tag("wall", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > LX - 1e-9))
    d.point_region("ppin", (0.5 * LX, 0.0))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xt, yt, tt = d.variable("top", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    xpn, ypn = d.variable("ppin", split=True)[:2]
    ci = d.variable("initial", split=True)
    ax = [xi, yi]
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    gu, gv = grad(u, ax), grad(v, ax)
    pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)

    mass = RHO * inner(ui.t, vi, n_contract=1)  # temporal order 1
    stiff = MU * inner(gu, gv, n_contract=2) - pp * trace(gv)  # temporal order 0
    con = -qq * trace(gu)
    mom = {
        "bare": lambda: mass + stiff,
        "wrapped": lambda: 1.0 * (mass + stiff),  # the defect
        "distributed": lambda: 1.0 * mass + 1.0 * stiff,
        "negated": lambda: -(-mass - stiff),  # Literal(-1) * (a + b), twice
        "divided": lambda: (mass + stiff) / 1.0,
        "scaled2": lambda: 2.0 * (mass + stiff),
    }[spelling]()
    if spelling == "scaled2":
        con = 2.0 * con
    fem = jno.fem(
        [
            mom,
            con,
            -TRACTION * v.bind(x=xt, y=yt, t=tt)[0],
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
            u(xt, yt)[1] - 0.0,
            p(xpn, ypn) - 0.0,
            u(*ci)[0] - 0.0,
            u(*ci)[1] - 0.0,
        ]
    )
    traj = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")).fn())
    b = fem.blocks[fem.block_index(u)]
    U = traj[:, b.start : b.stop].reshape(len(traj), -1, 2)
    return np.linalg.norm(U, axis=-1).max(axis=1)


@pytest.mark.parametrize("spelling", ["wrapped", "distributed", "negated", "divided"])
def test_a_scalar_on_a_mixed_order_term_does_not_change_the_solution(spelling):
    """The whole point: `1.0 * (mass + stiff)` must equal `mass + stiff`."""
    ref, got = _solve("bare"), _solve(spelling)
    assert np.allclose(got, ref, rtol=1e-10, atol=0.0), (
        f"{spelling}: final |u| {got[-1]:.6e} against the bare spelling's {ref[-1]:.6e}"
    )


def test_the_flow_actually_develops_so_the_comparison_has_teeth():
    """Guard against the test passing because everything is ~0: the traction must drive a real flow
    that RISES and saturates. The defect showed up as a linear ramp, so pin the curvature too."""
    sp = _solve("bare")
    assert sp[-1] > 1e-2, f"traction produced no flow ({sp[-1]:.3e}) -- the oracle is vacuous"
    # Measured: the correct march reaches 0.558 of a linear ramp over 20 steps; with the stiffness
    # missing it grew EXACTLY linearly (ratio 1.000). 0.85 sits clear of both.
    linear_ramp = sp[1] * (len(sp) - 1)
    assert sp[-1] < 0.85 * linear_ramp, (
        f"velocity grew linearly with step count (ratio {sp[-1] / linear_ramp:.3f}) -- the viscous "
        "term is missing from the stiffness"
    )


def test_scaling_the_equation_scales_the_answer_the_way_the_math_says():
    """Scaling momentum AND continuity but not the load solves `M u_t + A u = f/2`, so the velocity
    must halve -- distribution must not silently drop the factor."""
    ref, got = _solve("bare"), _solve("scaled2")
    assert np.allclose(got, 0.5 * ref, rtol=1e-8, atol=0.0), (
        f"scaled: {got[-1]:.6e} against the expected {0.5 * ref[-1]:.6e}"
    )
