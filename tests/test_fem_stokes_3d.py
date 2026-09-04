"""Taylor-Hood (P2 velocity / P1 pressure) Stokes on tetrahedra -- the 3-D mixed saddle system.

jNO assembled this correctly long before anything exercised it: no test, example or doc row in the
repository combined a 3-component velocity with a pressure field. These tests pin the three things
that were unverified.

Exact fields, both chosen to lie *in* the discrete space so a correct assembler recovers them to
machine precision (any error here is assembly or solver, never discretisation):

    u = (y^2 + z^2, z^2 + x^2, x^2 + y^2)     div u == 0 identically,  Delta u = (4, 4, 4)
    p = x + y + z - 3/2                        int p dx == 0 over the unit cube
    f = -mu Delta u + grad p = (1 - 4 mu) (1, 1, 1)

``p`` integrates to zero over the unit cube by construction, so the zero-mean gauge and the exact
solution are in the *same* gauge and can be compared directly.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
MU = 1.0


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly is float64; opt in per-test and restore (the session default is x64-off)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stokes_3d(size=0.35, mean_gauge=True):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    gu, gv = grad(u, [xi, yi, zi]), grad(v, [xi, yi, zi])
    pp, qq = p.bind(x=xi, y=yi, z=zi), q.bind(x=xi, y=yi, z=zi)
    vv = v.bind(x=xi, y=yi, z=zi)
    f = 1.0 - 4.0 * MU  # the same constant in every component
    momentum = MU * inner(gu, gv, n_contract=2) - pp * trace(gv) - f * (vv[0] + vv[1] + vv[2])
    gauge = p.pin(mean=True) if mean_gauge else p.pin(-1.5)  # p(0,0,0) = -3/2
    fem = jno.fem(
        [
            momentum,
            -qq * trace(gu),
            u(xb, yb, zb)[0] - (yb**2 + zb**2),
            u(xb, yb, zb)[1] - (zb**2 + xb**2),
            u(xb, yb, zb)[2] - (xb**2 + yb**2),
            gauge,
        ]
    )
    return d, fem, u, p, pp, qq


def _exact(fem):
    pv, pp_ = np.asarray(fem.field_points[0]), np.asarray(fem.field_points[1])
    x, y, z = pv[:, 0], pv[:, 1], pv[:, 2]
    vel = np.stack([y**2 + z**2, z**2 + x**2, x**2 + y**2], axis=1)
    pre = pp_[:, 0] + pp_[:, 1] + pp_[:, 2] - 1.5
    return vel, pre


def _split(fem, sol):
    off = fem.offsets
    return np.asarray(sol)[off[0] : off[1]].reshape(-1, 3), np.asarray(sol)[off[1] :]


def test_taylor_hood_tets_recover_a_quadratic_stokes_field():
    """The assembly test. P2 represents the quadratic velocity and P1 the linear pressure exactly,
    so a correct 3-D mixed assembler reproduces both to machine precision."""
    _, fem, *_ = _stokes_3d()
    assert fem._mode == "linear" and len(fem.offsets) == 3, "expected a two-block P2/P1 saddle system"
    sol = fem.solve(linear=jno.solve.lu(backend="host"))
    vel, pre = _split(fem, sol)
    ev, ep = _exact(fem)
    assert np.abs(vel - ev).max() < 1e-10, f"velocity off by {np.abs(vel - ev).max():.2e}"
    assert np.abs(pre - ep).max() < 1e-8, f"pressure off by {np.abs(pre - ep).max():.2e}"


def test_the_recovered_velocity_is_divergence_free():
    """``u`` is div-free identically and lies in the space, so the discrete field must be too --
    measured through the continuity block's own residual rather than a restatement of the solve."""
    _, fem, u, p, pp, qq = _stokes_3d()
    sol = fem.solve(linear=jno.solve.lu(backend="host"))
    d = fem.domain
    xi, yi, zi = d.variable("interior", split=True)[:3]
    gu = grad(u, [xi, yi, zi])
    r = np.asarray(fem.eval(-qq * trace(gu), sol))
    off = fem.offsets
    assert np.abs(r[off[1] :]).max() < 1e-9, f"weak divergence residual {np.abs(r[off[1] :]).max():.2e}"


def test_the_fieldsplit_recipe_solves_the_3d_saddle():
    """The path that scales past the direct solver's fill-in ceiling: AMG on the velocity block, a
    pressure-mass Schur approximation. Verified in 2-D already; this pins it in 3-D.

    Oracle is the direct solve, not a tolerance pulled from the air.
    """
    pytest.importorskip("pyamg", reason="pyamg required for the AMG velocity block")
    _, fem, u, p, pp, qq = _stokes_3d()
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    got = np.asarray(
        fem.solve(
            linear=jno.solve.fgmres(tol=1e-10, maxiter=2000),
            precond=jno.precond.triangular(
                (u, jno.precond.amg()),
                (p, jno.precond.form([(1.0 / MU) * pp * qq], inner=jno.solve.lu(backend="host"))),
            ),
        )
    )
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    # Measured 6.7e-6 at these settings. The gate is 5e-5 -- roughly 7x headroom over the measured
    # value, and well inside the float32 Krylov floor this path carries (the 2-D sibling in
    # test_fem_fieldsplit.py measures 6.3e-4 and gates at 5e-3). A real breakage moves the first
    # digit, not the fifth.
    assert err < 5e-5, f"fieldsplit drifted from the direct oracle: rel err {err:.2e}"


def test_the_zero_mean_gauge_is_what_makes_the_3d_pressure_right():
    """A point pin fixes one vertex's discrete value to a continuous one; the constant it leaves does
    not shrink with the mesh. Here the exact pressure is zero-mean, so the gauge is directly checkable:
    ``mean=True`` lands on it, the point pin is offset by a constant, and the velocity is the same field."""
    _, fem_m, *_ = _stokes_3d(mean_gauge=True)
    _, fem_p, *_ = _stokes_3d(mean_gauge=False)
    s_m = fem_m.solve(linear=jno.solve.lu(backend="host"))
    s_p = fem_p.solve(linear=jno.solve.lu(backend="host"))
    vel_m, pre_m = _split(fem_m, s_m)
    vel_p, pre_p = _split(fem_p, s_p)
    ev, ep = _exact(fem_m)

    np.testing.assert_allclose(vel_m, vel_p, rtol=0, atol=1e-10)  # the gauge cannot move the velocity
    assert np.abs(pre_m - ep).max() < 1e-8
    delta = pre_p - pre_m
    assert np.ptp(delta) < 1e-8, "the two gauges must differ by ONE constant, not by a field"
