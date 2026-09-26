"""jno.solve.sdirk(): Alexander's stiffly-accurate, L-stable SDIRK methods (SIAM J. Numer. Anal. 14(6), 1977).

Oracles: the ORDER (error against a fine same-scheme reference on the same mesh, so the spatial error cancels),
L-stability (a stiff, boundary-incompatible start that Crank-Nicolson rings on), exactness where the discrete
solution is exact, and agreement with the default march.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import box

import jno

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nsteps, T=0.1, h=0.2, nonlinear=False, ic_one=False):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h, time=(0.0, T, nsteps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = 1.0 if ic_one else jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    form = ui.t * vi + ui.x * vi.x + ui.y * vi.y
    if nonlinear:
        form = form + ui**3 * vi
    return jno.fem([form, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def _final(fem, **kw):
    return np.asarray(fem.solve(**kw).fn())[-1]


@pytest.mark.parametrize("order", [2, 3])
def test_sdirk_has_its_order(order):
    ref = _final(_heat(320), time=jno.solve.sdirk(order))
    errs = [float(np.linalg.norm(_final(_heat(n), time=jno.solve.sdirk(order)) - ref)) for n in (5, 10, 20)]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert min(rates) > order - 0.3, f"sdirk({order}) rates {rates} from errors {errs}"


@pytest.mark.parametrize("order", [2, 3])
def test_sdirk_is_l_stable(order):
    """A stiff, boundary-incompatible start: Crank-Nicolson inverts it at full amplitude, SDIRK damps it."""
    cn = np.asarray(_heat(8, T=0.5, h=0.14, ic_one=True).solve(time=jno.solve.theta(0.5)).fn())
    sd = np.asarray(_heat(8, T=0.5, h=0.14, ic_one=True).solve(time=jno.solve.sdirk(order)).fn())
    assert cn.min() < -0.5
    # L-stable is damped, not monotone: SDIRK2 dips to -0.086 and SDIRK3 less, against CN's -1.0
    assert abs(sd.min()) < 0.1 * abs(cn.min()) and np.abs(sd[-1]).max() < 0.2 * np.abs(cn[-1]).max()


def test_nonlinear_block_and_solver_slots_agree_with_a_fine_reference():
    ref = _final(_heat(320, nonlinear=True), time=jno.solve.sdirk(3))
    got = _final(_heat(20, nonlinear=True), time=jno.solve.sdirk(3))
    slot = _final(_heat(20, nonlinear=True), time=jno.solve.sdirk(3), linear=jno.solve.gmres(tol=1e-12))
    assert np.linalg.norm(got - ref) < 1e-4 * np.linalg.norm(ref)
    np.testing.assert_allclose(slot, got, rtol=1e-7, atol=1e-10)


def test_linear_slot_and_adaptive():
    fem = _heat(10)
    a = _final(fem, time=jno.solve.sdirk(3))
    b = _final(fem, time=jno.solve.sdirk(3), linear=jno.solve.cg(tol=1e-12), precond=jno.precond.jacobi())
    np.testing.assert_allclose(b, a, rtol=1e-7, atol=1e-10)
    ref = _final(_heat(320), time=jno.solve.sdirk(3))
    ad = _final(_heat(10), time=jno.solve.sdirk(3).adaptive(rtol=1e-7, atol=1e-9))
    assert np.linalg.norm(ad - ref) < 1e-5 * np.linalg.norm(ref)


def test_time_varying_dirichlet_is_exact():
    """u = x + t is exact in P1 and in any consistent time scheme: the stiffly accurate last stage lands on
    g(t_{n+1}) exactly."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.2, 5))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - (xb + tb), u(ci[0], ci[1]) - ci[0]])
    traj = np.asarray(fem.solve(time=jno.solve.sdirk(3)).fn())
    pts = np.asarray(d.mesh.points)[:, 0]
    t = np.linspace(0.0, 0.2, traj.shape[0])
    np.testing.assert_allclose(traj, pts[None, :] + t[:, None], atol=1e-9)


def test_gradient_through_the_march_matches_finite_differences():
    fem = _heat(6, T=0.05)
    blk = fem.operator
    save = jnp.linspace(blk.t0, blk.t1, 3)
    sch = jno.solve.sdirk(3)

    def loss(s):
        import dataclasses

        b = dataclasses.replace(blk, state0=s * blk.state0)
        return jnp.sum(sch.integrate(b, {}, save, linear_solve=None, nonlinear_solve=None) ** 2)

    g = float(jax.grad(loss)(1.3))
    fd = float((loss(1.3 + 1e-6) - loss(1.3 - 1e-6)) / 2e-6)
    np.testing.assert_allclose(g, fd, rtol=1e-6)


def test_refusals():
    with pytest.raises(ValueError, match="order"):
        jno.solve.sdirk(4)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 0.2, 5))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, t0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    wave = jno.fem(
        [ui.tt * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(x0, y0) - 0.0, u.bind(x=x0, y=y0, t=t0).t - 0.0]
    )
    with pytest.raises(NotImplementedError, match="undamped"):
        wave.solve(time=jno.solve.sdirk(3)).fn()
