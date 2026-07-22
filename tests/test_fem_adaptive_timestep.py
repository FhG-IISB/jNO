"""Adaptive step-size transient integration — ``fem.solve(time=jno.solve.adaptive(...))``.

Step-doubling (Richardson) error control on the block's own implicit step, so it works uniformly across
**all** transient cases: real / complex, scalar / vector, plain / periodic. The step size adapts to the
local error instead of the fixed ``dt`` from ``domain(time=(t0,t1,n))``; it is reverse-mode differentiable
(a fixed-length ``lax.scan`` with the controller ``stop_gradient``-ed — the state differentiates at the
realized step schedule) and fails loud (NaN) if the step budget is exhausted before ``t1``.

Run with x64 (the FEM assembly is float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno._fem import _solve_complex_transient  # noqa: E402
from jno.solve import adaptive  # noqa: E402
from jno.utils.solver.backend_blocks import _default_transient_integrate  # noqa: E402
from jno.utils.solver.timeschemes import _AdaptiveScheme  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(mesh_size=0.2, nsteps=6, coeff=1.0, param=False):
    """u_t = coeff·Δu on the unit square, u0 = sin(πx)sin(πy), zero Dirichlet (analytic mode-(1,1) decay
    exp(-2π²·coeff·t)). ``param`` makes ``coeff`` a runtime parameter ``k`` for the gradient tests."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, 0.05, nsteps))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    k = jno.np.parameter((1,), name="k") if param else coeff
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return d, jno.fem([ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])


def _analytic(d, save, coeff=1.0):
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    u0 = np.sin(PI * nodes[:, 0]) * np.sin(PI * nodes[:, 1])
    interior = ~(
        np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], 1) | np.isclose(nodes[:, 1], 0) | np.isclose(nodes[:, 1], 1)
    )
    return u0, interior, np.exp(-2 * PI**2 * coeff * np.asarray(save))


def test_adaptive_real_scalar_matches_analytic_and_beats_coarse_fixed():
    """The headline: on a coarse 6-point save grid the adaptive marcher takes finer *internal* steps, so
    it tracks the analytic heat decay far better than fixed backward-Euler on the same 6 points."""
    d, fem = _heat(mesh_size=0.15, nsteps=6)
    save = jnp.linspace(0.0, 0.05, 6)
    ad = np.asarray(
        _AdaptiveScheme(1e-5, 1e-8, 2000).integrate(fem.operator, {}, save, linear_solve=None, nonlinear_solve=None)
    )
    be = np.asarray(_default_transient_integrate(fem.operator, {}, save))
    assert not np.any(np.isnan(ad))
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # initial condition preserved exactly
    u0, interior, decay = _analytic(d, save)

    def rel(traj, k):
        e = (u0 * decay[k])[interior]
        return np.linalg.norm(traj[k][interior] - e) / np.linalg.norm(e)

    assert rel(ad, -1) < 2e-2, f"adaptive off analytic: {rel(ad, -1):.2e}"
    assert rel(ad, -1) < rel(be, -1), "adaptive must beat coarse fixed backward-Euler on the same save grid"


def test_adaptive_real_scalar_is_differentiable():
    """Reverse-mode differentiable in a runtime parameter: the AD gradient matches a finite difference."""
    d, fem = _heat(mesh_size=0.2, nsteps=6, param=True)
    (name,) = fem.operator.runtime_parameter_exprs
    save = jnp.linspace(0.0, 0.05, 6)
    sch = _AdaptiveScheme(1e-4, 1e-6, 1000)

    def loss(kv):
        return jnp.sum(
            sch.integrate(fem.operator, {name: jnp.asarray([kv])}, save, linear_solve=None, nonlinear_solve=None) ** 2
        )

    g = float(jax.grad(loss)(0.8))
    fd = float((loss(0.8 + 1e-5) - loss(0.8 - 1e-5)) / 2e-5)
    assert np.isfinite(g)
    assert abs(g - fd) <= 1e-2 * max(abs(fd), 1.0), f"AD {g} vs FD {fd}"


def test_adaptive_real_vector():
    """A vector field is a flat DOF block to the integrator — adaptive marches it unchanged."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.vector(jno.np.sin(PI * ci[0]), jno.np.sin(PI * ci[1]))
    fem = jno.fem(
        [ui.t.dot(vi) + (ui.x.dot(vi.x) + ui.y.dot(vi.y)), u(xb, yb) - jno.np.vector(0.0, 0.0), u(ci[0], ci[1]) - u0]
    )
    save = jnp.linspace(0.0, 0.05, 6)
    traj = np.asarray(
        _AdaptiveScheme(1e-4, 1e-6, 1000).integrate(fem.operator, {}, save, linear_solve=None, nonlinear_solve=None)
    )
    assert not np.any(np.isnan(traj))
    assert np.linalg.norm(traj[-1]) < np.linalg.norm(traj[0])  # diffusion decays both components


def test_adaptive_real_periodic():
    """Periodic ties pre-reduce the block; adaptive marches the reduced master-DOF space (``block.step``
    runs there), transparently — same reduced trajectory shape as the default stepper, no NaN."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(2 * PI * ci[0]) * jno.np.cos(PI * ci[1])
    fem = jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xl, yl) - u(xr, yr), u(ci[0], ci[1]) - u0])
    save = jnp.linspace(0.0, 0.05, 6)
    ad = np.asarray(
        _AdaptiveScheme(1e-4, 1e-6, 1000).integrate(fem.operator, {}, save, linear_solve=None, nonlinear_solve=None)
    )
    be = np.asarray(_default_transient_integrate(fem.operator, {}, save))
    assert not np.any(np.isnan(ad))
    assert ad.shape == be.shape  # both march the same reduced (periodic) DOF layout
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # IC preserved


def _complex_heat(mesh_size=0.25, param=False, periodic=False):
    """Complex diffusion u_t + coeff·(1+0.5j)·Δu -> complex_transient."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    k = jno.np.parameter((1,), name="k") if param else 1.0
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    diff = k * (1.0 + 0.5j) * (ui.x * vi.x + ui.y * vi.y)
    if periodic:
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        cons = [ui.t * vi + diff, u(xl, yl) - u(xr, yr), u(ci[0], ci[1]) - u0]
    else:
        xb, yb, _ = d.variable("boundary", split=True)
        cons = [ui.t * vi + diff, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0]
    return d, jno.fem(cons)


def test_adaptive_complex_scalar_forward_and_grad():
    """Complex-transient: the same marcher runs on the real 2n block. Forward is complex, no NaN, IC
    preserved, and the parametric inverse gradient matches a finite difference."""
    d, fem = _complex_heat()
    assert fem._mode == "complex_transient"
    save = jnp.linspace(0.0, 0.05, 6)
    sch = adaptive(rtol=1e-4, atol=1e-6, max_steps=1000)
    ad = np.asarray(_solve_complex_transient(fem.operator, save_ts=save, periodic=None, time=sch))
    be = np.asarray(_solve_complex_transient(fem.operator, save_ts=save, periodic=None))
    assert ad.dtype == np.complex128 and not np.any(np.isnan(ad))
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # IC preserved

    _, femp = _complex_heat(param=True)
    fc = _solve_complex_transient(femp.operator, save_ts=save, periodic=None, time=sch)  # parametric -> FunctionCall

    def loss(kv):
        return jnp.sum(jnp.abs(fc.fn(jnp.asarray([kv]))) ** 2)

    g = float(jax.grad(loss)(0.8))
    fd = float((loss(0.8 + 1e-5) - loss(0.8 - 1e-5)) / 2e-5)
    assert np.isfinite(g)
    assert abs(g - fd) <= 1e-2 * max(abs(fd), 1.0), f"AD {g} vs FD {fd}"


def test_adaptive_complex_periodic():
    """The full compose — complex + periodic (the metasurface-in-time case): one adaptive marcher over the
    reduced 2n block, recombined to complex and prolonged. No NaN, complex, IC preserved."""
    d, fem = _complex_heat(periodic=True)
    assert fem._mode == "complex_transient"
    save = jnp.linspace(0.0, 0.05, 6)
    ad = np.asarray(
        _solve_complex_transient(
            fem.operator, save_ts=save, periodic=fem._periodic, time=adaptive(rtol=1e-4, max_steps=1000)
        )
    )
    be = np.asarray(_solve_complex_transient(fem.operator, save_ts=save, periodic=fem._periodic))
    assert ad.dtype == np.complex128 and not np.any(np.isnan(ad))
    assert ad.shape == be.shape
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # IC preserved


def test_adaptive_fail_loud_on_exhausted_budget():
    """Never silently under-resolve: a tolerance far too tight for the step budget poisons the result with
    NaN (raise ``max_steps``), rather than returning a truncated march."""
    d, fem = _heat(mesh_size=0.2, nsteps=6)
    save = jnp.linspace(0.0, 0.05, 6)
    out = np.asarray(
        _AdaptiveScheme(1e-9, 1e-12, 20).integrate(fem.operator, {}, save, linear_solve=None, nonlinear_solve=None)
    )
    assert np.all(np.isnan(out)), "an exhausted step budget must fail loud (NaN), not silently truncate"


def test_adaptive_threads_through_fem_solve():
    """The user-facing path: ``fem.solve(time=jno.solve.adaptive(...))`` runs end to end for a real and a
    complex transient (the dispatch threads the scheme into both the real stepper and the 2n-block marcher)."""

    def _eval(x):  # a non-parametric transient solve is a lazy FunctionCall (real) or a concrete array (complex)
        return np.asarray(x.fn() if hasattr(x, "fn") else jnp.asarray(x))

    _, fem = _heat(mesh_size=0.3, nsteps=5)
    real = _eval(fem.solve(time=adaptive(rtol=1e-3, max_steps=500)))
    assert not np.any(np.isnan(real))
    _, femc = _complex_heat(mesh_size=0.3)
    cx = _eval(femc.solve(time=adaptive(rtol=1e-3, max_steps=500)))
    assert cx.dtype == np.complex128 and not np.any(np.isnan(cx))


def test_adaptive_complex_transient_slot_guards():
    """Scope limits fail loud on the complex-transient time path: only ``jno.solve.adaptive`` is wired
    (θ/exponential raise), and the other solver slots are not threaded there yet."""
    _, fem = _complex_heat(mesh_size=0.3)
    with pytest.raises(NotImplementedError, match="adaptive"):
        fem.solve(time=jno.solve.theta(0.5))
    with pytest.raises(NotImplementedError, match="time=|slots"):
        fem.solve(time=adaptive(), linear=jno.solve.lu())
