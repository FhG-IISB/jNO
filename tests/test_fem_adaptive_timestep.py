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
from jno.solve import adaptive  # noqa: E402
from jno.utils.solver.backend_blocks import _default_transient_integrate  # noqa: E402

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


def _semidiscrete(mesh_size, t_end, coeff=1.0, n_coarse=1001):
    """The ``dt -> 0`` solution on the SAME mesh: two fine fixed-dt solves, Richardson-extrapolated
    (backward Euler is first order, so ``2·u(dt/2) - u(dt)`` cancels the O(dt) term).

    Time-stepping accuracy must be measured against THIS, not against the analytic solution. On this
    problem the P1 spatial error and the backward-Euler time error have **opposite signs** — the discrete
    eigenvalue over-decays, backward Euler under-decays — so error-vs-analytic is non-monotonic in dt and
    partially cancels. A scheme with *more* time error can look closer to the analytic answer, which is an
    artefact, not accuracy."""
    ts = jnp.asarray([0.0, t_end])
    _, coarse = _heat(mesh_size=mesh_size, nsteps=n_coarse, coeff=coeff)
    _, fine = _heat(mesh_size=mesh_size, nsteps=2 * n_coarse - 1, coeff=coeff)
    return (
        2 * np.asarray(_default_transient_integrate(fine.operator, {}, ts))[-1]
        - np.asarray(_default_transient_integrate(coarse.operator, {}, ts))[-1]
    )


def test_adaptive_real_scalar_is_time_converged_and_beats_coarse_fixed():
    """The headline: on a coarse 6-point save grid the adaptive marcher takes finer *internal* steps, so
    its **time-discretization error** is far below fixed backward-Euler's on the same 6 points — and low
    enough that what remains is the mesh's own spatial error, i.e. it is time-converged."""
    d, fem = _heat(mesh_size=0.15, nsteps=6)
    save = jnp.linspace(0.0, 0.05, 6)
    ad = np.asarray(
        adaptive(rtol=1e-5, atol=1e-8, max_steps=2000).integrate(
            fem.operator, {}, save, linear_solve=None, nonlinear_solve=None
        )
    )
    be = np.asarray(_default_transient_integrate(fem.operator, {}, save))
    assert not np.any(np.isnan(ad))
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # initial condition preserved exactly

    u0, interior, decay = _analytic(d, save)
    ref = _semidiscrete(0.15, 0.05)
    t_err = lambda traj: np.linalg.norm(traj[-1][interior] - ref[interior]) / np.linalg.norm(ref[interior])
    assert t_err(ad) < 5e-3, f"adaptive time error: {t_err(ad):.2e}"
    assert t_err(ad) < 0.1 * t_err(be), f"adaptive {t_err(ad):.2e} vs fixed BE {t_err(be):.2e}"

    # Time-converged: the residual against the analytic solution is now the MESH error, not the scheme's.
    exact = (u0 * decay[-1])[interior]
    floor = np.linalg.norm(ref[interior] - exact) / np.linalg.norm(exact)
    off_analytic = np.linalg.norm(ad[-1][interior] - exact) / np.linalg.norm(exact)
    assert off_analytic < 1.2 * floor, f"off analytic {off_analytic:.2e} vs mesh floor {floor:.2e}"


def test_adaptive_starts_from_below_not_from_the_output_grid():
    """Regression: ``dt0`` must NOT default to the output grid's ``dt``.

    There is no step rejection (a discarded state would zero the per-step solve adjoint's cotangent and
    return a NaN gradient), so an over-large first step is committed permanently — only the *next* one
    shrinks, and by at most 5x per attempt. Seeding from the caller's output grid therefore bakes the
    first few attempts' error into the answer for good: on this problem attempt 1 lands ~740x over
    tolerance and attempt 2 ~34x. Growing from the floor instead cannot commit an out-of-tolerance step,
    because an under-sized step costs work, not accuracy.

    Measured against a Richardson-extrapolated semi-discrete reference (same mesh, dt -> 0, so this is
    PURE time error with no spatial contamination): 1.23e-2 seeded from the grid vs 2.94e-3 from below."""
    d, fem = _heat(mesh_size=0.15, nsteps=11, coeff=1.0)
    save = jnp.linspace(0.0, 0.05, 11)
    block = fem.operator
    kw = dict(rtol=1e-5, atol=1e-8, max_steps=800)

    ref = _semidiscrete(0.15, 0.05)
    time_err = lambda traj: np.linalg.norm(np.asarray(traj)[-1] - ref) / np.linalg.norm(ref)

    from_below = adaptive(**kw).integrate(block, {}, save, linear_solve=None, nonlinear_solve=None)
    from_grid = adaptive(**kw, dt0=float(block.dt)).integrate(block, {}, save, linear_solve=None, nonlinear_solve=None)
    e_below, e_grid = time_err(from_below), time_err(from_grid)
    assert not np.any(np.isnan(np.asarray(from_below)))
    assert e_below < 0.5 * e_grid, f"growing from below must beat grid-seeded: {e_below:.2e} vs {e_grid:.2e}"
    assert e_below < 5e-3, f"time error from below regressed: {e_below:.2e}"


def test_adaptive_second_order_base_is_far_cheaper_per_digit():
    """The base method's ORDER dominates everything the controller does.

    Step doubling spends 3 implicit solves per step. On a first-order base (θ=1, backward Euler) those
    solves buy error that only falls linearly in dt, so the scheme cannot beat a first-order fixed march
    by much — which is exactly what the work-precision numbers showed. Driving the SAME controller on a
    second-order base (θ=1/2) is dramatically cheaper per digit; the exponent follows via ``_theta_order``.

    Measured on this benchmark (x64, CPU): θ=1 -> 5.1e-3 in 54 steps / 162 solves; θ=1/2 -> 2.2e-4 in
    16 steps / 48 solves — ~23x the accuracy on ~3x less work. (Step count read off by bisecting
    ``max_steps``: the march NaNs when the budget is exhausted, and ``dt_min``/``dt_max`` depend only on
    the time span, so the step sequence is independent of the budget.) Asserted loosely so it pins the
    effect, not the exact arithmetic — the step count shifts by one or two with dtype and platform."""
    _, fem = _heat(mesh_size=0.15, nsteps=11)
    save = jnp.linspace(0.0, 0.05, 11)
    ref = _semidiscrete(0.15, 0.05)
    err = lambda traj: np.linalg.norm(np.asarray(traj)[-1] - ref) / np.linalg.norm(ref)

    kw = dict(rtol=1e-4, atol=1e-6, max_steps=4000)
    run = lambda sch: sch.integrate(fem.operator, {}, save, linear_solve=None, nonlinear_solve=None)
    first = run(adaptive(**kw))  # bare: the block's own θ=1 step
    second = run(jno.solve.theta(0.5).adaptive(**kw))  # same controller on a 2nd-order step
    e1, e2 = err(first), err(second)
    assert not np.any(np.isnan(np.asarray(second)))
    assert e2 < 0.2 * e1, f"second-order base should be much more accurate: {e2:.2e} vs {e1:.2e}"


def test_theta_order_drives_the_controller_exponent():
    """θ=1/2 is the ONLY second-order θ; everything else is first order. The controller reads this off the
    BASE scheme to pick its step-size exponent 1/(p+1), so a wrong answer here mis-sizes every step (it was
    previously hardwired to ½, which also mis-sized the trapezoidal second-order-system blocks)."""
    from jno.utils.solver.timeschemes import _theta_order

    assert _theta_order(0.5) == 2
    assert _theta_order(1.0) == 1 and _theta_order(0.0) == 1 and _theta_order(0.7) == 1
    assert jno.solve.theta(0.5).step_order == 2 and jno.solve.theta(1.0).step_order == 1


def test_adaptive_composes_onto_a_base_scheme():
    """Adaptivity is a step-size POLICY attached to a base step, not a scheme of its own: ``which step``
    and ``how big`` are separate axes. The bare ``jno.solve.adaptive()`` keeps meaning "the block's own
    θ-step", so it stays a drop-in."""
    assert adaptive().base is None  # bare form defers to the block
    wrapped = jno.solve.theta(0.5).adaptive(rtol=1e-5)
    assert wrapped.base.theta == 0.5 and wrapped.rtol == 1e-5
    assert wrapped.base.step_order == 2  # the controller exponent follows the base, not a hardwired 1/2

    # base=None resolves to the block's own θ at integrate time
    _, fem = _heat(mesh_size=0.3, nsteps=5)
    assert adaptive().base_for(fem.operator).theta == 1.0

    with pytest.raises(NotImplementedError, match="does not nest"):
        jno.solve.theta(0.5).adaptive().adaptive()
    with pytest.raises(NotImplementedError, match="exact in time|exponential"):
        jno.solve.exponential().adaptive()


def test_adaptive_rejection_would_break_the_gradient():
    """Documents WHY every attempt is accepted, so nobody "fixes" it back.

    The no-rejection design is load-bearing for AD, not for stability: rejecting means discarding an
    attempt's state, whose cotangent is then exactly zero, so the matrix-free per-step solve adjoint runs
    on ``b = 0`` and its relative-residual test divides 0/0. This test pins the *property that makes
    rejection unavailable* — that the accepted state is the one the gradient flows through — by checking
    the gradient stays finite and matches a finite difference at a tolerance loose enough to make the
    controller work hard (many attempts, wide dt range)."""
    d, fem = _heat(mesh_size=0.2, nsteps=6, param=True)
    (name,) = fem.operator.runtime_parameter_exprs
    save = jnp.linspace(0.0, 0.05, 6)
    sch = adaptive(rtol=1e-6, atol=1e-9, max_steps=1200)

    def loss(kv):
        return jnp.sum(
            sch.integrate(fem.operator, {name: jnp.asarray([kv])}, save, linear_solve=None, nonlinear_solve=None) ** 2
        )

    g = float(jax.grad(loss)(0.8))
    fd = float((loss(0.8 + 1e-5) - loss(0.8 - 1e-5)) / 2e-5)
    assert np.isfinite(g), "gradient must stay finite — a discarded state would make it NaN"
    assert abs(g - fd) <= 1e-2 * max(abs(fd), 1.0), f"AD {g} vs FD {fd}"


def test_adaptive_real_scalar_is_differentiable():
    """Reverse-mode differentiable in a runtime parameter: the AD gradient matches a finite difference."""
    d, fem = _heat(mesh_size=0.2, nsteps=6, param=True)
    (name,) = fem.operator.runtime_parameter_exprs
    save = jnp.linspace(0.0, 0.05, 6)
    sch = adaptive(rtol=1e-4, atol=1e-6, max_steps=1000)

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
        adaptive(rtol=1e-4, atol=1e-6, max_steps=1000).integrate(
            fem.operator, {}, save, linear_solve=None, nonlinear_solve=None
        )
    )
    assert not np.any(np.isnan(traj))
    assert np.linalg.norm(traj[-1]) < np.linalg.norm(traj[0])  # diffusion decays both components


def test_adaptive_real_periodic():
    """Periodic ties pre-reduce the block; adaptive marches the reduced main-DOF space (``block.step``
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
        adaptive(rtol=1e-4, atol=1e-6, max_steps=1000).integrate(
            fem.operator, {}, save, linear_solve=None, nonlinear_solve=None
        )
    )
    be = np.asarray(_default_transient_integrate(fem.operator, {}, save))
    assert not np.any(np.isnan(ad))
    assert ad.shape == be.shape  # both march the same reduced (periodic) DOF layout
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # IC preserved


def _complex_heat(mesh_size=0.25, param=False, periodic=False, nsteps=6):
    """Complex diffusion u_t + coeff·(1+0.5j)·Δu -> complex_transient."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, 0.05, nsteps))
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
    assert fem.is_complex and fem.is_transient
    save = jnp.linspace(0.0, 0.05, 6)
    sch = adaptive(rtol=1e-4, atol=1e-6, max_steps=1000)
    ad = np.asarray(fem.solve(save_ts=save, time=sch))
    be = np.asarray(fem.solve(save_ts=save))
    assert ad.dtype == np.complex128 and not np.any(np.isnan(ad))
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # IC preserved

    _, femp = _complex_heat(param=True)
    fc = femp.solve(save_ts=save, time=sch)  # parametric -> FunctionCall

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
    assert fem.is_complex and fem.is_transient
    assert fem._periodic is not None, "the u(left)-u(right) tie must reduce the complex transient block"
    save = jnp.linspace(0.0, 0.05, 6)
    ad = np.asarray(fem.solve(save_ts=save, time=adaptive(rtol=1e-4, max_steps=1000)))
    be = np.asarray(fem.solve(save_ts=save))
    assert ad.dtype == np.complex128 and not np.any(np.isnan(ad))
    assert ad.shape == be.shape
    assert np.max(np.abs(ad[0] - be[0])) < 1e-12  # IC preserved


def test_adaptive_fail_loud_on_exhausted_budget():
    """Never silently under-resolve: a tolerance far too tight for the step budget poisons the result with
    NaN (raise ``max_steps``), rather than returning a truncated march."""
    d, fem = _heat(mesh_size=0.2, nsteps=6)
    save = jnp.linspace(0.0, 0.05, 6)
    out = np.asarray(
        adaptive(rtol=1e-9, atol=1e-12, max_steps=20).integrate(
            fem.operator, {}, save, linear_solve=None, nonlinear_solve=None
        )
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


def test_theta_on_complex_transient():
    """jno.solve.theta threads into the complex-transient marcher: θ=1 reproduces the default backward
    Euler exactly, and θ=1/2 (Crank–Nicolson) is 2nd-order — much closer to a fine-dt reference on the same
    mesh than backward Euler at the same coarse step — and reverse-mode differentiable."""
    save = jnp.linspace(0.0, 0.05, 6)
    _, fem = _complex_heat(mesh_size=0.3, nsteps=6)
    default = np.asarray(fem.solve(save_ts=save))
    th1 = np.asarray(fem.solve(save_ts=save, time=jno.solve.theta(1.0)))
    assert np.max(np.abs(th1 - default)) < 1e-12  # θ=1 is exactly the default backward Euler

    _, fine = _complex_heat(mesh_size=0.3, nsteps=401)  # fine-dt reference on the same mesh
    ref = np.asarray(fine.solve(save_ts=save))
    cn = np.asarray(fem.solve(save_ts=save, time=jno.solve.theta(0.5)))
    err_be = np.linalg.norm(default[-1] - ref[-1]) / np.linalg.norm(ref[-1])
    err_cn = np.linalg.norm(cn[-1] - ref[-1]) / np.linalg.norm(ref[-1])
    assert not np.any(np.isnan(cn))
    assert err_cn < 0.2 * err_be, f"Crank–Nicolson (2nd-order) should beat backward Euler: CN={err_cn:.2e} BE={err_be:.2e}"

    _, femp = _complex_heat(mesh_size=0.3, param=True)
    fc = femp.solve(save_ts=save, time=jno.solve.theta(0.5))

    def loss(kv):
        return jnp.sum(jnp.abs(fc.fn(jnp.asarray([kv]))) ** 2)

    g = float(jax.grad(loss)(0.8))
    fd = float((loss(0.8 + 1e-5) - loss(0.8 - 1e-5)) / 2e-5)
    assert np.isfinite(g) and abs(g - fd) <= 1e-2 * max(abs(fd), 1.0), f"CN AD {g} vs FD {fd}"


def test_complex_transient_composes_solver_slots():
    """The complex transient is assembled as ONE real 2n block, so it is an ordinary transient block and
    the per-step solver slots apply to it like any other — ``linear=``, ``precond=``, and ``time=``
    together. Each of these raised ``NotImplementedError`` while the Re/Im legs were fused only at solve
    time by a second, bespoke marcher that none of the slots had been threaded into.

    The slots change *how* the step is solved, not *what* it solves, so every choice must land on the
    same trajectory as the default to solver tolerance."""
    _, fem = _complex_heat(mesh_size=0.3)
    base = np.asarray(fem.solve())
    assert base.dtype == np.complex128

    for label, kwargs in [
        ("linear=lu", dict(linear=jno.solve.lu())),
        ("linear=gmres", dict(linear=jno.solve.gmres())),
        ("linear=gmres+precond=jacobi", dict(linear=jno.solve.gmres(), precond=jno.precond.jacobi())),
        ("time=theta(1)+linear=lu", dict(time=jno.solve.theta(1.0), linear=jno.solve.lu())),
    ]:
        _, femk = _complex_heat(mesh_size=0.3)
        out = np.asarray(femk.solve(**kwargs))
        assert out.dtype == np.complex128 and not np.any(np.isnan(out)), f"{label} produced no complex result"
        rel = float(np.linalg.norm(out[-1] - base[-1]) / np.linalg.norm(base[-1]))
        assert rel < 1e-8, f"{label} disagrees with the default complex-transient solve: rel {rel:.3e}"


def test_complex_transient_exponential_scheme_routes():
    """``jno.solve.exponential`` on a complex transient was refused outright ("wired for adaptive and θ
    only") because the bespoke complex marcher implemented just those two. The fused block routes into the
    ordinary scheme dispatch, so the exponential integrator now reaches it like any real transient."""
    pytest.importorskip("matfree", reason="jno.solve.exponential needs the optional matfree package")
    _, fem = _complex_heat(mesh_size=0.3)
    out = np.asarray(fem.solve(time=jno.solve.exponential()))
    assert out.dtype == np.complex128 and not np.any(np.isnan(out))


def test_complex_transient_adapt_composes():
    """The scope limit this test used to pin is GONE: the adaptive transient driver now carries the
    fused ``[Re; Im]`` halves separately (a doubled field layout drives the transfer, the modulus
    drives the metric), so a complex transient composes with ``adapt=`` instead of failing loud.
    Smoke here — the analytic-recovery and zero-IC extremes live in test_fem_adapt_complex.py."""
    # remeshing goes through mmgpy, an OPTIONAL dependency the adapt-dedicated files skip at module
    # level; this file is not one of them, so guard per-test (as the matfree case above does).
    pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing")
    _, fem = _complex_heat(mesh_size=0.3)
    traj = fem.solve(adapt=jno.solve.remesh(every=2, max_dofs=80))
    final, _mesh = traj.final()
    final = np.asarray(final)
    assert np.iscomplexobj(final), "adaptive frames of a complex transient must be complex"
    assert not np.isnan(final).any()


def test_exhausted_budget_poison_reaches_the_adjoint():
    """A starved ``max_steps`` must poison the **gradient**, not only the value.

    ``adaptive_march`` NaN-poisons its trajectory when it cannot reach ``t1``.  Applied as
    ``jnp.where(reached, out, nan)`` that poisons the value alone: the VJP of ``where`` w.r.t. the taken
    branch is ``where(c, g, 0)``, so the gradient of a failed march came back as exactly **zero** — and an
    inverse problem reads the gradient, so a starved budget looked like a converged optimisation.  The
    poison is multiplicative, so the NaN reaches the cotangent.

    Driven directly with a scalar step so the property is pinned at the marcher, independent of what any
    particular block's adjoint does downstream.
    """
    from jno.utils.solver.timeschemes import adaptive_march

    def step_fn(u, t, dt):
        return u * (1.0 - dt * 3.0)

    def march(p, max_steps):
        out = adaptive_march(
            lambda u, t, dt: step_fn(u * p, t, dt),
            jnp.ones(1),
            0.0,
            1.0,
            jnp.linspace(0.0, 1.0, 4),
            rtol=1e-3,
            atol=1e-6,
            max_steps=max_steps,
        )
        return jnp.sum(out**2)

    starved = float(jax.grad(lambda p: march(p, 2))(1.0))
    assert np.isnan(float(march(1.0, 2))), "a march that cannot reach t1 must return a NaN trajectory"
    assert np.isnan(starved), f"the poison must reach the adjoint, got {starved}"

    fed = float(jax.grad(lambda p: march(p, 200))(1.0))
    assert np.isfinite(float(march(1.0, 200))), "a march that reaches t1 must not be poisoned"
    assert np.isfinite(fed), "and its gradient must stay finite — the poison is a no-op when reached"
