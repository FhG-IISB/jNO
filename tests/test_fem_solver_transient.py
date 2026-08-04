"""Solver slots threaded into the transient stepper: ``fem.solve(linear=, precond=, nonlinear=)``
now configures the per-step solves of the default theta-method integrator.

Pins: on a linear transient block the slot solvers reproduce the default backward-Euler
trajectory (Krylov+jacobi, sparse-direct, flexible-Krylov with an iterative inner, and AMG —
whose hierarchy is built ONCE on the step operator ``M + θ·dt·A`` and reused by every step);
a nonlinear transient block takes the composed Newton driver; the **second-order-in-time**
(augmented ``[u, v]``) block flows through the same machinery unchanged; a **parametric**
transient inverse stays differentiable through slot solvers (per-step materialization path,
``operator_fn(t, args)``); and the guard rails (``x0`` on transient; ``nonlinear=`` on a linear
block).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nonlinear=False, mesh_size=0.2, time=(0.0, 0.2, 21)):
    """Transient heat (optionally + u^3 reaction), homogeneous Dirichlet, sine-bump IC."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=time)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    terms = [ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic]
    if nonlinear:
        terms[0] = ui.t * vi + ui.x * vi.x + ui.y * vi.y + ui**3 * vi
    return jno.fem(terms)


def test_linear_transient_slots_match_default():
    fem = _heat()
    ref = np.asarray(fem.solve().fn())
    for name, kw in [
        ("cg+jacobi", dict(linear=jno.solve.cg(tol=1e-10), precond=jno.precond.jacobi())),
        ("lu", dict(linear=jno.solve.lu())),
        (
            "fgmres+inner",  # iterative preconditioner per step -> flexible outer required
            dict(linear=jno.solve.fgmres(tol=1e-10), precond=jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=30))),
        ),
    ]:
        sol = np.asarray(fem.solve(**kw).fn())
        assert np.abs(sol - ref).max() < 1e-8, f"{name} trajectory deviates from the default"


def test_linear_transient_amg_built_once_on_step_operator(monkeypatch):
    pytest.importorskip("pyamg", reason="pyamg required for the AMG setup (optional dep)")
    import jno.utils.solver.amg as amgmod

    fem = _heat(mesh_size=0.1)
    ref = np.asarray(fem.solve().fn())
    builds = []
    orig = amgmod.build_hierarchy
    monkeypatch.setattr(amgmod, "build_hierarchy", lambda A, **kw: (builds.append(1), orig(A, **kw))[1])
    sol = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10), precond=jno.precond.amg()).fn())
    assert np.abs(sol - ref).max() < 1e-8
    # the hierarchy is set up ONCE for the whole run (constant step operator M + θ·dt·A), before the scan —
    # not per step (amg() no longer self-caches; the transient compose materialises it once up front)
    assert len(builds) == 1


def test_nonlinear_transient_slots_match_default():
    fem = _heat(nonlinear=True)
    ref = np.asarray(fem.solve().fn())
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.fgmres(tol=1e-11)).fn())
    assert np.abs(sol - ref).max() < 1e-8


def test_nonlinear_transient_direct_newton_matches_default():
    """A sparse-DIRECT Newton drives each implicit step of the nonlinear transient march: the step
    tangent ``M/dt + J(u)`` is assembled and factorized (sparse LU) rather than solved matrix-free.
    It must survive the ``lax.scan`` time-march and match the matrix-free default trajectory --
    proving ``jno.solve.newton(direct=True)`` composes through the transient stepper."""
    fem = _heat(nonlinear=True)
    ref = np.asarray(fem.solve().fn())
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True)).fn())
    assert np.abs(sol - ref).max() < 1e-8


def test_nonlinear_transient_direct_newton_inverse():
    """Reverse-mode adjoint through the nonlinear transient inverse with the DIRECT Newton: each
    per-step ``custom_root`` uses a direct, transposable tangent solve (``sparse_lu_solve`` on ``J``
    and ``Jᵀ``), and those chain through the time-march ``lax.scan`` to recover the parameter."""
    import optax

    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(2), name="alpha_dt")
    alpha.initialize(jax.nn.initializers.constant(2.0))
    alpha.dtype(jnp.float64)
    alpha.optimizer(optax.adam(5e-2))

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.2, 11))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y + alpha * ui**3 * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])
    u_obs = jnp.asarray(_heat(nonlinear=True, mesh_size=0.25, time=(0.0, 0.2, 11)).solve().fn())  # alpha_true = 1

    node = fem.solve(nonlinear=jno.solve.newton(direct=True))
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    crux = jno.core([(node - u_obs).mse], domain=dummy)
    crux.solve(120)
    a = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
    assert abs(a - 1.0) < 0.05, f"alpha not recovered through the transient DIRECT Newton adjoint: {a}"


def test_nonlinear_transient_inverse_through_slots():
    """Reverse-mode adjoint through a NONLINEAR transient inverse with a *slot* inner solver.

    Regression for the transient-nonlinear adjoint: each per-step Newton solve reuses the inner
    linear solver as ``custom_root``'s implicit-diff tangent/adjoint solve. A raw slot solver
    (``jno.solve.gmres``) has no transpose rule of its own, so *chaining* these solves through the
    time-march ``lax.scan`` used to raise ``NotImplementedError`` in JAX's ``custom_linear_solve``
    transpose rule. ``newton_krylov`` now firewalls the slot in ``custom_linear_solve`` with an
    explicit ``A^T`` transpose solve, so the transient nonlinear gradient reaches the parameter and
    recovers it (steady nonlinear + linear transient already worked; this covers their intersection).
    """
    import optax

    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="alpha")
    alpha.initialize(jax.nn.initializers.constant(2.0))  # start far from truth = 1
    alpha.dtype(jnp.float64)
    alpha.optimizer(optax.adam(5e-2))

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.2, 11))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    # NONLINEAR reaction alpha*u^3 -> each step is a Newton solve (backward-Euler custom_root)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y + alpha * ui**3 * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])
    assert not fem.is_linear
    u_obs = jnp.asarray(_heat(nonlinear=True, mesh_size=0.25, time=(0.0, 0.2, 11)).solve().fn())  # alpha_true = 1

    node = fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.gmres(maxiter=2000))
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    crux = jno.core([(node - u_obs).mse], domain=dummy)
    crux.solve(120)
    a = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
    assert abs(a - 1.0) < 0.05, f"alpha not recovered through transient NONLINEAR slot adjoint: {a}"


def test_second_order_time_slots():
    """The u_tt (wave) path builds an augmented [u, v] linear block — the slots apply unchanged."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.3, 31))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    weak = ui.tt * vi + (ui.x * vi.x + ui.y * vi.y)
    u0 = u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])
    fem = jno.fem([weak, u(xb, yb) - 0.0, u0, ui0.t - 0.0])
    ref = np.asarray(fem.solve().fn())
    sol = np.asarray(fem.solve(linear=jno.solve.gmres(tol=1e-11), precond=jno.precond.jacobi()).fn())
    assert np.abs(sol - ref).max() < 1e-7


def test_parametric_transient_inverse_through_slots():
    """The per-step materialization path: a runtime parameter makes the operator
    ``operator_fn(t, args)``, so the step system is re-formed per step — the slot solver +
    jacobi (exact step diagonal via diag_fn) must stay differentiable through the scan."""
    import optax

    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="alpha")
    alpha.initialize(jax.nn.initializers.constant(2.0))
    alpha.dtype(jnp.float64)
    alpha.optimizer(optax.adam(5e-2))

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.2, 11))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])
    # observations at alpha_true = 1
    obs_fem = _heat(mesh_size=0.25, time=(0.0, 0.2, 11))
    u_obs = jnp.asarray(obs_fem.solve().fn())

    node = fem.solve(linear=jno.solve.bicgstab(tol=1e-12), precond=jno.precond.jacobi())
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    crux = jno.core([(node - u_obs).mse], domain=dummy)
    crux.solve(120)
    a = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
    assert abs(a - 1.0) < 0.05, f"alpha not recovered through transient slot solvers: {a}"


def test_transient_guards():
    fem = _heat()
    with pytest.raises(ValueError, match="initial conditions"):
        fem.solve(x0=jnp.zeros(4))
    with pytest.raises(ValueError, match="linear"):
        fem.solve(nonlinear=jno.solve.newton())


def test_nonlinear_transient_direct_linear_slot_matches_default():
    """The failure that motivated this: ``fem.solve(linear=jno.solve.lu(...))`` over a nonlinear
    transient block raised ``LinearOperator.dense(): a matvec-only operator cannot densify`` from
    inside the per-step Krylov loop. A direct linear slot now selects the driver that assembles the
    step tangent, and the trajectory matches the matrix-free default."""
    fem = _heat(nonlinear=True)
    ref = np.asarray(fem.solve().fn())
    for spec in (jno.solve.lu(), jno.solve.lu(host=True)):
        sol = np.asarray(fem.solve(linear=spec).fn())
        assert np.abs(sol - ref).max() < 1e-8, f"{spec.name} diverged from the default trajectory"


def test_nonlinear_transient_direct_linear_slot_is_the_solver_that_runs():
    """Per-step: the requested factorization must be the one that actually runs on every step of the
    march, not just be accepted at compose time."""
    from jno.utils.solver import linear as linear_mod

    calls = []
    original = linear_mod.host_lu_solve
    linear_mod.host_lu_solve = lambda A, b: calls.append(1) or original(A, b)
    try:
        sol = np.asarray(_heat(nonlinear=True).solve(linear=jno.solve.lu(host=True)).fn())
    finally:
        linear_mod.host_lu_solve = original
    assert calls, "lu(host=True) was requested but the host factorization never ran"
    assert np.abs(sol - np.asarray(_heat(nonlinear=True).solve().fn())).max() < 1e-8
