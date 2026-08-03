"""The slot-based solver API: ``fem.solve(x0=, nonlinear=, linear=, precond=)`` composed from the
callables-only ``jno.solve`` / ``jno.precond`` namespaces (see ``plans/fem-solver-api.md``).

Pins: every shipped linear solver reproduces the historic default on a Poisson system (slot
solvers receive the BCOO operator -- no densification); ``x0`` warm-starts (exact guess is a
fixed point); the direct/Krylov contract (``LinearSolver`` rejects ``M`` on direct solvers);
jit + vmap of the pure-JAX solvers at the contract level; slot composition on the nonlinear
path (driver + injected inner linear solve) matches the Newton-Krylov default; the parametric
(inverse) path stays differentiable through slot solvers; and the guard rails (slots xor
``solve_fn``; transient unsupported; ``precond`` on the matrix-free nonlinear path unsupported).
"""

from __future__ import annotations

import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.experimental.sparse as jsp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
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


def _poisson(mesh_size=0.2):
    """Poisson with exact solution u = x(1-x)y(1-y); homogeneous Dirichlet."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def _nonlinear(mesh_size=0.25):
    """Reaction-diffusion -lap u + u^3 = f with exact u = sin(pi x) sin(pi y)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    ss = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    f = 2.0 * PI**2 * ss + ss**3
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui**3 * vi - f * vi, u(xb, yb) - 0.0], quad_degree=3)


# ---------------------------------------------------------------------------
# steady linear slots
# ---------------------------------------------------------------------------


def test_linear_slot_solvers_match_default():
    fem = _poisson()
    u_ref = np.asarray(fem.solve())
    for solver, precond in [
        (jno.solve.cg(), jno.precond.jacobi()),  # SPD system: CG applies
        (jno.solve.bicgstab(), jno.precond.jacobi()),  # == the historic default, via slots
        (jno.solve.gmres(), jno.precond.jacobi()),
        (jno.solve.gmres(), None),
        (jno.solve.lu(), None),
        (jno.solve.dense(), None),
    ]:
        u = np.asarray(fem.solve(linear=solver, precond=precond))
        assert np.abs(u - u_ref).max() < 1e-7, f"{solver.name} (precond={precond}) deviates from the default"


def test_x0_warm_start():
    fem = _poisson()
    u_ref = jnp.asarray(fem.solve())
    # the exact solution as warm start is a fixed point of a converged Krylov solve
    u = fem.solve(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi(), x0=u_ref)
    assert np.abs(np.asarray(u - u_ref)).max() < 1e-12
    # a pure warm start (all other slots defaulted) also solves
    u2 = fem.solve(x0=jnp.zeros_like(u_ref))
    assert np.abs(np.asarray(u2 - u_ref)).max() < 1e-7


def test_user_written_slot_callables():
    """The documented extension contract: bare callables drop into the slots."""
    fem = _poisson()
    u_ref = np.asarray(fem.solve())

    def my_linear(A, b, *, M=None, x0=None):  # LinearOperator in, solution out
        return jnp.linalg.solve(A.dense(), b)

    def my_precond(ctx):  # ctx -> (v -> M^{-1} v)
        inv = 1.0 / ctx.diag()
        return lambda v: inv * v

    assert np.abs(np.asarray(fem.solve(linear=my_linear)) - u_ref).max() < 1e-9
    u = fem.solve(linear=jno.solve.cg(), precond=my_precond)
    assert np.abs(np.asarray(u) - u_ref).max() < 1e-7


# ---------------------------------------------------------------------------
# contract level: LinearOperator / LinearSolver under jit + vmap
# ---------------------------------------------------------------------------


def _spd(n=24, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    Ad = Q @ Q.T + n * np.eye(n)
    return jsp.BCOO.fromdense(jnp.asarray(Ad)), Ad


def test_solver_contract_jit_and_vmap():
    A, Ad = _spd()
    solver = jno.solve.cg(tol=1e-12)
    op = jno.solve.LinearOperator(A)
    b = jnp.asarray(np.random.default_rng(1).standard_normal(Ad.shape[0]))
    x = jax.jit(lambda bb: solver(op, bb))(b)
    assert np.allclose(np.asarray(x), np.linalg.solve(Ad, np.asarray(b)), atol=1e-8)
    B = jnp.stack([b, 2.0 * b, b - 1.0])
    X = jax.vmap(lambda bb: solver(op, bb))(B)  # shipped Krylov solvers are vmap-native
    assert np.allclose(np.asarray(X), np.linalg.solve(Ad, np.asarray(B).T).T, atol=1e-8)


def test_linear_operator_transpose_and_diag():
    A, Ad = _spd(n=8, seed=2)
    op = jno.solve.LinearOperator(A)
    v = jnp.arange(8, dtype=jnp.float64)
    assert np.allclose(np.asarray(op.T.mv(v)), Ad.T @ np.asarray(v))
    assert np.allclose(np.asarray(op.diag()), np.diag(Ad))
    mv_op = jno.solve.LinearOperator.from_matvec(lambda w: A @ w)
    assert np.allclose(np.asarray(mv_op.T.mv(v)), Ad.T @ np.asarray(v))  # via jax.linear_transpose
    with pytest.raises(TypeError):
        mv_op.diag()


# ---------------------------------------------------------------------------
# nonlinear slots
# ---------------------------------------------------------------------------


def test_nonlinear_slots_match_default():
    fem = _nonlinear()
    u_ref = np.asarray(fem.solve())
    u = np.asarray(fem.solve(nonlinear=jno.solve.newton()))
    assert np.abs(u - u_ref).max() < 1e-8
    # inner linear solve injected from the linear slot (adapted to the matrix-free contract)
    u2 = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.bicgstab(tol=1e-12)))
    assert np.abs(u2 - u_ref).max() < 1e-7
    # x0 is the Newton initial guess
    u3 = np.asarray(fem.solve(nonlinear=jno.solve.newton(), x0=jnp.asarray(u_ref)))
    assert np.abs(u3 - u_ref).max() < 1e-8


# ---------------------------------------------------------------------------
# sparse-DIRECT nonlinear Newton -- jno.solve.newton(direct=True)
# ---------------------------------------------------------------------------


def test_nonlinear_direct_newton_matches_default():
    """``direct=True`` factorizes the ASSEMBLED tangent each Newton step (sparse LU) instead of the
    matrix-free BiCGStab inner solve. On a well-behaved system it solves the SAME discrete nonlinear
    problem as the default -- to solver tolerance -- and reproduces the manufactured solution."""
    fem = _nonlinear()
    u_ref = np.asarray(fem.solve())  # matrix-free Newton-Krylov default
    u = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True)))
    assert np.abs(u - u_ref).max() < 1e-7
    p = np.asarray(fem.points)
    u_exact = np.sin(PI * p[:, 0]) * np.sin(PI * p[:, 1])  # exact u = sin(pi x) sin(pi y)
    assert np.abs(u - u_exact).max() < 5e-2  # coarse-mesh (0.25) discretization error, not a solver error


def test_nonlinear_direct_newton_line_search():
    """The direct inner solve composes with the shared globalization options (damped + line search)."""
    fem = _nonlinear()
    u_ref = np.asarray(fem.solve())
    u = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, line_search=True, damping=0.8)))
    assert np.abs(u - u_ref).max() < 1e-7


def test_nonlinear_direct_newton_differentiable():
    """The direct Newton stays differentiable in a parameter: ``custom_root`` supplies the implicit-
    function gradient via a DIRECT, transposable tangent solve on the assembled tangent at the root."""
    import optax

    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(3), name="alpha_direct")
    alpha.initialize(jax.nn.initializers.constant(2.0))
    alpha.dtype(jnp.float64)
    alpha.optimizer(optax.adam(5e-2))

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    ss = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    f = 2.0 * PI**2 * ss + ss**3
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + alpha * ui**3 * vi - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    u_obs = jnp.asarray(_nonlinear(mesh_size=0.25).solve())  # alpha_true = 1 data
    node = fem.solve(nonlinear=jno.solve.newton(direct=True))
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    crux = jno.core([(node - u_obs).mse], domain=dummy)
    crux.solve(120)
    a = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
    assert abs(a - 1.0) < 0.05, f"alpha not recovered through the direct Newton adjoint: {a}"


def test_direct_newton_without_assembled_jacobian_raises():
    """As a bare driver (no assembled Jacobian threaded in) ``direct=True`` fails loud: a sparse-direct
    step needs the assembler's tangent, provided only on the native nonlinear / transient paths."""
    solver = jno.solve.newton(direct=True)
    with pytest.raises(ValueError, match="ASSEMBLED Jacobian"):
        solver(lambda w: w, jnp.zeros(3))


# ---------------------------------------------------------------------------
# parametric (inverse) path stays differentiable through slot solvers
# ---------------------------------------------------------------------------


def test_parametric_linear_differentiable_through_slots():
    import optax

    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="alpha")
    alpha.initialize(jax.nn.initializers.constant(2.0))
    alpha.dtype(jnp.float64)
    alpha.optimizer(optax.adam(5e-2))

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    fem = jno.fem([alpha * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    u_obs = jnp.asarray(_poisson(mesh_size=0.25).solve())  # alpha_true = 1 data
    node = fem.solve(linear=jno.solve.cg(tol=1e-12), precond=jno.precond.jacobi())
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    crux = jno.core([(node - u_obs).mse], domain=dummy)
    crux.solve(120)
    a = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
    assert abs(a - 1.0) < 0.05, f"alpha not recovered through slot solvers: {a}"


# ---------------------------------------------------------------------------
# guard rails
# ---------------------------------------------------------------------------


def test_slots_and_solve_fn_are_exclusive():
    fem = _poisson()
    with pytest.raises(ValueError, match="not both"):
        fem.solve(solve_fn=lambda A, b: b, linear=jno.solve.cg())


def test_direct_solver_rejects_preconditioner():
    fem = _poisson()
    with pytest.raises(ValueError, match="direct solver"):
        fem.solve(linear=jno.solve.lu(), precond=jno.precond.jacobi())


def test_nonlinear_slot_on_linear_problem_raises():
    fem = _poisson()
    with pytest.raises(ValueError, match="no linearization"):
        fem.solve(nonlinear=jno.solve.newton())


def test_jacobi_on_matrix_free_nonlinear_raises():
    """precond= composes with the nonlinear path (materialized per linearization against the JVP
    operator) — but jacobi needs the assembled diagonal, which a matvec-only operator lacks."""
    fem = _nonlinear()
    with pytest.raises(TypeError, match="matvec-only"):
        fem.solve(nonlinear=jno.solve.newton(), precond=jno.precond.jacobi())


# NOTE: the former ``test_complex_transient_slots_raise`` is gone. A complex transient is now assembled
# as ONE real 2n block, so it *is* an ordinary transient block and the solver slots apply to it like any
# other. The inverted assertion — the slots compose, and every choice lands on the same trajectory — lives
# in ``test_fem_adaptive_timestep.py::test_complex_transient_composes_solver_slots``, on a real FEM rather
# than the synthetic ``FEM(None, None, [], mode=...)`` this test had to fabricate.


def test_x0_u0_conflict_raises():
    fem = _nonlinear()
    z = jnp.zeros(4)
    with pytest.raises(ValueError, match="same initial guess"):
        fem.solve(x0=z, u0=z)


# ---------------------------------------------------------------------------
# the COMPILED slot path: one compilation reused across solves
# ---------------------------------------------------------------------------
# The slot path used to call its Krylov solver from eager Python, paying dispatch per iteration
# (bicgstab+jacobi 104.7 ms vs 6.3 compiled, n=13861). Compiling it is only a win if `jax.jit`'s
# cache actually SPANS calls, and `FEM.solve` re-composes on every call -- so both the function and
# its static arguments have to be stable. These tests pin the three ways that goes wrong: a fresh
# spec object per call (recompile), a slot that cannot trace at all (crash), and a slot that quietly
# skips work the eager path does.


def _compiled():
    from jno.utils.solver.solver_api import _composed_compiled

    return _composed_compiled


def test_inline_specs_share_one_compilation():
    """`fem.solve(linear=jno.solve.cg())` -- the spec written inline, as the docs write it.

    Specs are handed to `jax.jit` as static arguments, and jax keys its cache on `hash`. With
    identity hashing every call is a fresh object, hence a fresh cache entry and a recompile:
    measured 0.4 ms hoisted against 83.5 ms inline on this 513-DOF system, i.e. compiling made the
    common usage 200x SLOWER than leaving it eager. Value keys are what make the cache work.
    """
    fem = _poisson(mesh_size=0.05)
    u_ref = np.asarray(fem.solve())
    before = _compiled()._cache_size()
    for _ in range(3):
        u = fem.solve(linear=jno.solve.bicgstab(tol=1e-10), precond=jno.precond.jacobi())
        assert np.abs(np.asarray(u) - u_ref).max() < 1e-7
    assert _compiled()._cache_size() == before + 1, "an inline spec recompiled instead of hitting the cache"

    # `linear` defaulted: the composer builds its own bicgstab() internally, on every call. That one
    # is invisible to the user, so identity hashing would recompile a solve nobody wrote a spec for.
    before = _compiled()._cache_size()
    for _ in range(3):
        fem.solve(precond=jno.precond.jacobi())
    assert _compiled()._cache_size() == before + 1, "the composer's own default solver recompiled per call"


def test_differently_configured_specs_do_not_share_a_cache_entry():
    """The risk value keys introduce: a key that misses a parameter would serve a solve configured
    the other way -- a wrong answer, not a slow one. Every argument that changes the iteration must
    be in the key, including the per-method extras (`gmres(restart=)`) that live in `**fixed`."""
    assert jno.solve.cg(tol=1e-8) == jno.solve.cg(tol=1e-8)
    assert hash(jno.solve.cg(tol=1e-8)) == hash(jno.solve.cg(tol=1e-8))
    for a, b in [
        (jno.solve.cg(tol=1e-8), jno.solve.cg(tol=1e-3)),
        (jno.solve.cg(maxiter=10), jno.solve.cg(maxiter=20)),
        (jno.solve.cg(atol=0.0), jno.solve.cg(atol=1e-12)),
        (jno.solve.gmres(restart=10), jno.solve.gmres(restart=30)),
        (jno.solve.fgmres(restart=10), jno.solve.fgmres(restart=30)),
        (jno.solve.minres(tol=1e-8), jno.solve.minres(tol=1e-6)),
        (jno.solve.cg(), jno.solve.bicgstab()),  # same arguments, different method
    ]:
        assert a != b, f"{a!r} and {b!r} would share a compiled solve"

    # End to end, in the order that would hide a collision: converge first, so a cache keyed on the
    # method name alone would then hand the 2-iteration budget the converged compilation and pass.
    fem = _poisson()
    u = fem.solve(linear=jno.solve.cg(tol=1e-12, maxiter=5000), precond=jno.precond.jacobi())
    assert np.abs(np.asarray(u) - np.asarray(fem.solve())).max() < 1e-7
    with pytest.raises(Exception, match="did not solve"):
        fem.solve(linear=jno.solve.cg(tol=1e-12, maxiter=2), precond=jno.precond.jacobi())


def test_host_side_slots_stay_eager_and_correct():
    """Not every slot can be materialized inside a trace, and the ones that cannot must fall back
    silently rather than fail. `chebyshev` measures spectrum bounds and then branches on what it
    measured (`if hi <= 0`); `amg`/`ams`/`form` assemble an auxiliary operator host-side through
    scipy/pyamg. A bare callable -- the documented extension contract -- declares nothing at all."""
    from jno.utils.solver.solver_api import _compilable

    fem = _poisson()
    u_ref = np.asarray(fem.solve())

    def my_linear(A, b, *, M=None, x0=None):
        return jnp.linalg.solve(A.dense(), b)

    def my_precond(ctx):
        return lambda v: (1.0 / ctx.diag()) * v

    cg = jno.solve.cg(tol=1e-12, maxiter=5000)
    eager = [
        (jno.solve.chebyshev(maxiter=2000), None),  # host-side branching in the SOLVER
        (cg, jno.precond.chebyshev(degree=6)),  # ... and in the PRECONDITIONER
        (cg, jno.precond.amg()),  # pyamg hierarchy, built host-side
        (cg, jno.precond.nystrom(rank=12)),
        (my_linear, None),  # bare callables: no traits, no key
        (cg, my_precond),
        (cg, jno.precond.jacobi().cached()),  # .cached() owns its own reuse; do not compile over it
    ]
    for linear, precond in eager:
        assert not _compilable(linear, precond), f"{linear} + {precond} must not be compiled"
        u = fem.solve(linear=linear, precond=precond)
        assert np.abs(np.asarray(u) - u_ref).max() < 1e-6, f"{linear} + {precond} deviates on Poisson"

    assert _compilable(cg, None) and _compilable(cg, jno.precond.jacobi()), "the traceable pair must compile"


def test_extreme_magnitude_is_scaled_before_the_jit_boundary():
    """`_normalize_extreme_scale` fires only on a CONCRETE operator -- it reads `float(max|A|)`
    raising on a tracer as "traced, leave alone". Run it inside the compiled function and every
    operator is a tracer, so the eddy regime (|A| ~ jw*sigma ~ 1e12) sails past the guard into the
    Arnoldi breakdown it exists to prevent: a returned ~0 with relative residual 1, silently. So the
    scaling has to happen OUTSIDE the jit, and this is the test that it still does.
    """
    from jno.utils.solver.solver_api import compose_linear_solve_fn

    n = 60
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((n, n))
    Ad = (Q @ Q.T + n * np.eye(n)) * 1e12  # mass-dominated eddy scaling
    A = jsp.BCOO.fromdense(jnp.asarray(Ad))
    b = jnp.asarray(rng.standard_normal(n)) * 1e12
    ref = np.linalg.solve(Ad, np.asarray(b))

    solve = compose_linear_solve_fn(jno.solve.bicgstab(tol=1e-12, maxiter=5000), jno.precond.jacobi(), None, None)
    x = np.asarray(solve(A, b))
    assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-6, "extreme-magnitude solve broke down under jit"


def test_compiled_path_still_raises_on_an_unconverged_solve():
    """Every Krylov solver ends in `_maybe_residual_check`, which needs a CONCRETE residual and
    steps aside on a tracer. So compiling the solve does not disable that guard loudly -- it
    disables it silently, and an exhausted iteration budget starts returning its unconverged vector
    as if it had succeeded. Caught by an AMS test whose *negative control* (Jacobi cannot converge
    in 60 iterations) stopped failing, which is the kind of thing that reads as a passing suite.
    """
    fem = _poisson(mesh_size=0.05)
    with pytest.raises(Exception, match="did not solve"):
        fem.solve(linear=jno.solve.cg(tol=1e-12, maxiter=2), precond=jno.precond.jacobi())
    # and the guard does not fire on a solve that DID converge
    u = fem.solve(linear=jno.solve.cg(tol=1e-12, maxiter=5000), precond=jno.precond.jacobi())
    assert np.abs(np.asarray(u) - np.asarray(fem.solve())).max() < 1e-7
