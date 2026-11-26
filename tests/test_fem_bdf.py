"""``jno.solve.bdf()``: variable-step, variable-order BDF / NDF (orders 1-5) with error control.

Oracle: a fine fixed-step SDIRK3 march (1600 steps) on the same mesh, whose grid contains every save time,
so what is measured is the time error of the scheme under test (Shampine & Reichelt, SIAM J. Sci. Comput.
18 (1997) for the method; Hairer & Wanner II, Sec. III.5 for variable-coefficient BDF).

Run with x64 (the FEM assembly is float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
T1 = 0.05
N_SAVE = 81
SAVE = jnp.linspace(0.0, T1, N_SAVE)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _form(nsteps, *, cubic=False, param=False, mesh_size=0.25):
    """u_t = k lap u (- u^3 when ``cubic``) on the unit square, zero walls, two decay modes in u0."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, T1, nsteps))
    u, phi = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    k = jno.np.parameter((1,), name="k") if param else 1.0
    s = jno.np.sin
    u0 = 3.0 * (s(PI * ci[0]) * s(PI * ci[1]) + s(2 * PI * ci[0]) * s(3 * PI * ci[1]))
    weak = ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y)
    if cubic:
        weak = weak + ui**3 * vi
    return jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])


def _run(scheme, fem, save=SAVE, args=None):
    return scheme.integrate(fem.operator, args or {}, save, linear_solve=None, nonlinear_solve=None)


def _reference(cubic):
    fem = _form(1601, cubic=cubic)
    full = np.asarray(_run(jno.solve.sdirk(3), fem, jnp.linspace(0.0, T1, 1601)))
    return full[::20]


@pytest.fixture(scope="module")
def ref_linear():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return _reference(False)
    finally:
        jax.config.update("jax_enable_x64", prev)


def _worst(out, ref):
    return max(np.max(np.abs(out[i] - ref[i])) / np.max(np.abs(ref[i])) for i in range(1, N_SAVE))


def test_bdf_error_follows_the_tolerance(ref_linear):
    fem = _form(6)
    errs = []
    for rtol in (1e-4, 1e-6):
        out = np.asarray(_run(jno.solve.bdf().adaptive(rtol=rtol, atol=1e-3 * rtol), fem))
        assert not np.any(np.isnan(out))
        errs.append(_worst(out, ref_linear))
        assert errs[-1] < 20 * rtol, f"rtol={rtol}: error {errs[-1]:.2e}"
    assert errs[1] < errs[0] / 10, f"tightening the tolerance 100x gained only {errs[0] / errs[1]:.1f}x"


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("ndf", [True, False], ids=["ndf", "bdf"])
def test_every_fixed_order_reaches_the_tolerance(ref_linear, order, ndf):
    out = np.asarray(_run(jno.solve.bdf(order=order, ndf=ndf).adaptive(rtol=1e-4, atol=1e-7), _form(6)))
    assert not np.any(np.isnan(out))
    assert _worst(out, ref_linear) < 50 * 1e-4


def test_higher_order_takes_fewer_steps():
    """At rtol = 1e-6 the variable-order march reaches t1 within 100 attempts; held at order 2 it does not
    (it needs a few hundred), and backward Euler would need thousands. The budget is a step count."""
    fem = _form(6)
    tight = dict(rtol=1e-6, atol=1e-9, max_steps=100)
    assert not np.any(np.isnan(np.asarray(_run(jno.solve.bdf().adaptive(**tight), fem, SAVE[-1:]))))
    assert np.all(np.isnan(np.asarray(_run(jno.solve.bdf(order=2).adaptive(**tight), fem, SAVE[-1:]))))


def test_fixed_grid_orders_one_and_two_are_backward_euler_and_bdf2():
    fem = _form(21)
    grid = jnp.linspace(0.0, T1, 21)
    assert np.array_equal(
        np.asarray(_run(jno.solve.bdf(order=2), fem, grid)), np.asarray(_run(jno.solve.bdf2(), fem, grid))
    )
    assert np.array_equal(
        np.asarray(_run(jno.solve.bdf(order=1), fem, grid)), np.asarray(_run(jno.solve.theta(1.0), fem, grid))
    )


def test_bdf2_adaptive_is_the_bdf_march_at_order_two(ref_linear):
    out = np.asarray(_run(jno.solve.bdf2().adaptive(rtol=1e-4, atol=1e-7), _form(6)))
    assert _worst(out, ref_linear) < 50 * 1e-4


def test_bdf_marches_a_nonlinear_block():
    """u_t = lap u - u^3: every BDF step is a Newton solve (the backward-Euler step from the shifted state)."""
    ref = _reference(True)
    out = np.asarray(_run(jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8), _form(6, cubic=True)))
    assert not np.any(np.isnan(out))
    assert _worst(out, ref) < 2e-4


def test_bdf_is_differentiable_through_the_march():
    fem = _form(6, param=True, mesh_size=0.3)
    (name,) = fem.operator.runtime_parameter_exprs
    sch = jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8)

    def loss(kv):
        return jnp.sum(_run(sch, fem, SAVE[::10], {name: jnp.asarray([kv])}) ** 2)

    g = float(jax.grad(loss)(0.8))
    fd = float((loss(0.8 + 1e-5) - loss(0.8 - 1e-5)) / 2e-5)
    assert np.isfinite(g)
    assert abs(g - fd) <= 1e-2 * max(abs(fd), 1.0), f"AD {g} vs FD {fd}"


def test_bdf_rejections_keep_the_gradient_finite():
    """An oversized first step is rejected; its discarded state must give a zero, not NaN, cotangent."""
    fem = _form(6, param=True, mesh_size=0.3)
    (name,) = fem.operator.runtime_parameter_exprs
    sch = jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8, dt0=0.02)
    g = jax.grad(lambda kv: jnp.sum(_run(sch, fem, SAVE[::10], {name: jnp.asarray([kv])}) ** 2))(0.8)
    assert np.isfinite(float(g))


@pytest.mark.parametrize("scheme", [jno.solve.bdf(), jno.solve.bdf(order=2), jno.solve.bdf(order=1).adaptive()])
def test_bdf_refuses_a_second_order_in_time_block(scheme):
    """A ``u_tt`` block is assembled undamped (theta = 1/2); a BDF march would damp the wave silently."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4, time=(0.0, T1, 6))
    u, phi = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.tt * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    with pytest.raises(NotImplementedError, match="second-order-in-time"):
        _run(scheme, fem)


def test_bdf_composes_through_fem_solve_complex_and_periodic():
    """The step is the block's own backward-Euler step, so the complex (2n real block) and periodic (reduced)
    paths and a state-dependent mass with a time-varying wall march through ``fem.solve`` unchanged. The
    oracle is a tight SDIRK3 march of the same form."""
    from tests.test_fem_adaptive_timestep import _complex_heat

    def ev(x):
        return np.asarray(x.fn() if hasattr(x, "fn") else jnp.asarray(x))

    save = jnp.linspace(0.0, T1, 6)
    ref = jno.solve.sdirk(3).adaptive(rtol=1e-7, atol=1e-10)
    sch = jno.solve.bdf().adaptive(rtol=1e-4, atol=1e-7)

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, T1, 6))
    u, phi = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    mass_wall = jno.fem(
        [(1 + ui**2) * ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - jno.np.exp(-2 * tb), u(ci[0], ci[1]) - 1.0]
    )

    for label, fem, cplx in [
        ("c(u) u_t + g(t)", mass_wall, False),
        ("complex", _complex_heat(mesh_size=0.3)[1], True),
        ("complex periodic", _complex_heat(mesh_size=0.3, periodic=True)[1], True),
    ]:
        got, want = ev(fem.solve(save_ts=save, time=sch)), ev(fem.solve(save_ts=save, time=ref))
        assert np.iscomplexobj(got) == cplx and not np.any(np.isnan(got)), label
        rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert rel < 2e-3, f"{label}: {rel:.2e}"


def test_bdf_refuses_a_bad_order():
    for bad in (0, 6, 2.5, True):
        with pytest.raises(ValueError, match="order must be"):
            jno.solve.bdf(order=bad)


def test_the_step_solve_starts_from_the_predictor():
    """Each BDF step's solve starts from the predictor P(t_{n+1}), O(h^(k+1)) from the answer, not from the
    shifted state psi the formula forms, O(h) from it. The warm start's relative residual is then near the
    local error (measured ~4e-7 at rtol 1e-6) instead of ~1e-2, and the Krylov solve has less to do."""
    from jno.utils.solver.backend_blocks import _default_step_solve

    seen = []

    def counting(matvec, rhs, x0, diag_fn):
        r0 = jnp.linalg.norm(rhs - matvec(x0)) / jnp.linalg.norm(rhs)
        jax.debug.callback(lambda v: seen.append(float(v)), r0)
        return _default_step_solve(matvec, rhs, x0, diag_fn())

    fem = _form(6)
    scheme = jno.solve.bdf().adaptive(rtol=1e-6, atol=1e-9, max_steps=150)
    out = np.asarray(scheme.integrate(fem.operator, {}, SAVE[-1:], linear_solve=counting, nonlinear_solve=None))
    assert not np.any(np.isnan(out)) and len(seen) > 20
    assert np.median(seen) < 1e-4, f"median warm-start residual {np.median(seen):.1e}"


@pytest.mark.parametrize("rtol", [1e-4, 1e-6, 1e-9])
@pytest.mark.parametrize("make", [lambda: jno.solve.bdf(), lambda: jno.solve.sdirk(3)], ids=["bdf", "sdirk3"])
def test_step_solve_tolerance_follows_the_march_tolerance(monkeypatch, make, rtol):
    """The default step solve of an error-controlled march stops at a fraction of the march's tolerance (not at a
    fixed 1e-10): ``min(1e-2 rtol, 1e-1 atol / max|u0|)``, within ``[10 eps, 1e-6]``. A loose march must not pay for
    tight inner solves, a tight one must not drown its error estimate in solver noise. The fixed grid keeps 1e-10."""
    import jax.scipy.sparse.linalg as jsl

    seen = []
    real = jsl.bicgstab

    def spy(A, b, **kw):
        seen.append(float(kw["tol"]))
        return real(A, b, **kw)

    monkeypatch.setattr(jsl, "bicgstab", spy)
    fem = _form(6)
    atol = 1e-3 * rtol
    u0max = float(np.max(np.abs(np.asarray(fem.operator.state0))))
    want = min(1e-2 * rtol, 1e-1 * atol / u0max)
    want = min(max(want, 10 * np.finfo(np.float64).eps), 1e-6)
    _run(make().adaptive(rtol=rtol, atol=atol, max_steps=40), fem, SAVE[-1:])
    assert seen and all(np.isclose(v, want, rtol=1e-6) for v in seen), (seen, want)
    seen.clear()
    _run(jno.solve.sdirk(3), fem, SAVE)
    assert seen and all(v == 1e-10 for v in seen), seen


def _exact_semidiscrete(fem, save):
    """The exact solution of the semidiscrete heat system ``M u' = -A u`` (zero Dirichlet rows held at 0), by the
    generalised eigendecomposition of the interior block: no time-stepping error, so a march's own is measured."""
    import scipy.linalg as sl

    blk = fem.operator
    dense = lambda X: np.asarray(X.todense() if hasattr(X, "todense") else X)  # noqa: E731
    M, A = dense(blk.M), dense(blk.A)
    u0 = np.asarray(blk.state0).reshape(-1)
    inner = np.abs(np.diag(M)) > 1e-14
    lam, V = sl.eigh(A[np.ix_(inner, inner)], M[np.ix_(inner, inner)])
    c = V.T @ (M[np.ix_(inner, inner)] @ u0[inner])
    out = np.zeros((len(save), len(u0)))
    for i, t in enumerate(np.asarray(save)):
        out[i, inner] = V @ (np.exp(-lam * t) * c)
    return out


@pytest.mark.parametrize(
    "make, gain", [(lambda: jno.solve.bdf(), 1.0), (lambda: jno.solve.sdirk(3), 8.0)], ids=["bdf", "sdirk3"]
)
def test_tight_tolerance_march_reaches_the_end_and_is_accurate(make, gain):
    """A march at rtol 1e-8 (atol 1e-11) must finish inside a modest budget and be more accurate than at 1e-6.
    With the fixed 1e-10 step solve the solver noise (about 1e-10 max|u| per component) sat above atol, the error
    estimate never fell below 1 and bdf crawled on the step floor until the budget ran out (NaN).

    bdf's gain is only asserted to be positive: its first steps sit on the step floor ``1e-4 * span`` and are accepted
    there whatever the estimate says, which leaves a start-up error near 1e-6 that no rtol removes (a lower floor
    does); sdirk3 starts with a third-order step and gains most of the 100x (its dense output limits the rest)."""
    fem = _form(6)
    ref = _exact_semidiscrete(fem, SAVE)
    errs = []
    for rtol in (1e-6, 1e-8):
        out = np.asarray(_run(make().adaptive(rtol=rtol, atol=1e-3 * rtol, max_steps=1500), fem))
        assert not np.any(np.isnan(out)), f"rtol={rtol}: budget exhausted"
        errs.append(_worst(out, ref))
    assert errs[1] < errs[0] / gain, f"tightening 100x gained only {errs[0] / errs[1]:.1f}x ({errs})"
    if gain > 1:
        assert errs[1] < 20 * 1e-8


def _count_solves(scheme, fem, args=None):
    """Run ``scheme`` with a counting ``linear=`` slot: how many step solves were EXECUTED (a skipped branch of
    a ``lax.cond`` does not fire the callback)."""
    from jno.utils.solver.backend_blocks import _default_step_solve

    n = []

    def counting(matvec, rhs, x0, diag_fn):
        jax.debug.callback(lambda: n.append(1))
        return _default_step_solve(matvec, rhs, x0, diag_fn())

    out = scheme.integrate(fem.operator, args or {}, SAVE, linear_solve=counting, nonlinear_solve=None)
    jax.block_until_ready(out)
    jax.effects_barrier()
    return len(n), np.asarray(out)


@pytest.mark.parametrize("make", [lambda ms: jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8, max_steps=ms),
                                  lambda ms: jno.solve.sdirk(3).adaptive(rtol=1e-5, atol=1e-8, max_steps=ms)],
                         ids=["bdf", "sdirk3"])  # fmt: skip
def test_the_unused_part_of_the_step_budget_costs_nothing(make):
    """An error-controlled march pays for the attempts it needs, not for ``max_steps``: the executed step solves
    and the result are the same for a budget of 150 and of 4000 attempts (the settled tail is skipped a chunk at
    a time instead of running one idle iteration per attempt)."""
    fem = _form(6)
    n_small, out_small = _count_solves(make(150), fem)
    n_big, out_big = _count_solves(make(4000), fem)
    assert 0 < n_small
    assert n_big == n_small, (n_small, n_big)
    np.testing.assert_array_equal(out_small, out_big)


@pytest.mark.parametrize("make", [lambda ms: jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8, max_steps=ms),
                                  lambda ms: jno.solve.sdirk(3).adaptive(rtol=1e-5, atol=1e-8, max_steps=ms)],
                         ids=["bdf", "sdirk3"])  # fmt: skip
def test_adaptive_gradient_is_the_same_for_any_budget_and_matches_finite_differences(make):
    """Reverse mode runs through the checkpointed chunks: the gradient does not depend on how large the budget is
    (the skipped chunks send nothing back) and matches a central finite difference; ``jit`` and ``vmap`` agree."""
    fem = _form(6, param=True, mesh_size=0.3)
    (name,) = fem.operator.runtime_parameter_exprs

    def loss(ms):
        sch = make(ms)
        return lambda kv: jnp.sum(_run(sch, fem, SAVE[::10], {name: jnp.asarray([kv])}) ** 2)

    g_small = float(jax.grad(loss(150))(0.8))
    g_big = float(jax.jit(jax.grad(loss(2500)))(0.8))
    fd = float((loss(150)(0.8 + 1e-5) - loss(150)(0.8 - 1e-5)) / 2e-5)
    assert np.isfinite(g_small)
    assert abs(g_small - g_big) <= 1e-9 * max(abs(g_small), 1.0), (g_small, g_big)
    assert abs(g_small - fd) <= 1e-2 * max(abs(fd), 1.0), f"AD {g_small} vs FD {fd}"
    ks = jnp.asarray([0.7, 0.8, 0.9])
    batched = jax.vmap(jax.grad(loss(150)))(ks)
    looped = jnp.stack([jax.grad(loss(150))(k) for k in ks])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(looped), rtol=1e-8)


def test_reverse_mode_memory_does_not_grow_with_the_step_budget():
    """The reverse pass keeps the chunk-boundary carries and one chunk's intermediates, not those of every attempt
    of the budget: the compiled gradient's temporary buffers grow much slower than ``max_steps``."""
    fem = _form(6, param=True, mesh_size=0.3)
    (name,) = fem.operator.runtime_parameter_exprs

    def temp_bytes(ms):
        sch = jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8, max_steps=ms)
        g = jax.jit(jax.grad(lambda kv: jnp.sum(_run(sch, fem, SAVE[::10], {name: jnp.asarray([kv])}) ** 2)))
        return g.lower(jnp.asarray(0.8)).compile().memory_analysis().temp_size_in_bytes

    small, big = temp_bytes(250), temp_bytes(4000)
    assert big < 8 * small, f"temp {small} -> {big} bytes for a 16x budget"  # was 16x (10.7 MB -> 171 MB) before


def test_bdf_traces_the_step_solve_once_not_once_per_order():
    """The order switch sits around the cheap formula weights and error estimates, not around the step: the block's
    step (a Krylov solve, its adjoint, the residual gate) is traced once, so bdf compiles about like an sdirk march
    instead of five times the program (a switch over the whole step traced it once per order)."""
    from jno.utils.solver.backend_blocks import _default_step_solve

    traced = []

    def counting(matvec, rhs, x0, diag_fn):
        traced.append(1)
        return _default_step_solve(matvec, rhs, x0, diag_fn())

    fem = _form(6)
    jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8, max_steps=40).integrate(
        fem.operator, {}, SAVE[-1:], linear_solve=counting, nonlinear_solve=None
    )
    assert len(traced) == 1, f"the step was traced {len(traced)} times"
