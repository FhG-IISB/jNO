"""The pure-JAX Krylov trio jNO implements itself (absent from the JAX ecosystem): FGMRES
(Saad 1993, Alg. 2.2), MINRES (Paige & Saunders 1975), Chebyshev (Saad 2003 §12.3).

Pins: correctness vs dense/scipy oracles on the structure each method targets (non-symmetric for
FGMRES, symmetric *indefinite* for MINRES, SPD for Chebyshev); FGMRES's distinguishing property
(an *iterative*, per-call-varying preconditioner — illegal for plain GMRES); restart smaller than
the Krylov dimension; the Chebyshev polynomial preconditioner accelerating CG; ``jit`` + ``vmap``
at the contract level; reverse-mode differentiability through the ``custom_linear_solve``
firewall (vs finite differences, non-symmetric so a transpose bug cannot hide); and end-to-end
use as ``fem.solve`` slots.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver import krylov


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _spd(n=40, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    return Q @ Q.T + n * np.eye(n)


def _sym_indefinite(n=40, seed=1):
    """Symmetric with a genuinely mixed spectrum — CG diverges here, MINRES is the method."""
    rng = np.random.default_rng(seed)
    D = np.diag(np.concatenate([np.linspace(1.0, 10.0, n // 2), -np.linspace(1.0, 5.0, n - n // 2)]))
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return U @ D @ U.T


def _nonsym(n=40, seed=2):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + n * np.eye(n)


def _op(Ad):
    return jno.solve.LinearOperator(jsp.BCOO.fromdense(jnp.asarray(Ad)))


def _b(n, seed=3):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(n))


# ---------------------------------------------------------------------------
# correctness vs oracles
# ---------------------------------------------------------------------------


def _f32_tridiag(n=200):
    """SPD tridiagonal in float32, plus its float64 reference solution. Explicit f32 arrays rather
    than toggling ``jax_enable_x64`` — precision is per-array, and the global flag is process-wide."""
    d, off = np.full(n, 2.0), np.full(n - 1, -1.0)
    A64 = np.diag(d) + np.diag(off, 1) + np.diag(off, -1)
    b64 = np.random.default_rng(0).standard_normal(n)
    A32, b32 = jnp.asarray(A64.astype(np.float32)), jnp.asarray(b64.astype(np.float32))
    return (lambda v: A32 @ v), b32, np.linalg.solve(A64, b64)


def test_breakdown_guard_is_representable_in_float32():
    """The guard floor must exist in the dtype it guards.

    The historical constant was ``1e-300``, which **underflows to exactly 0.0 in float32** (smallest
    normal ~1.18e-38). Every ``maximum(x, tiny)`` breakdown guard in the three solvers then became
    ``maximum(x, 0.0)`` and divided by exact zero, giving inf/NaN instead of a clamped value — and
    jNO's session default is float32 (``tests/conftest.py``), so that path was reachable by default."""
    from jno.utils.solver.krylov import _tiny_of

    assert np.float32(1e-300) == 0.0, "the premise: the old constant is not representable in f32"
    for dt in (np.float32, np.float64):
        tiny = float(_tiny_of(dt))
        assert tiny > 0.0 and np.asarray(tiny, dt) > 0, f"{dt.__name__} floor underflowed"
        assert np.isfinite(1.0 / np.asarray(tiny, dt)), f"1/tiny overflowed in {dt.__name__}"


def test_float32_breakdown_returns_finite_not_nan():
    """A zero right-hand side drives ``beta`` to exactly 0 — the case the guard exists for. Both
    solvers must return the (finite) zero solution rather than NaN."""
    mv, b32, _ = _f32_tridiag()
    zero = jnp.zeros_like(b32)
    for name, fn in (("minres", krylov.minres), ("fgmres", krylov.fgmres)):
        x = np.asarray(fn(mv, zero, tol=1e-8, maxiter=50))
        assert np.all(np.isfinite(x)), f"{name} returned non-finite values on a zero RHS"
        assert np.max(np.abs(x)) == 0.0, f"{name} did not return the zero solution"


def test_tolerance_floor_is_reachable_and_leaves_float64_alone():
    """A tolerance below unit round-off cannot be met: the residual norm never falls below
    ~eps*||b||, so the loop runs to ``maxiter`` on a system it already solved. The shipped default
    ``tol=1e-8`` is exactly that request in float32 (eps 1.2e-7).

    float64 must be untouched — this is what makes the fix free of regression surface."""
    from jno.utils.solver.krylov import _effective_tol

    assert _effective_tol(1e-8, np.float64) == 1e-8, "float64 tolerances must not move"
    assert _effective_tol(1e-2, np.float32) == 1e-2, "a reachable f32 tolerance must not move"
    floored = _effective_tol(1e-8, np.float32)
    assert floored > float(np.finfo(np.float32).eps), "the f32 floor must exceed unit round-off"
    assert floored < 1e-5, f"the floor should stay tight, got {floored:.2e}"


def test_float32_tracks_float64_at_matched_restart():
    """The end-to-end property: at the SAME restart, float32 must track float64 rather than diverge.

    Guards against reading restarted-GMRES stagnation as a precision bug. On this deliberately
    ill-conditioned tridiagonal (kappa ~ 1.6e4) a restart of 30 stagnates around 1.1e-2 in BOTH
    precisions — the restart is the limit, not the dtype. Without a restart the two separate as
    expected (~1e-13 vs ~1e-5), which is the honest float32 floor."""
    mv32, b32, x_ref = _f32_tridiag()
    n = b32.shape[0]
    A64 = np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    b64 = jnp.asarray(np.asarray(b32, np.float64))
    mv64 = lambda v: jnp.asarray(A64) @ v  # noqa: E731
    rel = lambda x: float(np.linalg.norm(np.asarray(x, np.float64) - x_ref) / np.linalg.norm(x_ref))  # noqa: E731

    r32 = rel(krylov.fgmres(mv32, b32, tol=1e-8, restart=30, maxiter=1200))
    r64 = rel(krylov.fgmres(mv64, b64, tol=1e-8, restart=30, maxiter=1200))
    assert r32 == pytest.approx(r64, rel=0.1), f"f32 must track f64 at matched restart: {r32:.2e} vs {r64:.2e}"
    assert np.isfinite(r32)

    # and unrestarted, float32 reaches its own floor rather than stalling or blowing up
    assert rel(krylov.fgmres(mv32, b32, tol=1e-8, restart=n, maxiter=1200)) < 1e-4


def test_minres_symmetric_indefinite_vs_scipy():
    Ad = _sym_indefinite()
    b = _b(Ad.shape[0])
    x = jno.solve.minres(tol=1e-12)(_op(Ad), b)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9
    sp = pytest.importorskip("scipy.sparse.linalg")
    x_sp, info = sp.minres(Ad, np.asarray(b), rtol=1e-12)
    assert info == 0
    assert np.abs(np.asarray(x) - x_sp).max() < 1e-7


def test_minres_spd_preconditioner():
    Ad = _spd(seed=4) + np.diag(np.linspace(0, 500, 40))  # spread the diagonal so Jacobi bites
    b = _b(40)
    op = _op(Ad)
    inv = 1.0 / jnp.asarray(np.diag(Ad))
    x = jno.solve.minres(tol=1e-12)(op, b, M=lambda v: inv * v)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9


def test_fgmres_nonsymmetric_and_restarted():
    Ad = _nonsym()
    b = _b(Ad.shape[0])
    x_ref = np.linalg.solve(Ad, np.asarray(b))
    x = jno.solve.fgmres(tol=1e-12)(_op(Ad), b)
    assert np.abs(np.asarray(x) - x_ref).max() < 1e-9
    # restart << n exercises the outer cycle loop
    x_r = jno.solve.fgmres(tol=1e-12, restart=7)(_op(Ad), b)
    assert np.abs(np.asarray(x_r) - x_ref).max() < 1e-9


def test_fgmres_flexible_iterative_preconditioner():
    """The defining FGMRES property: M is itself an inexact Krylov solve (varies per call)."""
    Ad = _spd(seed=5)
    b = _b(Ad.shape[0])
    op = _op(Ad)
    M_inner = lambda v: jax.scipy.sparse.linalg.cg(op.mv, v, tol=1e-2, maxiter=5)[0]
    x = jno.solve.fgmres(tol=1e-12)(op, b, M=M_inner)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9


def test_chebyshev_solver_true_and_estimated_bounds():
    Ad = _spd(seed=6)
    b = _b(Ad.shape[0])
    x_ref = np.linalg.solve(Ad, np.asarray(b))
    lam = np.linalg.eigvalsh(Ad)
    x = jno.solve.chebyshev(lmin=float(lam[0]), lmax=float(lam[-1]), tol=1e-12)(_op(Ad), b)
    assert np.abs(np.asarray(x) - x_ref).max() < 1e-9
    x_auto = jno.solve.chebyshev(tol=1e-12, maxiter=2000)(_op(Ad), b)  # power-iteration bounds
    assert np.abs(np.asarray(x_auto) - x_ref).max() < 1e-8


def _spd_with_outliers(n=300, n_out=15, seed=21):
    """SPD with a spread bulk plus a few large outlying eigenvalues.

    The regime that motivates a low-rank preconditioner: Jacobi can rescale a spectrum but cannot
    *separate* a handful of large outliers, which is what stalls CG. A FEM operator gets this shape
    from a stiff coefficient contrast or a near-null-space."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.concatenate([np.linspace(2000.0, 300.0, n_out), np.linspace(1.0, 8.0, n - n_out)])
    A = (Q * lam) @ Q.T
    return 0.5 * (A + A.T)


def _cg_iters(Ad, b, M, xref, cap=500):
    """Smallest CG iteration count reaching 1e-10 relative error (``None`` if never)."""
    for k in range(1, cap + 1):
        x, _ = jax.scipy.sparse.linalg.cg(lambda v: Ad @ v, b, M=M, tol=0.0, atol=0.0, maxiter=k)
        if np.linalg.norm(np.asarray(x) - xref) / np.linalg.norm(xref) < 1e-10:
            return k
    return None


def test_nystrom_preconditioner_beats_jacobi_on_outlying_eigenvalues():
    """Randomized Nyström (Frangella, Tropp & Udell 2023, Alg. 2.1 + §3) deflates the top of the
    spectrum — the part a diagonal preconditioner cannot reach.

    On a spectrum with large outliers Jacobi is not merely weak, it is *worse than nothing*
    (measured: 124 iterations vs 98 unpreconditioned), because rescaling cannot separate outliers.
    A rank-20 sketch captures the 15 outliers and roughly halves the unpreconditioned count, for a
    setup cost of 20 matvecs."""
    from jno.utils.solver.solver_api import PrecondContext, materialize_precond

    Ad = jnp.asarray(_spd_with_outliers())
    b = _b(Ad.shape[0], seed=22)
    xref = np.linalg.solve(np.asarray(Ad), np.asarray(b))
    op = _op(Ad)

    plain = _cg_iters(Ad, b, None, xref)
    jac = _cg_iters(Ad, b, materialize_precond(jno.precond.jacobi(), PrecondContext(op)), xref)
    nys = _cg_iters(Ad, b, materialize_precond(jno.precond.nystrom(rank=20), PrecondContext(op)), xref)
    assert None not in (plain, jac, nys), f"all three must converge, got {plain}, {jac}, {nys}"
    assert nys < 0.6 * plain, f"nystrom({nys}) should roughly halve unpreconditioned CG ({plain})"
    assert nys < jac, f"nystrom({nys}) must beat jacobi({jac}) where the spectrum has outliers"

    # more rank captures more of the spectrum -> never worse
    nys40 = _cg_iters(Ad, b, materialize_precond(jno.precond.nystrom(rank=40), PrecondContext(op)), xref)
    assert nys40 <= nys, f"rank 40 ({nys40}) should not be worse than rank 20 ({nys})"


def test_nystrom_applier_is_linear_symmetric_and_reproducible():
    """Contract required to precondition CG: the application must be a *linear* symmetric operator
    (a fixed sketch, no per-call randomness), and a fixed ``seed`` must reproduce it exactly."""
    from jno.utils.solver.solver_api import PrecondContext, materialize_precond

    Ad = jnp.asarray(_spd_with_outliers(n=120, seed=23))
    op = _op(Ad)
    M = materialize_precond(jno.precond.nystrom(rank=12, seed=3), PrecondContext(op))
    v, w = _b(120, seed=24), _b(120, seed=25)

    lin = np.asarray(M(2.0 * v + w) - 2.0 * M(v) - M(w))
    assert np.abs(lin).max() < 1e-9, "the preconditioner application must be linear"
    # symmetry: <Mv, w> == <v, Mw>
    assert abs(float(M(v) @ w - v @ M(w))) < 1e-9, "the Nystrom preconditioner must be symmetric"
    # the same seed reproduces the sketch exactly; a different one still gives a valid operator
    M_same = materialize_precond(jno.precond.nystrom(rank=12, seed=3), PrecondContext(op))
    assert np.abs(np.asarray(M(v) - M_same(v))).max() < 1e-12, "seed must make the sketch reproducible"
    M_other = materialize_precond(jno.precond.nystrom(rank=12, seed=99), PrecondContext(op))
    assert np.all(np.isfinite(np.asarray(M_other(v))))


def test_nystrom_rejects_ranks_and_operators_it_cannot_serve():
    """Extremes, each failing loud rather than silently wasting work or returning garbage."""
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

    Ad = jnp.asarray(_spd(n=30, seed=26))
    ctx = PrecondContext(_op(Ad))
    # rank >= n costs more matvecs than a direct solve -- refuse rather than "work" pointlessly
    with pytest.raises(ValueError, match="rank"):
        materialize_precond(jno.precond.nystrom(rank=30), ctx)
    with pytest.raises(ValueError, match="rank"):
        materialize_precond(jno.precond.nystrom(rank=64), ctx)
    # a matvec-only operator has no shape to size the sketch from
    mv_only = LinearOperator.from_matvec(lambda v: Ad @ v)
    if getattr(mv_only, "shape", None) is None:
        with pytest.raises(TypeError, match="shape|matvec-only"):
            materialize_precond(jno.precond.nystrom(rank=4), PrecondContext(mv_only))
    # the smallest useful sketch still produces a working, finite operator
    M1 = materialize_precond(jno.precond.nystrom(rank=1), ctx)
    assert np.all(np.isfinite(np.asarray(M1(_b(30, seed=27)))))


def test_lanczos_bounds_bracket_the_spectrum_where_the_ratio_guess_does_not():
    """Both ends of the spectrum, measured rather than guessed (Lanczos 1950 §II).

    The historical estimate took ``lmax`` from power iteration and *fabricated*
    ``lmin = lmax / 30``. A Chebyshev polynomial is a contraction only **inside** the interval it
    is fitted to, so whenever the true ``lmin/lmax`` ratio is smaller than the assumed 1/30 the
    fabricated lower end sits above the bottom of the spectrum and the lowest modes are amplified
    instead of damped. This matrix is built with a true ratio (~1/80) well below the guess, which
    is exactly the regime the guess gets wrong.

    Pins that Lanczos recovers both ends closely, that the guess does not, and that the guess errs
    in the dangerous direction (too *high* a lower bound, i.e. modes left outside the interval)."""
    from jno.utils.solver.krylov import lanczos_spectrum_bounds, power_iteration_bound

    pytest.importorskip("matfree", reason="Lanczos bounds need the optional matfree package")
    n = 160
    key = jax.random.PRNGKey(11)
    B = jax.random.normal(key, (n, n))
    Ad = B @ B.T / n + 0.05 * jnp.eye(n)  # SPD with a wide spectrum
    true = np.linalg.eigvalsh(np.asarray(Ad))
    lo_t, hi_t = float(true[0]), float(true[-1])
    assert lo_t / hi_t < 1.0 / 30.0, "fixture must sit in the regime the lmax/30 guess gets wrong"

    got = lanczos_spectrum_bounds(lambda v: Ad @ v, n, iters=40)
    assert got is not None, "matfree is installed, so Lanczos must produce bounds"
    lo_l, hi_l = float(got[0]), float(got[1])
    assert abs(hi_l - hi_t) / hi_t < 1e-3, f"Lanczos lmax {hi_l:.5f} vs true {hi_t:.5f}"
    assert abs(lo_l - lo_t) / lo_t < 0.25, f"Lanczos lmin {lo_l:.5f} vs true {lo_t:.5f}"
    # Ritz values are interior to the true spectrum -- never outside it
    assert lo_l >= lo_t - 1e-9 and hi_l <= hi_t + 1e-9, "Ritz values must bracket from the inside"

    # the guess it replaces: lmax is fine, lmin lands ABOVE the true smallest eigenvalue
    hi_p = float(power_iteration_bound(lambda v: Ad @ v, n, iters=40))
    lo_guess = hi_p / 30.0
    assert lo_guess > lo_t, "the fixture is only interesting if the guess overshoots the true lmin"
    assert abs(lo_l - lo_t) < abs(lo_guess - lo_t), (
        f"Lanczos lmin {lo_l:.5f} must beat the guess {lo_guess:.5f} against true {lo_t:.5f}"
    )


def test_spectrum_bounds_honours_explicit_and_falls_back():
    """The bound chooser: caller-supplied ends always win, and a missing/degenerate Lanczos still
    yields a usable interval (the power-iteration fallback) rather than failing the solve."""
    from jno.utils.solver.krylov import lanczos_spectrum_bounds, spectrum_bounds

    Ad = _spd(seed=12)
    n = Ad.shape[0]
    mv = lambda v: Ad @ v  # noqa: E731

    lo, hi = spectrum_bounds(mv, n, lmin=0.25, lmax=4.0)
    assert (lo, hi) == (0.25, 4.0), "explicit bounds must pass through untouched"

    lo, hi = spectrum_bounds(mv, n, iters=30)  # estimated (Lanczos, or fallback)
    true = np.linalg.eigvalsh(Ad)
    assert 0.0 < lo < hi, f"estimated interval must be non-degenerate, got [{lo}, {hi}]"
    assert hi >= float(true[-1]) * 0.98, "lmax must not badly under-estimate the top of the spectrum"

    # a 1-DOF operator is the degenerate extreme: Lanczos cannot run, the fallback must still work
    one = jnp.asarray([[3.0]])
    assert lanczos_spectrum_bounds(lambda v: one @ v, 1, iters=8) is None
    lo1, hi1 = spectrum_bounds(lambda v: one @ v, 1, iters=8)
    assert 0.0 < lo1 < hi1 and hi1 >= 3.0 * 0.98


def test_chebyshev_polynomial_preconditioner_accelerates_cg():
    from jno.utils.solver.solver_api import PrecondContext, materialize_precond

    Ad = _spd(seed=7)
    b = _b(Ad.shape[0])
    op = _op(Ad)
    M = materialize_precond(jno.precond.chebyshev(degree=8), PrecondContext(op))
    x = jno.solve.cg(tol=1e-12)(op, b, M=M)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9
    # the fixed-degree application is linear in v (required for CG): p(A)(a v + w) = a p(A)v + p(A)w
    v, w = _b(40, seed=8), _b(40, seed=9)
    lin = np.asarray(M(2.0 * v + w) - 2.0 * M(v) - M(w))
    assert np.abs(lin).max() < 1e-10


# ---------------------------------------------------------------------------
# transforms: jit, vmap, grad (the custom_linear_solve firewall)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [lambda: jno.solve.fgmres(tol=1e-12), lambda: jno.solve.minres(tol=1e-12)])
def test_jit_and_vmap(factory):
    Ad = _sym_indefinite(seed=10)
    op = _op(Ad)
    solver = factory()
    b = _b(Ad.shape[0])
    x = jax.jit(lambda bb: solver(op, bb))(b)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-8
    B = jnp.stack([b, -b, 3.0 * b + 1.0])
    X = jax.vmap(lambda bb: solver(op, bb))(B)
    assert np.abs(np.asarray(X) - np.linalg.solve(Ad, np.asarray(B).T).T).max() < 1e-8


def test_grad_through_firewall_nonsymmetric():
    """Reverse-mode through fgmres on a NON-symmetric parametric system vs finite differences —
    exercises the transpose solve of the custom_linear_solve wrapper."""
    A0 = jnp.asarray(_nonsym(n=12, seed=11))
    P = jnp.asarray(np.random.default_rng(12).standard_normal((12, 12)))
    b = _b(12, seed=13)
    solver = jno.solve.fgmres(tol=1e-13)

    def loss(theta):
        op = jno.solve.LinearOperator(A0 + theta * P)
        return jnp.sum(solver(op, b) ** 2)

    g = float(jax.grad(loss)(0.3))
    eps = 1e-6
    g_fd = (float(loss(0.3 + eps)) - float(loss(0.3 - eps))) / (2 * eps)
    assert abs(g - g_fd) / (abs(g_fd) + 1e-12) < 1e-5


# ---------------------------------------------------------------------------
# end-to-end as fem.solve slots
# ---------------------------------------------------------------------------


def test_fem_solve_with_new_krylov_slots():
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    u_ref = np.asarray(fem.solve())
    for solver, precond in [
        (jno.solve.fgmres(), jno.precond.jacobi()),
        (jno.solve.minres(), jno.precond.jacobi()),  # Poisson stiffness is SPD: MINRES applies
        (jno.solve.cg(), jno.precond.chebyshev(degree=6)),
        (jno.solve.chebyshev(maxiter=2000), None),
        # the low-rank slot rides fem.solve like any other preconditioner (SPD Poisson operator)
        (jno.solve.cg(), jno.precond.nystrom(rank=12)),
    ]:
        uu = np.asarray(fem.solve(linear=solver, precond=precond))
        assert np.abs(uu - u_ref).max() < 1e-6, f"{solver.name} deviates on Poisson"
