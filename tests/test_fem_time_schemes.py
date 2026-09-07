"""Time-integration schemes via ``fem.solve(time=...)`` — ``jno.solve.theta`` and ``jno.solve.exponential``.

Oracle is transient heat with a fundamental-mode IC (``sin πx sin πy``), which decays as ``e^{-2π²t}``.
The defining property of the exponential integrator is that it is **exact in time**: its answer is
independent of the number of steps, where backward-Euler's is not — and it beats backward-Euler at a
fixed (coarse) step count. The θ-scheme test checks the override (θ=1 ≡ default, θ=½ differs, both valid).
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import box

import jno

_HAS_MATFREE = importlib.util.find_spec("matfree") is not None
PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nsteps, T=0.03, h=0.11):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h, time=(0.0, T, nsteps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def _advection_diffusion(nsteps, T=0.02, h=0.14, beta=6.0):
    """Transient **advection–diffusion** — the convection term ``β·∂ₓu`` makes ``A`` non-symmetric."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h, time=(0.0, T, nsteps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return jno.fem([ui.t * vi + beta * ui.x * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def _final(fem, **kw):
    return np.asarray(fem.solve(**kw).fn())[-1]


def test_theta_override():
    """θ=1 reproduces the default (backward-Euler); θ=½ (Crank–Nicolson) gives a different, valid answer."""
    default = _final(_heat(6))
    be = _final(_heat(6), time=jno.solve.theta(1.0))
    cn = _final(_heat(6), time=jno.solve.theta(0.5))
    assert np.allclose(be, default, atol=1e-10)  # θ=1 ≡ the assembly default
    assert not np.allclose(cn, default, atol=1e-4)  # θ=½ is a genuinely different scheme
    assert np.isfinite(cn).all()


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_is_exact_in_time():
    """The exponential integrator is *exact in time*: its answer does not depend on the step count, where
    backward-Euler's does. exp(2 steps) ≈ exp(8 steps); BE(2) is far from BE(8)."""
    e2 = _final(_heat(2), time=jno.solve.exponential(order=40))
    e8 = _final(_heat(8), time=jno.solve.exponential(order=40))
    assert np.linalg.norm(e2 - e8) / np.linalg.norm(e8) < 1e-5  # step-independent ⇒ exact in time

    b2, b8 = _final(_heat(2)), _final(_heat(8))
    assert np.linalg.norm(b2 - b8) / np.linalg.norm(b8) > 5e-2  # backward-Euler depends strongly on the step


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_beats_backward_euler():
    """At a fixed coarse step count the exponential integrator is much closer to the time-converged answer."""
    ref = _final(_heat(400))  # time-converged reference
    exp_err = np.linalg.norm(_final(_heat(4), time=jno.solve.exponential(order=40)) - ref) / np.linalg.norm(ref)
    be_err = np.linalg.norm(_final(_heat(4)) - ref) / np.linalg.norm(ref)
    assert exp_err < 0.5 * be_err  # exact-in-time wins at coarse steps


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_consistent_mass_is_more_accurate():
    """``mass='consistent'`` (full M, no lumping error, matrix-free M-inner-product Lanczos) is closer to
    the time-converged reference than ``mass='lumped'`` — and is still exact in time (step-independent)."""
    h = 0.16  # coarse: the consistent path runs a CG M-solve per Lanczos step
    ref = _final(_heat(300, h=h))
    lump = np.linalg.norm(_final(_heat(4, h=h), time=jno.solve.exponential(mass="lumped")) - ref) / np.linalg.norm(ref)
    cons = np.linalg.norm(
        _final(_heat(4, h=h), time=jno.solve.exponential(mass="consistent", order=30)) - ref
    ) / np.linalg.norm(ref)
    assert cons < lump  # consistent mass removes the lumping error

    c4 = _final(_heat(4, h=h), time=jno.solve.exponential(mass="consistent", order=30))
    c8 = _final(_heat(8, h=h), time=jno.solve.exponential(mass="consistent", order=30))
    assert np.linalg.norm(c4 - c8) / np.linalg.norm(c8) < 1e-5  # still exact in time


def test_m_inner_product_lanczos_matches_dense_and_differentiates():
    """The matrix-free M-inner-product Lanczos (the consistent-mass engine) computes ``f(M⁻¹A)·v``
    exactly and is differentiable — pure JAX, no host factorization, so consistent mass is scalable
    *and* autodiff-friendly (unlike a dense Cholesky with a concrete interior extraction)."""
    from jno.utils.solver.mass import m_inner_funm

    rng = np.random.default_rng(0)
    n = 50
    Bm = rng.standard_normal((n, n))
    M = jnp.asarray(Bm @ Bm.T + n * np.eye(n))
    Ba = rng.standard_normal((n, n))
    A = jnp.asarray(Ba @ Ba.T + np.eye(n))
    Minv = jnp.linalg.inv(M)
    L = Minv @ A
    m_inner = lambda a, b: a @ (M @ b)
    ones = jnp.ones(n)
    e0 = ones / jnp.sqrt(m_inner(ones, ones))
    v = jnp.asarray(rng.standard_normal(n))
    t = 0.05

    fv = m_inner_funm(lambda x: Minv @ (A @ x), m_inner, e0, v, lambda lam: jnp.exp(-t * lam), order=40)
    true = jax.scipy.linalg.expm(-t * L) @ v
    assert float(jnp.linalg.norm(fv - true) / jnp.linalg.norm(true)) < 1e-8  # exact vs dense expm

    def loss(scale):
        fw = m_inner_funm(lambda x: scale * (Minv @ (A @ x)), m_inner, e0, v, lambda lam: jnp.exp(-t * lam), order=40)
        return jnp.sum(fw**2)

    g = float(jax.grad(loss)(1.0))
    fd = float((loss(1.0 + 1e-5) - loss(1.0 - 1e-5)) / 2e-5)
    assert abs(g - fd) / abs(fd) < 1e-3  # gradient flows through the matrix-free Lanczos


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_nonsymmetric_advection_diffusion():
    """``symmetric=False`` advances a **non-symmetric** advection–diffusion operator with an Arnoldi + Padé
    exponential: exact in time (step-independent) and closer to the time-converged reference than backward
    Euler. The symmetric path is *invalid* here (it assumes ``A = Aᵀ``), so it must be visibly worse."""
    ns = jno.solve.exponential(symmetric=False, order=40)
    e3 = _final(_advection_diffusion(3), time=ns)
    e9 = _final(_advection_diffusion(9), time=ns)
    assert np.linalg.norm(e3 - e9) / np.linalg.norm(e9) < 1e-5  # exact in time

    ref = _final(_advection_diffusion(300))  # fine backward-Euler reference
    exp_err = np.linalg.norm(_final(_advection_diffusion(4), time=ns) - ref) / np.linalg.norm(ref)
    be_err = np.linalg.norm(_final(_advection_diffusion(4)) - ref) / np.linalg.norm(ref)
    sym_err = np.linalg.norm(_final(_advection_diffusion(4), time=jno.solve.exponential(order=40)) - ref) / np.linalg.norm(
        ref
    )
    assert exp_err < be_err  # exact-in-time beats backward Euler at coarse steps
    assert sym_err > 3 * exp_err  # the symmetric path is wrong on a non-symmetric operator


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_nonsymmetric_is_differentiable():
    """A gradient flows through the non-symmetric (Arnoldi + Padé) exponential integrator — the whole point:
    differentiable transport for inverse problems. ``∂/∂β`` of the final-state energy matches central FD."""
    ns = jno.solve.exponential(symmetric=False, order=40)

    def loss(beta):
        return jnp.sum(jnp.asarray(_advection_diffusion(3, beta=beta).solve(time=ns).fn())[-1] ** 2)

    g = float(jax.grad(loss)(6.0))
    fd = (float(loss(6.0 + 1e-4)) - float(loss(6.0 - 1e-4))) / 2e-4
    assert abs(g - fd) / abs(fd) < 1e-4


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_nonsymmetric_time_varying_forcing_is_step_independent():
    """The non-symmetric (Arnoldi + Padé) path also integrates a TIME-VARYING source by ETD2 — a ramp row
    in the augmented generator. For an AFFINE-in-time source it stays exact-in-time, so a coarse march
    equals a fine one on the non-symmetric advection–diffusion operator (backward Euler would not)."""

    def build(nsteps):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.14, time=(0.0, 0.02, nsteps))
        u, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
        f = jno.np.sin(PI * xi) * jno.np.sin(PI * yi) * (2.0 + 30.0 * ti)  # source affine in t
        return jno.fem(
            [ui.t * vi + 6.0 * ui.x * vi + ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic]
        )

    ns = jno.solve.exponential(symmetric=False, order=40)
    coarse, fine = _final(build(3), time=ns), _final(build(12), time=ns)
    assert np.linalg.norm(coarse - fine) / np.linalg.norm(fine) < 1e-6, (
        "non-symmetric ETD2 step-independent for affine f(t)"
    )
    assert np.abs(fine).max() > 1e-3  # a non-trivial forced solution (a real oracle)


def test_transient_with_source_converges_to_manufactured_solution():
    """A transient heat problem **with a source term** must converge to the exact manufactured solution.

    Regression: the per-step linear solve must be preconditioned. The composed default was a *bare*
    ``bicgstab()`` (no preconditioner) that silently under-converged each step — invisible on homogeneous
    decay (the warm-start already sits near the answer) but ~30% wrong on any forced/growing solution.
    Constant source ``f=SS`` ⇒ ``u=SS·(1−e^{−2π²T})/(2π²)``; time-varying ``f=SS(1+2π²t)`` ⇒ ``u=SS·t``."""
    T = 0.3

    def build(time_varying):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06, time=(0.0, T, 60))
        u, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        ss = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
        f = ss * (1.0 + 2.0 * PI**2 * ti) if time_varying else ss
        return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 0.0])

    for time_varying in (False, True):
        fem = build(time_varying)
        pts = np.asarray(fem.points)
        ss = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
        exact = ss * T if time_varying else ss * (1.0 - np.exp(-2.0 * PI**2 * T)) / (2.0 * PI**2)
        for scheme in (None, jno.solve.theta(0.5), jno.solve.theta(1.0)):
            got = _final(fem) if scheme is None else _final(fem, time=scheme)
            assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 5e-3, (time_varying, scheme)


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_handles_time_varying_forcing_via_etd2():
    """The exponential integrator now integrates a TIME-VARYING source by ETD2 (the exponential trapezoidal
    rule): the source is sampled at both ends of each step and its ramp rides a φ₂ weight. For a source
    AFFINE in time it stays exact-in-time, so it (a) recovers the manufactured solution u = ss·t to the
    spatial-discretisation tolerance and (b) is step-count independent — a coarse 4-step march equals a
    fine 40-step one (where backward Euler needs many steps). ``f = ss·(1 + 2π²t)`` ⇒ ``u = ss·t``."""

    def build(nstep):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06, time=(0.0, 0.3, nstep))
        u, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        ss = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
        f = ss * (1.0 + 2.0 * PI**2 * ti)  # affine in t → exact continuous solution u = ss·t
        return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 0.0])

    fem = build(40)
    pts = np.asarray(fem.points)
    exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1]) * 0.3  # u(T=0.3) = ss·T
    exp = jno.solve.exponential(mass="consistent", order=40)
    fine = _final(fem, time=exp)
    assert np.linalg.norm(fine - exact) / np.linalg.norm(exact) < 5e-3, "ETD2 must recover the manufactured u = ss·t"
    coarse = _final(build(4), time=exp)  # exact-in-time for affine forcing ⇒ step-count independent
    assert np.linalg.norm(coarse - fine) / np.linalg.norm(fine) < 1e-6, (
        "ETD2 is step-count independent for an affine source"
    )


def test_time_scheme_rejects_a_steady_problem():
    """``time=`` selects a TIME integrator, so it is an error on a non-transient problem."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi])
    with pytest.raises(ValueError, match="transient"):
        fem.solve(time=jno.solve.theta(0.5))


# ======================================================================================
# BDF2 -- second order AND L-stable, which the theta-method cannot be at once
# ======================================================================================
def _heat_exact(fem, T=0.03):
    """The analytic decay of the first eigenmode, sampled at the FE nodes."""
    pts = np.asarray(fem.points)
    return np.exp(-2 * PI**2 * T) * np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])


def test_bdf2_is_second_order_in_time():
    """Halving dt must quarter the time error. Measured against a fine-dt reference on the SAME mesh,
    so the (fixed) spatial error cancels and only the temporal rate is under test."""
    ref = _final(_heat(400), time=jno.solve.bdf2())
    errs = []
    for n in (10, 20, 40):
        errs.append(float(np.linalg.norm(_final(_heat(n), time=jno.solve.bdf2()) - ref)))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert min(rates) > 1.7, f"BDF2 must show ~2nd order, got rates {rates} from errors {errs}"


def test_bdf2_beats_backward_euler_at_the_same_cost():
    """Same number of steps, same per-step work -- the accuracy is the whole difference."""
    ref = _final(_heat(400), time=jno.solve.bdf2())
    e_be = float(np.linalg.norm(_final(_heat(16), time=jno.solve.theta(1.0)) - ref))
    e_b2 = float(np.linalg.norm(_final(_heat(16), time=jno.solve.bdf2()) - ref))
    assert e_b2 < e_be / 5.0, f"BDF2 {e_b2:.3e} vs backward Euler {e_be:.3e}"


def _rough_heat(nsteps, T, h=0.14):
    """Heat with IC = 1 everywhere against a u = 0 boundary. The incompatibility excites the STIFFEST
    modes of the discrete operator, which is where a merely A-stable scheme misbehaves."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h, time=(0.0, T, nsteps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])


def test_bdf2_damps_the_stiff_modes_crank_nicolson_rings_on():
    """**The reason the scheme exists.** Second order and L-stable at once, which the theta-method
    cannot manage: theta=1 is L-stable but first order, and theta=1/2 (Crank-Nicolson) is second order
    but only A-stable -- its amplification factor tends to -1 for a stiff mode, so the mode does not
    decay, it alternates in sign at full amplitude.

    Measured with 8 steps over T = 0.5, an initial condition incompatible with the boundary (so the
    stiffest modes are excited), and a solution that is physically dead long before the end:

        Crank-Nicolson   min u = -1.00   (the IC amplitude -- undecayed, just inverted)
        BDF2             min u = -0.023

    A 43x smaller spurious excursion, and the final field is decayed rather than ringing. It is stated
    as an amplitude and not as "no oscillation": BDF2 is not monotone either, it is *damped*.
    """
    n, T = 8, 0.5
    cn = np.asarray(_rough_heat(n, T).solve(time=jno.solve.theta(0.5)).fn())
    b2 = np.asarray(_rough_heat(n, T).solve(time=jno.solve.bdf2()).fn())

    assert cn.min() < -0.5, f"Crank-Nicolson must actually ring here ({cn.min():.3f}), or this proves nothing"
    assert b2.min() > -0.05, f"BDF2 must damp rather than ring ({b2.min():.3f})"
    assert abs(cn.min()) > 20.0 * abs(b2.min())
    # and the stiff content is gone by the end, where CN still carries it
    assert np.abs(b2[-1]).max() < 0.2 * np.abs(cn[-1]).max()


def test_bdf2_composes_with_the_solver_slots():
    """`linear=`/`precond=` reach the per-step solve through BDF2 exactly as through the theta march --
    it reuses `block.step`, so there is one implementation and it cannot drift."""
    plain = _final(_heat(12), time=jno.solve.bdf2())
    for kw in (
        dict(linear=jno.solve.lu()),
        dict(linear=jno.solve.gmres(tol=1e-12)),
        dict(linear=jno.solve.bicgstab(tol=1e-12), precond=jno.precond.jacobi()),
    ):
        got = _final(_heat(12), time=jno.solve.bdf2(), **kw)
        rel = float(np.linalg.norm(got - plain) / np.linalg.norm(plain))
        assert rel < 1e-8, f"slot {kw} moved the answer by {rel:.2e}"


def test_bdf2_marches_a_nonsymmetric_block():
    """Advection-diffusion: `A` is non-symmetric, which is the case the per-step BiCGStab rescue in
    `block.step` exists for. BDF2 inherits it rather than re-implementing it."""
    ref = _final(_advection_diffusion(300), time=jno.solve.bdf2())
    got = _final(_advection_diffusion(30), time=jno.solve.bdf2())
    assert np.isfinite(got).all()
    assert float(np.linalg.norm(got - ref) / np.linalg.norm(ref)) < 5e-2


def test_bdf2_refuses_an_adaptive_step():
    """Step doubling sizes a ONE-step method; BDF2 needs two previous states. Refused by name rather
    than falling through to the base class's generic message."""
    with pytest.raises(NotImplementedError, match="two previous states"):
        jno.solve.bdf2().adaptive(rtol=1e-5)


def test_bdf2_refuses_a_steady_problem():
    """The `time=` slot itself checks this, so BDF2 gets it for free."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    with pytest.raises(ValueError, match="transient"):
        fem.solve(time=jno.solve.bdf2())


def test_the_default_save_grid_returns_the_trajectory_without_resampling(monkeypatch):
    """`save_ts` defaults to the block's own grid, so the sampling is an identity -- and it must be
    taken as one. The general path allocates a second copy of the whole trajectory plus the blend
    workspace, which is what made a 6000-step x 18k-DOF case fail to allocate 5.72 GiB on an 8 GB card.

    Pinned because the check is inside a broad `except`: when this was refactored, a NameError in the
    comparison silently sent every march down the slow path with nothing said."""
    import jno.utils.solver.backend_blocks as bb

    orig, fired = bb._resample_trajectory, []

    def spy(traj, grid_ts, save_ts, dtype):
        out = orig(traj, grid_ts, save_ts, dtype)
        fired.append(out is traj)
        return out

    monkeypatch.setattr(bb, "_resample_trajectory", spy)
    for scheme in (jno.solve.theta(1.0), jno.solve.bdf2()):
        fired.clear()
        _heat(6).solve(time=scheme).fn()
        assert fired == [True], f"{scheme!r}: the identity fast path did not fire ({fired})"
