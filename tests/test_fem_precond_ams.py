"""``jno.precond.ams()`` — the H(curl) auxiliary-space Maxwell preconditioner, through ``fem.solve``.

M4a: the spec is exercised on the REAL curl-curl+β·mass system via the ordinary steady-linear path
(no complex-solve changes yet). It must reproduce the direct solve and converge in the handful of
iterations M1/M2 established — where the gradient near-null-space makes an un-preconditioned CG crawl.
"""

import contextlib

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import pytest

import jno

inner, vec = jno.np.inner, jno.np.vector
# The lu auxiliary runs on CPU (jax-spsolve/cuSolver is flaky on GPU — the same reason the complex
# eddy path is CPU-pinned); the GPU-native aux is a multigrid inner solver (M5). Pin the solves.
_CPU = next(iter(jax.devices("cpu")), None)
_ON_CPU = jax.default_device(_CPU) if _CPU else contextlib.nullcontext()


def _solve(fem, **kw):
    with _ON_CPU:
        return np.asarray(jnp.asarray(fem.solve(**kw))).reshape(-1)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _curlcurl_source(mesh_size, beta):
    """A driven real curl-curl (+ β·mass) N1E problem: inner(curl u, curl v) + β⟨u,v⟩ − ⟨Js, v⟩."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    Js = vec(0.0 * x, 0.0 * y, 1.0 + 0.0 * z)
    term = inner(cu, cv) + beta * inner(ui, vi) - inner(Js, vi)
    return jno.fem([term])


def _complex_eddy(mesh_size, freq, eps):
    """A complex eddy operator νK + jω(σ+ε)M with σ nonzero only on a sub-region (copper analog) and a
    small ε mass floor everywhere (the σ=0-in-air ε-gauge), plus a source in the conductor."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    sig = jno.np.where(x > 0.5, 1.0, 0.0)
    Js = vec(0.0 * x, 0.0 * x, sig)
    omega = 2 * np.pi * freq
    return jno.fem([inner(cu, cv) + 1j * omega * (sig + eps) * inner(ui, vi) - inner(Js, vi)])


def test_ams_complex_eddy_matches_sparse_lu():
    """M4b: the complex eddy operator solved by complex GMRES + AMS (through the new complex-direct
    wiring — sparse ``A_r + i·A_i``, never the dense 2n block) matches the sparse-direct solve. This is
    the σ=0-in-air ε-gauge regime, complex-symmetric ⇒ GMRES (not CG). Also the CI guard for the M5b
    complex→real auxiliary reformulation: the default ``lu`` aux here runs through it (factor-jω
    gradient + real 2n-block Π), so a passing solve confirms the reformulation is exact."""
    fem = _complex_eddy(0.3, freq=1e6, eps=1e-3)
    x_lu = _solve(fem)  # default: complex sparse-LU on the real-equivalent block
    x_ams = _solve(fem, linear=jno.solve.gmres(tol=1e-8, restart=120, maxiter=600), precond=jno.precond.ams())
    assert x_ams.dtype == np.complex128
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-8


def test_ams_complex_eddy_extreme_scale_matches_lu():
    """Regression for the extreme-magnitude Krylov breakdown. The *physical* eddy operator has huge
    absolute coefficients (ν ~ 1/μ₀ ~ 1e6, jωσ ~ jω·σ_cu ~ 1e12), and jax GMRES's Arnoldi breaks down
    on it — it returns ~0 (relative residual 1.0) *regardless of preconditioner* — unless the system is
    normalized to O(1) first. The slot path auto-scales the operator+RHS by a concrete scalar (solution-
    invariant), so a realistic-scale eddy solve reproduces the sparse-direct result. Every other test
    here uses O(1) coefficients, so they miss this; without the normalization x_ams would be ~0 and the
    solve would raise the residual-check error."""
    mu0 = 4 * np.pi * 1e-7
    nu, sigma, omega = 1.0 / mu0, 5.8e7, 2 * np.pi * 1e4  # ν~8e5, jωσ~3.6e12  ⇒  |A| ~ 1e12
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    sig = jno.np.where(x > 0.5, sigma, 0.0)  # conductor half; ε-gauge mass floor everywhere
    Js = vec(0.0 * x, 0.0 * x, jno.np.where(x > 0.5, 1.0, 0.0))
    fem = jno.fem([nu * inner(cu, cv) + 1j * omega * (sig + sigma * 1e-3) * inner(ui, vi) - inner(Js, vi)])
    x_lu = _solve(fem)
    x_ams = _solve(fem, linear=jno.solve.gmres(tol=1e-8, restart=200, maxiter=20), precond=jno.precond.ams())
    assert x_ams.dtype == np.complex128
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-7


def test_non_native_precond_complex_extreme_scale_does_not_break_down():
    """A non-complex-native precond (jacobi) on a complex problem solves the real-equivalent 2n block.
    That block must be handed to the slot solver as a SPARSE BCOO so the composed solver's extreme-scale
    normalization fires — otherwise the mass-dominated eddy block (|A| ~ 1e12) breaks the Krylov Arnoldi
    and the solve returns ~0 (relative residual 1.0 → raises). Jacobi is a weak preconditioner for the
    indefinite eddy block, so it does not reach tight accuracy (that is what AMS is for), but the solve
    now *converges to a real answer* instead of breaking down. Regression for the sparse-2n-block route."""
    mu0 = 4 * np.pi * 1e-7
    nu, sigma, omega = 1.0 / mu0, 5.8e7, 2 * np.pi * 1e4  # |A| ~ 1e12
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    sig = jno.np.where(x > 0.5, sigma, 0.0)
    Js = vec(0.0 * x, 0.0 * x, jno.np.where(x > 0.5, 1.0, 0.0))
    fem = jno.fem([nu * inner(cu, cv) + 1j * omega * (sig + sigma * 1e-3) * inner(ui, vi) - inner(Js, vi)])
    x_lu = _solve(fem)
    # Without the fix this raises (residual 1.0). With it, jacobi iterates to a real (if loose) solution.
    x_jac = _solve(fem, linear=jno.solve.gmres(tol=1e-8, restart=300, maxiter=40), precond=jno.precond.jacobi())
    assert np.all(np.isfinite(x_jac))
    assert np.linalg.norm(x_jac - x_lu) / np.linalg.norm(x_lu) < 0.1  # real solution, not the x≈0 breakdown


def test_ams_matches_direct_solve_through_fem_solve():
    """fem.solve(linear=cg, precond=ams()) reproduces the sparse-direct solve — the spec is wired
    end-to-end (prepare builds G/Π from the mesh, materialize assembles the nodal aux, CG converges)."""
    fem = _curlcurl_source(0.3, beta=1e-4)
    x_lu = _solve(fem, linear=jno.solve.lu())
    x_ams = _solve(fem, linear=jno.solve.cg(tol=1e-9, maxiter=400), precond=jno.precond.ams())
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-7


def test_ams_converges_in_a_small_iteration_budget():
    """The whole point: AMS collapses the count. It reaches tol in a budget (60) far below the
    hundreds an un-preconditioned/Jacobi CG needs on this gradient-dominated (β→small) operator."""
    fem = _curlcurl_source(0.3, beta=1e-5)
    x_lu = _solve(fem, linear=jno.solve.lu())
    x_ams = _solve(fem, linear=jno.solve.cg(tol=1e-7, maxiter=60), precond=jno.precond.ams())
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-5  # converged within the budget
    with pytest.raises(Exception):  # Jacobi cannot, in the same budget — the null-space is undamped
        _solve(fem, linear=jno.solve.cg(tol=1e-7, maxiter=60), precond=jno.precond.jacobi())


def test_ams_custom_aux_solver():
    """The auxiliary nodal solves are a pluggable ``jno.solve`` inner solver (default lu()); an
    iterative inner solver works too — the hook M5 swaps a multigrid solver into."""
    fem = _curlcurl_source(0.3, beta=1e-4)
    x_lu = _solve(fem, linear=jno.solve.lu())
    spec = jno.precond.ams(aux=jno.solve.cg(tol=1e-10, maxiter=500))
    x = _solve(fem, linear=jno.solve.cg(tol=1e-9, maxiter=400), precond=spec)
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_ams_inexact_aux_pairs_with_flexible_outer():
    """M5 mechanism: an *inexact/iterative* auxiliary solver (here a loose CG — a stand-in for a
    multigrid ``aux`` such as jaxamg/AmgX) is a **variable** preconditioner, so it must be driven by a
    **flexible** outer solver (fgmres). This is the exact path the scalable GPU AMG aux runs through;
    the real jaxamg aux is verified separately (it needs a GPU + the AmgX stack)."""
    fem = _curlcurl_source(0.3, beta=1e-4)
    x_lu = _solve(fem, linear=jno.solve.lu())
    inexact_aux = jno.solve.cg(tol=1e-3, maxiter=80)  # solved loosely → M varies iteration to iteration
    x = _solve(fem, linear=jno.solve.fgmres(tol=1e-7, restart=80, maxiter=400), precond=jno.precond.ams(aux=inexact_aux))
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-5


def test_ams_requires_a_gauge_on_pure_curl_curl():
    """A bare curl-curl has GᵀAG ≡ 0 (curl∘grad = 0) — no coercivity on the gradient space. The spec
    must refuse with actionable guidance rather than silently forming a singular auxiliary solve."""
    fem = _curlcurl_source(0.3, beta=0.0)
    with pytest.raises(ValueError, match="curl-curl"):
        _solve(fem, linear=jno.solve.cg(maxiter=50), precond=jno.precond.ams())


def test_ams_build_makes_the_solve_differentiable():
    """The eager ``.build(fem)`` hook runs the host aux-assembly ONCE outside the trace and freezes it,
    so an AMS-preconditioned solve runs — and **differentiates** — under a trace. The gradient flows by
    implicit differentiation through the traced operator (the frozen preconditioner only accelerates);
    it matches finite differences and the analytic ``d/ds ‖(sA)⁻¹b‖² = -2‖A⁻¹b‖²`` at ``s=1``."""
    import jax.experimental.sparse as jsp

    from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

    fem = _curlcurl_source(0.3, beta=1e-4)
    A0 = fem.operator[0]
    A0b = A0 if hasattr(A0, "indices") else jsp.BCOO.fromdense(jnp.asarray(A0))
    b = jnp.asarray(fem.operator[1]).reshape(-1)
    spec = jno.precond.ams().build(fem)  # eager host setup → frozen auxiliaries

    def loss(s):  # the operator s·A is traced in s → materialize must not host-assemble
        op = LinearOperator(jsp.BCOO((A0b.data * s, A0b.indices), shape=A0b.shape))
        M = materialize_precond(spec, PrecondContext(op, fem))
        return jnp.sum(jno.solve.fgmres(tol=1e-10, restart=100, maxiter=600)(op, b, M=M) ** 2)

    with _ON_CPU:
        val = float(loss(1.0))
        g = float(jax.grad(loss)(1.0))
        fd = float((loss(1.0 + 1e-5) - loss(1.0 - 1e-5)) / 2e-5)
    assert abs(g - fd) / abs(fd) < 1e-3  # matches finite differences
    assert abs(g - (-2.0 * val)) / abs(2.0 * val) < 1e-3  # and the analytic -2‖A⁻¹b‖²


def test_ams_reference_build_preconditions_a_parametric_solve():
    """Parametric-inverse (design-loop) use: the operator ``A(θ)`` is traced, so freeze the AMS
    auxiliaries once from a **concrete reference** at ``θ₀`` — ``jno.precond.ams().build(fem0)`` — and
    reuse that spec for the parametric solve. The parametric node evaluates to the forward solution (so
    the frozen preconditioner solved it correctly); it is a trace node, so ``∂/∂θ`` flows through it."""
    spec = jno.precond.ams().build(_curlcurl_source(0.35, beta=1e-3))  # concrete reference
    assert spec._frozen is not None

    beta = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="beta_ams")
    beta.initialize(jax.nn.initializers.constant(1e-3))
    beta.dtype(jnp.float64)
    fem_p = _curlcurl_source(0.35, beta)  # β is now a jno.np.parameter → parametric FemLinearSystem

    with _ON_CPU:
        node = fem_p.solve(linear=jno.solve.fgmres(tol=1e-9, restart=80, maxiter=400), precond=spec)
        assert not isinstance(node, jax.Array), "a parametric solve must be a differentiable trace node"
        u_ref = np.asarray(_curlcurl_source(0.35, 1e-3).solve(linear=jno.solve.lu())).reshape(-1)
        u = np.asarray(jno.core([(node - jnp.asarray(u_ref)).mae], domain=None).eval([node])).reshape(-1)
    assert np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref) < 1e-6


def test_ams_needs_the_owning_fem():
    """Materialised on a bare operator (no ctx.fem → no edge topology) it errors clearly, since G/Π
    are mesh objects, not recoverable from the matrix alone."""
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    n = 6
    op = LinearOperator(jsparse.BCOO.fromdense(jnp.eye(n)))
    with pytest.raises(TypeError, match="owning FEM"):
        jno.precond.ams().materialize(PrecondContext(op, None))


# ------------------------------------------------------------------------------------
# Driven time-harmonic Maxwell with SURFACE-ONLY absorption (impedance / first-order ABC).
#
# The gradient auxiliary used to keep only ``Im A_G``, on the eddy-case reasoning that GᵀKG = 0 makes
# A_G purely imaginary. For a driven wave problem that fails twice: Re A_G = -k₀²·GᵀεMG ≠ 0, and with
# no volume loss ``Im A_G`` is a *boundary* mass — identically zero on every interior node, hence
# singular. The aux solve returned garbage and the outer Krylov stalled at residual ~1 with NO error
# (the ``GᵀAG ≈ 0`` guard does not catch it). Inverting the full complex A_G fixes it.
# ------------------------------------------------------------------------------------
def _driven_maxwell(mesh_size, k0, volume_loss=0.0):
    """curl-curl − k₀²·mass, Silver–Müller impedance ABC on the whole boundary, plane-wave source."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
    xi, yi, zi, _ = d.variable("interior", split=True)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    nb = d.variable("boundary", normals=True)
    tu, tv = u.vector.cross(nb), v.vector.cross(nb)
    cb = d.variable("boundary", split=True)
    g = vec(1.0 + 0.0 * cb[0], 0.0 * cb[1], 0.0 * cb[2])
    vol = inner(cu, cv) - k0**2 * inner(ui, vi)
    if volume_loss:
        vol = vol + 1j * volume_loss * inner(ui, vi)
    return jno.fem([vol, 1j * k0 * inner(tu, tv), 2j * k0 * inner(g, tv)])


@pytest.mark.parametrize("k0", [1.0, 3.0])
def test_ams_driven_maxwell_surface_only_absorption(k0):
    """The regression: with absorption ONLY on the boundary, AMS must actually solve the system. Before
    the full-complex auxiliary this stalled silently — GMRES returned relative residual ~1 at every k₀."""
    fem = _driven_maxwell(0.3, k0)
    x_lu = _solve(fem)
    spec = jno.precond.ams().build(fem)
    x_ams = _solve(fem, linear=jno.solve.gmres(tol=1e-10, restart=60, maxiter=400), precond=spec)
    rel = np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu)
    assert rel < 1e-8, f"AMS did not solve the surface-absorbing wave problem (rel err {rel:.2e})"


def test_ams_driven_maxwell_volume_loss_still_works():
    """The other side of the fix: a volume imaginary mass (a physical ε'' — what a lossy dielectric
    actually has) was the only case that worked before, and must keep working."""
    fem = _driven_maxwell(0.3, 1.0, volume_loss=1e-3)
    x_lu = _solve(fem)
    spec = jno.precond.ams().build(fem)
    x_ams = _solve(fem, linear=jno.solve.gmres(tol=1e-10, restart=60, maxiter=400), precond=spec)
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-8


def test_ams_accepts_a_custom_aux_solver():
    """A user-supplied ``aux=`` solver takes the non-default (per-apply) branch of the build-once
    ``materialize`` — the same path ``aux=jno.solve.amg()`` (jaxamg on the GPU) uses. An explicit ``lu()``
    exercises it and must reproduce the default (frozen-SuperLU) answer."""
    fem = _driven_maxwell(0.3, 1.0)
    x_lu = _solve(fem)
    spec = jno.precond.ams(aux=jno.solve.lu()).build(fem)
    x_ams = _solve(fem, linear=jno.solve.gmres(tol=1e-10, restart=60, maxiter=400), precond=spec)
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-8


def test_built_ams_reaches_the_compiled_slot_path():
    """AMS applies a multi-level auxiliary solve on EVERY Krylov iteration, so leaving it on the eager
    path dispatches that whole structure from Python per iteration -- it had the most to lose of any
    spec from the class-level ``traceable = False`` (the same default made a built AMG hierarchy
    measure 21x slower than Jacobi).

    Traceability is a property of STATE here: unbuilt, ``materialize`` runs scipy on the host; built,
    every ingredient traces -- ``ctx.diag()``, the frozen G/Π constants, and the default auxiliary's
    SuperLU factor applied through ``jax.pure_callback``, which ``jit`` supports.
    """
    from jno.utils.solver.solver_api import _compilable

    fem = _curlcurl_source(0.4, beta=1e-4)
    cg = jno.solve.cg(tol=1e-9, maxiter=400)

    unbuilt = jno.precond.ams()
    assert unbuilt.traceable is False, "an unbuilt AMS still has to assemble its auxiliaries on the host"
    assert unbuilt.key is None
    assert not _compilable(cg, unbuilt)

    built = jno.precond.ams().build(fem)
    assert built.traceable is True, "a built AMS is frozen auxiliaries + a pure-JAX applier"
    assert built.key is not None
    assert _compilable(cg, built), "a built AMS must reach the compiled path -- that is the point"


def test_compiled_ams_gives_the_same_answer_as_eager():
    """Compiling the solve around AMS must not change what it returns -- checked against BOTH the
    eager AMS solve and the sparse-direct reference."""
    fem = _curlcurl_source(0.4, beta=1e-4)
    x_lu = _solve(fem, linear=jno.solve.lu())
    cg = jno.solve.cg(tol=1e-9, maxiter=400)
    x_eager = _solve(fem, linear=cg, precond=jno.precond.ams())
    x_compiled = _solve(fem, linear=cg, precond=jno.precond.ams().build(fem))
    assert np.linalg.norm(x_eager - x_lu) / np.linalg.norm(x_lu) < 1e-7
    assert np.linalg.norm(x_compiled - x_lu) / np.linalg.norm(x_lu) < 1e-7
    assert np.linalg.norm(x_compiled - x_eager) / np.linalg.norm(x_lu) < 1e-7


def test_ams_with_an_undeclared_aux_stays_eager():
    """A user ``aux`` is called on every apply, so AMS can only trace if that solver can. Anything
    that does not declare the ``jit`` trait keeps AMS eager -- costing speed, never correctness."""
    from jno.utils.solver.solver_api import _compilable

    fem = _curlcurl_source(0.4, beta=1e-4)
    cg = jno.solve.cg(tol=1e-9, maxiter=400)

    bare = jno.precond.ams(aux=lambda op, rhs: rhs).build(fem)  # a bare callable declares nothing
    assert bare.traceable is False
    assert not _compilable(cg, bare)

    declared = jno.precond.ams(aux=jno.solve.cg(tol=1e-8, maxiter=200)).build(fem)
    assert declared.traceable is True, "a jno.solve spec declares jit=True, so AMS may trace"
    assert _compilable(cg, declared)


def test_complex_ams_drives_a_flexible_outer():
    """Complex AMS through the FLEXIBLE outer solver, which needed fgmres to be complex-correct.

    fgmres orthogonalised with ``V @ w`` (bilinear, not Hermitian) and built Givens rotations from
    ``h**2``; on a complex system the same problem that now solves to ~1e-13 came back with relative
    error 1.0e+4. Nothing exercised complex fgmres, so nothing caught it -- this is the end-to-end
    guard, and ``test_fgmres_complex_matches_dense`` is the unit-level one.

    **Scope, measured rather than assumed:** this passes with the DEFAULT (exact, factored) auxiliary.
    An *inexact* Krylov aux -- the cheap auxiliary that would let complex AMS scale -- does NOT yet
    converge here: cg stalls at 7.6e-4 and bicgstab at 5.8e-4 relative residual, identically at aux
    tolerances 1e-3 and 1e-6, so the aux tolerance is not the limiter; the complex→real 2n auxiliary
    blocks are simply not yielding to a plain Krylov sweep inside its iteration budget. That wants a
    multigrid aux (the M5 jaxamg path), not a looser tolerance, and is not claimed here.
    """
    fem = _complex_eddy(0.3, freq=1e6, eps=1e-3)
    x_lu = _solve(fem, linear=jno.solve.lu())
    spec = jno.precond.ams().build(fem)
    x = _solve(fem, linear=jno.solve.fgmres(tol=1e-9, restart=120, maxiter=1200), precond=spec)
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_complex_ams_takes_an_amg_auxiliary():
    """The auxiliary that makes complex AMS cheap -- and the reason it did not work before.

    AMS splits every complex auxiliary into the real-equivalent 2n block ``[[Re,-Im],[Im,Re]]``,
    because AmgX-style multigrid is real-only. That block is skew-dominated by construction (measured
    ‖A-Aᵀ‖/‖A‖ = 2.0 with the mass term dominating), and algebraic multigrid cannot coarsen it:
    smoothed aggregation diverged to 1e+20, ``air_solver`` made no progress, Ruge-Stuben returned NaN.
    A plain Krylov aux fared no better -- cg stalls at 7.6e-4 and bicgstab at 5.8e-4, identically at
    aux tolerances 1e-3 and 1e-6, so the tolerance was never the limiter.

    The underlying COMPLEX auxiliary is exactly complex-symmetric (8e-17), where the same solver
    converges in 5-7 iterations. pyamg builds complex hierarchies natively, so an aux that declares
    ``complex_ok`` skips the reformulation and gets the block multigrid can actually solve.
    """
    fem = _complex_eddy(0.3, freq=1e6, eps=1e-3)
    x_lu = _solve(fem, linear=jno.solve.lu())
    spec = jno.precond.ams(aux=jno.precond.amg())
    x = _solve(fem, linear=jno.solve.fgmres(tol=1e-9, restart=120, maxiter=1200), precond=spec)
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_amg_aux_keeps_the_complex_blocks_unreformulated():
    """The mechanism, pinned directly: a ``complex_ok`` aux must leave the auxiliaries COMPLEX, and a
    real-only one must still get the 2n reformulation. Getting this backwards is silent -- the solve
    still runs, it just stalls."""
    fem = _complex_eddy(0.3, freq=1e6, eps=1e-3)

    amg_aux = jno.precond.ams(aux=jno.precond.amg()).build(fem)._frozen
    assert amg_aux["complex"] and amg_aux["aux_complex"]
    assert np.iscomplexobj(amg_aux["rg_csr"].data), "a complex-capable aux must get complex blocks"
    assert amg_aux["rg_csr"].shape[0] == amg_aux["rg_nv"], "not the doubled 2n block"

    default_aux = jno.precond.ams().build(fem)._frozen  # exact SuperLU: real-only path
    assert not default_aux["aux_complex"]
    assert default_aux["rg_csr"].shape[0] == 2 * default_aux["rg_nv"], "expected the real 2n block"


def test_amg_aux_still_lets_ams_compile():
    """An AMG aux is applied through jNO's own ``pure_callback`` wrapper, which ``jit`` supports, so it
    must not knock AMS back onto the eager path."""
    from jno.utils.solver.solver_api import _compilable

    fem = _curlcurl_source(0.4, beta=1e-4)
    spec = jno.precond.ams(aux=jno.precond.amg()).build(fem)
    assert spec.traceable is True
    assert _compilable(jno.solve.cg(tol=1e-9, maxiter=400), spec)
