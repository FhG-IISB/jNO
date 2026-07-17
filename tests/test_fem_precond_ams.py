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
