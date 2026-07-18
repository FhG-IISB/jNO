"""``jno.solve`` -- the callables-only solver namespace for ``fem.solve``'s slots.

Every factory returns a configured **callable** (no strings anywhere): a linear solver
``(A, b, *, M=None, x0=None) -> x`` or a nonlinear driver ``(residual_fn, u0, *,
linear_solve=None) -> u``. All shipped solvers are pure JAX -- ``jit``- and ``vmap``-native,
differentiable through ``lax.custom_linear_solve`` / ``lax.custom_root`` -- and they *reuse*
existing implementations (``jax.scipy.sparse.linalg`` Krylov, the differentiable sparse-direct
``spsolve``) rather than re-implementing them. A user-written callable with the same signature
drops into the same slot; if it is pure JAX it inherits every transform automatically.

Usage::

    fem.solve(linear=jno.solve.cg(tol=1e-10), precond=jno.precond.jacobi())
    fem.solve(linear=jno.solve.lu())                      # differentiable sparse-direct
    fem.solve(nonlinear=jno.solve.newton(), x0=u_guess)   # warm-started Newton-Krylov

Defaults when a slot is ``None`` are unchanged from the historic behaviour: Jacobi-preconditioned
BiCGStab (steady linear) and Jacobian-free Newton-Krylov (nonlinear).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from .utils.solver.solver_api import (
    LinearOperator,
    LinearSolver,
    NonlinearSolver,
    PrecondApplier,
    _maybe_residual_check,
)

__all__ = [
    "LinearOperator",
    "LinearSolver",
    "NonlinearSolver",
    "lu",
    "dense",
    "cg",
    "bicgstab",
    "gmres",
    "fgmres",
    "minres",
    "chebyshev",
    "amg",
    "newton",
    "picard",
    "eigs",
    "logdet",
    "trace",
    "applyfun",
    "diagonal",
    "lstsq",
    "theta",
    "exponential",
]


def lu() -> LinearSolver:
    """Differentiable sparse-direct solve (JAX ``spsolve``: cuSolver on GPU, native LU on CPU).

    Wraps the existing :func:`jno.utils.solver.linear.sparse_lu_solve` -- robust on the
    indefinite saddle-point systems where Jacobi-preconditioned Krylov stalls, reverse-mode
    differentiable in the matrix entries and the right-hand side. Direct: ignores ``x0`` and
    rejects a preconditioner. ``jit`` yes; **no vmap batching rule upstream** (trait
    ``vmap="no"``) -- use a Krylov solver inside vmapped/batched solves.
    """

    def _fn(op: LinearOperator, b, *, M, x0):
        from .utils.solver.linear import sparse_lu_solve

        if op.bcoo is not None:
            return sparse_lu_solve(op.bcoo, b)
        # a dense operator gets the dense direct solve — BCOO.fromdense would need a concrete
        # nse, which does not exist under jit/vmap tracing
        return jnp.linalg.solve(op.dense(), b)

    return LinearSolver(_fn, name="lu", direct=True, traits={"vmap": "no"})


def dense() -> LinearSolver:
    """Dense LAPACK solve (``jnp.linalg.solve``) on the densified operator.

    ``O(N^2)`` memory / ``O(N^3)`` time -- the right answer for small systems and coarse
    blocks, and the only shipped direct solver with a native vmap batching rule. Direct:
    ignores ``x0``, rejects a preconditioner.
    """

    def _fn(op: LinearOperator, b, *, M, x0):
        return jnp.linalg.solve(op.dense(), b)

    return LinearSolver(_fn, name="dense", direct=True)


def _krylov(name: str, tol: float, atol: float, maxiter: Optional[int], **fixed):
    def _fn(op: LinearOperator, b, *, M, x0):
        method = getattr(jax.scipy.sparse.linalg, name)
        x, _info = method(op.mv, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M, **fixed)
        return _maybe_residual_check(op, b, x, name)

    return LinearSolver(_fn, name=name)


def cg(*, tol: float = 1e-8, atol: float = 0.0, maxiter: Optional[int] = 20_000) -> LinearSolver:
    """Conjugate gradients (``jax.scipy.sparse.linalg.cg``) -- **symmetric positive-definite**
    systems only (Poisson, elasticity, mass matrices). Cheapest per iteration; takes ``M`` and
    ``x0``. Implicitly differentiable upstream via ``lax.custom_linear_solve``."""
    return _krylov("cg", tol, atol, maxiter)


def bicgstab(*, tol: float = 1e-8, atol: float = 0.0, maxiter: Optional[int] = 20_000) -> LinearSolver:
    """BiCGStab (``jax.scipy.sparse.linalg.bicgstab``) -- general non-symmetric systems.
    With ``precond=jno.precond.jacobi()`` this reproduces the historic ``fem.solve()``
    steady-linear default exactly."""
    return _krylov("bicgstab", tol, atol, maxiter)


def gmres(*, tol: float = 1e-8, atol: float = 0.0, maxiter: Optional[int] = None, restart: int = 30) -> LinearSolver:
    """Restarted GMRES (``jax.scipy.sparse.linalg.gmres``) -- non-symmetric systems where
    BiCGStab's erratic convergence hurts; memory grows with ``restart``. For an *iterative*
    (e.g. multigrid-with-tolerance) preconditioner a flexible variant (FGMRES) is required --
    planned; see ``plans/fem-solver-api.md``."""
    return _krylov("gmres", tol, atol, maxiter, restart=restart, solve_method="batched")


def _firewalled(raw, op: LinearOperator, b, *, M, x0, symmetric: bool, name: str):
    """Run a raw (non-differentiable) iteration inside ``lax.custom_linear_solve``.

    The differentiability firewall: gradients w.r.t. ``b`` and anything the matvec closes over
    come from the implicit transpose solve — the loop itself is never differentiated, and the
    preconditioner needs no gradient path at all. The transpose solve runs the same iteration
    on ``A^T`` (on ``A`` itself when ``symmetric``), reusing ``M`` (legitimate: a preconditioner
    only affects convergence speed, never the converged solution).
    """
    fwd = lambda _mv, rhs: raw(op.mv, rhs, M=M, x0=x0)
    if symmetric:
        rev = fwd
    else:
        # The reverse pass solves A^T y = v and MUST be preconditioned by M^T, not M: a
        # preconditioner never changes the converged solution, but for a non-symmetric M
        # (block-triangular Schur, ILU) M approximates A^{-1} and is near-useless for A^T, so the
        # adjoint Krylov solve runs almost unpreconditioned -- orders of magnitude more iterations
        # than the forward (empirically ~90x the forward step for a Taylor-Hood Navier-Stokes
        # step). jno.precond appliers carry a structural transpose (PrecondApplier.T); a bare
        # callable preconditioner has none, so we fall back to reusing M (correct, maybe slow).
        M_T = M.T if isinstance(M, PrecondApplier) else M
        rev = lambda _mv, rhs: raw(op.T.mv, rhs, M=M_T, x0=None)
    x = jax.lax.custom_linear_solve(op.mv, b, fwd, transpose_solve=rev, symmetric=symmetric)
    return _maybe_residual_check(op, b, x, name)


def fgmres(*, tol: float = 1e-8, restart: int = 30, maxiter: int = 1000) -> LinearSolver:
    """Flexible restarted GMRES (Saad 1993, Alg. 2.2; see
    :func:`jno.utils.solver.krylov.fgmres`) — the outer solver to use when the preconditioner is
    itself **iterative** (an inner Krylov sweep, a multigrid cycle with a tolerance, a block/Schur
    recipe with inexact inner solves), which plain GMRES's fixed-``M`` assumption forbids.
    Memory: two ``(restart, n)`` bases."""

    def _fn(op: LinearOperator, b, *, M, x0):
        from .utils.solver.krylov import fgmres as _raw

        raw = lambda mv, rhs, M, x0: _raw(mv, rhs, M=M, x0=x0, tol=tol, restart=restart, maxiter=maxiter)
        return _firewalled(raw, op, b, M=M, x0=x0, symmetric=False, name="fgmres")

    return LinearSolver(_fn, name="fgmres")


def minres(*, tol: float = 1e-8, maxiter: int = 2000) -> LinearSolver:
    """MINRES (Paige & Saunders 1975, §5; see :func:`jno.utils.solver.krylov.minres`) — the
    Krylov method for **symmetric indefinite** systems: Stokes/Biot saddle points, biharmonic
    (Argyris/Morley), shifted Helmholtz-like operators. Monotone residual where BiCGStab is
    erratic; ``O(1)`` memory where GMRES grows with ``restart``. The preconditioner must be
    symmetric positive definite even when ``A`` is indefinite."""

    def _fn(op: LinearOperator, b, *, M, x0):
        from .utils.solver.krylov import minres as _raw

        raw = lambda mv, rhs, M, x0: _raw(mv, rhs, M=M, x0=x0, tol=tol, maxiter=maxiter)
        return _firewalled(raw, op, b, M=M, x0=x0, symmetric=True, name="minres")

    return LinearSolver(_fn, name="minres")


def chebyshev(
    *,
    lmin: Optional[float] = None,
    lmax: Optional[float] = None,
    tol: float = 1e-8,
    maxiter: int = 500,
    bound_iters: int = 30,
    lmin_ratio: float = 1.0 / 30.0,
    safety: float = 1.05,
) -> LinearSolver:
    """Chebyshev semi-iteration for **SPD** systems (Golub & Varga 1961; Saad 2003 §12.3,
    Alg. 12.1; see :func:`jno.utils.solver.krylov.chebyshev_iteration`). Inner-product free —
    matvecs and AXPYs only, no reductions — so it shines under ``vmap`` and on GPU where CG's
    dot products serialise. Needs spectrum bounds of ``M^{-1} A``: pass ``lmin``/``lmax`` when
    known; otherwise ``lmax`` is estimated by ``bound_iters`` power-iteration steps (inflated by
    ``safety``) and ``lmin = lmin_ratio * lmax`` — a smoother-style default that converges but
    slower than true bounds; prefer real bounds for a *solver* use."""

    def _fn(op: LinearOperator, b, *, M, x0):
        from .utils.solver.krylov import chebyshev_iteration, power_iteration_bound

        hi = lmax
        if hi is None:
            hi = safety * power_iteration_bound(op.mv, b.shape[0], dtype=b.dtype, iters=bound_iters, M=M)
        lo = lmin if lmin is not None else lmin_ratio * hi
        raw = lambda mv, rhs, M, x0: chebyshev_iteration(mv, rhs, lmin=lo, lmax=hi, M=M, x0=x0, tol=tol, maxiter=maxiter)
        return _firewalled(raw, op, b, M=M, x0=x0, symmetric=True, name="chebyshev")

    return LinearSolver(_fn, name="chebyshev")


def _require_jaxamg():
    try:
        import jaxamg
    except ImportError as e:  # optional GPU dependency — see the message for the system requirements
        raise ImportError(
            "jno.solve.amg() needs the optional dependency `jaxamg` (NVIDIA AmgX wrapped as a JAX "
            "primitive). Install it with `pip install jax-neural-operators[amg]` (the `amg` extra); it "
            "also requires a prebuilt AmgX 2.5+, CUDA Toolkit 12+, JAX-with-CUDA, and an MPI stack "
            "(mpi4py / mpi4jax). Install those into your environment, then retry."
        ) from e
    return jaxamg


def amg(
    *, tol: float = 1e-6, maxiter: int = 500, krylov: Optional[str] = "PBICGSTAB", config: Optional[dict] = None
) -> LinearSolver:
    """GPU algebraic-multigrid solve via **jaxamg** (NVIDIA AmgX wrapped as a JAX primitive).

    A self-contained solver: it runs an AMG-preconditioned Krylov iteration (or pure AMG) entirely on
    the GPU/device — the on-device counterpart to the host-side pyamg used by ``jno.precond.amg``.
    Ideal for large H¹-elliptic systems (Poisson, diffusion, elasticity) and, via
    ``jno.precond.inner(jno.solve.amg(...))``, as the smoother inside an outer flexible-Krylov solve
    (e.g. the auxiliary nodal solves of a future H(curl) AMS preconditioner). Plain AMG will **not**
    converge on a raw curl-curl (H(curl)) system — that needs AMS on top.

    ``config`` (a full AmgX-format dict) overrides the convenience args entirely; otherwise a config is
    built from ``tol``/``maxiter``/``krylov`` (``krylov=None`` → pure AMG, else an AMG-preconditioned
    ``krylov`` Krylov, e.g. ``"PBICGSTAB"``/``"GMRES"``/``"PCG"``). Needs an **assembled** operator
    (hands the sparse matrix to jaxamg); errors on a matrix-free operator. Direct-style: takes no outer
    ``precond=`` (it owns its AMG preconditioner) and ignores ``x0``.

    Optional dependency — jaxamg is imported lazily; see :func:`_require_jaxamg` for the requirements.

    Reference: Liu, Fan & Wang, *JAX-AMG: A GPU-Accelerated Differentiable Sparse Linear Solver Library
    for JAX*, arXiv:2606.09001 (2026); wraps NVIDIA AmgX (Naumov et al., 2015).
    """

    def _fn(op: LinearOperator, b, *, M, x0):
        A = op.bcoo
        if A is None:
            raise ValueError(
                "jno.solve.amg() needs an assembled (sparse) operator — it cannot solve a matrix-free "
                "operator. Use a Krylov solver (cg/bicgstab/fgmres) for matrix-free systems."
            )
        jaxamg = _require_jaxamg()
        if config is not None:
            cfg = dict(config)
        elif krylov:
            cfg = {
                "solver": krylov,
                "preconditioner": {"solver": "AMG"},
                "tolerance": float(tol),
                "max_iters": int(maxiter),
            }
        else:
            cfg = {"solver": "AMG", "tolerance": float(tol), "max_iters": int(maxiter)}
        x, _info = jaxamg.solve(A, b, config=cfg)
        return jnp.asarray(x).reshape(-1)

    # AmgX owns the solve; treat as direct (no outer preconditioner). vmap left "no" (AmgX/MPI primitive
    # is not a pure-JAX batching op) — loop for batched solves.
    return LinearSolver(_fn, name="amg", direct=True, traits={"vmap": "no"})


def _root_driver(
    name, *, damping, rtol, atol, max_steps, inner_tol, inner_maxit, line_search, ls_max, ls_c
) -> NonlinearSolver:
    def _fn(residual_fn, u0, *, linear_solve):
        from .utils.solver.newton_krylov import newton_krylov

        return newton_krylov(
            residual_fn,
            u0,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            inner_tol=inner_tol,
            inner_maxit=inner_maxit,
            linear_solve=linear_solve,
            damping=damping,
            line_search=line_search,
            ls_max=ls_max,
            ls_c=ls_c,
        )

    return NonlinearSolver(_fn, name=name)


def newton(
    *,
    damping: float = 1.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 100,
    inner_tol: float = 1e-10,
    inner_maxit: int = 2000,
    line_search: bool = False,
    ls_max: int = 25,
    ls_c: float = 1e-4,
) -> NonlinearSolver:
    """Jacobian-free Newton-Krylov -- the (unchanged) nonlinear default, as a configurable slot.

    Wraps :func:`jno.utils.solver.newton_krylov.newton_krylov`: ``J @ v`` from a JVP, inner
    matrix-free solve (default BiCGStab, or the ``linear=`` slot when given), implicit
    differentiation via ``lax.custom_root`` so gradients reach parameters without unrolling.
    ``damping < 1`` relaxes each update for strongly nonlinear residuals; ``line_search=True`` adds
    residual-norm Armijo backtracking (up to ``ls_max`` halvings, sufficient-decrease constant
    ``ls_c``) so a stiff problem converges without hand-tuning ``damping``."""
    return _root_driver(
        "newton",
        damping=damping,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        inner_tol=inner_tol,
        inner_maxit=inner_maxit,
        line_search=line_search,
        ls_max=ls_max,
        ls_c=ls_c,
    )


def picard(
    *,
    damping: float = 1.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 200,
    inner_tol: float = 1e-10,
    inner_maxit: int = 2000,
    line_search: bool = False,
    ls_max: int = 25,
    ls_c: float = 1e-4,
) -> NonlinearSolver:
    """Damped Picard (lagged-coefficient / fixed-point) iteration — pair with :func:`jno.lag`.

    Freeze the troublesome solution-dependent coefficients in the weak form with
    ``jno.lag(...)``; the linearization of the residual is then the *Picard* operator (the
    lagged system re-solved at each iterate), and this driver iterates it with optional damping.
    The classic trade: more outer iterations than Newton's quadratic convergence, but each
    linearized system keeps the structure (symmetry, definiteness) that block preconditioners
    and multigrid need — e.g. a non-Newtonian Stokes flow whose full-Newton velocity block is
    strongly nonsymmetric while its Picard block is a plain symmetric Stokes operator.

    Without any ``jno.lag`` marker in the residual this is exactly damped Newton. The default
    ``max_steps`` is higher than Newton's — linear (not quadratic) convergence. ``line_search=True``
    adds residual-norm Armijo backtracking (up to ``ls_max`` halvings, sufficient-decrease constant
    ``ls_c``): essential when the lagged operator's step overshoots from a stiff initial state (a
    rigid-plastic cold start whose effective viscosity spans orders of magnitude), where fixed
    damping alone either diverges or crawls. See the ``jno.lag`` docstring for the inverse-problem
    (Picard-adjoint) caveat.
    """
    return _root_driver(
        "picard",
        damping=damping,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        inner_tol=inner_tol,
        inner_maxit=inner_maxit,
        line_search=line_search,
        ls_max=ls_max,
        ls_c=ls_c,
    )


def eigs(*, k: int = 6, which: str = "smallest"):
    """Generalized **symmetric eigensolver** ``K x = λ M x`` (K symmetric, M SPD). Returns a callable
    ``(K, M=None) -> (λ, X)``: the ``k`` eigenvalues at the requested end (``which='smallest'`` /
    ``'largest'``) and their **M-orthonormal** eigenvectors (``Xᵀ M X = I``). ``M=None`` is the standard
    problem ``K x = λ x``.

    Use it for modal analysis (vibration), buckling, EM cavity/waveguide resonances and photonic band
    structure — everything that is ``Kx=λMx`` rather than ``Ax=b``. Build ``K``/``M`` as source-less
    ``jno.fem`` bilinear forms (or via :meth:`FEM.eigs`).

    **Differentiable.** V1a reduces the pencil densely (Cholesky ``M=LLᵀ`` → ``jnp.linalg.eigh`` on
    ``L⁻¹KL⁻ᵀ``), so ``∂λ/∂θ`` flows for free — for **simple** eigenvalues (degenerate/crossing
    eigenvalues make the eigen-JVP singular; use the trace of a degenerate cluster there). Preconditioned
    LOBPCG (reusing ``jno.precond.*``) and shift-invert to a target ``σ`` are the planned scale paths.
    """
    if which not in ("smallest", "largest", "SM", "LM", "SA", "LA"):
        raise ValueError(f"jno.solve.eigs: which={which!r} — use 'smallest' or 'largest'.")

    def _fn(K, M=None):
        from .utils.solver.eigen import dense_geneigh

        return dense_geneigh(K, M, k, which)

    return _fn


def logdet(A, *, samples: int = 32, order: int = 25, key=None):
    """Differentiable, matrix-free ``log det A`` (symmetric positive-definite) — stochastic Lanczos
    quadrature via the optional ``matfree`` package. Scales where a direct factorisation cannot; the
    key use is **Bayesian log-evidence / marginal likelihood** of a FEM precision operator. Returns an
    unbiased estimate (variance ↓ with ``samples``, bias ↓ with ``order``). See
    :func:`jno.utils.solver.matfun.logdet`."""
    from .utils.solver.matfun import logdet as _logdet

    return _logdet(A, samples=samples, order=order, key=key)


def trace(A, *, fun=None, samples: int = 32, order: int = 25, key=None):
    """Differentiable, matrix-free ``tr A`` (Hutchinson) or ``tr f(A)`` (``fun=``, Lanczos quadrature) —
    e.g. ``fun=lambda z: 1/z`` for ``tr(A⁻¹)`` (uncertainty / effective degrees of freedom). Optional
    ``matfree``. See :func:`jno.utils.solver.matfun.trace`."""
    from .utils.solver.matfun import trace as _trace

    return _trace(A, fun=fun, samples=samples, order=order, key=key)


def applyfun(A, v, *, fun, order: int = 30, symmetric: bool = True):
    """Matrix-free ``f(A)·v`` — e.g. one exact exponential-integrator step ``exp(-dt·A)·v`` with
    ``fun=lambda z: jnp.exp(-dt*z)``. ``symmetric=True`` (default, Lanczos) assumes ``A = Aᵀ``;
    ``symmetric=False`` (Arnoldi + an eigendecomposition of the Hessenberg with an analytic Daleckii–Krein
    derivative) handles a **non-symmetric** ``A`` (advection–diffusion). Both are **differentiable and
    GPU-capable** — the non-symmetric path for any **holomorphic** ``fun`` on a **diagonalizable** ``A``.
    Optional ``matfree``. See :func:`jno.utils.solver.matfun.applyfun`."""
    from .utils.solver.matfun import applyfun as _applyfun

    return _applyfun(A, v, fun=fun, order=order, symmetric=symmetric)


def diagonal(A, *, fun=None, samples: int = 32, order: int = 25, key=None):
    """Differentiable, matrix-free estimate of the **diagonal** of ``A`` (Hutchinson) or ``f(A)`` (``fun=``,
    ``A`` symmetric) — the per-DOF **field** counterpart of :func:`trace`. The key use is
    ``fun=lambda z: 1/z`` → ``diag(A⁻¹)``, the **pointwise posterior variance / uncertainty map** of a FEM
    precision, plottable on the mesh. Stochastic (variance ↓ ``samples``, bias ↓ ``order``); optional
    ``matfree``. See :func:`jno.utils.solver.matfun.diagonal`."""
    from .utils.solver.matfun import diagonal as _diagonal

    return _diagonal(A, fun=fun, samples=samples, order=order, key=key)


def lstsq(A, b, *, damp: float = 0.0, atol: float = 1e-6, btol: float = 1e-6, maxiter: int = 100_000, x0=None):
    """Differentiable, matrix-free **least-squares** ``min_x ‖A x − b‖²`` for a **rectangular** ``A`` (LSMR) —
    the gap left by the square ``Ax=b`` solvers. ``damp`` adds Tikhonov ``+ damp²‖x‖²`` (ill-posed /
    rank-deficient inverse problems); ``x0`` an initial guess. Real operators; optional ``matfree``.
    See :func:`jno.utils.solver.matfun.lstsq`."""
    from .utils.solver.matfun import lstsq as _lstsq

    return _lstsq(A, b, damp=damp, atol=atol, btol=btol, maxiter=maxiter, x0=x0)


def theta(theta: float = 1.0):
    """θ-method **time scheme** for ``fem.solve(time=...)``: ``θ=1`` backward Euler (default),
    ``θ=1/2`` Crank–Nicolson / trapezoidal (2nd-order accurate), ``θ=0`` forward Euler. Overrides the
    scheme the assembly picks; composes with ``linear=``/``precond=`` (the per-step solve)."""
    from .utils.solver.timeschemes import _ThetaScheme

    return _ThetaScheme(theta)


def exponential(*, order: int = 40, mass: str = "lumped", symmetric: bool = True):
    """Matrix-**exponential** time scheme for ``fem.solve(time=...)`` — advances a *linear autonomous*
    parabolic block ``M u̇ + A u = c`` by ``u(t+dt) = exp(-dt·M⁻¹A) u(t) (+ φ₁ forcing)``, matrix-free.
    **Exact in time and unconditionally stable**, so it takes large stiff steps a θ-step cannot. ``order``
    is the Krylov size. ``mass='lumped'`` (default) is the row-sum diagonal — cheapest, discrete maximum
    principle; ``mass='consistent'`` uses the full ``M`` (no lumping error) via a matrix-free
    M-inner-product Lanczos.

    ``symmetric=True`` (default, ``A = Aᵀ``) uses Lanczos. ``symmetric=False`` handles a **non-symmetric**
    operator (**advection–diffusion / transport**): it advances by Arnoldi + a differentiable **Padé**
    exponential, with forcing carried exactly through an augmented generator — still matrix-free, GPU, and
    reverse-mode differentiable. All paths are differentiable; time-varying coefficients / nonlinear → use
    :func:`theta`."""
    from .utils.solver.timeschemes import _ExponentialScheme

    return _ExponentialScheme(order, mass, symmetric)
