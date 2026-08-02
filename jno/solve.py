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
    "svd",
    "lstsq",
    "theta",
    "exponential",
    "adaptive",
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
    known; otherwise **both** ends are measured by ``bound_iters`` steps of Lanczos (Lanczos 1950,
    §II — the extreme Ritz values of the tridiagonal), for the same one-matvec-per-step cost as
    the power iteration it replaces. Without the optional :mod:`matfree` package this falls back
    to power iteration for ``lmax`` and the ``lmin = lmin_ratio * lmax`` guess, which converges
    more slowly and, when the true ratio is smaller than assumed, leaves the lowest modes outside
    the fitted interval where the polynomial amplifies them."""

    def _fn(op: LinearOperator, b, *, M, x0):
        from .utils.solver.krylov import chebyshev_iteration, spectrum_bounds

        lo, hi = spectrum_bounds(
            op.mv,
            b.shape[0],
            dtype=b.dtype,
            iters=bound_iters,
            M=M,
            lmin=lmin,
            lmax=lmax,
            safety=safety,
            lmin_ratio=lmin_ratio,
        )
        raw = lambda mv, rhs, M, x0: chebyshev_iteration(mv, rhs, lmin=lo, lmax=hi, M=M, x0=x0, tol=tol, maxiter=maxiter)
        return _firewalled(raw, op, b, M=M, x0=x0, symmetric=True, name="chebyshev")

    return LinearSolver(_fn, name="chebyshev")


def _require_jaxamg():
    try:
        import jaxamg
    except ImportError as e:  # optional GPU dependency — see the message for the system requirements
        raise ImportError(
            "jno.solve.amg() needs the optional dependency `jaxamg` (NVIDIA AmgX wrapped as a JAX "
            "primitive). Install it with `pip install jax-numerical-operators[amg]` (the `amg` extra); it "
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
    name, *, damping, rtol, atol, max_steps, inner_tol, inner_maxit, line_search, ls_max, ls_c, direct=False
) -> NonlinearSolver:
    def _fn(residual_fn, u0, *, linear_solve=None, jacobian=None):
        if direct:
            # Sparse-direct Newton: factorize the ASSEMBLED tangent each step (robust on saddles / stiff
            # drag where the matrix-free Krylov inner solve stalls). Needs the assembler-provided Jacobian.
            if jacobian is None:
                raise ValueError(
                    "jno.solve.newton(direct=True) needs the ASSEMBLED Jacobian, which only the native "
                    "nonlinear FEM assembler / transient stepper provides. Use it via "
                    "fem.solve(nonlinear=jno.solve.newton(direct=True)) on a native nonlinear problem "
                    "(a matrix-free residual has no assembled tangent to factorize)."
                )
            from .utils.solver.newton_krylov import newton_direct

            return newton_direct(
                residual_fn,
                jacobian,
                u0,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
                damping=damping,
                line_search=line_search,
                ls_max=ls_max,
                ls_c=ls_c,
            )
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

    return NonlinearSolver(_fn, name=name, direct=direct)


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
    direct: bool = False,
) -> NonlinearSolver:
    """Newton root-find, as a configurable slot. Two inner-solve modes:

    * **default (matrix-free)** -- ``J @ v`` from a JVP, inner matrix-free solve (default BiCGStab, or the
      ``linear=`` slot), implicit differentiation via ``lax.custom_root``. The historic behaviour.
    * **``direct=True`` (sparse-direct)** -- factorize the ASSEMBLED tangent each step with a sparse LU
      instead of an iterative inner solve. Robust on **indefinite / ill-conditioned** systems -- a
      Taylor-Hood velocity/pressure saddle, a stiff Carman-Kozeny phase-change drag -- where the
      matrix-free BiCGStab has no saddle-point preconditioner and stalls. Still differentiable (implicit
      diff with a *direct*, transposable tangent solve at the root). Composes only where the assembler
      provides the tangent: ``fem.solve(nonlinear=jno.solve.newton(direct=True))`` on a native nonlinear
      problem (steady or the transient stepper); the ``linear=``/``precond=`` slots are then unused.

    ``damping < 1`` relaxes each update; ``line_search=True`` adds residual-norm Armijo backtracking (up
    to ``ls_max`` halvings, constant ``ls_c``) so a stiff problem converges without hand-tuning."""
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
        direct=direct,
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


def eigs(*, k: int = 6, which: str = "smallest", precond=None, tol=None, maxiter=None):
    """Generalized **symmetric eigensolver** ``K x = λ M x`` (K symmetric, M SPD). Returns a callable
    ``(K, M=None) -> (λ, X)``: the ``k`` eigenvalues at the requested end (``which='smallest'`` /
    ``'largest'``) and their **M-orthonormal** eigenvectors (``Xᵀ M X = I``). ``M=None`` is the standard
    problem ``K x = λ x``.

    Use it for modal analysis (vibration), buckling, EM cavity/waveguide resonances and photonic band
    structure — everything that is ``Kx=λMx`` rather than ``Ax=b``. Build ``K``/``M`` as source-less
    ``jno.fem`` bilinear forms (or via :meth:`FEM.eigs`).

    **Two paths, selected by the arguments.** With no iterative argument the pencil is reduced
    **densely** (Cholesky ``M=LLᵀ`` → ``jnp.linalg.eigh`` on ``L⁻¹KL⁻ᵀ``) — exact, and the right answer
    when you want the whole low spectrum of a small problem, but ``O(N²)`` memory because it
    materializes the operator. Passing ``precond=`` switches to **preconditioned LOBPCG**
    (:func:`jno.utils.solver.eigen.lobpcg_geneigh`, Knyazev 2001), which only ever applies ``K``/``M`` as
    matvecs and so runs at a scale the dense reduction cannot::

        lam, X = K.eigs(mass=mass, k=6)                              # dense
        lam, X = K.eigs(mass=mass, k=6, precond=jno.precond.amg())   # LOBPCG, matrix-free

    The Rayleigh-Ritz runs in the **M-inner product**, so the consistent (non-lumped) mass matrix of an
    ordinary FEM form is handled directly. ``tol``/``maxiter`` tune that iteration (defaults ``1e-6`` /
    ``200``) and are rejected without ``precond=``, so a tolerance can never be silently ignored on the
    dense path. If the sweep budget is exhausted before ``tol``, the result is **NaN-poisoned** rather
    than silently under-converged — jNO never fails silently.

    **Differentiable** on both paths: ``∂λ/∂θ`` for **simple** eigenvalues (degenerate/crossing
    eigenvalues make the derivative ill-defined — use the trace of the cluster). The dense path
    differentiates through ``eigh``; LOBPCG freezes the converged eigenvector and differentiates the
    Rayleigh quotient, which is the same derivative exactly (``∂R/∂x = 0`` at an eigenvector) — but its
    **eigenvectors** carry no gradient, where the dense path's do.
    """
    if which not in ("smallest", "largest", "SM", "LM", "SA", "LA"):
        raise ValueError(f"jno.solve.eigs: which={which!r} — use 'smallest' or 'largest'.")
    if precond is None and (tol is not None or maxiter is not None):
        raise ValueError(
            "jno.solve.eigs: tol=/maxiter= configure the iterative (LOBPCG) path, but no precond= was "
            "given, so the dense reduction would run and silently ignore them. Pass precond= (e.g. "
            "jno.precond.jacobi() for unpreconditioned-strength LOBPCG), or drop tol=/maxiter=."
        )
    _tol = 1e-6 if tol is None else float(tol)
    _maxiter = 200 if maxiter is None else int(maxiter)

    def _fn(K, M=None):
        from .utils.solver.eigen import dense_geneigh, lobpcg_geneigh

        if precond is None:
            return dense_geneigh(K, M, k, which)

        from .utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

        op = K if isinstance(K, LinearOperator) else LinearOperator(K)
        apply = materialize_precond(precond, PrecondContext(op))
        lam, X, res = lobpcg_geneigh(K, M, k, which, precond=apply, tol=_tol, maxiter=_maxiter)
        bad = res > _tol  # budget exhausted -> poison, never a quietly under-converged spectrum
        return jnp.where(bad, jnp.nan, lam), jnp.where(bad, jnp.nan, X)

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


def svd(A, *, k: int = 6, depth: int | None = None, v0=None):
    """Differentiable, matrix-free **partial SVD** — the ``k`` largest singular triplets of a possibly
    **rectangular** operator (Golub–Kahan bidiagonalization, 1965). The non-symmetric counterpart to
    :func:`eigs`: use it for **POD / reduced-order bases** from a snapshot matrix, and for the
    **ill-posedness** of an inverse problem — the singular spectrum of the parameter-to-observable map
    says which modes are recoverable at all. ``A`` is touched only through its matvec, so it may be the
    JVP of a differentiable FEM solve rather than an assembled matrix. ``depth`` (default ``2k+10``)
    must exceed ``k`` — see :func:`jno.utils.solver.matfun.svd` for why, and for the clustered-spectrum
    caveat. Returns ``(U, s, Vt)``. Optional ``matfree``."""
    from .utils.solver.matfun import svd as _svd

    return _svd(A, k=k, depth=depth, v0=v0)


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
    scheme the assembly picks; composes with ``linear=``/``precond=`` (the per-step solve).

    Marches the domain's fixed time grid. Call ``.adaptive(...)`` on the result to have the step size
    chosen from an error estimate instead — ``jno.solve.theta(0.5).adaptive(rtol=1e-5)`` — which is
    substantially cheaper per digit than the first-order default (see :func:`adaptive`)."""
    from .utils.solver.timeschemes import _ThetaScheme

    return _ThetaScheme(theta)


def exponential(*, order: int = 40, mass: str = "lumped", symmetric: bool = True):
    """Matrix-**exponential** time scheme for ``fem.solve(time=...)`` — advances a linear block
    ``M u̇ + A u = f(t)`` with **time-independent** ``M``, ``A`` by ``u(t+dt) = exp(-dt·M⁻¹A) u(t) +
    forcing``, matrix-free. The homogeneous decay is **exact in time and unconditionally stable**, so it
    takes large stiff steps a θ-step cannot. A **constant** source rides a ``φ₁`` weight (exact); a
    **time-varying** source ``f(t)`` is integrated by **ETD2** (the exponential trapezoidal rule) — sampled
    at both step ends with a ``φ₂`` ramp weight, so it is **exact for a source affine in time** and
    second-order for a general one (Hochbruck & Ostermann, *Exponential integrators*, Acta Numerica 19
    (2010) 209–286, §2.3). ``order`` is the Krylov size. ``mass='lumped'`` (default) is the row-sum diagonal
    — cheapest, discrete maximum principle; ``mass='consistent'`` uses the full ``M`` (no lumping error) via
    a matrix-free M-inner-product Lanczos.

    ``symmetric=True`` (default, ``A = Aᵀ``) uses Lanczos. ``symmetric=False`` handles a **non-symmetric**
    operator (**advection–diffusion / transport**): it advances by Arnoldi + a differentiable **Padé**
    exponential, with forcing carried exactly through an augmented generator (a ramp row for ETD2) — still
    matrix-free, GPU, and reverse-mode differentiable. All paths are differentiable; time-varying
    **coefficients** ``M(t)``/``A(t)`` (a moving/parametric operator) or a nonlinear form → use
    :func:`theta`."""
    from .utils.solver.timeschemes import _ExponentialScheme

    return _ExponentialScheme(order, mass, symmetric)


def adaptive(*, rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 1000, dt0: float | None = None):
    """**Adaptive step-size** time scheme for ``fem.solve(time=...)``: the step size is chosen per step
    from a **step-doubling** (Richardson) local-error estimate — one full step compared with two
    half-steps — so a stiff / sharp / multi-rate transient takes small steps only where it needs them and
    large steps elsewhere, instead of the fixed ``dt`` from ``domain(time=(t0,t1,n))``.

    Built on the block's own implicit θ-step, so it inherits the DAE (Dirichlet) handling and works for a
    **linear or nonlinear**, **scalar or vector**, **plain, periodic, or complex** transient. It is a
    **fixed-length** ``lax.scan`` of ``max_steps`` attempts (a static trip count — the settled tail just
    consumes an attempt), which keeps it **reverse-mode differentiable** (the gradient flows through the
    realized step sequence). ``rtol``/``atol`` set the mixed relative/absolute tolerance; ``max_steps`` is
    the step **budget** — if it is exhausted before ``t1`` the trajectory is returned as ``NaN`` (raise it),
    never silently under-resolved. Composes with ``linear=``/``precond=`` (the per-step solve).
    Backward-Euler order (first-order in time); pair with a fine tolerance for accuracy.

    ``dt0`` is the **first** step. It defaults to the smallest allowed step (``1e-4`` of the time span) so
    the controller **approaches the right step size from below**. That default matters: there is no step
    rejection (see :func:`jno.utils.solver.timeschemes.adaptive_march` — a discarded state would make the
    per-step solve adjoint run at zero cotangent and return a ``NaN`` gradient), so an over-large step is
    *committed*, not retried, and only the next one shrinks. Growing into the step size can never commit an
    over-tolerance step — an under-sized step is wasteful, not inaccurate — whereas starting at the output
    grid's ``dt`` bakes in the error of the first few steps permanently. On a 2-D heat benchmark, taking
    ``dt0`` from the output grid left the result **4.2x** less accurate than growing from below, for the same
    tolerance and ~18% fewer steps. Pass ``dt0`` explicitly only if you know the correct scale; the growth
    cap is 5x per step, so the default reaches any scale in a handful of attempts.

    This bare form sizes **whatever step the assembly picked** for the block — backward Euler for a
    parabolic block, which is **first order**. That is the single biggest accuracy lever here, and it is
    worth moving: step doubling costs 3 implicit solves per step, so spending them on a first-order base
    cannot beat a first-order fixed march by much. Attach the controller to a **second-order** step instead
    and the same tolerance is far cheaper per digit — the step-size exponent follows the base's order
    automatically (``1/(p+1)``)::

        fem.solve(time=jno.solve.adaptive(rtol=1e-4))                    # 5.1e-3 for 162 solves
        fem.solve(time=jno.solve.theta(0.5).adaptive(rtol=1e-4))         # 2.2e-4 for  48 solves

    (~23x the accuracy on ~3x less work, measured on the 2-D heat benchmark of
    ``tests/test_fem_adaptive_timestep.py`` — ``mesh_size=0.15``, t=0.05, x64, against the semidiscrete
    reference; step doubling costs 3 implicit solves per step). Every scheme exposing a single
    step carries ``.adaptive(...)``, so this composes with future base methods without new arguments;
    :func:`exponential` raises, since it is already exact in time for the homogeneous decay.

    Second order is opt-in rather than the default because θ=1/2 is A-stable but **not L-stable**: on a
    stiff problem with rough or incompatible initial data it rings instead of damping, where backward Euler
    is unconditionally smooth. Prefer ``jno.solve.theta(0.5).adaptive(...)`` for smooth parabolic and wave
    problems; keep the bare form when robustness matters more than order.

    NOTE on what adaptivity does and does not buy: on the benchmarks measured here a *well-chosen fixed*
    dt matches or beats it at equal work, because the optimal step size is nearly constant and the error
    estimate costs 3x. Reach for ``adaptive`` when you **cannot** pick dt in advance — unknown or
    parameter-dependent stiffness, sweeps, inverse problems whose fitted parameter moves the timescale —
    not as a speed optimization."""
    from .utils.solver.timeschemes import _AdaptiveScheme

    return _AdaptiveScheme(None, rtol, atol, max_steps, dt0)
