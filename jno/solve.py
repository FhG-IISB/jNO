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

from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # runtime import stays lazy inside remesh()/relocate()
    from .utils.solver.fem_adapt import AdaptSpec

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
    "remesh",
    "relocate",
]


def lu(*, backend: str = "device", host: bool | None = None) -> LinearSolver:
    """Differentiable sparse-direct solve (JAX ``spsolve``: cuSolver on GPU, native LU on CPU).

    Wraps the existing :func:`jno.utils.solver.linear.sparse_lu_solve` -- robust on the
    indefinite saddle-point systems where Jacobi-preconditioned Krylov stalls, reverse-mode
    differentiable in the matrix entries and the right-hand side. Direct: ignores ``x0`` and
    rejects a preconditioner. ``jit`` yes; **no vmap batching rule upstream** (trait
    ``vmap="no"``) -- use a Krylov solver inside vmapped/batched solves.

    Args:
        backend: WHERE the factorization happens. All three obey the same ``(A, b) -> x`` contract and
            are equally differentiable; they differ in speed, reach, and what they need installed.

            ``"device"`` (default) -- JAX's ``spsolve``: cuSolver on a CUDA GPU, a native LU on CPU.
            No extra dependency, and the only one of the three that needs nothing installed.

            ``"host"`` -- SuperLU on the CPU via ``scipy``, keeping the rest of the solve where it is.
            Affordable because a direct solve factorises ONCE, so the operator crosses PCIe once
            rather than per iteration. It runs meshes cuSolver refuses outright (Stokes 26,908,
            H(curl) 26,154, 3-D Poisson 87,284), and on this machine beat cuSolver in all 12 measured
            points (0.15x-0.81x of its time). Read that as *cuSolver's sparse LU is weak*, not as
            *GPUs lose*: see ``"cudss"``. Its factorization is cached on the operator's CONTENT, so a
            constant-operator march factorizes once for the trajectory (measured 2.9x at 23,934 DOFs
            over 51 steps) and the adjoint reuses it; a Newton loop gets nothing and pays a hash.

            ``"cudss"`` -- NVIDIA cuDSS on the GPU. **The fastest of the three wherever it runs**, and
            the one to reach for on a Newton loop or a shift-invert eigensolve, because it separates
            the symbolic plan from the numeric factorization and jNO caches on the SPARSITY: the plan
            survives a change of values. Measured on an RTX 3070 (fp64 at 1/64 rate -- the
            *unfavourable* card) against host SuperLU: factorization 3.4 ms vs 79.9 ms on a
            Taylor-Hood Stokes saddle, 576 ms vs 64,856 ms on lap3d 50^3, and **64.7x per Newton step**
            at n=64,000. It also factors that Stokes saddle cuSolver calls singular, with smaller
            residuals. Needs the optional stack (``nvmath-python``, ``cudss``, ``cupy``) and a GPU;
            raises a clear ``ImportError`` otherwise. Fill-in still governs 3-D (69x-218x nnz growth
            at lap3d 20^3-40^3), so it moves the ceiling and makes device memory the binding
            constraint -- it is not a substitute for a preconditioner.

            ``"pardiso"`` -- Intel MKL PARDISO on the CPU, multithreaded. **The fastest
            FACTORIZATION of the four**, and the one to pick for a Newton loop: like cuDSS it splits
            symbolic analysis from numeric factorization, so the analysis survives a change of values.
            Measured on lap3d 50^3 (n=125,000) against single-threaded SuperLU's 65,212 ms:
            factorization **298 ms**, and a Newton re-factorization **296 ms -- 220x**, where cuDSS
            reaches 115x. Being on the CPU it is also the answer when a factorization will not fit in
            device memory. Its adjoint is cheaper than cuDSS's too: ``A^T x = b`` comes from the SAME
            factorization rather than a second one. Needs ``pypardiso`` (x86-64).

            **Choosing between the last two: pick by the phase your problem repeats.** A Newton loop
            re-FACTORIZES, so PARDISO wins. A shift-invert eigensolve or a constant-operator transient
            re-SOLVES against one factorization, and there cuDSS is 11x faster per solve (3.5 ms vs
            40 ms at lap3d 50^3) and additionally takes a whole block of right-hand sides at once.

            There is deliberately no ``"auto"``: which backend wins depends on hardware jNO cannot
            inspect, and silently choosing would violate the no-surprises rule.
        host: Deprecated alias for ``backend="host"``, kept so existing calls keep working. Passing
            both is an error.
    """
    if host is not None:
        if backend != "device":
            raise ValueError(
                f"jno.solve.lu() got both backend={backend!r} and host={host!r}. `host=` is the "
                f'deprecated spelling of backend="host" -- pass only backend=.'
            )
        backend = "host" if host else "device"
    if backend not in ("device", "host", "cudss", "pardiso"):
        raise ValueError(
            f"jno.solve.lu(backend={backend!r}) is not a known backend. Use 'device' (JAX spsolve: "
            f"cuSolver on GPU, native LU on CPU), 'host' (CPU SuperLU via scipy), 'cudss' (NVIDIA "
            f"cuDSS on GPU -- fastest repeated SOLVE), or 'pardiso' (Intel MKL PARDISO on CPU -- "
            f"fastest FACTORIZATION). Install the last two with jax-numerical-operators[fem]."
        )

    def _fn(op: LinearOperator, b, *, M, x0):
        from .utils.solver.linear import cudss_lu_solve, host_lu_solve, pardiso_lu_solve, sparse_lu_solve

        solve = {
            "device": sparse_lu_solve,
            "host": host_lu_solve,
            "cudss": cudss_lu_solve,
            "pardiso": pardiso_lu_solve,
        }[backend]
        if op.bcoo is not None:
            return solve(op.bcoo, b)
        # a dense operator gets the dense direct solve — BCOO.fromdense would need a concrete
        # nse, which does not exist under jit/vmap tracing
        return jnp.linalg.solve(op.dense(), b)

    # No `key`, so the composer leaves this on the eager path. What compiling the composed solve buys
    # is the removal of per-ITERATION Python dispatch; a direct solver issues one `spsolve` and has
    # none to remove, so it would pay a compile for nothing.
    # the name carries the placement, so two specs that factor in different memories are not
    # reported (or cached) as though they were the same solver
    name = {"device": "lu", "host": "lu-host", "cudss": "lu-cudss", "pardiso": "lu-pardiso"}[backend]
    # `multi_rhs` lets a caller holding a BLOCK of right-hand sides (the shift-invert eigensolver's
    # subspace iteration) hand the whole block over in one call instead of looping its columns.
    # `host_kernel` names the numpy-level solve this spec corresponds to, for the callers that run
    # on the host and cannot go back through JAX -- notably ARPACK's shift-invert OPinv in the
    # non-symmetric eigensolver. "device" has none: it IS a JAX primitive.
    traits = {
        "vmap": "no",
        "multi_rhs": backend == "cudss",
        "host_kernel": None if backend == "device" else backend,
    }
    return LinearSolver(_fn, name=name, direct=True, traits=traits)


def dense() -> LinearSolver:
    """Dense LAPACK solve (``jnp.linalg.solve``) on the densified operator.

    ``O(N^2)`` memory / ``O(N^3)`` time -- the right answer for small systems and coarse
    blocks, and the only shipped direct solver with a native vmap batching rule. Direct:
    ignores ``x0``, rejects a preconditioner.
    """

    def _fn(op: LinearOperator, b, *, M, x0):
        return jnp.linalg.solve(op.dense(), b)

    return LinearSolver(_fn, name="dense", direct=True)  # one LAPACK call: no iteration to compile away


def _krylov(name: str, tol: float, atol: float, maxiter: Optional[int], **fixed):
    def _fn(op: LinearOperator, b, *, M, x0):
        method = getattr(jax.scipy.sparse.linalg, name)
        x, _info = method(op.mv, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M, **fixed)
        return _maybe_residual_check(op, b, x, name)

    # `key` must name every argument that changes the iteration -- see LinearSolver. `fixed` is
    # per-method extra configuration (GMRES's restart), so it goes in sorted rather than by position.
    return LinearSolver(_fn, name=name, key=(tol, atol, maxiter, tuple(sorted(fixed.items()))))


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

    return LinearSolver(_fn, name="fgmres", key=(tol, restart, maxiter))


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

    return LinearSolver(_fn, name="minres", key=(tol, maxiter))


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

    # `jit: False` -- `spectrum_bounds` measures the spectrum and then branches on what it measured
    # (a collapsed or inverted Lanczos interval is rejected), which a tracer cannot answer. Passing
    # explicit lmin=/lmax= does not change that: the same code path still validates them.
    return LinearSolver(_fn, name="chebyshev", traits={"jit": False})


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
    # is not a pure-JAX batching op) — loop for batched solves. No `key`, so the composer keeps this
    # eager: AmgX builds its hierarchy from the matrix VALUES, and whether that setup survives a tracer
    # is unverified here (jaxamg needs a GPU + AmgX, absent from this environment). Nothing is lost —
    # the solve is one AmgX call, with no per-iteration Python dispatch to compile away.
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
                    "The sparse-direct Newton needs the ASSEMBLED Jacobian, which only the native "
                    "nonlinear FEM assembler / transient stepper provides -- a matrix-free residual has "
                    "no assembled tangent to factorize. Reached either via "
                    "fem.solve(nonlinear=jno.solve.newton(direct=True)) or by a direct linear slot "
                    "(fem.solve(linear=jno.solve.lu()/dense()/amg()), which selects this driver); on a "
                    "matrix-free-only problem use an iterative linear= (cg/bicgstab/gmres/fgmres)."
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
                # the composed ``linear=``/``precond=`` slots, over the ASSEMBLED tangent; None keeps
                # the historic sparse-LU default
                linear_solve=linear_solve,
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


def eigs(*, k: int = 6, which: str = "smallest", sigma=None, linear=None, precond=None, tol=None, maxiter=None, X0=None):
    """Generalized **symmetric eigensolver** ``K x = λ M x`` (K symmetric, M SPD). Returns a callable
    ``(K, M=None) -> (λ, X)``: the ``k`` eigenvalues at the requested end (``which='smallest'`` /
    ``'largest'``) and their **M-orthonormal** eigenvectors (``Xᵀ M X = I``). ``M=None`` is the standard
    problem ``K x = λ x``.

    Use it for modal analysis (vibration), buckling, EM cavity/waveguide resonances and photonic band
    structure — everything that is ``Kx=λMx`` rather than ``Ax=b``. Build ``K``/``M`` as source-less
    ``jno.fem`` bilinear forms (or via :meth:`FEM.eigs`).

    **Three paths, selected by the arguments.** With no iterative argument the pencil is reduced
    **densely** (Cholesky ``M=LLᵀ`` → ``jnp.linalg.eigh`` on ``L⁻¹KL⁻ᵀ``) — exact, and the right answer
    when you want the whole low spectrum of a small problem, but ``O(N²)`` memory because it
    materializes the operator. Passing ``precond=`` switches to **preconditioned LOBPCG**
    (:func:`jno.utils.solver.eigen.lobpcg_geneigh`, Knyazev 2001), which only ever applies ``K``/``M`` as
    matvecs and so runs at a scale the dense reduction cannot. Passing ``sigma=`` targets the ``k``
    eigenvalues **nearest the shift** — interior modes (cavity resonances, band structure away from the
    band edge), which no extremal-end iteration can reach — via the spectral transformation
    ``θ = 1/(λ−σ)`` (:func:`jno.utils.solver.eigen.shift_invert_geneigh`, Ericsson & Ruhe 1980 §2); the
    inner solves against ``K − σM`` default to a once-factorized host sparse LU, and ``linear=`` picks a
    different inner solver (e.g. ``jno.solve.amg()`` when a factorization is too big)::

        lam, X = K.eigs(mass=mass, k=6)                              # dense
        lam, X = K.eigs(mass=mass, k=6, precond=jno.precond.amg())   # LOBPCG, matrix-free
        lam, X = K.eigs(mass=mass, k=4, sigma=60.0)                  # the 4 modes nearest λ = 60

    ``sigma=`` replaces ``which=`` (the target is "nearest σ") and needs no ``precond=`` — the
    transformation is its own preconditioner. The Rayleigh-Ritz runs in the **M-inner product**, so the
    consistent (non-lumped) mass matrix of an ordinary FEM form is handled directly. ``tol``/``maxiter``
    tune the iterative paths (defaults ``1e-6`` / ``200``) and are rejected on the dense path, so a
    tolerance can never be silently ignored. If the budget is exhausted before ``tol`` — or a shift
    lands on an eigenvalue and the inner factorization degenerates — the result is **NaN-poisoned**
    rather than silently under-converged — jNO never fails silently.

    **Differentiable** on both paths: ``∂λ/∂θ`` for **simple** eigenvalues (degenerate/crossing
    eigenvalues make the derivative ill-defined — use the trace of the cluster). The dense path
    differentiates through ``eigh``; LOBPCG freezes the converged eigenvector and differentiates the
    Rayleigh quotient, which is the same derivative exactly (``∂R/∂x = 0`` at an eigenvector) — but its
    **eigenvectors** carry no gradient, where the dense path's do.
    """
    if which not in ("smallest", "largest", "SM", "LM", "SA", "LA"):
        raise ValueError(f"jno.solve.eigs: which={which!r} — use 'smallest' or 'largest'.")
    if sigma is not None:
        if which not in ("smallest", "SM", "SA"):  # the default — anything else is a contradiction
            raise ValueError(
                "jno.solve.eigs: sigma= targets the k eigenvalues NEAREST the shift; which= does not "
                "apply. Drop which=, or drop sigma=."
            )
        if precond is not None:
            raise ValueError(
                "jno.solve.eigs: sigma= needs no precond= — the spectral transformation is its own "
                "preconditioner (the gaps near the shift become the largest in the transformed "
                "spectrum). Pass linear= to change the INNER solver against K - sigma*M instead."
            )
    elif linear is not None:
        raise ValueError(
            "jno.solve.eigs: linear= picks the inner solver of the shift-invert path and means "
            "nothing without sigma=. Pass sigma=, or drop linear=."
        )
    if X0 is not None:
        if sigma is not None:
            raise NotImplementedError(
                "jno.solve.eigs: X0= (warm start) is not wired into the shift-invert path yet — the "
                "spectral transformation converges from random in a handful of sweeps anyway. Drop "
                "X0=, or drop sigma= and warm-start the LOBPCG path."
            )
        if precond is None:
            raise ValueError(
                "jno.solve.eigs: X0= seeds the iterative (LOBPCG) block, but no precond= was given, "
                "so the dense reduction would run and silently ignore it. Pass precond= (e.g. "
                "jno.precond.jacobi()), or drop X0=."
            )
    if precond is None and sigma is None and (tol is not None or maxiter is not None):
        raise ValueError(
            "jno.solve.eigs: tol=/maxiter= configure the iterative paths, but neither precond= nor "
            "sigma= was given, so the dense reduction would run and silently ignore them. Pass "
            "precond= (e.g. jno.precond.jacobi() for unpreconditioned-strength LOBPCG) or sigma=, "
            "or drop tol=/maxiter=."
        )
    _tol = 1e-6 if tol is None else float(tol)
    _maxiter = 200 if maxiter is None else int(maxiter)

    def _fn(K, M=None):
        from .utils.solver.eigen import (
            _require_symmetric,
            _symmetry_verdict,
            dense_geneigh,
            lobpcg_geneigh,
            nonsymmetric_geneigh,
            shift_invert_geneigh,
        )

        # The symmetric paths below Hermitianize by construction, so a non-self-adjoint operator
        # would be silently answered with the spectrum of its symmetric part. Probe the bilinear
        # form and ROUTE on the answer: Arnoldi (complex spectrum) rather than a refusal. A traced
        # or unsized operator cannot be probed and keeps the historical symmetric assumption.
        if "nonsymmetric" in (_symmetry_verdict(K), _symmetry_verdict(M)):
            if precond is not None:
                raise ValueError(
                    "jno.solve.eigs: precond= is not used by the non-symmetric (Arnoldi) path -- its "
                    "shift-invert is ARPACK's own sparse LU. Drop precond=, and pass sigma= to target "
                    "an interior region."
                )
            return nonsymmetric_geneigh(
                K,
                M,
                k,
                sigma,
                which,
                inner_solve=linear,
                tol=0.0 if tol is None else float(tol),
                maxiter=maxiter,
            )
        _require_symmetric(K, "K")
        _require_symmetric(M, "M")
        if sigma is not None:
            return shift_invert_geneigh(K, M, k, sigma, inner_solve=linear, tol=_tol, maxiter=_maxiter)
        if precond is None:
            return dense_geneigh(K, M, k, which)

        from .utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

        op = K if isinstance(K, LinearOperator) else LinearOperator(K)
        apply = materialize_precond(precond, PrecondContext(op))
        lam, X, res = lobpcg_geneigh(K, M, k, which, precond=apply, tol=_tol, maxiter=_maxiter, X0=X0)
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


def remesh(
    *,
    anisotropic: bool = False,
    max_dofs: int | None = None,
    every: int = 5,
    metric_field: int = 0,
    hmin: float | None = None,
    hmax: float | None = None,
    theta: float = 0.5,
    refine_factor: float = 2.0,
    max_iters: int = 8,
    tol: float | None = None,
    eps: float | None = None,
) -> AdaptSpec:
    """**h-adaptivity** for ``fem.solve(adapt=...)``: change the mesh to follow the solution.

    On a **steady** problem this is the refine loop — solve, estimate (Zienkiewicz–Zhu), mark
    (Dörfler ``theta``), refine by ``refine_factor``, repeat up to ``max_iters`` — growing the mesh
    toward convergence. On a **transient** problem it remeshes every ``every`` steps at a *constant*
    budget and carries the state across (basis-aware transfer), so the mesh tracks a moving feature and
    coarsens its wake instead of ratcheting up::

        fem.solve(adapt=jno.solve.remesh(anisotropic=True, max_dofs=6000, every=4))

    ``anisotropic=True`` refines on a Hessian metric (stretched elements aligned to the solution's
    curvature) instead of isotropic ZZ marking — far fewer DOFs for a layer or a front, and the right
    choice for an interface. ``hmin``/``hmax`` bound the edge sizes; ``metric_field`` picks which coupled
    field drives the metric. Metric-based DOF control is approximate, so ``max_dofs`` is honoured only
    loosely in that mode.

    Steady-only: ``max_iters``, ``tol``, ``eps`` (a relative-change plateau detector, not a certified
    bound). Transient-only: ``every``, ``metric_field``.

    Args:
        anisotropic: Hessian-metric refinement instead of isotropic ZZ + Dörfler marking.
        max_dofs: Vertex budget. Steady: stop once reached. Transient: the constant target.
        every: Transient only — remesh every ``every`` time steps.
        metric_field: Transient multifield only — index of the field driving the metric.
        hmin: Smallest allowed edge length (default: mean edge / 50).
        hmax: Largest allowed edge length (default: 2 × mean edge).
        theta: Dörfler bulk-marking fraction (0..1).
        refine_factor: Local edge-size reduction applied to marked cells each round.
        max_iters: Steady only — maximum refine-solve rounds.
        tol: Steady only — stop once the global error estimate falls below this.
        eps: Steady only — stop once the round's figure of merit stops moving by more than this
            (two consecutive rounds required).

    Returns:
        AdaptSpec: The adaptation spec to pass as ``fem.solve(adapt=...)``.

    See :func:`relocate` for the fixed-connectivity (r-adaptive) alternative: it keeps the topology, so
    there is no mesh schedule to freeze and no cross-mesh transfer, and its vertex map is differentiable
    in the monitor. The two compose — remesh first, then relocate on the result.
    """
    from .utils.solver.fem_adapt import AdaptSpec

    return AdaptSpec(
        theta=theta,
        max_iters=max_iters,
        refine_factor=refine_factor,
        tol=tol,
        max_dofs=max_dofs,
        eps=eps,
        anisotropic=anisotropic,
        hmin=hmin,
        hmax=hmax,
        every=every,
        metric_field=metric_field,
    )


def relocate(
    *,
    method: str = "descent",
    max_iters: int = 8,
    lr: float = 3e-3,
    quality_floor: float = 0.1,
    relax: int = 60,
    relax_step: float = 0.1,
) -> AdaptSpec:
    """**r-adaptivity** for ``fem.solve(adapt=...)``: move the mesh vertices, keep the connectivity.

    Moves the vertices tagged ``domain.variable(region)[i].trainable()`` so the mesh **equidistributes**
    the solution's features, at fixed connectivity and no new DOFs::

        xm, ym, _ = domain.variable("core", where=interior, split=True)
        xm.trainable(); ym.trainable()                  # BEFORE jno.fem(...)
        u = fem.solve(adapt=jno.solve.relocate())

    Requires at least one coordinate tagged ``.trainable()`` before ``jno.fem`` (else it raises).
    Tagging is **literal and per-axis**: ``xm.trainable()`` frees only the x column. That is the lever for
    boundary vertices — free an edge's *along-edge* axis and its nodes slide within the wall; leave the
    normal axis untagged and the domain shape is preserved exactly.

    **Two methods.** ``"descent"`` (default) walks the vertices down the equidistribution defect of an
    arclength monitor, evaluated *through the differentiable solve*, with a backtracking ``det J`` line
    search — on a stiff problem neither a stock optimiser nor an energy barrier can guarantee validity from
    outside the step control. ``"monge_ampere"`` instead solves ``m·det(I + H(φ)) = θ`` for a mesh potential
    and takes ``x = ξ + ∇φ`` (McRae, Cotter & Budd, *Optimal-transport-based mesh adaptivity on the plane
    and sphere using finite elements*, SIAM J. Sci. Comput. **40**(2) (2018) A1121–A1148, arXiv:1612.08077,
    §3.1); the displacement is a gradient, so the *whole* map cannot fold and no line search is needed.

    Measured on the Allen–Cahn front the suite uses (``h = 0.06``, ``eps = 0.03``, 377 nodes), error on a
    common fine grid so the comparison does not depend on where each mesh puts its nodes:

    ==================  ===========  ============  ==================
    method              rel-L2       vs uniform    min element quality
    ==================  ===========  ============  ==================
    uniform             1.096e-01    1.000         0.834
    ``"descent"``       3.951e-02    **0.361**     0.503
    ``"monge_ampere"``  8.879e-02    0.811         0.160
    ==================  ===========  ============  ==================

    So descent stays the default: Monge–Ampère converges in far fewer rounds (3–6 against 30) and reaches a
    comparable equidistribution defect, but it degrades element quality badly here and the answer with it.
    Lowering ``relax_step`` recovers part of the gap (0.811 → 0.633 at ``relax_step=0.02``).

    Works in **2D and 3D**, on a scalar or vector field of **any nodal-Lagrange order**, and across linear,
    nonlinear, transient, periodic and complex problems (all but complex-*transient*). It does not compose
    with a moving mesh (``coord.d(t) - v``) — that driver owns the march.

    Further limits, measured rather than argued:

    - **The monitor reads vertex values only**, whatever the element order, so at P2 and above it adapts to
      the P1 sub-sampling of the field rather than to everything the field resolves. Higher order still
      relocates correctly; it just does not get a sharper monitor for the extra DOFs.
    - **Monge–Ampère's non-folding is a property of the whole map.** Holding a subset of vertices truncates
      it, and the truncation is what can tangle: on a 21² square with a diagonal front, freezing the whole
      boundary reached ``min det J = -1.2e-03`` where the full map stayed positive throughout. Freeing
      tangential axes recovers nearly all of it. Either method checks ``det J`` each round and keeps the
      last valid mesh, so a bad tagging costs accuracy, not correctness.
    - Its relaxation is explicit in ``relax_step``: past the stability limit more iterations make things
      *worse* (spread 0.111 → 0.292 going from ``relax=40`` to ``300`` at ``relax_step=0.2``).
    - The monitor is arclength-based, which suits an **under-resolved** feature; on an already
      well-resolved mesh a curvature monitor wins. Not yet selectable.
    - Relocation beats :func:`remesh` when features are few and sharp; loses when they are spread through
      the domain (with four separated fronts the crossover moved below one element per feature width) or
      when the mesh already over-resolves them. The two **compose** — remesh, then relocate on the result.

    Args:
        method: ``"descent"`` or ``"monge_ampere"``.
        max_iters: Outer relocation rounds.
        lr: ``"descent"`` only — base step for the RMS-normalised descent.
        quality_floor: ``"descent"`` only — a step is halved until no element's ``|det J|`` falls below this
            fraction of the initial worst element.
        relax: ``"monge_ampere"`` only — relaxation iterations per round (McRae et al. eq. (3.7)). Each is
            one Poisson solve against a matrix factorized once for the whole run, so these are cheap.
        relax_step: ``"monge_ampere"`` only — the relaxation pseudo-step ``Δt``.

    Returns:
        AdaptSpec: The adaptation spec to pass as ``fem.solve(adapt=...)``.
    """
    if method not in ("descent", "monge_ampere"):
        raise ValueError(f"jno.solve.relocate(method={method!r}): expected 'descent' or 'monge_ampere'.")
    from .utils.solver.fem_adapt import AdaptSpec

    return AdaptSpec(
        relocate=True,
        relocate_method=method,
        max_iters=max_iters,
        lr=lr,
        quality_floor=quality_floor,
        ma_relax=relax,
        ma_dt=relax_step,
    )


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
