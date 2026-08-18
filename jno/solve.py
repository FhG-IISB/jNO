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

from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp
import numpy as _np

if TYPE_CHECKING:  # runtime import stays lazy inside remesh()/relocate()
    from .utils.solver.fem_adapt import AdaptSpec

from .utils.solver.solver_api import (
    ContinuationSpec,
    LinearOperator,
    LinearSolver,
    NonlinearSolver,
    PrecondApplier,
    _maybe_residual_check,
)

__all__ = [
    "continuation",
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
    "staggered",
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
    "enrich",
    "refine",
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

    # The tolerances travel WITH the spec, not just inside the closure: a driver run under `lax.scan`
    # (the load-path march) cannot raise on non-convergence from inside the trace, so the caller
    # re-checks outside it — and it must judge against the tolerance the user actually asked for.
    return NonlinearSolver(_fn, name=name, direct=direct, traits={"rtol": rtol, "atol": atol})


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


def continuation(keep: str = "last", **params: Any) -> ContinuationSpec:
    """**Parameter continuation** for ``fem.solve(continuation=...)``: march runtime parameters across
    a value sequence, warm-starting each solve from the previous one::

        cap = jno.np.parameter((1,), name="cap")
        fem = jno.fem([... cap ...])                      # built ONCE
        u = fem.solve(continuation=jno.solve.continuation(cap=np.geomspace(G / 2000, G, 8)))

    One driver, three names for one mechanism: a **frequency or material sweep** in EM, **load
    stepping** in mechanics, **homotopy** in numerics -- reaching a parameter value the cold solve
    cannot. Sequences given together are marched **zipped**, not as a grid, so two coefficients can
    ramp in step.

    ``keep="last"`` returns the final solution (homotopy); ``keep="all"`` returns the whole family,
    ``(n_values, n_dofs)`` -- a sweep.

    **Why this instead of a Python loop that rebuilds the form.** The form is traced and compiled once
    and the parameter arrives as a runtime argument, so an 8-step ramp is 8 solves, not 8 rebuilds
    and 8 XLA compilations. That is the difference between a continuation being a tool and being the
    dominant cost of a run.

    Fixed (unswept) parameter values are passed as ordinary keywords to ``fem.solve`` alongside this
    spec; naming one in both places raises rather than silently picking a winner.

    Scope, refused by name elsewhere: **steady** problems (linear or nonlinear), real or
    fused-complex. A transient sweep is a plain loop over ``fem.solve()`` -- there is no warm start to
    carry between independent trajectories, so this driver would add nothing.
    """
    return ContinuationSpec(params=dict(params), keep=keep)


def staggered(
    fields,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_sweeps: int = 200,
    inner_steps: int = 20,
    inner_tol: float = 1e-10,
    inner_maxit: int = 2000,
    line_search="backtrack",
    damping: float = 1.0,
    ls_max: int = 25,
    ls_c: float = 1e-4,
    direct: bool = False,
    over_relax: float = 1.0,
) -> NonlinearSolver:
    """**Alternate minimization** — solve a coupled system one field at a time, sweeping until the full
    residual converges. ``fields`` is the trial symbols in the order to sweep them::

        fem.solve(nonlinear=jno.solve.staggered([u, dm]))

    Reach for it when the coupled energy is **non-convex in the fields jointly but convex in each
    separately** — the case where a monolithic Newton has no descent guarantee and diverges outright.
    Variational phase-field fracture is the canonical one: the ``(1-d)^2 |grad u|^2`` coupling is quartic
    in the pair, while ``u`` alone solves a linear elasticity problem and ``d`` alone a linear elliptic
    one. Alternate minimization turns that into a sequence of convex solves, each decreasing the energy.
    Fixed-stress Biot poroelasticity and thermo-mechanical staggering have the same shape.

    Algorithm: Bourdin, Francfort & Marigo, *Numerical experiments in revisited brittle fracture*,
    J. Mech. Phys. Solids **48** (2000), §3 — as the staggered operator split with a history field,
    Miehe, Welschinger & Hofacker, IJNME **83** (2010).

    **The trade is the convergence rate, and it is not small.** Alternate minimization converges
    *linearly* where Newton is quadratic, so it can need hundreds of sweeps near a propagating crack —
    hence ``max_sweeps=200``. It buys robustness, not speed; on a problem where Newton converges,
    Newton is the better choice (Farrell & Maurini, CMAME **312**, 2017, compare the two directly).

    Sweeping is Gauss-Seidel: each sub-solve sees the updates made earlier in the same sweep, so the
    ORDER of ``fields`` matters. Every block must be listed — an omitted field's equations would never be
    solved, which is rejected rather than silently skipped.

    Differentiable in the ordinary way: at convergence the full residual is zero, so the sweep is just a
    way of *finding* that root, and ``lax.custom_root`` supplies the gradient from the full Jacobian.

    **``direct=True`` factorizes each field's assembled diagonal block** instead of solving it
    matrix-free, and pairs with a ``linear=`` slot::

        fem.solve(nonlinear=jno.solve.staggered([u, dm], direct=True), linear=jno.solve.lu(backend="pardiso"))

    Reach for it when a *sub-block* is ill-conditioned — near-incompressible elasticity (ν → 0.5) is the
    common one. The matrix-free default cannot help there: a ``precond=`` spec materializes against an
    assembled operator, and a sub-solve is a restriction *closure* with none, so the block is solved by
    **unpreconditioned** BiCGStab. The trade is that the full tangent is assembled to use one block of
    it; a sparsity-caching backend (``pardiso``/``cudss``) then pays only the numeric re-factorization
    per sweep. On a well-conditioned problem the matrix-free default is cheaper — this is not a
    free upgrade, and it is not the default.

    Scope: composes through ``fem.solve(nonlinear=...)`` on a multifield problem, which is where the
    block layout comes from; it has no meaning on a single field and says so. Each field is solved on
    its own — solving a GROUP of fields together (a Stokes velocity/pressure pair inside one sweep) is
    not wired.
    """
    resolved: dict = {"blocks": None, "names": None, "constrained": None}

    def _prepare(fem):
        blocks = getattr(fem, "blocks", None)
        if blocks is None or len(blocks) < 2:
            raise ValueError(
                "jno.solve.staggered: this problem has a single field block, so there is nothing to "
                "alternate between. Use jno.solve.newton() (or picard) instead."
            )
        want = list(fields)
        idxs = [fem.block_index(f) for f in want]
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"jno.solve.staggered: a field is listed twice (resolved block indices {idxs}).")
        if set(idxs) != set(range(len(blocks))):
            missing = sorted(set(range(len(blocks))) - set(idxs))
            raise ValueError(
                f"jno.solve.staggered: every field block must be swept, but blocks {missing} were not "
                f"listed (got {idxs} of {len(blocks)}). An unlisted field's equations would never be "
                "solved — list all of them, in the order you want them swept."
            )
        resolved["blocks"] = [_np.arange(int(blocks[i].start), int(blocks[i].stop), dtype=_np.int32) for i in idxs]
        resolved["names"] = idxs
        # Essential-condition dofs, so over-relaxation can leave them alone (see staggered_newton).
        _dd = getattr(getattr(fem, "_op", None), "dirichlet_dofs", None)
        resolved["constrained"] = None if _dd is None else _np.asarray(_dd, dtype=_np.int64)

    if line_search not in (True, False, "backtrack"):
        raise ValueError(
            f"jno.solve.staggered: line_search={line_search!r} is not a known choice. True (the default) "
            "is the EXACT line search — bisection for the minimizer of the energy along the Newton "
            'direction; "backtrack" is the older residual-norm Armijo; False takes the fixed `damping`.'
        )
    if not (0.0 < float(over_relax) < 2.0):
        raise ValueError(
            f"jno.solve.staggered: over_relax must lie in (0, 2), got {over_relax}. Kahan's condition — "
            "outside that range the over-relaxed Gauss-Seidel iteration is not guaranteed to converge "
            "(Farrell & Maurini, IJNME 109 (2017), section 2.1). over_relax=1 is plain alternate "
            "minimization."
        )

    def _fn(residual_fn, u0, *, linear_solve=None, jacobian=None, project=None, constrained=None):
        if resolved["blocks"] is None:
            raise ValueError(
                "jno.solve.staggered needs the problem's block layout, which only `fem.solve(...)` can "
                "supply — it resolves the trial symbols you passed to their DOF blocks. Used as a bare "
                "callable there is nothing to alternate over."
            )
        from .utils.solver.newton_krylov import staggered_newton

        if direct and jacobian is None:
            raise ValueError(
                "jno.solve.staggered(direct=True) factorizes each field's ASSEMBLED diagonal block, and "
                "nothing supplied the tangent. It composes where the assembler provides one — a native "
                "nonlinear FEM problem, steady or on a `tau=` load-path march — reached via "
                "fem.solve(nonlinear=jno.solve.staggered([...], direct=True)). On a matrix-free-only "
                "problem drop direct= (the sub-solves are then unpreconditioned Krylov)."
            )
        return staggered_newton(
            residual_fn,
            u0,
            resolved["blocks"],
            rtol=rtol,
            atol=atol,
            max_sweeps=max_sweeps,
            inner_steps=inner_steps,
            inner_tol=inner_tol,
            inner_maxit=inner_maxit,
            linear_solve=linear_solve,
            jacobian=jacobian if direct else None,
            over_relax=float(over_relax),
            project=project,
            constrained=constrained if constrained is not None else resolved["constrained"],
            damping=damping,
            line_search=line_search,
            ls_max=ls_max,
            ls_c=ls_c,
        )

    spec = NonlinearSolver(
        _fn, name="staggered", direct=direct, traits={"vmap": "native", "jit": True, "rtol": rtol, "atol": atol}
    )
    # Over-relaxation steps PAST the sub-solve's answer, so a box-constrained field needs the projector
    # the `bounds` wrapper owns; ask for it only when it is actually needed.
    spec.wants_project = float(over_relax) != 1.0
    spec.prepare = _prepare  # eager block resolution at compose time (mirrors a precond spec's .prepare)
    return spec


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
    criterion: Any = None,
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

    **``criterion=``** refines on a **traced expression** instead of the recovery error estimator --
    which is how production AMR codes actually mark, on a physical quantity rather than an error
    estimate. It carries **no test function**; a criterion is a field, not an equation::

        ui = u.bind(x=xi, y=yi)

        remesh(criterion=jno.np.sqrt(ui.x**2 + ui.y**2))   # gradient / shock detector
        remesh(criterion=phi * (1.0 - phi))                # phase-field interface
        remesh(criterion=jno.np.abs(uy.x - ux.y))          # 2-D vorticity, on a vector field
        remesh(criterion=d.by_region({"weld": 1.0, "plate": 0.0}))   # refine one material

    The expression is assembled against this problem's own test function, normalised by the lumped
    mass to a nodal field, and integrated per cell; ``theta``, ``refine_factor`` and the remesh
    mechanism are unchanged, so it composes with everything else here. On a multifield problem
    ``metric_field`` selects which field's space it is assembled in. A term that already carries the
    test function is accepted too.

    Second derivatives in a criterion (a Löhner-style ``|D2u|/|Du|`` detector) need ``order >= 2``:
    a P1 Hessian is identically zero, so at order 1 such a criterion evaluates to nothing.

    **A criterion may instead be a per-CELL geometry quantity** -- ``d.cell_aspect()``,
    ``d.cell_volume()``, or anything built from them. It carries no trial or test function, which is
    how it is told apart, and it is evaluated rather than assembled. It must be one value per cell;
    reduce a multi-component one yourself (``jno.np.min(d.cell_angles(), axis=1)``).

    **A criterion may be a CONDITION rather than a ranking** -- ``jno.le(expr, bound)`` /
    ``jno.ge(...)``. Then its signed margin marks every cell that breaks it (not a ``theta`` fraction,
    and ``theta`` is refused), and the march stops when none does. This is what lets a mesh condition
    be its own trigger, with no cadence or threshold argument::

        remesh(criterion=lambda d: jno.le(d.cell_aspect(), 2.0))   # keep every element decent

    Measured on a deliberately stretched mesh: worst aspect 2.87 -> 1.57 in one round, 0 marked on the
    next. Set a bound the mesher can reach -- an unstructured 2-D mesh bottoms out near 1.2-1.5, and a
    tighter bound never settles. A bare comparison (``q > 2.0``) is refused: it says which cells are
    bad but not by how much, so marking would take a fraction of them and leave the rest.

    **Pass a callable** (``criterion=lambda d: ...``) for a geometry criterion. A geometry node
    captures the cell table when it is constructed, so a single node keeps answering for the mesh it
    was built on; once refinement changes the topology it is refused by name rather than read as a
    shape mistake.

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
        criterion=criterion,
    )


def refine(
    *,
    criterion: Any = None,
    theta: float = 0.5,
    max_iters: int = 8,
    max_dofs: int | None = None,
    tol: float | None = None,
    eps: float | None = None,
    metric_field: int = 0,
) -> AdaptSpec:
    """**h-adaptivity by local refinement** for ``fem.solve(adapt=...)``: split the marked cells.

    Beside :func:`remesh` rather than a flag on it, because it is a different algorithm. ``remesh``
    rebuilds the mesh at a finer size field, so it needs a geometry to rebuild from and returns a mesh
    that does not nest inside the old one. ``refine`` splits each marked cell into 4 (a quadrilateral)
    or 8 (a hexahedron): local, needs no mesher, works on a mesh loaded from a file, and every existing
    node survives with its value::

        d = jno.Shape.rect(0, 0, 1, 1).quad().structured(n=8).domain()
        u = fem.solve(adapt=jno.solve.refine(theta=0.4, max_iters=4))

        d = jno.Shape.box(0, 0, 0, 1, 1, 1).structured(n=4).quad().domain()   # hexes
        u = fem.solve(adapt=jno.solve.refine(criterion=jno.np.abs(ui.x)))     # composes with a criterion

    **For hexahedra this is the only h-adaptivity there is.** No general all-hex mesher exists -- gmsh's
    ``Recombine3DAll`` on a plain box returns tetrahedra and no hexahedra -- so ``remesh`` has nothing
    to remesh *to* and refuses by name.

    The price is that the mesh stops being conforming: a split cell's edge midpoint is not a vertex of
    its unrefined neighbour, so its value is not free. Such **hanging nodes** are constrained to the
    coarse facet they lie on, ``u = sum_i w_i u_parent_i`` -- the same relation a periodic tie and a
    mortar coupling impose, and carried by the same prolongation. Neighbours are kept within one
    refinement level (a 2:1 balance), so no constrained node ever has a constrained parent.

    ``theta`` (Dörfler marking), ``criterion``, ``max_iters``, ``max_dofs``, ``tol`` and ``eps`` mean
    exactly what they do on :func:`remesh`. There is no ``refine_factor``: a split halves the cell by
    construction. There is no ``anisotropic``: the split is isotropic, so there is no direction to
    stretch along -- that needs a simplex mesh and ``remesh(anisotropic=True)``.

    **Limitations, measured.** Quadrilateral and hexahedral meshes only; a simplex mesh refuses by name
    and should use :func:`remesh`, whose mmg path is local already. A hanging node landing on a tied or
    periodic interface is refused rather than composed. Steady problems only -- the transient driver
    transfers state across a remesh, and that path does not yet carry a constraint set. Geometry is
    exact for affine cells; a warped hexahedron's faces are non-planar, so refining it moves the volume
    by an O(h^2) amount (measured: 3.9e-04 for a 0.06 warp on a 0.25 cell).

    Returns:
        AdaptSpec: The adaptation spec to pass as ``fem.solve(adapt=...)``.
    """
    from .utils.solver.fem_adapt import AdaptSpec

    return AdaptSpec(
        split=True,
        criterion=criterion,
        theta=theta,
        max_iters=max_iters,
        max_dofs=max_dofs,
        tol=tol,
        eps=eps,
        metric_field=metric_field,
    )


def relocate(
    *,
    objective: Any = "equidistribution",
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

    **Two objectives, and the choice is the problem's, not a preference.** ``objective=`` picks *what*
    descent minimises:

    - ``"equidistribution"`` (default) equidistributes an **arclength monitor** — it targets *resolution*,
      and wins where a feature is under-resolved or moving.
    - ``"energy"`` descends the **FE Dirichlet energy**. For a Ritz method
      ``E_h - E_exact = 1/2 ||u - u_h||_E^2``, so on a steady problem the energy *is* the error norm and
      descending it minimises the error directly.
    - ``"huang"`` is Huang's equidistribution–alignment functional (see :class:`AdaptSpec`).

    **Or a weak-form expression**, when the mesh has a job the three functionals cannot state. They are
    mesh-*quality* measures: they see the solution only through a monitor, so they can ask for
    resolution but not for a physical condition. An expression is assembled exactly as ``criterion=``
    is and summed to a scalar, over a **volume or a boundary** region::

        xs, ys, ns = domain.variable("side", normals=True, split=True)
        ys.trainable()                                   # the wall may move along y only
        us = u.bind(x=xs, y=ys)
        fem.solve(adapt=jno.solve.relocate(objective=jno.np.inner(us, ns) ** 2))

    That is a **free surface**: the wall is moved until the flow through it vanishes. The facet normals
    are rebuilt from the moving vertices, so ``n`` is the current mesh's normal, not the initial one.
    The gradient runs through the solve, as it does for the strings — matched to central differences at
    7.5e-09 on a Stokes channel whose no-slip bottom couples the flow to the wall's position, where the
    through-flow falls 11.4x over 60 rounds (12.5x at 120: this is a descent, not a root-find).

    Two things to know. The objective is a **scalar**, so it needs a scalar test function: on a
    velocity/pressure saddle the pressure test is picked automatically. And when the expression reaches
    its region only through a **bound view** (``u.bind(x=xr, y=yr)``, which absorbs its coordinates),
    the test function cannot be auto-bound — carry it yourself, ``objective=<expr> * v_r[0]``. That
    case raises with this instruction rather than a trace-level binding error.

    The region's facet quadrature tables are built only when the **form** carries a surface term, so a
    surface objective needs the boundary term to be in the ``jno.fem([...])`` list (a traction-free
    wall, ``0.0 * v_r[0]``, is enough).

    Neither dominates, measured on the two problem types the test suite pins:

    ==========================  =========================  =========================
    objective                   L-shape corner (fixed)     Allen–Cahn front (moving)
    ==========================  =========================  =========================
    ``"energy"``                **55 % error cut**         10.7x WORSE than uniform
    ``"equidistribution"``      12 % worse                 **0.51x uniform**
    ==========================  =========================  =========================

    So: a **fixed singularity** wants ``"energy"``; an **under-resolved or moving front** wants the
    default. The energy is also not scale-free — a vector field carries the energy of all its components,
    so ``lr`` is problem-scaled — where the monitor functionals are.

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
        objective=objective,
        relocate_method=method,
        max_iters=max_iters,
        lr=lr,
        quality_floor=quality_floor,
        ma_relax=relax,
        ma_dt=relax_step,
    )


def enrich(
    *,
    criterion=None,
    theta: float = 0.5,
    max_iters: int = 8,
    max_dofs: int | None = None,
    tol: float | None = None,
    eps: float | None = None,
    metric_field: int = 0,
):
    """**p-adaptivity** for ``fem.solve(adapt=...)``: raise the polynomial order where it is needed,
    leaving the mesh alone.

    The fourth adaptivity beside :func:`remesh` (rebuild finer), :func:`refine` (split cells) and
    :func:`relocate` (move nodes). This one changes neither the points nor the connectivity: it
    switches **interpolation covers** on at the marked nodes, so the field gains coefficients where
    the solution needs them and stays P1 elsewhere. Requires a field declared ``space="cover"``::

        u, phi = d.fem_symbols(space="cover")
        fem = jno.fem([...])
        fem.solve(adapt=jno.solve.enrich(criterion=jno.np.sqrt(ui.x**2 + ui.y**2)))

    Why this is the cheap route to variable ``p``: enrichment rides the partition of unity, so an
    enriched node next to an unenriched one blends automatically. There are no constraint equations
    at an order interface and no edge-mode bookkeeping, which is what a hierarchical p-basis needs.
    Compose with :func:`refine` across successive solves for **hp** — h where the solution is rough,
    p where it is smooth.

    Args:
        criterion: A **traced expression** marking where to enrich, exactly as in :func:`remesh` --
            a field, carrying no test function (``jno.np.sqrt(ui.x**2 + ui.y**2)``, ``phi*(1-phi)``,
            ``d.by_region({...})``). Omitted, the Zienkiewicz–Zhu recovery estimator is used,
            spread from cells to their nodes.
        theta: Dörfler bulk-marking fraction, over NODES rather than cells: the fewest nodes whose
            indicator reaches ``theta`` of the total are enriched each round.
        max_iters: Maximum enrich-solve rounds.
        max_dofs: Stop once the system reaches this many ACTIVE DOFs. Active, not total: the padded
            layout gives every node its cover slots and an unenriched node simply has them pinned, so
            the total never changes and only the free count tracks the enrichment.
        tol: Stop once the global indicator falls below this.
        eps: Stop when the indicator plateaus for two consecutive rounds.
        metric_field: Which field of a coupled problem drives the marking.

    Scope: simplices only, first-order covers, and the enriched field must be ``space="cover"``.
    ``fem.adapt_history`` records ``n_enriched`` per round beside the usual ``n_dofs``/``estimate``.
    """
    from .utils.solver.fem_adapt import AdaptSpec

    return AdaptSpec(
        enrich=True,
        criterion=criterion,
        theta=theta,
        max_iters=max_iters,
        max_dofs=max_dofs,
        tol=tol,
        eps=eps,
        metric_field=metric_field,
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


def adaptive(
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 1000,
    dt0: float | None = None,
    limit=None,
    shrink: float = 0.5,
    grow: float = 1.5,
):
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
    not as a speed optimization.

    **On a pseudo-time LOAD PATH** — ``fem.solve(tau=jno.solve.adaptive(limit=...))`` on a
    ``domain(tau=...)`` history march — the criterion is different, and it has to be. A rate-independent
    load path has no local truncation error to estimate: each step is an *equilibrium*, not an
    approximation to a trajectory, so Richardson measures nothing. ``limit`` instead bounds how much the
    solution may change in one step::

        fem.solve(tau=jno.solve.adaptive(limit=0.05))            # every DOF
        fem.solve(tau=jno.solve.adaptive(limit=[(dm, 0.05)]))    # per field — the usual case

    A step is **rejected** (and the step size cut by ``shrink``) when the solve fails to converge or the
    change exceeds ``limit``; a comfortable step grows by ``grow``. That matters beyond cost: with a
    fixed grid a step can converge perfectly and still skip an entire propagation event, giving a valid
    sequence of equilibria with no resolved event between them — and because the march is path-dependent
    (history + irreversibility), that is a different answer, not just a coarser one.

    ``limit`` is **required** in the ``tau=`` slot and rejected in ``time=`` (and vice versa for
    ``rtol``/``atol``) — the two controllers measure different things and silently applying one where the
    other was meant would be a plausible wrong answer.
    """
    from .utils.solver.timeschemes import _AdaptiveScheme

    return _AdaptiveScheme(None, rtol, atol, max_steps, dt0, limit=limit, shrink=shrink, grow=grow)
