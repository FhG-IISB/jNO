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

from .utils.solver.solver_api import LinearOperator, LinearSolver, NonlinearSolver, _maybe_residual_check

__all__ = ["LinearOperator", "LinearSolver", "NonlinearSolver", "lu", "dense", "cg", "bicgstab", "gmres", "newton"]


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

        A = op.bcoo if op.bcoo is not None else op.dense()
        return sparse_lu_solve(A, b)

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


def newton(
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 100,
    inner_tol: float = 1e-10,
    inner_maxit: int = 2000,
) -> NonlinearSolver:
    """Jacobian-free Newton-Krylov -- the (unchanged) nonlinear default, as a configurable slot.

    Wraps :func:`jno.utils.solver.newton_krylov.newton_krylov`: ``J @ v`` from a JVP, inner
    matrix-free solve (default BiCGStab, or the ``linear=`` slot when given), implicit
    differentiation via ``lax.custom_root`` so gradients reach parameters without unrolling."""

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
        )

    return NonlinearSolver(_fn, name="newton")
