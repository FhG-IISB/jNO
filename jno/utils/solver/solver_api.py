"""Slot-based solver API: the contracts behind ``fem.solve(x0=, nonlinear=, linear=, precond=)``.

Design (see ``plans/fem-solver-api.md``): the FEM solver space factorises into four orthogonal
slots -- warm start, linearization driver, inner linear solve, preconditioner. Every slot takes a
**callable** (no strings); the shipped defaults live in the ``jno.solve`` / ``jno.precond``
namespaces and are pure JAX (jit- + vmap-native), reusing ``jax.scipy.sparse.linalg`` and the
existing ``sparse_lu_solve`` rather than re-implementing solvers. ``fem.solve(solve_fn=...)``
remains the total override and is unchanged.

Contracts
---------
* linear solver: ``fn(A, b, *, M=None, x0=None) -> x`` with ``A`` a :class:`LinearOperator`
  (``.mv(v)``, ``.T.mv(v)``, ``.diag()``, ``.bcoo``, ``.dense()``). Pure-JAX implementations
  inherit ``jit``/``vmap``/AD; the ``jax.scipy`` Krylov wrappers are built on
  ``lax.custom_linear_solve`` upstream, so they are implicitly differentiable already.
* nonlinear driver: ``fn(residual_fn, u0, *, linear_solve=None) -> u`` where ``linear_solve`` is
  a matrix-free ``(matvec, rhs) -> x`` built from the ``linear`` slot.
* preconditioner spec: an object with ``materialize(ctx) -> (v -> M^{-1} v)`` (or a bare
  ``ctx -> apply`` callable). Specs are declarative -- jno materializes them at solve time with a
  :class:`PrecondContext` carrying the assembled operator. A preconditioner only changes
  convergence *speed*, never the converged solution, so it needs **no** gradient path
  (the differentiability firewall of ``custom_linear_solve`` / ``custom_root``).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .linear import matrix_diagonal

__all__ = [
    "LinearOperator",
    "LinearSolver",
    "NonlinearSolver",
    "PrecondContext",
    "materialize_precond",
    "compose_linear_solve_fn",
    "compose_nonlinear_solve_fn",
]


class LinearOperator:
    """Uniform handle over an assembled operator (BCOO, dense array, or bare matvec).

    Gives every linear solver one interface regardless of the storage the assembler produced:
    ``.mv(v)`` (also ``@``), lazy transpose ``.T``, ``.diag()``, ``.dense()``, and ``.bcoo``
    (``None`` when not sparse). A matvec-only operator (``from_matvec``) supports ``mv`` and a
    transpose via ``jax.linear_transpose``; ``diag``/``dense`` raise -- direct solvers and
    diagonal preconditioners need an assembled matrix.
    """

    def __init__(self, A: Any, *, _transposed: bool = False):
        if callable(A) and not hasattr(A, "shape"):
            raise TypeError("Wrap a bare matvec with LinearOperator.from_matvec(fn).")
        self._A = A
        self._mv = None
        self._transposed = _transposed

    @classmethod
    def from_matvec(cls, mv: Callable, *, _transposed: bool = False) -> "LinearOperator":
        op = cls.__new__(cls)
        op._A = None
        op._mv = mv
        op._transposed = _transposed
        return op

    @property
    def shape(self):
        if self._A is None:
            return None
        s = self._A.shape
        return (s[1], s[0]) if self._transposed else s

    def mv(self, v):
        if self._mv is not None:
            if self._transposed:
                # transpose of a linear map, via JAX's transposition (exact, not an approximation)
                (out,) = jax.linear_transpose(self._mv, jnp.zeros_like(v))(v)
                return out
            return self._mv(v)
        # ``v @ A`` is the transposed matvec for both BCOO and dense -- no transpose materialised
        return (v @ self._A) if self._transposed else (self._A @ v)

    __matmul__ = mv
    __call__ = mv

    @property
    def T(self) -> "LinearOperator":
        if self._mv is not None:
            return LinearOperator.from_matvec(self._mv, _transposed=not self._transposed)
        return LinearOperator(self._A, _transposed=not self._transposed)

    def diag(self):
        if self._A is None:
            raise TypeError("LinearOperator.diag(): a matvec-only operator has no assembled diagonal.")
        return matrix_diagonal(self._A)  # transpose shares the diagonal

    def dense(self):
        if self._A is None:
            raise TypeError("LinearOperator.dense(): a matvec-only operator cannot densify.")
        M = self._A.todense() if hasattr(self._A, "todense") else jnp.asarray(self._A)
        return M.T if self._transposed else M

    @property
    def bcoo(self):
        return self._A if hasattr(self._A, "todense") else None


def _maybe_residual_check(op: LinearOperator, b, x, who: str, *, rtol: float = 1e-4):
    """Eager-only convergence guard: raise on a garbage solution when values are concrete.

    Under ``jit``/``vmap``/``grad`` (tracers) it is a no-op -- the check would either fail to
    concretise or break the transform; there the solver's own iteration cap is the guard.
    """
    if any(isinstance(v, jax.core.Tracer) for v in (x, b)):
        return x
    import numpy as np

    rel = float(jnp.linalg.norm(b - op.mv(x)) / (jnp.linalg.norm(b) + 1e-30))
    if not np.isfinite(rel) or rel > rtol:
        raise RuntimeError(
            f"jno.solve.{who} did not solve the system (relative residual {rel:.1e}); the problem may be "
            "singular/ill-posed or need a preconditioner. Try jno.solve.lu(), a precond= spec, or your own solve_fn."
        )
    return x


class LinearSolver:
    """A configured linear solver: ``solver(A, b, *, M=None, x0=None) -> x``.

    ``fn`` receives ``(op: LinearOperator, b, M, x0)``. ``traits`` documents transform support
    (``vmap: "native" | "sequential" | "no"``) so composition layers (and, later, the auto
    policy) can pick honestly instead of silently host-looping.
    """

    def __init__(self, fn: Callable, *, name: str, traits: Optional[dict] = None, direct: bool = False):
        self._fn = fn
        self.name = name
        self.traits = {"vmap": "native", "jit": True, **(traits or {})}
        self.direct = direct  # a direct solver ignores x0 and takes no preconditioner

    def __call__(self, A, b, *, M=None, x0=None):
        op = A if isinstance(A, LinearOperator) else LinearOperator(A)
        b = jnp.asarray(b).reshape(-1)
        if self.direct and M is not None:
            raise ValueError(f"jno.solve.{self.name} is a direct solver -- it takes no preconditioner (precond=).")
        return self._fn(op, b, M=M, x0=x0)

    def __repr__(self):
        return f"jno.solve.{self.name}({', '.join(f'{k}={v}' for k, v in self.traits.items())})"


class NonlinearSolver:
    """A configured nonlinear driver: ``driver(residual_fn, u0, *, linear_solve=None) -> u``."""

    def __init__(self, fn: Callable, *, name: str, traits: Optional[dict] = None):
        self._fn = fn
        self.name = name
        self.traits = {"vmap": "native", "jit": True, **(traits or {})}

    def __call__(self, residual_fn, u0, *, linear_solve=None):
        return self._fn(residual_fn, u0, linear_solve=linear_solve)

    def __repr__(self):
        return f"jno.solve.{self.name}()"


class PrecondContext:
    """What a preconditioner spec sees at materialization time.

    ``ctx.A`` is the assembled :class:`LinearOperator` (matvec-only on the Jacobian-free
    nonlinear path), ``ctx.fem`` the owning :class:`jno.FEM` (``None`` outside ``fem.solve``),
    and ``ctx.diag()`` the operator diagonal. Block extraction and auxiliary weak-form assembly
    arrive with the block-preconditioner phase (see ``plans/fem-solver-api.md``).
    """

    def __init__(self, A: LinearOperator, fem: Any = None):
        self.A = A
        self.fem = fem

    def diag(self):
        return self.A.diag()


def materialize_precond(spec: Any, ctx: PrecondContext) -> Callable:
    """Turn a preconditioner spec into the ``v -> M^{-1} v`` applier for this solve."""
    if hasattr(spec, "materialize"):
        return spec.materialize(ctx)
    if callable(spec):  # duck-typed: a bare ``ctx -> apply`` factory
        return spec(ctx)
    raise TypeError(
        f"precond= expects a jno.precond.* spec or a callable (ctx) -> (v -> M^-1 v); got {type(spec).__name__}."
    )


def compose_linear_solve_fn(linear, precond, x0, fem=None) -> Callable:
    """Compose the linear-mode slots into the classic ``(A, b) -> x`` ``solve_fn`` contract.

    The composed callable is handed to the *existing* dispatch (plain / periodic-reduced /
    parametric ``FemLinearSystem`` / complex real-block), so every path keeps its current
    reduction and implicit-differentiation behaviour. It accepts the assembler's BCOO operator
    directly -- no densification.
    """
    if linear is None:
        from ... import solve as _solve_ns

        linear = _solve_ns.bicgstab()  # matches the historic matrix-free default
    x0_flat = None if x0 is None else jnp.asarray(x0).reshape(-1)

    def composed(A, b):
        op = A if isinstance(A, LinearOperator) else LinearOperator(A)
        M = materialize_precond(precond, PrecondContext(op, fem)) if precond is not None else None
        return linear(op, jnp.asarray(b).reshape(-1), M=M, x0=x0_flat)

    return composed


def compose_nonlinear_solve_fn(nonlinear, linear, precond, fem=None) -> Callable:
    """Compose the nonlinear-mode slots into the ``(residual_fn, u0) -> u`` ``solve_fn`` contract.

    The inner Newton/Picard linear solve is matrix-free (a JVP matvec), so a ``precond=`` spec
    that needs the assembled matrix cannot be materialized here yet -- form-based preconditioners
    (assembled auxiliary operators) are the planned route; until then this raises.
    """
    if precond is not None:
        raise NotImplementedError(
            "precond= on the matrix-free nonlinear path needs a form-based preconditioner "
            "(assembled auxiliary operator) -- not implemented yet; see plans/fem-solver-api.md."
        )
    if nonlinear is None:
        from ... import solve as _solve_ns

        nonlinear = _solve_ns.newton()

    inner = None
    if linear is not None:
        # adapt the linear slot to the driver's matrix-free ``(matvec, rhs) -> x`` inner contract
        inner = lambda matvec, rhs: linear(LinearOperator.from_matvec(matvec), rhs)

    return lambda residual_fn, u0: nonlinear(residual_fn, u0, linear_solve=inner)
