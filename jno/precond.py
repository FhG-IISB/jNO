"""``jno.precond`` -- preconditioner **specs** for ``fem.solve(precond=...)``.

A spec is declarative: it says *what* preconditioner to build, and jno materializes it at solve
time against a :class:`jno.utils.solver.solver_api.PrecondContext` (the assembled operator; later
per-field blocks and auxiliary weak-form assembly). The materialized applier is just
``v -> M^{-1} v`` and composes with any Krylov solver from ``jno.solve``.

Preconditioners change convergence *speed*, never the converged solution, so a spec needs no
gradient path -- arbitrary (even non-JAX, via a future callback tier) appliers stay compatible
with differentiable solves. A user spec is any object with ``materialize(ctx)`` or a bare
``ctx -> (v -> M^{-1} v)`` callable::

    def my_precond(ctx):
        inv = 1.0 / ctx.diag()
        return lambda v: inv * v

    fem.solve(linear=jno.solve.cg(), precond=my_precond)

Block/Schur composition and form-based (auxiliary weak-form) specs are the next phase -- see
``plans/fem-solver-api.md``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .utils.solver.solver_api import PrecondContext  # noqa: F401  (re-export for user specs)

__all__ = ["PrecondContext", "jacobi", "chebyshev"]


class _Jacobi:
    """Spec for the diagonal (Jacobi) preconditioner; see :func:`jacobi`."""

    def materialize(self, ctx: PrecondContext):
        d = ctx.diag()
        safe = jnp.where(jnp.abs(d) > 1e-30, d, 1.0)  # zero diagonals (saddle blocks) left unscaled
        inv = 1.0 / safe
        return lambda v: inv * v

    def __repr__(self):
        return "jno.precond.jacobi()"


class _Chebyshev:
    """Spec for the fixed-degree Chebyshev polynomial preconditioner; see :func:`chebyshev`."""

    def __init__(self, degree, lmin, lmax, lmin_ratio, safety, bound_iters):
        self.degree = degree
        self.lmin, self.lmax = lmin, lmax
        self.lmin_ratio, self.safety, self.bound_iters = lmin_ratio, safety, bound_iters

    def materialize(self, ctx: PrecondContext):
        from .utils.solver.krylov import chebyshev_apply, power_iteration_bound

        if ctx.A.shape is None and self.lmax is None:
            raise TypeError(
                "jno.precond.chebyshev on a matvec-only operator needs explicit spectrum bounds "
                "(lmin=, lmax=) — there is no assembled matrix to estimate them from."
            )
        hi = self.lmax
        if hi is None:
            n = ctx.A.shape[0]
            hi = self.safety * power_iteration_bound(ctx.A.mv, n, iters=self.bound_iters)
        lo = self.lmin if self.lmin is not None else self.lmin_ratio * hi
        return lambda v: chebyshev_apply(ctx.A.mv, v, lmin=lo, lmax=hi, degree=self.degree)

    def __repr__(self):
        return f"jno.precond.chebyshev(degree={self.degree})"


def jacobi() -> _Jacobi:
    """Diagonal (Jacobi) preconditioner ``M^{-1} v = v / diag(A)``.

    The cheapest useful preconditioner: one elementwise multiply per application, ``jit``- and
    ``vmap``-native, effective on diagonally-dominant (elliptic) systems -- heat, diffusion,
    elasticity. Zero/near-zero diagonals (e.g. the pressure block of a saddle-point system) are
    left unscaled so it never produces ``inf``/``NaN`` -- but it does not *rescue* saddle
    systems; use ``jno.solve.lu()`` there (block/Schur specs are planned).

    ``fem.solve(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi())`` reproduces the
    historic steady-linear default exactly.
    """
    return _Jacobi()


def chebyshev(
    *,
    degree: int = 8,
    lmin: float | None = None,
    lmax: float | None = None,
    lmin_ratio: float = 1.0 / 30.0,
    safety: float = 1.05,
    bound_iters: int = 30,
) -> _Chebyshev:
    """Fixed-degree Chebyshev **polynomial** preconditioner ``M^{-1} = p_degree(A) ≈ A^{-1}``
    for SPD operators (Saad 2003, §12.3 / Golub & Varga 1961 — the same recurrence as
    ``jno.solve.chebyshev``, truncated at ``degree`` with no convergence test, which keeps the
    application a fixed *linear* map so it may precondition CG and MINRES).

    The GPU-era substitute for Gauss-Seidel/ILU smoothing: only matvecs and AXPYs — no
    reductions, no triangular solves — ``jit``- and ``vmap``-native. Spectrum bounds of ``A``
    are taken from ``lmin``/``lmax`` when given, else estimated by power iteration (``safety``
    inflation, ``lmin = lmin_ratio * lmax``).
    """
    return _Chebyshev(degree, lmin, lmax, lmin_ratio, safety, bound_iters)
