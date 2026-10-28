"""Subspaces of the unknowns, as values -- for coarse spaces, deflation and constrained eigensolves.

A **space** is a set of vectors ``Z`` (``n x k``) on the unknowns of whatever problem it is used with. Every
consumer takes the same values -- :func:`jno.precond.coarse` (and ``schwarz(coarse=...)``),
``jno.precond.amg(nullspace=...)``, ``fem.eigs(deflate=...)``:

* **formulas** in the domain's coordinates, one per vector: ``[1]``, ``[1, x, y]``, the 2-D rigid-body modes
  ``[(1, 0), (0, 1), (-y, x)]`` (a tuple per vector field), ``{p: [1]}`` for one field of several. Evaluated
  at the DOF nodes of the problem when the consumer sets up, so one value serves every problem on its domain;
* an ``(n, k)`` **array** on the system being solved (the reduced one when a tie reduces it);
* a **callable** ``ctx -> (n, k)``: ``ctx.fem`` the problem, ``ctx.n`` the size, ``ctx.A`` the operator;
* a **computed space** from this module -- an object with ``basis(ctx) -> (n, k)``.

A computed space is resolved once, when its consumer sets up (it is host work, outside the solve's trace). A
space that changes from solve to solve -- a Krylov space recycled along a time march or a Newton loop -- is
not supported yet: an object with an ``update`` method is refused rather than silently frozen.
"""

from __future__ import annotations

import numpy as np

__all__ = ["eigvecs"]


class _EigVecs:
    """The ``k`` lowest eigenvectors of the operator; see :func:`eigvecs`."""

    def __init__(self, k, tol, maxiter):
        self.k, self.tol, self.maxiter = k, tol, maxiter

    def basis(self, ctx):
        """The eigenvectors, ``(n, k)``, of ``ctx.A`` (LOBPCG with a Jacobi preconditioner).

        Args:
            ctx: the setup context: ``ctx.A`` the concrete operator, ``ctx.n`` its size.

        Returns:
            The ``(n, k)`` eigenvectors, orthonormal.
        """
        import jax.numpy as jnp

        from . import precond as _precond
        from .utils.solver.eigen import lobpcg_geneigh
        from .utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

        A = getattr(ctx, "A", None)
        if A is None:
            raise ValueError(
                f"{self!r}: the eigenvectors are those of the operator, and this consumer set up without one. "
                "Use it through fem.solve(precond=...), or give the vectors as an array."
            )
        op = A if isinstance(A, LinearOperator) else LinearOperator(A)
        M = materialize_precond(_precond.jacobi(), PrecondContext(op))
        lam, X, res = lobpcg_geneigh(A, None, self.k, "smallest", precond=M, tol=self.tol, maxiter=self.maxiter)
        if not float(res) <= self.tol:
            raise RuntimeError(
                f"{self!r}: LOBPCG reached a relative residual of {float(res):.1e}, short of tol={self.tol:g}, in "
                f"{self.maxiter} sweeps. Raise maxiter, loosen tol (a slightly inexact deflation space still "
                "deflates), or ask for fewer vectors."
            )
        return np.asarray(jnp.asarray(X))

    def __repr__(self):
        return f"jno.space.eigvecs({self.k})"


def eigvecs(k: int, *, tol: float = 1e-6, maxiter: int = 500) -> _EigVecs:
    """The ``k`` lowest eigenvectors of the operator being solved, as a space.

    The classical deflation space: removing the smallest eigenvalues from a Krylov solve shrinks the
    effective condition number to ``lambda_max / lambda_{k+1}`` (Nicolaides, SIAM J. Numer. Anal. 24(2),
    1987; Frank & Vuik, SIAM J. Sci. Comput. 23(2), 2001). Computed once, at setup, by preconditioned LOBPCG
    (Knyazev 2001) on the operator the solve runs on -- so it pays when the operator is reused: a march, a
    sweep of right-hand sides, ``reuse=True``. ::

        fem.solve(linear=jno.solve.fgmres(), precond=jno.precond.jacobi() >> jno.precond.coarse(jno.space.eigvecs(8)))

    Args:
        k: number of eigenvectors.
        tol: LOBPCG's relative residual; not reaching it raises.
        maxiter: LOBPCG sweep budget.

    Returns:
        A space for :func:`jno.precond.coarse`, ``amg(nullspace=)`` or ``fem.eigs(deflate=)``.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"jno.space.eigvecs(k={k!r}): must be a positive int.")
    return _EigVecs(int(k), float(tol), int(maxiter))
