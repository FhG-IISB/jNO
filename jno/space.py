"""Subspaces of the unknowns, as values -- for coarse spaces, deflation and constrained eigensolves.

A **space** is a set of vectors ``Z`` (``n x k``) on the unknowns of whatever problem it is used with. Every
consumer takes the same values -- :func:`jno.precond.coarse` (and ``schwarz(coarse=...)``),
``jno.precond.amg(nullspace=...)``, ``fem.eigs(deflate=...)``:

* **formulas** in the domain's coordinates, one per vector: ``[1]``, ``[1, x, y]``, the 2-D rigid-body modes
  ``[(1, 0), (0, 1), (-y, x)]`` (a tuple per vector field), ``{p: [1]}`` for one field of several. Evaluated
  at the DOF nodes of the problem when the consumer sets up, so one value serves every problem on its domain;
* an ``(n, k)`` **array** on the system being solved (the reduced one when a tie reduces it);
* a **callable** ``ctx -> (n, k)``: ``ctx.fem`` the problem, ``ctx.n`` the size, ``ctx.A`` the operator;
* a **computed space** from this module -- an object with ``basis(ctx) -> (n, k)``: :func:`eigvecs` (the lowest
  eigenvectors of the operator) or :func:`geneo` (GenEO, per Schwarz subdomain).

A computed space is resolved once, when its consumer sets up (it is host work, outside the solve's trace). A
space that changes from solve to solve -- a Krylov space recycled along a time march or a Newton loop -- is
not supported yet: an object with an ``update`` method is refused rather than silently frozen.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["eigvecs", "geneo"]


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


class _GenEO:
    """GenEO vectors on a Schwarz preconditioner's subdomains; see :func:`geneo`."""

    on_parts = True  # computed on the consumer's Schwarz subdomains; its columns keep their own supports

    def __init__(self, tau, nev):
        self.tau, self.nev = tau, nev
        self.counts = None

    def basis(self, ctx):
        """The vectors, a sparse ``(n, K)``, from ``ctx.fem``'s element matrices on the subdomains ``ctx.schwarz``.

        Args:
            ctx: the setup context: ``ctx.fem`` the problem, ``ctx.A`` the concrete operator, ``ctx.schwarz``
                the Schwarz pattern whose subdomains the eigenproblems are solved on.

        Returns:
            A ``scipy.sparse`` CSC ``(n, K)``; ``self.counts`` holds the number kept per subdomain.
        """
        from .utils.solver.geneo import geneo_space

        pat = self._schwarz_of(ctx)
        Z, self.counts = geneo_space(ctx.fem, ctx.A, pat, self.tau, self.nev, who=repr(self))
        return Z

    def static_setup(self, ctx, *, extra=None, headroom: bool = False):
        """``(plan, pattern)``: the space with a STATIC structure -- ``nev`` columns on every subdomain, the ones
        above ``tau`` masked -- and its values at ``ctx.A``. The coarse pattern's structure depends on the Schwarz
        pattern alone, so the values can be recomputed from a later operator, inside a compiled solve too
        (:func:`~jno.utils.solver.geneo.geneo_values`). ``extra``: the problem's structural entries; ``headroom``:
        size ``nev=None`` for an operator that changes (twice the count found now)."""
        import logging

        from .utils.solver.deflation import setup_operator, static_pattern
        from .utils.solver.geneo import geneo_pieces, geneo_plan, geneo_values

        pat = self._schwarz_of(ctx)
        A = setup_operator(ctx.fem, ctx.A)
        plan, w, V, vals = geneo_plan(ctx.fem, A, pat, self.tau, self.nev, headroom=headroom, who=repr(self))
        row, part = geneo_pieces(plan, pat)
        cpat = static_pattern(row, part, pat.p, plan.nev, ctx.A, extra)
        with jax.ensure_compile_time_eval():
            null, W, kmask, counts = geneo_values(plan, pat, A, vals, eigs=(w, V))
        self.counts = np.asarray(counts)
        logging.getLogger("jno").info(
            f"{self!r}: {int(self.counts.sum())} coarse vectors on {pat.p} subdomains ({self.counts.min()}-"
            f"{self.counts.max()} per subdomain, room for {plan.nev})."
        )
        return plan, cpat._replace(null=null, W=W, kmask=kmask)

    def _schwarz_of(self, ctx):
        pat = getattr(ctx, "schwarz", None)
        if pat is None:
            raise ValueError(
                f"{self!r}: the vectors are computed per subdomain of an overlapping Schwarz preconditioner. Use "
                f"jno.precond.coarse({self!r}, parts=jno.precond.schwarz(coarse=False)) or "
                f"jno.precond.schwarz(coarse={self!r})."
            )
        return pat

    def __repr__(self):
        nev = "" if self.nev is None else f", nev={self.nev}"
        return f"jno.space.geneo(tau={self.tau:g}{nev})"


def geneo(tau: float = 0.2, *, nev=None) -> _GenEO:
    """The **GenEO** coarse space: per overlapping Schwarz subdomain, the slow modes its local solve cannot fix.

    For every subdomain ``i`` it solves the generalized eigenproblem ``A_i^N v = lambda D_i A_ii D_i v`` and keeps
    the eigenvectors with ``lambda < tau`` as coarse vectors ``R_i^T D_i v`` (Spillane, Dolean, Hauret, Nataf,
    Pechstein & Scheichl, Numer. Math. 126, 2014; Dolean, Jolivet & Nataf, *An Introduction to Domain
    Decomposition Methods*, SIAM 2015, ch. 7). ``A_i^N`` is the operator assembled over the subdomain's own cells
    (natural conditions on its cut), ``A_ii`` its block of the global operator, ``D_i`` the partition of unity.
    The space adapts to the coefficients: a subdomain cut by a stiffness jump gets the extra near-free motions
    that rigid-body modes miss, a homogeneous one only its rigid modes. Two-level Schwarz with it converges in a
    number of iterations bounded independently of the subdomain count and the coefficient contrast (for an SPD
    operator, with the bound set by ``tau``). ::

        S = jno.precond.schwarz(parts=64, coarse=False)
        Q = jno.precond.coarse(jno.space.geneo(), parts=S)
        fem.solve(linear=jno.solve.cg(), precond=Q >> S >> Q)     # or schwarz(parts=64, coarse=jno.space.geneo())

    The eigenproblems are dense, batched over the subdomains on the device; computed once, when the
    preconditioner sets up -- or, with ``reuse=k`` on the spec that holds it (``coarse(..., reuse=1)``,
    ``schwarz(coarse=geneo(), reuse=1)``), recomputed every k-th solve, and with ``reuse=1`` at every
    linearization inside ``jno.solve.newton()`` and ``staggered(direct=True)``, for a stiffness pattern that
    moves (a crack, a design, yielding). Needs a real, symmetric positive definite problem, on native FEM of any element or
    else with a symmetric diagonally dominant operator (FDM diffusion, say: its edges stand in for the cells). A
    periodic tie, slip or hanging nodes are fine (a real ``P``: each cell's matrix is carried over as
    ``P^T K_e P``), and so is a march (the cells combine as the step operator does; its mass must not change). No
    eigenvalue below ``tau`` -- a very short time step -- gives an empty space and no correction. Anything else
    raises.

    Args:
        tau: keep the eigenvalues below it (between 0 and 1; larger keeps more vectors and converges faster).
        nev: at most this many vectors per subdomain; ``None`` keeps every one below ``tau`` (under ``reuse=k``:
            room for twice the count the setup's operator needs; a subdomain that fills it is logged).

    Returns:
        A space for ``jno.precond.coarse(..., parts=schwarz(coarse=False))`` or ``jno.precond.schwarz(coarse=)``.
    """
    if isinstance(tau, bool) or not isinstance(tau, (int, float, np.floating, np.integer)) or not tau > 0:
        raise ValueError(f"jno.space.geneo(tau={tau!r}): must be a positive number.")
    if nev is not None and (isinstance(nev, bool) or not isinstance(nev, (int, np.integer)) or nev < 1):
        raise ValueError(f"jno.space.geneo(nev={nev!r}): must be a positive int or None.")
    return _GenEO(float(tau), None if nev is None else int(nev))
