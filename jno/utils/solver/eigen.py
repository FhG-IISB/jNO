"""Generalized symmetric eigensolver ``K x = λ M x`` (K symmetric, M symmetric positive-definite).

V1a is a **dense** reduction: Cholesky ``M = L Lᵀ`` turns the pencil into the standard problem for
``C = L⁻¹ K L⁻ᵀ`` (solved by :func:`jax.numpy.linalg.eigh`, which carries a JVP, so the eigenvalues are
**differentiable for free**), then maps the eigenvectors back ``x = L⁻ᵀ y`` — leaving them
**M-orthonormal** (``XᵀMX = I``), the invariant every iterative variant must also preserve. Exact and
cheap for the small problems where you want the whole low spectrum; :mod:`jno.solve` exposes it as
``jno.solve.eigs`` and adds a preconditioned LOBPCG path for scale.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def _as_dense(A):
    if A is None:
        return None
    return jnp.asarray(A.todense() if hasattr(A, "todense") else A)


def dense_geneigh(K, M, k: int, which: str = "smallest"):
    """The ``k`` eigenpairs of ``K x = λ M x`` at the requested end of the spectrum.

    Args:
        K: symmetric operator (dense / BCOO / anything with ``.todense()``).
        M: symmetric positive-definite mass operator, or ``None`` for the standard problem ``Kx=λx``.
        k: number of eigenpairs.
        which: ``"smallest"`` (default) or ``"largest"`` by algebraic value.

    Returns:
        ``(λ, X)`` — eigenvalues ``(k,)`` ascending (or descending), and M-orthonormal eigenvectors
        ``(n, k)`` (columns), so ``Xᵀ M X = I``.
    """
    Kd = _as_dense(K)
    Kd = 0.5 * (Kd + Kd.T)  # symmetrise away assembly roundoff
    Md = _as_dense(M)
    if Md is None:
        lam, V = jnp.linalg.eigh(Kd)
    else:
        Md = 0.5 * (Md + Md.T)
        L = jnp.linalg.cholesky(Md)
        C = solve_triangular(L, solve_triangular(L, Kd, lower=True).T, lower=True)  # L⁻¹ K L⁻ᵀ
        lam, Y = jnp.linalg.eigh(0.5 * (C + C.T))
        V = solve_triangular(L.T, Y, lower=False)  # x = L⁻ᵀ y  →  M-orthonormal
    order = jnp.argsort(lam)
    if which in ("largest", "LM", "LA"):
        order = order[::-1]
    idx = order[:k]
    return lam[idx], V[:, idx]
