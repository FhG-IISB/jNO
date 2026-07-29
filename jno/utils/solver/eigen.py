"""Generalized symmetric eigensolver ``K x = λ M x`` (K symmetric, M symmetric positive-definite).

Two paths, both returning **M-orthonormal** eigenvectors (``XᵀMX = I``) — the invariant every variant
must preserve:

* :func:`dense_geneigh` — a **dense** reduction: Cholesky ``M = L Lᵀ`` turns the pencil into the standard
  problem for ``C = L⁻¹ K L⁻ᵀ`` (solved by :func:`jax.numpy.linalg.eigh`, which carries a JVP, so the
  eigenvalues are **differentiable for free**), then maps the eigenvectors back ``x = L⁻ᵀ y``. Exact and
  cheap when you want the whole low spectrum of a small problem, and the oracle the iterative path is
  checked against. It densifies, so it is ``O(N²)`` memory.

* :func:`lobpcg_geneigh` — **preconditioned LOBPCG** for scale: matvecs against ``K``/``M`` plus a
  ``jno.precond.*`` apply, so a sparse/matrix-free operator is never densified.

:mod:`jno.solve` exposes both through ``jno.solve.eigs``; an iterative-only argument (``precond=`` /
``sigma=``) selects LOBPCG, otherwise the dense reduction runs exactly as before.
"""

from __future__ import annotations

import jax
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
    if _as_dense(M) is None:
        lam, V = jnp.linalg.eigh(Kd)
    else:
        from .mass import cholesky_spd

        L = cholesky_spd(M)  # M = L Lᵀ (shared with the consistent-mass exponential integrator)
        C = solve_triangular(L, solve_triangular(L, Kd, lower=True).T, lower=True)  # L⁻¹ K L⁻ᵀ
        lam, Y = jnp.linalg.eigh(0.5 * (C + C.T))
        V = solve_triangular(L.T, Y, lower=False)  # x = L⁻ᵀ y  →  M-orthonormal
    order = jnp.argsort(lam)
    if which in ("largest", "LM", "LA"):
        order = order[::-1]
    idx = order[:k]
    return lam[idx], V[:, idx]


# --------------------------------------------------------------------------------------------------
# Preconditioned LOBPCG (the scale path)
# --------------------------------------------------------------------------------------------------


def _as_op(A):
    """Uniform matvec handle over a BCOO / dense / already-wrapped operator."""
    from .solver_api import LinearOperator

    if A is None:
        return None
    return A if isinstance(A, LinearOperator) else LinearOperator(A)


def _blockmv(op, X):
    """Apply an operator column-wise to an ``(n, m)`` block. ``None`` is the identity (``M = I``)."""
    if op is None:
        return X
    return jax.vmap(op.mv, in_axes=1, out_axes=1)(X)


def _m_orth_basis(V, MV, rtol):
    """A transform ``Z`` making ``V Z`` M-orthonormal, via the eigendecomposition of the M-Gram.

    Returns ``(Z, keep)``. Numerically M-rank-deficient directions (the zero ``P`` block on the first
    sweep, or a search direction that has collapsed into the current subspace) get a **zero** column
    rather than a huge one, so the basis stays fixed-shape — dropping columns would change shapes and
    break ``jit``. ``keep`` marks the usable directions so the caller can push the dead ones out of the
    part of the spectrum it is selecting from.
    """
    G = V.conj().T @ MV
    G = 0.5 * (G + G.conj().T)
    w, U = jnp.linalg.eigh(G)
    keep = w > rtol * jnp.maximum(jnp.max(jnp.abs(w)), jnp.finfo(w.dtype).tiny)
    scale = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, w, 1.0)), 0.0)
    return U * scale[None, :], keep


def _m_orth_ordered(V, MV, eps):
    """M-orthonormalize ``V`` **without reordering its columns** — Cholesky ``VᵀMV = L Lᵀ`` → ``V L⁻ᵀ``.

    The eigendecomposition in :func:`_m_orth_basis` is fine for a subspace *basis* (only the span
    matters) but must not be used on the returned eigenvectors: when the Gram is ≈ I every eigenvalue
    is ≈ 1, so ``eigh`` returns an essentially arbitrary rotation and the columns would no longer line
    up with their eigenvalues. ``L⁻ᵀ`` is triangular, so column ``i`` only ever mixes in columns before
    it — the Ritz ordering survives.
    """
    G = V.conj().T @ MV
    G = 0.5 * (G + G.conj().T)
    L = jnp.linalg.cholesky(G + eps * jnp.eye(G.shape[0], dtype=G.dtype))
    return solve_triangular(L, V.conj().T, lower=True).conj().T


def lobpcg_geneigh(
    K,
    M,
    k: int,
    which: str = "smallest",
    *,
    precond=None,
    tol: float = 1e-6,
    maxiter: int = 200,
    seed: int = 0,
):
    """The ``k`` eigenpairs of ``K x = λ M x`` by **preconditioned LOBPCG**, without densifying either
    operator.

    Locally Optimal Block Preconditioned Conjugate Gradient — Knyazev, *Toward the Optimal Preconditioned
    Eigensolver: Locally Optimal Block Preconditioned Conjugate Gradient Method*, SIAM J. Sci. Comput.
    **23**(2), 517-541 (2001), Algorithm 4.1 (§4). Each sweep does a Rayleigh-Ritz over the block
    ``S = [X, W, P]`` — current iterate, preconditioned residual ``W = T(K X − M X Λ)``, and the previous
    search direction — in the **M-inner product**, so the consistent (non-lumped) mass matrix of an
    ordinary FEM form is handled directly rather than requiring a lumping approximation. The
    B-orthonormal Rayleigh-Ritz basis follows Hetmaniuk & Lehoucq, *Basis selection in LOBPCG*, J. Comput.
    Phys. **218**(1), 324-332 (2006).

    Args:
        K: symmetric operator (BCOO / dense / :class:`LinearOperator`) — never densified.
        M: symmetric positive-definite mass operator, or ``None`` for the standard problem ``Kx = λx``.
        k: number of eigenpairs.
        which: ``"smallest"`` (default) or ``"largest"`` by algebraic value.
        precond: a materialized applier ``v -> T v`` with ``T ≈ K⁻¹`` (from ``jno.precond.*``), or
            ``None`` for unpreconditioned LOBPCG. This is the whole point of the method: on an
            ill-conditioned FEM stiffness the unpreconditioned iteration converges at the rate of the
            condition number.
        tol: convergence tolerance on ``‖K x − λ M x‖`` of the worst wanted pair, normalized by the
            block's **spectrum scale** (its largest Ritz value) so the gate is invariant under
            ``K -> sK`` and finite for a null mode. Do not set it near machine precision: on an
            ill-conditioned pencil the residual floors well above that (measured ``4.4e-8`` on a
            singular all-Neumann Laplacian with ``cond(K) ≈ 2e16``), and a tolerance below the floor
            just burns the whole budget and NaN-poisons a perfectly good spectrum.
        maxiter: sweep budget. Reaching it is **not** an error — check the returned residual.
        seed: PRNG seed for the random initial block (deterministic by default, so runs reproduce).

    Returns:
        ``(λ, X, res)`` — eigenvalues ``(k,)`` in the requested order, M-orthonormal eigenvectors
        ``(n, k)``, and the final worst-pair relative residual (a scalar, for the caller to gate on).

    **Differentiability.** The iteration itself runs under ``stop_gradient`` and the eigenvalues are
    recovered from the Rayleigh quotient ``λ = xᵀKx / xᵀMx`` at the converged (frozen) ``x``. That is
    not an approximation: ``∂R/∂x = 0`` at an exact eigenvector, so for a **simple** eigenvalue
    ``∂λ/∂θ = xᵀ(∂K/∂θ − λ ∂M/∂θ)x`` exactly — the same quantity the dense path's ``eigh`` JVP produces,
    without differentiating through the sweeps. Degenerate/crossing eigenvalues make the derivative
    ill-defined for either path (use the trace of the cluster). The **eigenvectors** carry no gradient
    here, unlike the dense path.
    """
    smallest = which in ("smallest", "SM", "SA")
    Kop, Mop = _as_op(K), _as_op(M)
    n = int(jnp.shape(K)[0] if hasattr(K, "shape") and K.shape is not None else Kop.shape[0])
    if k < 1 or k > n:
        raise ValueError(f"jno.solve.eigs: k={k} out of range for an operator of size {n}.")
    dtype = jnp.zeros((), dtype=_as_dense_dtype(K))
    rtol_rank = jnp.finfo(dtype.dtype).eps * 1e2
    eps_chol = jnp.finfo(dtype.dtype).eps * 1e2

    # GUARD VECTORS. Iterate on a block of kb > k. The k-th Ritz pair converges at a rate set by the gap
    # to eigenvalue k+1, which for a clustered FEM spectrum is tiny -- so a block of exactly k stalls on
    # its last vector long after the first k-1 are converged (measured on a 40x40 pencil: theta[3] still
    # 7.10 against a true 4.77 while theta[0] was already exact to 4 digits). The guards absorb that
    # slow direction; only the first k are gated on and returned. Standard practice, Knyazev 2001 §5.
    kb = min(n, k + max(3, (k + 1) // 2))
    T = (lambda R: R) if precond is None else (lambda R: jax.vmap(precond, in_axes=1, out_axes=1)(R))

    def ritz(X, gate_k):
        KX, MX = _blockmv(Kop, X), _blockmv(Mop, X)
        num = jnp.sum(X.conj() * KX, axis=0).real
        den = jnp.sum(X.conj() * MX, axis=0).real
        lam = num / jnp.where(jnp.abs(den) > 0, den, 1.0)
        R = KX - MX * lam[None, :]
        # Normalize by the SPECTRUM scale (largest Ritz value in the block), not by the per-pair terms
        # `‖Kx‖ + |λ|‖Mx‖`. That textbook denominator is identically ‖Kx‖ for a NULL mode (λ = 0 makes
        # R = Kx), so the ratio pins at exactly 1.0 and the gate can never be met -- which is precisely
        # what an all-Neumann Laplacian hands you as its first eigenvector. Dividing by the block's
        # spectrum scale stays invariant under K -> sK (both R and λ scale by s) and lets a converged
        # null mode read as converged. X is M-orthonormal here, so ‖x‖_M = 1 needs no extra factor.
        lam_scale = jnp.maximum(jnp.max(jnp.abs(lam)), jnp.finfo(lam.dtype).tiny)
        rel = jnp.linalg.norm(R, axis=0) / lam_scale
        return lam, R, jnp.max(rel[:gate_k])  # guards need not converge, so they do not gate

    def sweep(state):
        i, X, P, _ = state
        _lam, R, _res = ritz(X, kb)
        S = jnp.concatenate([X, T(R), P], axis=1)
        KS, MS = _blockmv(Kop, S), _blockmv(Mop, S)
        Z, keep = _m_orth_basis(S, MS, rtol_rank)
        A = Z.conj().T @ (S.conj().T @ KS) @ Z
        A = 0.5 * (A + A.conj().T)
        # Exile the dropped (zero-column) directions to the far end of the spectrum so the selection
        # below never picks one: they carry an exact 0 eigenvalue that would otherwise look "smallest".
        big = 1e6 * (jnp.max(jnp.abs(A)) + 1.0)
        A = A + jnp.diag(jnp.where(keep, 0.0, big if smallest else -big).astype(A.dtype))
        theta, C = jnp.linalg.eigh(A)
        idx = jnp.arange(kb) if smallest else (theta.shape[0] - 1 - jnp.arange(kb))
        Ccol = (Z @ C)[:, idx]
        Xn = S @ Ccol
        Pn = S @ Ccol.at[:kb, :].set(0.0)  # the [W, P] part only — the LOBPCG search direction
        Xn = _m_orth_ordered(Xn, _blockmv(Mop, Xn), eps_chol)
        _l, _r, res_n = ritz(Xn, k)
        return (i + 1, Xn, Pn, res_n)

    X0 = jax.random.normal(jax.random.PRNGKey(seed), (n, kb), dtype=dtype.dtype)
    X0 = _m_orth_ordered(X0, _blockmv(Mop, X0), eps_chol)
    init = (0, X0, jnp.zeros_like(X0), jnp.asarray(jnp.inf, dtype.dtype))
    # stop_gradient: the sweeps are a search for the eigenvector, not part of the value's definition.
    # `while_loop` is reverse-mode-hostile, and it does not need to be differentiable — the gradient is
    # recovered exactly from the Rayleigh quotient below (see the docstring).
    _i, X, _P, res = jax.lax.stop_gradient(jax.lax.while_loop(lambda s: (s[0] < maxiter) & (s[3] > tol), sweep, init))

    # Differentiable readout: Rayleigh quotient at the frozen eigenvector. Guards are dropped here.
    X = X[:, :k]
    KX, MX = _blockmv(Kop, X), _blockmv(Mop, X)
    lam = jnp.sum(X.conj() * KX, axis=0).real / jnp.sum(X.conj() * MX, axis=0).real
    return lam, X, res


def _as_dense_dtype(A):
    """Result dtype of the operator, without materializing it."""
    dt = getattr(A, "dtype", None)
    return dt if dt is not None else jnp.zeros(()).dtype
