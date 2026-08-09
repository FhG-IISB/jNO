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

* :func:`shift_invert_geneigh` — the ``k`` eigenpairs **nearest a shift** ``σ`` (interior modes:
  cavity resonances, band structure away from the band edge), by block subspace iteration on the
  spectrally transformed operator ``C = (K−σM)⁻¹M`` with ``θ = 1/(λ−σ)``.

:mod:`jno.solve` exposes all three through ``jno.solve.eigs``: ``precond=`` selects LOBPCG,
``sigma=`` selects shift-invert, otherwise the dense reduction runs exactly as before.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def _as_dense(A):
    if A is None:
        return None
    if hasattr(A, "todense"):  # BCOO
        return jnp.asarray(A.todense())
    if hasattr(A, "dense") and callable(getattr(A, "dense")):  # LinearOperator (incl. matvec-only)
        return jnp.asarray(A.dense())
    return jnp.asarray(A)


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
        from .mass import cholesky_spd

        # Pass the DENSIFIED mass: `M` may be a matvec-only LinearOperator (the constraint-reduced
        # pencil PᵀMP), which cholesky_spd cannot consume.
        L = cholesky_spd(Md)  # M = L Lᵀ (shared with the consistent-mass exponential integrator)
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
    """Apply an operator column-wise to an ``(n, m)`` block. ``None`` is the identity (``M = I``).

    An operator tagged ``column_loop=True`` is applied by a static unrolled loop instead of ``vmap``:
    the shift-inverted operator's matvec runs a host-factorized direct solve through
    ``jax.pure_callback``, which has **no vmap batching rule** — ``vmap`` would fail where the plain
    per-column call is fine (the block width is a small static ``kb``)."""
    if op is None:
        return X
    if getattr(op, "column_loop", False):
        return jnp.stack([op.mv(X[:, i]) for i in range(X.shape[1])], axis=1)
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


def shift_invert_geneigh(
    K, M, k: int, sigma: float, *, inner_solve=None, tol: float = 1e-6, maxiter: int = 200, seed: int = 0
):
    """The ``k`` eigenpairs of ``K x = λ M x`` **nearest the shift** ``σ`` — interior eigenvalues.

    The extremal-end methods (LOBPCG, Lanczos) cannot target the middle of a spectrum: an interior
    eigenvalue is extremal in no ordering, and the FEM gaps there are relatively tiny. The **spectral
    transformation** (Ericsson & Ruhe, *The spectral transformation Lanczos method for the numerical
    solution of large sparse generalized symmetric eigenvalue problems*, Math. Comp. **35** (1980),
    §2) fixes both at once: with ``C = (K − σM)⁻¹M``, every eigenpair maps to ``C x = θ x`` with

        ``θ = 1/(λ − σ)``,

    so the eigenvalues **nearest σ become the largest |θ|** — the dominant subspace — and the
    transformed gaps are enormous exactly where the original gaps are tiny (the transform is its own
    preconditioner; no ``precond=`` is needed or accepted here). ``C`` is self-adjoint in the
    M-inner product (``M C = M(K−σM)⁻¹M`` is symmetric), so **block subspace iteration** on ``C``
    (Bathe & Wilson 1973; guards absorbing the slow boundary direction, as in LOBPCG) converges the
    ``k`` nearest pairs wherever they lie — both sides of σ or all on one — and, being a BLOCK
    method, converges a degenerate cluster near σ (the double modes of a symmetric cavity) as a
    block, where single-vector shift-invert Lanczos finds one copy.

    Every sweep closes with a Rayleigh–Ritz of the **original pencil** on the block, so the
    convergence gate is the caller's quantity — the λ-space relative residual ``‖Kx − λMx‖`` of the
    ``k`` wanted pairs, normalized by their spectrum scale — never a θ-space proxy, whose mapping
    back is amplified by ``‖K−σM‖/|θ|`` (from ``Kx − λMx = −(K−σM)(Cx−θx)/θ``). An exhausted budget
    NaN-poisons rather than returning a quietly under-converged interior spectrum. A shift that
    lands ON an eigenvalue makes ``K − σM`` singular; the inner factorization then yields garbage
    that fails the same gate — perturb σ off the eigenvalue.

    Args:
        K: **assembled** symmetric operator (BCOO or dense — the shifted operator is factorized, so
            a matvec-only operator is a ``TypeError``).
        M: assembled SPD mass operator, or ``None`` for the standard problem (``M = I``).
        k: number of eigenpairs, nearest σ.
        sigma: the shift (a plain float — differentiating through the shift location is not defined).
        inner_solve: ``(A, b) -> x`` for the inner solves against ``K − σM`` (from ``jno.solve.*``).
            Default: the host sparse-direct LU, whose content-keyed factorization cache means the
            matrix is **factorized once** and every subsequent inner solve is a pair of triangular
            substitutions.
        tol / maxiter / seed: as in :func:`lobpcg_geneigh` (per transformed run).

    Returns:
        ``(λ, X)`` — the ``k`` eigenvalues nearest σ (sorted by ``|λ − σ|``), M-orthonormal
        eigenvectors, NaN-poisoned if the final original-pencil residual gate fails. Eigenvalues are
        differentiable through the Rayleigh quotient at the frozen eigenvectors, like the LOBPCG
        path; eigenvectors carry no gradient.
    """
    for name, op in (("K", K), ("M", M)):
        if op is not None and not (hasattr(op, "todense") or isinstance(op, jnp.ndarray) or hasattr(op, "__array__")):
            raise TypeError(
                f"jno.solve.eigs(sigma=...): {name} must be an ASSEMBLED operator (BCOO or dense) — "
                "the shifted operator K - sigma*M is factorized for the inner solves, which a "
                "matvec-only operator cannot provide."
            )
    n = int(K.shape[0])
    if k < 1 or k > n:
        raise ValueError(f"jno.solve.eigs: k={k} out of range for an operator of size {n}.")
    sigma = float(sigma)

    # Small pencil: the two k-blocks of the transformed runs would overlap the whole spectrum, and at
    # this size the dense reduction is exact and cheaper than any iteration — take it and pick the
    # k nearest σ directly.
    if n <= max(64, 4 * k + 16):
        lam_all, V_all = dense_geneigh(K, M, n, "smallest")
        idx = jnp.argsort(jnp.abs(lam_all - sigma))[:k]
        return lam_all[idx], V_all[:, idx]

    from .solver_api import LinearOperator, _add_step_operator

    if M is None:
        import jax.experimental.sparse as jsp

        eye = jsp.BCOO((jnp.ones(n, _as_dense_dtype(K)), jnp.stack([jnp.arange(n)] * 2, axis=1)), shape=(n, n))
        A_sig = _add_step_operator(K, eye, -sigma)
        Mmv = lambda v: v  # noqa: E731
    else:
        A_sig = _add_step_operator(K, M, -sigma)
        Mop = LinearOperator(M)
        Mmv = Mop.mv

    if inner_solve is None:
        from .linear import host_lu_solve

        inner = lambda b: host_lu_solve(A_sig, b)  # noqa: E731  factorized once (content-keyed cache)
    else:
        inner = lambda b: inner_solve(A_sig, b)  # noqa: E731

    # Block shift-invert SUBSPACE ITERATION (Bathe & Wilson, *Solution methods for eigenvalue
    # problems in structural mechanics*, IJNME 6 (1973) — the classical pairing with the Ericsson-Ruhe
    # transformation). One application of ``C = (K−σM)⁻¹M`` per sweep multiplies every unwanted
    # direction by |θ_unwanted/θ_wanted| — tiny, because the transformation makes the near-σ |θ| the
    # dominant ones by construction — so the m-block converges to the k nearest pairs (wherever they
    # lie: both sides of σ, or all on one) with the m−k guards absorbing the slow boundary direction,
    # exactly as in LOBPCG. Each sweep closes with a Rayleigh–Ritz of the ORIGINAL pencil on the
    # block, so the convergence gate is the quantity the caller cares about — the λ-space residual —
    # never a θ-space proxy whose mapping back is amplified by ``‖K−σM‖/|θ|``.
    Kop = LinearOperator(K)
    Mop_full = LinearOperator(M) if M is not None else None
    m = min(n, k + max(3, (k + 1) // 2))  # guard vectors, as in LOBPCG
    dt = _as_dense_dtype(K)
    eps = jnp.finfo(dt).eps

    def _apply_C(V):  # C V = (K−σM)⁻¹ M V, column-wise: the host-factorized inner solve has no vmap rule
        return jnp.stack([inner(Mmv(V[:, i])) for i in range(m)], axis=1)

    def sweep(state):
        i, V, _res, _lam = state
        W = _apply_C(V)
        Z, keepc = _m_orth_basis(W, _blockmv(Mop_full, W), eps * 1e2)
        Sb = W @ Z  # M-orthonormal basis (a collapsed direction -> zero column)
        A = Sb.T @ _blockmv(Kop, Sb)
        A = 0.5 * (A + A.T)
        big = 1e6 * (jnp.max(jnp.abs(A)) + abs(sigma) + 1.0)  # exile dead columns far from the shift
        A = A + jnp.diag(jnp.where(keepc, 0.0, big).astype(A.dtype))
        mu, Q = jnp.linalg.eigh(A)
        order = jnp.argsort(jnp.abs(mu - sigma))  # nearest-σ first, guards after
        Vn = Sb @ Q[:, order]
        X = Vn[:, :k]
        KX = _blockmv(Kop, X)
        MX = _blockmv(Mop_full, X) if M is not None else X
        lam = mu[order][:k]
        scale = jnp.maximum(jnp.max(jnp.abs(lam)), jnp.finfo(lam.dtype).tiny)
        rel = jnp.max(jnp.linalg.norm(KX - MX * lam[None, :], axis=0) / scale)
        return (i + 1, Vn, rel, lam)

    key = jax.random.PRNGKey(seed)
    V0 = jax.random.normal(key, (n, m), dtype=dt)
    V0 = _m_orth_ordered(V0, _blockmv(Mop_full, V0), eps * 1e2)
    init = (0, V0, jnp.asarray(jnp.inf, dt), jnp.zeros((k,), dt))
    # stop_gradient: the sweeps locate the eigenvectors; the differentiable value is the Rayleigh
    # quotient at the frozen result (below) — identical to the LOBPCG path's contract.
    _i, V, res, _l = jax.lax.stop_gradient(jax.lax.while_loop(lambda s: (s[0] < maxiter) & (s[2] > tol), sweep, init))
    X = V[:, :k]

    # Differentiable readout + honesty gate on the ORIGINAL pencil: the Rayleigh quotient at the
    # frozen eigenvectors carries ∂λ/∂θ; a budget exhausted past ``tol`` — or the NaN residuals a
    # singular shift produces — NaN-poisons rather than returning a quietly wrong interior spectrum
    # (``res <= tol`` is False for NaN).
    KX = _blockmv(Kop, X)
    MX = _blockmv(Mop_full, X) if M is not None else X
    lam = jnp.sum(X * KX, axis=0) / jnp.sum(X * MX, axis=0)
    bad = ~(res <= tol)
    return jnp.where(bad, jnp.nan, lam), jnp.where(bad, jnp.nan, X)


def _as_dense_dtype(A):
    """Result dtype of the operator, without materializing it."""
    dt = getattr(A, "dtype", None)
    return dt if dt is not None else jnp.zeros(()).dtype
