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


def _require_symmetric(op, name: str, *, probes: int = 2, seed: int = 12345) -> None:
    """Fail loud when ``op`` is not symmetric — every path here solves the **symmetric** pencil.

    Both reductions Hermitianize by construction (:func:`dense_geneigh` forms ``½(K+Kᵀ)``; LOBPCG's
    Rayleigh–Ritz symmetrizes its projected matrix). That is right for *assembly roundoff* and
    silently **wrong** for a genuinely non-self-adjoint operator, because it answers a different
    question: measured on a deliberately non-symmetric ``K``, the values returned were exactly the
    spectrum of ``½(K+Kᵀ)`` and **not one of them was an eigenvalue of** ``K`` — whose true spectrum
    was complex. Non-self-adjoint operators are the normal case in plasma/flow stability (resistive
    tearing, drift waves, anything with a mean flow), where the answer is a complex growth rate.

    Tested through the **bilinear form** rather than by forming ``Kᵀ``: ``⟨w, Kv⟩ = ⟨Kw, v⟩`` for all
    ``v, w`` iff ``K = Kᵀ``. Two matvecs per probe, no transpose is ever materialized, and it works
    on a **matvec-only** operator — which the constraint-reduced pencil ``PᵀKP`` is. **Concrete-only**:
    under a trace the probes come back as tracers and the check is skipped, the same contract as
    :func:`jno.utils.solver.solver_api._maybe_residual_check`.
    """
    if _symmetry_verdict(op, probes=probes, seed=seed) == "nonsymmetric":
        raise ValueError(
            f"jno.solve.eigs: {name} is NOT symmetric. This solver reduces the SYMMETRIC pencil, so "
            f"it would silently return the spectrum of \u00bd({name}+{name}\u1d40) -- a different "
            "problem: a non-self-adjoint operator generally has COMPLEX eigenvalues, and none of the "
            f"symmetrized values need be an eigenvalue of {name} at all. Pass sigma= to use the "
            "non-symmetric shift-invert path (ARPACK), or, if you are certain you want the "
            f"symmetrized surrogate, pass it explicitly: 0.5*({name} + {name}.T)."
        )


def _symmetry_verdict(op, *, probes: int = 2, seed: int = 12345) -> str:
    """``"symmetric"`` / ``"nonsymmetric"`` / ``"unknown"`` -- which pencil this operator really is.

    Same bilinear-form probe :func:`_require_symmetric` has always used (``\u27e8w, Kv\u27e9 = \u27e8Kw, v\u27e9``
    for all ``v, w`` iff ``K = K\u1d40``): two matvecs per probe, no transpose materialized, and it works
    on a matvec-only operator. Split out so the dispatcher can ROUTE on the answer rather than only
    refuse -- a non-symmetric operator now has somewhere to go.

    ``"unknown"`` is returned for a traced or unsized operator, exactly where the old code declined to
    fabricate a verdict. Callers must treat it as "assume symmetric", which preserves the historical
    behaviour: under ``jit`` the symmetric path is still what runs.
    """
    if op is None:
        return "symmetric"
    o = _as_op(op)
    shape = getattr(o, "shape", None)
    if shape is None:  # an unsized matvec cannot be probed
        return "unknown"
    import numpy as np

    n = int(shape[0])
    rng = np.random.default_rng(seed)
    dt = _as_dense_dtype(op)
    tol = max(1e-8, 1e3 * float(jnp.finfo(jnp.zeros((), dt).dtype).eps))
    worst = 0.0
    for _ in range(probes):
        v = jnp.asarray(rng.standard_normal(n), dt)
        w = jnp.asarray(rng.standard_normal(n), dt)
        Kv, Kw = o.mv(v), o.mv(w)
        if isinstance(Kv, jax.core.Tracer) or isinstance(Kw, jax.core.Tracer):
            return "unknown"  # traced: cannot concretise, so do not fabricate a verdict
        # plain (non-conjugated) products: this tests K = Kᵀ, matching what dense_geneigh imposes
        num = abs(complex(jnp.sum(w * Kv) - jnp.sum(Kw * v)))
        den = float(jnp.linalg.norm(w) * jnp.linalg.norm(Kv)) + 1e-300
        worst = max(worst, num / den)
    if worst > tol:
        return "nonsymmetric"
    return "symmetric"


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
    X0=None,
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
        X0: warm-start eigenvector guesses, ``(n, j)`` columns with ``j ≤ k`` + guards (a single
            vector may be 1-D). The classic sweep accelerator: a parameter/frequency/k-point sweep
            seeds each solve with the previous point's eigenvectors, cutting the sweeps to the few
            that track the drift instead of re-finding the subspace from random. Missing columns are
            padded with the seeded random block; the columns should be linearly independent.

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

    V0 = jax.random.normal(jax.random.PRNGKey(seed), (n, kb), dtype=dtype.dtype)
    if X0 is not None:
        W0 = jnp.asarray(X0, dtype.dtype)
        W0 = W0[:, None] if W0.ndim == 1 else W0
        if W0.shape[0] != n or W0.shape[1] > kb:
            raise ValueError(
                f"jno.solve.eigs: X0 must be (n, j) warm-start columns with n={n} and j <= {kb} "
                f"(k + guard vectors); got {tuple(W0.shape)}."
            )
        # warm columns lead, the seeded random block pads the guards — _m_orth_ordered keeps the order
        V0 = jnp.concatenate([W0, V0[:, W0.shape[1] :]], axis=1)
    V0 = _m_orth_ordered(V0, _blockmv(Mop, V0), eps_chol)
    init = (0, V0, jnp.zeros_like(V0), jnp.asarray(jnp.inf, dtype.dtype))
    # stop_gradient: the sweeps are a search for the eigenvector, not part of the value's definition.
    # `while_loop` is reverse-mode-hostile, and it does not need to be differentiable — the gradient is
    # recovered exactly from the Rayleigh quotient below (see the docstring).
    _i, X, _P, res = jax.lax.stop_gradient(jax.lax.while_loop(lambda s: (s[0] < maxiter) & (s[3] > tol), sweep, init))

    # Differentiable readout: Rayleigh quotient at the frozen eigenvector. Guards are dropped here.
    X = X[:, :k]
    KX, MX = _blockmv(Kop, X), _blockmv(Mop, X)
    lam = jnp.sum(X.conj() * KX, axis=0).real / jnp.sum(X.conj() * MX, axis=0).real
    return lam, X, res


def _left_eigenvectors(A, B, lam, V, np, sp, spla, sla, dense):
    """Left eigenvectors ``w`` with ``w^H A = lambda w^H B``, paired to ``lam`` by construction.

    Needed only for the derivative: for a SIMPLE eigenvalue,
    ``dlambda = w^H (dA - lambda dB) v / (w^H B v)`` (Wilkinson, *The Algebraic Eigenvalue Problem*,
    1965, ch. 2). The symmetric case never needed this because there ``w = v``.

    Two routes, and the pairing is why:

    * **dense** -- ``scipy.linalg.eig(..., left=True)`` returns both families in ONE ordering, so
      ``w_i`` already belongs to ``lambda_i``. Free and unambiguous.
    * **sparse** -- INVERSE ITERATION on ``(A - lambda_i B)^H`` using the eigenvalue already in hand.
      The obvious alternative, a second Arnoldi run on ``A^H``, would return the conjugated spectrum
      in its own order and leave us matching each value to its partner -- ambiguous exactly when the
      spectrum clusters, which is exactly when the derivative is most delicate. Inverse iteration has
      no pairing step at all: it targets one eigenvalue by construction. The shifted matrix is
      near-singular, which is what makes the iteration converge in a step or two rather than a
      difficulty; the growing component IS the eigenvector.

    Cost, stated: the sparse route factors once PER EIGENVALUE, so a gradient costs ``k`` extra
    factorizations on top of the forward solve. Small ``k`` is the normal case here.
    """
    n = A.shape[0]
    k = len(lam)
    if dense:
        Aq = np.asarray(A)
        Bq = None if B is None else np.asarray(B)
        lam_all, W_all, V_all = sla.eig(Aq, Bq, left=True, right=True)
        # match by value: sla.eig's own order, restricted to the k we returned
        used = np.zeros(len(lam_all), bool)
        cols = []
        for value in lam:
            d = np.abs(lam_all - value) + np.where(used, np.inf, 0.0)
            j = int(np.argmin(d))
            used[j] = True
            cols.append(W_all[:, j])
        return np.stack(cols, axis=1)

    As = sp.csr_matrix(A)
    Bs = sp.csr_matrix(B) if B is not None else sp.identity(n, format="csr", dtype=As.dtype)
    # complex128 throughout: SuperLU's solve inherits the factorization's dtype and refuses a wider
    # right-hand side, and inverse iteration runs on a DELIBERATELY near-singular matrix, which is no
    # place to be in single precision. The forward eigenpairs keep the caller's dtype.
    cdt128 = np.complex128
    As = As.astype(cdt128)
    Bs = Bs.astype(cdt128)
    W = np.empty((n, k), dtype=cdt128)
    BH = Bs.conj().T.tocsc()
    for i in range(k):
        S = (As - lam[i] * Bs).conj().T.tocsc()
        try:
            lu = spla.splu(S)
        except RuntimeError:  # exactly singular: the eigenvalue is converged to machine precision
            lu = spla.splu(S + (1e3 * np.finfo(float).eps * abs(lam[i]) + 1e-300) * sp.identity(n, format="csc"))
        w = np.asarray(V[:, i], dtype=W.dtype)  # the right eigenvector is already a good start
        for _ in range(3):  # 2 is normally enough; the third is cheap insurance
            w = lu.solve(BH @ w)
            nrm = np.linalg.norm(w)
            if not np.isfinite(nrm) or nrm == 0:
                break
            w = w / nrm
        # Did it actually converge to a LEFT eigenvector? ||A^H w - conj(lambda) B^H w|| says so
        # directly. It does not converge at a DEFECTIVE eigenvalue -- where the derivative genuinely
        # does not exist -- and inverse iteration can wander inside a degenerate subspace, returning a
        # plausible vector that is not the partner of v. Poison the column rather than hand back a
        # finite wrong derivative; the NaN carries through w^H B v into the gradient.
        resid = np.linalg.norm(As.conj().T @ w - np.conj(lam[i]) * (BH @ w))
        floor = abs(np.abs(As).max()) * max(np.linalg.norm(w), 1e-300)
        W[:, i] = w if resid <= 1e-6 * floor else np.nan

    return W


@jax.custom_vjp
def _no_eigenvector_grad(V, A):
    """Identity on the eigenvectors, whose derivative is **NaN** rather than a silent zero.

    The eigenvectors come out of a ``stop_gradient``-ed host callback, so without this ``jax.grad``
    reports **zero** for them -- measured against finite differences the true derivative was 2.4e-04,
    and a silent zero is a wrong answer wearing the shape of a right one.

    Reverse mode, not forward, and that is the whole design. A ``custom_jvp`` rule runs during
    tracing, before dead-code elimination, so it cannot tell whether the eigenvectors are actually
    used -- a first attempt raised even for a loss built purely from eigenvalues. The VJP rule instead
    receives ``V``'s COTANGENT, which is exactly the question being asked: nonzero means something
    downstream really does depend on the eigenvectors, and only then is the operator's gradient
    poisoned. A loss that touches only the eigenvalues differentiates normally.

    NaN rather than an exception because the answer is only known inside the backward pass, where a
    Python ``raise`` would fire on a traced predicate. It propagates and cannot be mistaken for a
    gradient; see :func:`_attach_eigenvalue_grad` for what IS differentiable here.
    """
    del A
    return V


def _no_eigenvector_grad_fwd(V, A):
    return V, jnp.zeros_like(A)


def _no_eigenvector_grad_bwd(zeros_like_A, g):
    used = jnp.any(g != 0)
    return jnp.zeros_like(g), jnp.where(used, jnp.nan, 0.0) + zeros_like_A


_no_eigenvector_grad.defvjp(_no_eigenvector_grad_fwd, _no_eigenvector_grad_bwd)


@jax.custom_jvp
def _attach_eigenvalue_grad(A, B, lam, V, W):
    """Identity in ``lam``, carrying the eigenvalue derivative w.r.t. ``A`` and ``B``.

    The eigen-decomposition itself runs in a ``pure_callback`` and is not differentiable, so the
    gradient is *attached* here instead: ``lam``, ``V`` and ``W`` arrive as constants and this adds the
    analytic first-order rule. That keeps the host solver exactly as it was and still gives
    ``jax.grad`` the right answer -- the same trick ``custom_linear_solve`` plays for a direct solve.

    **Eigenvalues only.** ``V`` is returned straight from the callback and stays non-differentiable:
    an eigenvector derivative needs the rest of the spectrum (or a projected solve against the
    deflated operator), which this does not compute. Differentiating through the eigenvectors raises,
    rather than returning a plausible wrong number.
    """
    return lam


@_attach_eigenvalue_grad.defjvp
def _attach_eigenvalue_grad_jvp(primals, tangents):
    A, B, lam, V, W = primals
    dA, dB, _, _, _ = tangents
    Wc = jnp.conj(W)

    def quad(Mat):  # w_i^H Mat v_i for every i, without forming anything n x n
        return jnp.einsum("ni,nm,mi->i", Wc, Mat.astype(Wc.dtype), V)

    denom = quad(B) if B is not None else jnp.einsum("ni,ni->i", Wc, V)
    num = quad(dA) if type(dA) is not object and dA is not None else jnp.zeros_like(lam)
    if B is not None and dB is not None:
        num = num - lam * quad(dB)
    # |w^H B v| / (||w|| ||v||) is 1/kappa, the reciprocal CONDITION NUMBER of the eigenvalue -- the
    # cosine of the angle between its left and right eigenvectors. It goes to zero at a DEFECTIVE
    # eigenvalue, where the derivative does not exist at all: the perturbation series there is in
    # sqrt(eps), not eps, so no first-order rule can be right. sqrt(eps) is therefore the threshold,
    # not some arbitrarily small number -- measured on a Jordan-block pencil the cosine came out
    # 6.7e-09, comfortably above a 1e-12 cutoff, and the "gradient" returned was 1.6e+08. A huge
    # finite number is the worst possible answer here, so this returns NaN instead.
    real_dt = jnp.real(jnp.zeros((), denom.dtype)).dtype
    eps = jnp.finfo(real_dt).eps
    scale = jnp.maximum(jnp.linalg.norm(Wc, axis=0) * jnp.linalg.norm(V, axis=0), jnp.finfo(real_dt).tiny)
    safe = jnp.abs(denom) > jnp.sqrt(eps) * scale
    # The NaN goes in the DENOMINATOR, not in a where-branch. `where(safe, num/denom, nan)` puts it in
    # a CONSTANT branch, and constants transpose to zero under reverse mode -- measured, that returned
    # a gradient of exactly 0.0 for a defective eigenvalue, which is precisely the silent answer this
    # guard exists to prevent. Dividing by NaN keeps the tangent LINEAR in `num`, so the transpose
    # carries it into `jax.grad`.
    return lam, num / jnp.where(safe, denom, jnp.nan)


def _arnoldi_backend(inner_solve, sigma):
    """Which host kernel ARPACK's ``OPinv`` should use -- or ``None`` for ARPACK's own SuperLU."""
    if inner_solve is None:
        return None
    if sigma is None:
        raise ValueError(
            "jno.solve.eigs: linear= only applies to the shift-invert path, and no sigma= was given. "
            "Without a shift there is no (K - sigma*M) to factor -- Arnoldi runs on plain matvecs. "
            "Pass sigma= to target a region, or drop linear=."
        )
    backend = getattr(inner_solve, "traits", {}).get("host_kernel", "__missing__")
    name = getattr(inner_solve, "name", type(inner_solve).__name__)
    if backend == "__missing__" or not getattr(inner_solve, "direct", False):
        raise ValueError(
            f"jno.solve.eigs: linear={name} cannot drive the non-symmetric shift-invert. ARPACK asks "
            "for (K - sigma*M)^-1 as an operator it applies every step, which wants ONE factorization "
            "reused -- an iterative solver would need a tolerance tight enough to erase the saving. "
            'Use jno.solve.lu(backend="pardiso") (fastest factorization), "cudss" (fastest repeated '
            'solve, which is what this loop does) or "host".'
        )
    if backend is None:
        raise ValueError(
            'jno.solve.eigs: linear=jno.solve.lu(backend="device") cannot drive the non-symmetric '
            "shift-invert -- it is a JAX primitive and ARPACK calls its operator from host code, "
            'outside any trace. Use backend="pardiso", "cudss" or "host".'
        )
    return backend


def _arnoldi_opinv(shifted, backend, np, spla):
    """``(K - sigma*M)^-1`` as a scipy ``LinearOperator`` backed by a jNO host kernel.

    This is why the kernels were written as plain numpy functions: ARPACK calls back into Python from
    Fortran, so anything reached from here must work OUTSIDE a JAX trace. The backends' sparsity-keyed
    caches then do the rest -- the factorization happens on the first application and every subsequent
    Arnoldi step is a solve against it, which is exactly the workload cuDSS is fastest at.
    """
    import scipy.sparse as sp

    if backend == "host":
        lu = spla.splu(sp.coo_matrix(shifted).tocsc())
        return spla.LinearOperator(shifted.shape, matvec=lu.solve, dtype=shifted.data.dtype)

    from .linear import _cudss_available, _cudss_host_solve, _pardiso_available, _pardiso_host_solve

    available, kernel = {
        "cudss": (_cudss_available, _cudss_host_solve),
        "pardiso": (_pardiso_available, _pardiso_host_solve),
    }[backend]
    if not available():
        raise ImportError(
            f"jno.solve.eigs: linear=jno.solve.lu(backend={backend!r}) needs that backend installed. "
            "Install it with `pip install jax-numerical-operators[fem]`, or use backend='host'."
        )
    data = np.ascontiguousarray(shifted.data)
    idx = np.ascontiguousarray(np.stack([shifted.row, shifted.col], axis=1))
    shape = tuple(int(v) for v in shifted.shape)
    return spla.LinearOperator(
        shape,
        matvec=lambda b: kernel(data, idx, np.asarray(b).reshape(-1), shape, False),
        dtype=data.dtype,
    )


def nonsymmetric_geneigh(K, M, k: int, sigma, which: str = "smallest", *, inner_solve=None, tol: float = 0.0, maxiter=None):
    """The ``k`` eigenpairs of a **NON-self-adjoint** pencil ``K x = lambda M x`` -- COMPLEX spectrum.

    Everything else in this module reduces the *symmetric* pencil, and both of those reductions
    Hermitianize by construction, so on a non-self-adjoint operator they answer a different question:
    measured on a deliberately non-symmetric ``K``, the values returned were exactly the spectrum of
    ``1/2(K+K^T)`` and **not one of them was an eigenvalue of** ``K``, whose true spectrum was complex.
    That is the case this function exists for, and it is the normal case in plasma and flow stability
    -- resistive tearing, drift waves, anything with a mean flow -- where the answer *is* a complex
    growth rate and its sign is the physics.

    Implicitly-restarted Arnoldi (Lehoucq & Sorensen, *SIAM J. Matrix Anal. Appl.* 17(4):789, 1996)
    via ARPACK, reached through ``scipy.sparse.linalg.eigs``. Arnoldi rather than Lanczos precisely
    because it does not assume self-adjointness: it builds a Hessenberg (not tridiagonal) projection
    and so admits complex Ritz values. With ``sigma`` it runs the same Ericsson-Ruhe spectral
    transformation the symmetric path uses, ``theta = 1/(lambda-sigma)``, which is what makes INTERIOR
    eigenvalues reachable -- and interior is where a stability threshold lives.

    **Runs on the host** through a ``pure_callback``: ARPACK is Fortran, and this is a small dense-ish
    reduction over a handful of vectors, not a per-iteration inner loop. Consequences, stated plainly:

    * **The EIGENVALUES are differentiable, in reverse mode.** ``dlambda = w^H (dA - lambda dB) v /
      (w^H B v)`` for a simple eigenvalue (Wilkinson 1965, ch. 2), attached by a ``custom_jvp`` over
      the host solve; verified against finite differences to 1e-09. The eigenVECTORS are not -- their
      derivative needs the rest of the spectrum -- and differentiating through them yields **NaN**
      rather than the silent zero the plain callback would give. Because that guard is a
      ``custom_vjp``, **forward mode (``jax.jvp``/``jacfwd``) is unavailable on this function**; use
      ``jax.grad``/``jacrev``, which is what an inverse problem wants anyway.
    * A **defective** eigenvalue has no derivative at all (its perturbation series runs in
      ``sqrt(eps)``), and it is detected by the eigenvalue condition number ``|w^H B v|/(||w|| ||v||)``
      falling below ``sqrt(eps)``: the gradient is NaN there rather than the enormous finite number
      the formula would otherwise produce (measured 1.6e+08 on a Jordan-block pencil).
    * **``linear=`` selects the shift-invert factorization.** ARPACK asks for ``(K-sigma*M)^-1`` as an
      operator and applies it once per Arnoldi step, so this is the "factor once, solve many" shape:
      the factorization is built on the first application and every later step reuses it through the
      backend's sparsity-keyed cache. ``jno.solve.lu(backend="cudss"/"pardiso"/"host")`` are accepted,
      because those kernels are plain numpy-level functions that a host callback can call directly.
      ``backend="device"`` and the Krylov solvers are not: the first is a JAX primitive with no
      host-callable form, and an iterative inner solve would need tolerances tight enough to erase
      the saving. Both raise rather than being quietly ignored.

      **It is opt-in because it is not always a win, measured.** ARPACK applies the inverse ~50-70
      times per run, so the trade is one fast factorization against a per-application overhead. On a
      non-symmetric convection-like operator with PARDISO behind it: at n=3,000 **0.72x** (slower --
      scipy SuperLU factors that quickly enough that the overhead dominates), at n=20,000 **10.05x**
      (21.6 s against 217 s). Leave ``linear=None`` for small pencils; reach for it when the
      factorization is what hurts.
    * ARPACK needs ``k < n-1``; smaller pencils take an exact dense ``scipy.linalg.eig``.

    Returns ``(lam, V)`` with **complex** dtype always -- a real return would be a lie about what a
    non-self-adjoint operator can produce, even when a particular spectrum happens to come out real.
    """
    import numpy as np

    from .solver_api import LinearOperator

    n = int(K.shape[0])
    if k < 1 or k > n:
        raise ValueError(f"jno.solve.eigs: k={k} out of range for an operator of size {n}.")
    if sigma is None:
        _which_code(which)  # eagerly: a bad `which` must raise plainly, not wrapped by the callback
    backend = _arnoldi_backend(inner_solve, sigma)
    Kd = _as_dense_dtype(K)
    cdt = np.complex128 if jnp.finfo(jnp.zeros((), Kd).dtype).bits == 64 else np.complex64

    def _host(Kh, Mh):
        import scipy.linalg as sla
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        A = np.asarray(Kh)
        B = None if Mh is None or np.ndim(Mh) == 0 else np.asarray(Mh)
        if n <= max(64, 4 * k + 16) or k >= n - 1:
            # exact, and cheaper than an iteration at this size -- mirrors the symmetric path's cutoff
            lam, V = sla.eig(A, B)
            order = np.argsort(np.abs(lam - sigma)) if sigma is not None else _which_order(lam, which)
            idx = order[:k]
            lam, V = lam[idx], V[:, idx]
            W = _left_eigenvectors(A, B, lam, V, np, sp, spla, sla, dense=True)
            return lam.astype(cdt), V.astype(cdt), W.astype(cdt)
        As = sp.csr_matrix(A)
        Bs = sp.csr_matrix(B) if B is not None else None
        opinv = None
        if backend is not None:
            # ARPACK wants (K - sigma*M)^-1 as something it can APPLY; hand it a jNO backend so the
            # one factorization it needs is the fast one, and every Arnoldi step reuses it.
            Ms = Bs if Bs is not None else sp.identity(n, format="csr", dtype=As.dtype)
            opinv = _arnoldi_opinv((As - sigma * Ms).tocoo(), backend, np, spla)
        try:
            lam, V = spla.eigs(
                As,
                k=k,
                M=Bs,
                sigma=sigma,
                which="LM" if sigma is not None else _which_code(which),
                tol=tol,
                maxiter=maxiter,
                OPinv=opinv,
            )
        except RuntimeError as exc:
            if "singular" not in str(exc).lower():
                raise
            raise RuntimeError(
                f"jno.solve.eigs: shift-invert failed because K - {sigma}*M is exactly singular, "
                f"i.e. sigma={sigma} IS an eigenvalue. Shift-invert needs to factor that matrix, so "
                "the shift must not sit exactly on the spectrum. Move sigma slightly off it (the "
                "transformation still makes nearby eigenvalues dominant, so a small offset costs "
                "nothing)."
            ) from exc
        order = np.argsort(np.abs(lam - sigma)) if sigma is not None else _which_order(lam, which)
        lam, V = lam[order], V[:, order]
        W = _left_eigenvectors(As, Bs, lam, V, np, sp, spla, sla, dense=False)
        return lam.astype(cdt), V.astype(cdt), W.astype(cdt)

    Kop = K if isinstance(K, LinearOperator) else LinearOperator(K)
    Kdense = Kop.dense() if hasattr(Kop, "dense") else jnp.asarray(K)
    Mdense = None
    if M is not None:
        Mop = M if isinstance(M, LinearOperator) else LinearOperator(M)
        Mdense = Mop.dense() if hasattr(Mop, "dense") else jnp.asarray(M)

    lam, V, W = jax.pure_callback(
        _host,
        (
            jax.ShapeDtypeStruct((k,), cdt),
            jax.ShapeDtypeStruct((n, k), cdt),
            jax.ShapeDtypeStruct((n, k), cdt),
        ),
        jax.lax.stop_gradient(Kdense),
        jax.lax.stop_gradient(jnp.zeros(()) if Mdense is None else Mdense),
    )
    # the decomposition itself is a host callback and carries no derivative; attach the analytic
    # eigenvalue rule here, with lam/V/W entering as constants
    lam = _attach_eigenvalue_grad(Kdense, Mdense, lam, V, W)
    return lam, _no_eigenvector_grad(V, Kdense)


def _which_code(which: str) -> str:
    """jNO's ``which`` -> ARPACK's. Magnitude, not algebraic order: a complex spectrum has no order."""
    table = {"smallest": "SM", "largest": "LM", "SM": "SM", "LM": "LM", "LR": "LR", "SR": "SR"}
    if which not in table:
        raise ValueError(
            f"jno.solve.eigs: which={which!r} is not available for a NON-symmetric operator. Use "
            "'smallest'/'largest' (by |lambda|), 'LR'/'SR' (by real part -- the growth rate, which is "
            "usually what a stability question asks for), or pass sigma= to target an interior region."
        )
    return table[which]


def _which_order(lam, which: str):
    import numpy as np

    if which in ("largest", "LM"):
        return np.argsort(-np.abs(lam))
    if which == "LR":
        return np.argsort(-lam.real)
    if which == "SR":
        return np.argsort(lam.real)
    return np.argsort(np.abs(lam))


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
        block_inner = None
    else:
        inner = lambda b: inner_solve(A_sig, b)  # noqa: E731
        # A solver advertising ``multi_rhs`` takes the WHOLE subspace block in one call. This is the
        # method's inner loop -- one application of C per sweep, m columns each -- and a factorization
        # solved as a block beats the same factorization solved column by column by 1.9x at m=4 rising
        # to 5.5x at m=32 (cuDSS, measured). Solvers without the trait keep the column loop.
        block_inner = inner if getattr(inner_solve, "traits", {}).get("multi_rhs") else None

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

    def _apply_C(V):  # C V = (K−σM)⁻¹ M V
        MV = jnp.stack([Mmv(V[:, i]) for i in range(m)], axis=1)
        if block_inner is not None:  # one block solve, not m of them
            return block_inner(MV)
        # column-wise otherwise: the host-factorized inner solve runs through a ``pure_callback``,
        # which has no vmap batching rule, so a static unrolled loop is what is available
        return jnp.stack([inner(MV[:, i]) for i in range(m)], axis=1)

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
