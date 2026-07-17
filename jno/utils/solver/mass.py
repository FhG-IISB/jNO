"""Mass-matrix handling shared by the schemes that reduce a symmetric-definite pencil ``· = λ M ·`` (or
``M u̇ = ...``) to a plain symmetric form — the generalized eigensolver and the exponential integrator.

Two treatments:

* **lumped** — the row-sum diagonal ``d_i = Σ_j M_ij = ∫ φ_i`` (partition of unity): matrix-free, and it
  gives a discrete maximum principle. ``M^{-1/2}`` is then ``1/√d`` — trivial.
* **consistent** — the full ``M`` via a dense Cholesky ``M = L Lᵀ``: no lumping error, at the cost of a
  factorization (done **once**; a matrix-free M-inner-product Lanczos is the large-``n`` alternative).
"""

from __future__ import annotations

import jax.numpy as jnp


def _dense(M):
    return jnp.asarray(M.todense() if hasattr(M, "todense") else M)


def lumped_diagonal(M):
    """Row-sum (lumped) mass diagonal ``d`` — one matvec, matrix-free."""
    return M @ jnp.ones(M.shape[0], _dense(M).dtype if not hasattr(M, "data") else jnp.asarray(M.data).dtype)


def cholesky_spd(M):
    """Lower Cholesky factor ``L`` of an SPD (sub)matrix ``M = L Lᵀ`` — the symmetric reduction applies
    ``L⁻¹``/``L⁻ᵀ`` (triangular solves). Symmetrised for roundoff. Dense (moderate ``n``)."""
    Md = _dense(M)
    return jnp.linalg.cholesky(0.5 * (Md + Md.T))


def m_inner_funm(L_mv, m_inner, e0, v, fun, order):
    """``f(L)·v`` where ``L`` is self-adjoint in the **M-inner-product** ``m_inner(a,b) = aᵀ M b`` — via a
    generalized (M-orthonormal) **Lanczos** iteration. Fully **matrix-free** (``L_mv`` applies ``L=M⁻¹A``,
    ``m_inner`` uses only an M-matvec) and **differentiable** (the recurrence unrolls, ``eigh`` of the small
    tridiagonal is differentiable) — the scalable, autodiff-friendly consistent-mass path, with no dense
    factorization and no host-side interior extraction. ``e0`` is a fixed M-unit fallback for a null start.

    The three-term recurrence ``β_{j+1} q_{j+1} = L q_j − α_j q_j − β_j q_{j-1}`` uses the M-inner-product
    throughout (``α_j = ⟨q_j, L q_j⟩_M``), giving a symmetric tridiagonal ``T``; then
    ``f(L) v = ‖v‖_M · Q f(T) e_1``. ``order`` is the Krylov size."""
    beta0 = jnp.sqrt(jnp.maximum(m_inner(v, v), 0.0))
    q = jnp.where(beta0 > 1e-300, v / jnp.where(beta0 > 1e-300, beta0, 1.0), e0)
    Q, alphas, betas = [q], [], []
    q_prev, beta = jnp.zeros_like(q), jnp.zeros((), q.dtype)
    for _j in range(order):  # unrolled (order small) so full reorthogonalization is straightforward
        w = L_mv(q)
        alpha = m_inner(q, w)
        w = w - alpha * q - beta * q_prev
        for qk in Q:  # FULL M-reorthogonalization — without it ghost eigenvalues corrupt the gradient
            w = w - m_inner(qk, w) * qk
        beta_next = jnp.sqrt(jnp.maximum(m_inner(w, w), 0.0))
        q_prev, beta = q, beta_next
        q = jnp.where(beta_next > 1e-300, w / jnp.where(beta_next > 1e-300, beta_next, 1.0), q)
        alphas.append(alpha)
        betas.append(beta_next)
        Q.append(q)
    alphas, betas = jnp.stack(alphas), jnp.stack(betas)
    T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
    evals, evecs = jnp.linalg.eigh(0.5 * (T + T.T))
    fT_e1 = evecs @ (fun(evals) * evecs[0, :])
    return beta0 * (jnp.stack(Q[:order]).T @ fT_e1)  # ‖v‖_M · Q f(T) e₁
