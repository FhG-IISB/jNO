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
