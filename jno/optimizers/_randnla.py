"""Randomized Nyström approximation of a symmetric operator given only as a matrix-vector product,
the preconditioner built from it, and preconditioned conjugate gradients -- shared by ``dngd``
(``solver="nystrom_pcg"``), ``nystrom_ngd`` and ``nncg``.

Based on:
  Frangella, Z., Tropp, J. A., & Udell, M. (2023).
  Randomized Nyström Preconditioning.
  SIAM Journal on Matrix Analysis and Applications, 44(2), 718-752 -- the stable Nyström
  approximation and the preconditioner :math:`P^{-1}`.
  Rathore, P., Lei, W., Frangella, Z., Lu, L., & Udell, M. (2024).
  Challenges in Training PINNs: A Loss Landscape Perspective. ICML 2024, arXiv:2402.01868,
  Algorithm 5 -- the extra shift that keeps the sketch factorisable for an indefinite Hessian.

For a symmetric :math:`A` (``n x n``) and a test matrix :math:`\\Omega` (``n x ℓ``, orthonormal Gaussian
or ``ℓ`` sampled coordinate vectors), the stable Nyström approximation is
:math:`Y_\\nu = A\\Omega + \\nu\\Omega`, :math:`C = \\operatorname{chol}(\\Omega^\\top Y_\\nu)`,
:math:`U\\Sigma = \\operatorname{svd}(Y_\\nu C^{-\\top})`, :math:`\\hat\\Lambda = \\max(\\Sigma^2 - \\nu, 0)`, so
:math:`A \\approx U \\hat\\Lambda U^\\top`. For :math:`A + \\mu I` the preconditioner is

.. math::

    P^{-1} x = U(\\hat\\Lambda + \\mu I)^{-1} U^\\top x + \\tau\\,(x - U U^\\top x),

with :math:`\\tau = 1/(\\hat\\lambda_\\ell + \\mu)` (Frangella et al.; Rathore et al. scale it by
:math:`\\hat\\lambda_\\ell + \\mu`, which conjugate gradients does not see) or :math:`\\tau = 1/\\mu`
(Jnini & Vella 2025, Eq. (11)).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from ._residuals import JACOBIAN_CHUNK


class Nystrom(NamedTuple):
    U: jax.Array  # (n, ℓ) orthonormal
    lam: jax.Array  # (ℓ,) descending, >= 0


def nystrom_approx(matvec: Callable, n: int, rank: int, key, dtype, *, sketch: str = "gaussian") -> Nystrom:
    """Rank-``rank`` stable Nyström approximation of the symmetric operator ``matvec``."""
    if sketch == "gaussian":
        omega, _ = jnp.linalg.qr(jax.random.normal(key, (n, rank), dtype=dtype))
    elif sketch == "sampling":  # landmarks: coordinate vectors, uniform without replacement
        idx = jax.random.choice(key, n, (rank,), replace=False)
        omega = jax.nn.one_hot(idx, n, dtype=dtype).T
    else:
        raise ValueError(f"unknown Nyström sketch {sketch!r}; use 'gaussian' or 'sampling'.")
    Y = jax.lax.map(matvec, omega.T, batch_size=min(JACOBIAN_CHUNK, rank)).T  # (n, ℓ)
    eps = jnp.finfo(dtype).eps
    nu = jnp.sqrt(n) * eps * jnp.linalg.norm(Y)
    Y = Y + nu * omega
    B = omega.T @ Y
    B = 0.5 * (B + B.T)
    # An indefinite operator (a Hessian away from a minimum) can make B indefinite: shift it until it
    # is safely positive definite, and remove the same shift from the eigenvalues afterwards.
    b_min = jnp.linalg.eigvalsh(B)[0]
    extra = jnp.where(b_min <= nu, nu - b_min + eps * jnp.abs(jnp.trace(B)), 0.0)
    Y = Y + extra * omega
    C = jnp.linalg.cholesky(B + extra * jnp.eye(rank, dtype=dtype))
    F = solve_triangular(C, Y.T, lower=True).T  # Y C^{-T}
    U, S, _ = jnp.linalg.svd(F, full_matrices=False)
    return Nystrom(U, jnp.maximum(S**2 - (nu + extra), 0.0))


def nystrom_precond(ny: Nystrom, mu, tail) -> Callable:
    """``x -> P^{-1} x`` for ``A + mu I`` (see the module docstring; ``tail`` is τ)."""

    def apply(x):
        utx = ny.U.T @ x
        return ny.U @ (utx / (ny.lam + mu)) + tail * (x - ny.U @ utx)

    return apply


class PCGResult(NamedTuple):
    x: jax.Array
    iters: jax.Array
    rel_residual: jax.Array
    negative_curvature: jax.Array  # stopped because p^T A p <= 0 (A not positive definite there)


def pcg(matvec: Callable, b, precond: Callable, *, x0=None, tol: float = 1e-10, maxiter: int = 100) -> PCGResult:
    """Preconditioned conjugate gradients for ``A x = b``, stopping at ``||r|| <= tol ||b||``.

    For an operator that is not positive definite (a damped Hessian), it stops at the first direction
    of non-positive curvature and returns the iterate reached so far (Steihaug's truncation); when
    that happens on the first iteration the result is the preconditioned right-hand side, a descent
    direction for ``x -> ½xᵀAx - bᵀx``'s gradient at zero.
    """
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    z = precond(r)
    p = z
    rz = jnp.vdot(r, z)
    bnorm = jnp.linalg.norm(b)
    tiny = jnp.finfo(b.dtype).tiny

    def cond(c):
        i, x, r, z, p, rz, neg = c
        return (i < maxiter) & (jnp.linalg.norm(r) > tol * bnorm) & ~neg

    def body(c):
        i, x, r, z, p, rz, _ = c
        Ap = matvec(p)
        pAp = jnp.vdot(p, Ap)
        neg = pAp <= 0
        alpha = rz / jnp.where(neg, 1.0, pAp)
        x_new = jnp.where(neg, jnp.where(i == 0, z, x), x + alpha * p)
        r_new = jnp.where(neg, r, r - alpha * Ap)
        z_new = precond(r_new)
        rz_new = jnp.vdot(r_new, z_new)
        p_new = z_new + (rz_new / jnp.maximum(rz, tiny)) * p
        return i + 1, x_new, r_new, z_new, p_new, rz_new, neg

    i, x, r, _, _, _, neg = jax.lax.while_loop(cond, body, (0, x, r, z, p, rz, jnp.asarray(False)))
    return PCGResult(x, i, jnp.linalg.norm(r) / jnp.maximum(bnorm, tiny), neg)
