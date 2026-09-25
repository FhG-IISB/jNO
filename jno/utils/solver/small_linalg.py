"""Closed-form determinant and inverse of the small (1x1, 2x2, 3x3) matrices of element geometry.

``jnp.linalg.inv`` / ``det`` factorise with LU even for a 3x3: on a GPU a batch of them becomes separate
cuSOLVER/cuBLAS ``getrf`` + ``trsm`` kernels that XLA cannot fuse into the element loop around them. The
cofactor formulas are a few dozen multiply-adds per matrix, fused like any other elementwise work.
Measured on a 3-D P1 nonlinear Poisson solve (87k DOF, RTX 3070): the element Jacobian inverses were
~30 ms of a 163 ms element loop. Differentiable like any other jnp expression; larger matrices fall
back to ``jnp.linalg``.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["small_det", "small_inv"]


def _n(J):
    s = jnp.shape(J)
    return s[-1] if len(s) >= 2 and s[-1] == s[-2] else None


def small_det(J):
    """``det(J)`` over the last two axes; closed form up to 3x3."""
    n = _n(J)
    if n == 1:
        return J[..., 0, 0]
    if n == 2:
        return J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    if n == 3:
        a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]
        d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]
        g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return jnp.linalg.det(J)


def small_inv(J):
    """``inv(J)`` over the last two axes; adjugate over determinant up to 3x3."""
    n = _n(J)
    if n == 1:
        return 1.0 / J
    if n == 2:
        det = small_det(J)[..., None, None]
        adj = jnp.stack(
            [jnp.stack([J[..., 1, 1], -J[..., 0, 1]], -1), jnp.stack([-J[..., 1, 0], J[..., 0, 0]], -1)], -2
        )
        return adj / det
    if n == 3:
        a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]
        d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]
        g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]
        adj = jnp.stack(
            [
                jnp.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
                jnp.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
                jnp.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1),
            ],
            -2,
        )
        return adj / small_det(J)[..., None, None]
    return jnp.linalg.inv(J)
