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
import numpy as np

__all__ = ["small_det", "small_inv", "small_matmul", "small_einsum"]

#: Per-instance multiply-adds (``M·N·K`` of one pairwise contraction, batch axes excluded) up to which a
#: contraction is lowered as a broadcast multiply and a sum rather than a ``dot_general``: 8x8x8. Not a
#: speed measurement -- both lowerings do the same ``M·N·K`` multiply-adds, and a GEMM only earns its
#: keep by reusing operands across a tile, which a contraction this small does not have. On a GPU, XLA
#: sends a batched ``dot_general`` to a Triton GEMM whose smallest tile is 16 wide: a batch of 3x3
#: products ran ~200x slower than the elementwise form (see :func:`small_einsum`).
SMALL_CONTRACTION = 512


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
        adj = jnp.stack([jnp.stack([J[..., 1, 1], -J[..., 0, 1]], -1), jnp.stack([-J[..., 1, 0], J[..., 0, 0]], -1)], -2)
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


def small_matmul(a, b):
    """``a @ b`` over the last two axes as a broadcast multiply and a sum -- for the 2x2 / 3x3 matrices of
    element geometry, where a ``dot_general`` becomes a padded batched GEMM (see :func:`small_einsum`)."""
    return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)


def small_einsum(subscripts: str, *operands, limit: int | None = None):
    """``jnp.einsum(subscripts, *operands)``, with small contractions lowered as broadcast-multiply-sum.

    A per-point tensor contraction -- ``g gᵀ``, ``A:B``, a 3x3 metric applied to a vector -- inside an
    element loop is a BATCHED ``dot_general`` once ``vmap`` adds the cell axis, and XLA's GPU backend
    hands that to a Triton GEMM tiled for matrices, padding each 3x3 product into a 16x8 tile. In a 3-D
    stabilised Navier-Stokes residual (442k DOFs, RTX 3070) the Vreman invariant
    ``einsum("...ik,...jk,...jl,...il->...", g, g, g, g)`` ran as such a GEMM: 19 of 99 ms per residual
    for ~1 MFLOP of work per chunk. Written as a multiply and a reduction it fuses into the element loop.

    The operands are contracted pairwise, left to right. If EVERY pairwise contraction is at most
    ``limit`` (default :data:`SMALL_CONTRACTION`) multiply-adds per instance (its ``M·N·K``; axes shared by both operands and kept in the
    result are batch, and do not count), the multiply-sum form is used; otherwise the whole call is
    ``jnp.einsum``, which keeps its optimised contraction order and its GEMM for real matrices. A form
    this does not parse (no ``->``, a repeated index within one operand) is passed to ``jnp.einsum``
    unchanged. Same values up to summation order; differentiable like any jnp expression.
    """
    plan = _small_einsum_plan(subscripts, operands, SMALL_CONTRACTION if limit is None else limit)
    if plan is None:
        return jnp.einsum(subscripts, *operands)
    (xs, X), rest, out = plan
    for ys, Y, rs in rest:
        union = xs + "".join(c for c in ys if c not in xs)
        P = _expand(X, xs, union) * _expand(Y, ys, union)
        X = jnp.sum(P, axis=tuple(i for i, c in enumerate(union) if c not in rs)) if set(union) - set(rs) else P
        xs = "".join(c for c in union if c in rs)
    if set(xs) - set(out):  # one operand, or indices no operand pair summed away
        X = jnp.sum(X, axis=tuple(i for i, c in enumerate(xs) if c not in out))
        xs = "".join(c for c in xs if c in out)
    return jnp.transpose(X, [xs.index(c) for c in out]) if xs != out else X


def _expand(X, xs, union):
    """``X`` (indices ``xs``) transposed into ``union`` order, with a size-1 axis for each absent index."""
    present = [c for c in union if c in xs]
    X = jnp.transpose(X, [xs.index(c) for c in present]) if "".join(present) != xs else X
    it = iter(jnp.shape(X))
    return jnp.reshape(X, tuple(next(it) if c in xs else 1 for c in union))


def _small_einsum_plan(subscripts, operands, limit):
    """``((xs, X), [(ys, Y, result)...], out)`` with explicit indices, or ``None`` to use ``jnp.einsum``."""
    import string

    spec = subscripts.replace(" ", "")
    if "->" not in spec or not operands:
        return None
    lhs, out = spec.split("->")
    ins = lhs.split(",")
    if len(ins) != len(operands):
        return None
    used = set(spec) - set(".,->")
    free = [c for c in string.ascii_letters if c not in used]
    shapes = [tuple(jnp.shape(o)) for o in operands]
    n_ell = []
    for s, sh in zip(ins, shapes):
        if s.count("...") > 1 or ("." in s.replace("...", "")):
            return None
        k = len(sh) - len(s.replace("...", ""))
        if k < 0 or (k > 0 and "..." not in s):
            return None
        n_ell.append(k)
    E = "".join(free[: max(n_ell + [0])])
    if E and "..." not in out:
        return None
    if "..." in out and len(E) == 0:
        out = out.replace("...", "")
    ins = [s.replace("...", E[len(E) - k :] if k else "") for s, k in zip(ins, n_ell)]
    out = out.replace("...", E)
    if any(len(set(s)) != len(s) for s in ins) or len(set(out)) != len(out) or set(out) - set("".join(ins)):
        return None
    size = {}
    for s, sh in zip(ins, shapes):
        for c, n in zip(s, sh):
            size[c] = max(size.get(c, 1), int(n))
    prod = lambda cs: int(np.prod([size[c] for c in cs])) if cs else 1  # noqa: E731
    xs, steps = ins[0], []
    for k in range(1, len(ins)):
        ys = ins[k]
        need = set("".join(ins[k + 1 :])) | set(out)
        union = xs + "".join(c for c in ys if c not in xs)
        rs = "".join(c for c in union if c in need)
        M = prod([c for c in xs if c in rs and c not in ys])
        N = prod([c for c in ys if c in rs and c not in xs])
        K = prod([c for c in union if c not in rs])
        if M * N * K > limit:
            return None
        steps.append((ys, operands[k], rs))
        xs = rs
    return (ins[0], operands[0]), steps, out
