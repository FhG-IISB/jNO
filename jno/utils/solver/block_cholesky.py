"""Sparse block Cholesky ``E = L L^T`` of a symmetric positive definite matrix made of ``k x k`` blocks on a graph.

The coarse matrix of a space split over many subdomains is block sparse: block ``(i, j)`` is nonzero only when
subdomains ``i`` and ``j`` overlap or touch through the operator. Above a few thousand directions its dense
factorization is the cost (and ``K^2`` the memory), while the block graph has only ``p`` nodes.

Two phases, as everywhere in :mod:`deflation`: the symbolic one runs on the host on the block graph (a
nested-dissection order by METIS, Karypis & Kumar 1998; the elimination tree and the fill, George & Liu,
*Computer Solution of Large Sparse Positive Definite Systems*, 1981, ch. 5; columns grouped by their height in the
tree); the numeric one is pure JAX: per level of the tree, every column's diagonal block is factored, its
off-diagonal blocks solved and its updates scatter-added, each as one batched operation. Columns of one level are
never ancestors of each other, so their updates only reach later levels. The triangular solves run level by
level the same way, twice per application.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

__all__ = ["BlockCholeskyPlan", "block_cholesky_plan", "block_cholesky", "block_solve"]


class _Level(NamedTuple):
    nodes: jnp.ndarray  # (c,) the columns of this level
    diag: jnp.ndarray  # (c,) their diagonal block slots
    off: jnp.ndarray  # (o,) off-diagonal block slots (i, j), i a later column
    orow: jnp.ndarray  # (o,) i
    ocol: jnp.ndarray  # (o,) position of j within ``nodes``
    tgt: jnp.ndarray  # (u,) update targets (a, b)
    src1: jnp.ndarray  # (u,) slot (a, j)
    src2: jnp.ndarray  # (u,) slot (b, j)


class BlockCholeskyPlan(NamedTuple):
    """The symbolic factorization.

    Attributes:
        p: block rows.
        nb: stored blocks of ``L`` (diagonal, off-diagonal and fill).
        brow: ``(nb,)`` block row of each slot.
        bcol: ``(nb,)`` block column of each slot.
        levels: the columns by height in the elimination tree, leaves first.
        order: ``(p,)`` the elimination order.
        factor: ``blocks -> L``, :func:`block_cholesky` on this plan, compiled once.
        solve: ``(L, b) -> x``, :func:`block_solve` on this plan, compiled once.
    """

    p: int
    nb: int
    brow: np.ndarray
    bcol: np.ndarray
    levels: tuple
    order: np.ndarray
    factor: Any = None
    solve: Any = None

    def slots(self, i, j):
        """Slot of block ``(i, j)`` for arrays of block indices, ``nb`` where it is not stored (the upper
        triangle in elimination order, or not in the pattern)."""
        p = self.p
        key = self.brow.astype(np.int64) * p + self.bcol
        o = np.argsort(key)
        q = np.asarray(i, np.int64) * p + np.asarray(j, np.int64)
        at = np.minimum(np.searchsorted(key[o], q), key.size - 1)
        return np.where(key[o][at] == q, o[at], self.nb)


def _order(p, adj):
    """Nested-dissection order of the block graph (identity for a tiny one)."""
    if p <= 3 or not any(len(a) for a in adj):
        return np.arange(p)
    import pymetis

    perm, _iperm = pymetis.nested_dissection(adjacency=[list(map(int, a)) for a in adj])
    return np.asarray(perm, np.int64)


def block_cholesky_plan(p: int, ei, ej) -> BlockCholeskyPlan:
    """Symbolic phase (host) for a ``p x p`` block matrix whose off-diagonal nonzero blocks are ``(ei, ej)``
    (either triangle, either or both orientations)."""
    ei, ej = np.asarray(ei, np.int64), np.asarray(ej, np.int64)
    off = ei != ej
    ei, ej = ei[off], ej[off]
    adj = [set() for _ in range(p)]
    for a, b in zip(ei.tolist(), ej.tolist()):
        adj[a].add(b)
        adj[b].add(a)
    order = _order(p, adj)
    pos = np.empty(p, np.int64)
    pos[order] = np.arange(p)
    struct, parent, children = [None] * p, np.full(p, -1), [[] for _ in range(p)]
    for j in order.tolist():  # columns in elimination order: struct(j) = later neighbours + children's structs
        s = {a for a in adj[j] if pos[a] > pos[j]}
        for c in children[j]:
            s |= struct[c]
        s.discard(j)
        struct[j] = s
        if s:
            parent[j] = min(s, key=lambda a: pos[a])
            children[parent[j]].append(j)
    height = np.zeros(p, np.int64)
    for j in order.tolist():  # children come first in elimination order
        if parent[j] >= 0:
            height[parent[j]] = max(height[parent[j]], height[j] + 1)

    brow, bcol = list(range(p)), list(range(p))  # slot j: diagonal block (j, j)
    for j in range(p):
        for a in sorted(struct[j], key=lambda a: pos[a]):
            brow.append(a)
            bcol.append(j)
    brow, bcol = np.asarray(brow, np.int64), np.asarray(bcol, np.int64)
    plan = BlockCholeskyPlan(p, int(brow.size), brow, bcol, (), order)

    i32 = lambda a: jnp.asarray(np.asarray(a, np.int64), jnp.int32)  # noqa: E731
    levels = []
    for h in range(int(height.max(initial=-1)) + 1):
        nodes = np.flatnonzero(height == h)
        orow, ocol, ta, tb, s1j = [], [], [], [], []
        for c, j in enumerate(nodes.tolist()):
            s = sorted(struct[j], key=lambda a: pos[a])
            orow += s
            ocol += [c] * len(s)
            for x, a in enumerate(s):
                for b in s[: x + 1]:  # pos(b) <= pos(a): the lower triangle and the diagonal
                    ta.append(a)
                    tb.append(b)
                    s1j.append(j)
        orow, ocol, s1j = np.asarray(orow, np.int64), np.asarray(ocol, np.int64), np.asarray(s1j, np.int64)
        ta, tb = np.asarray(ta, np.int64), np.asarray(tb, np.int64)
        offs = plan.slots(orow, nodes[ocol]) if orow.size else orow
        tgt = plan.slots(ta, tb) if ta.size else ta
        src1 = plan.slots(ta, s1j) if ta.size else ta
        src2 = plan.slots(tb, s1j) if ta.size else ta
        if np.any(offs == plan.nb) or np.any(tgt == plan.nb) or np.any(src1 == plan.nb) or np.any(src2 == plan.nb):
            raise AssertionError("block_cholesky_plan: the fill pattern is not closed (a bug).")
        levels.append(_Level(i32(nodes), i32(nodes), i32(offs), i32(orow), i32(ocol), i32(tgt), i32(src1), i32(src2)))
    plan = plan._replace(levels=tuple(levels))
    # One compiled program each, not one per level: run eagerly, every level's operations have their own shapes
    # and would each be compiled on first use. Inside a compiled solve they are inlined as usual.
    return plan._replace(
        factor=jax.jit(lambda b: block_cholesky(plan, b)), solve=jax.jit(lambda L, b: block_solve(plan, L, b))
    )


def _t(x):
    return jnp.swapaxes(x, -1, -2)


def block_cholesky(plan: BlockCholeskyPlan, blocks):
    """Numeric phase (traceable): ``L``'s blocks ``(nb, k, k)`` from ``E``'s lower blocks in the plan's slots
    (fill slots zero). A block that is not positive definite gives NaN -- the caller checks the pivots."""
    Lb = blocks
    for lev in plan.levels:
        Ljj = jnp.linalg.cholesky(Lb[lev.diag])
        Lb = Lb.at[lev.diag].set(Ljj)
        if lev.off.size:
            Lij = _t(solve_triangular(Ljj[lev.ocol], _t(Lb[lev.off]), lower=True))  # A_ij L_jj^-T
            Lb = Lb.at[lev.off].set(Lij)
        if lev.tgt.size:
            Lb = Lb.at[lev.tgt].add(-(Lb[lev.src1] @ _t(Lb[lev.src2])))
    return Lb


def block_solve(plan: BlockCholeskyPlan, Lb, b):
    """``x = (L L^T)^-1 b`` for ``b`` ``(p, k)`` (traceable): forward by levels, then backward in reverse."""
    x = b
    for lev in plan.levels:
        yj = solve_triangular(Lb[lev.diag], x[lev.nodes][..., None], lower=True)[..., 0]
        x = x.at[lev.nodes].set(yj)
        if lev.off.size:
            x = x.at[lev.orow].add(-jnp.einsum("nkl,nl->nk", Lb[lev.off], yj[lev.ocol]))
    for lev in reversed(plan.levels):
        s = x[lev.nodes]
        if lev.off.size:
            s = s - jax.ops.segment_sum(
                jnp.einsum("nlk,nl->nk", Lb[lev.off], x[lev.orow]), lev.ocol, num_segments=int(lev.nodes.size)
            )
        x = x.at[lev.nodes].set(solve_triangular(_t(Lb[lev.diag]), s[..., None], lower=False)[..., 0])
    return x
