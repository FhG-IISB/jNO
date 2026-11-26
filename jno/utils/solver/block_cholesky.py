"""Sparse block Cholesky ``E = L L^T`` of a symmetric positive definite matrix made of ``k x k`` blocks on a graph.

The coarse matrix of a space split over many subdomains is block sparse: block ``(i, j)`` is nonzero only when
subdomains ``i`` and ``j`` overlap or touch through the operator. Above a few thousand directions its dense
factorization is the cost (and ``K^2`` the memory), while the block graph has only ``p`` nodes.

Two phases, as everywhere in :mod:`deflation`: the symbolic one runs on the host on the block graph (a
nested-dissection order by METIS, Karypis & Kumar 1998; the elimination tree and the fill, George & Liu,
*Computer Solution of Large Sparse Positive Definite Systems*, 1981, ch. 5). Consecutive columns with the same
structure below them are merged into SUPERNODES (Liu, Ng & Peyton, SIAM J. Matrix Anal. Appl. 14(1), 1993):
a nested-dissection separator is a chain of such columns, and as one dense block it is eliminated in one step
instead of one per column. The supernodes are grouped by their height in the supernodal tree, so the numeric
phase -- pure JAX -- runs one batched step per level: every supernode's dense diagonal block is factored, its
panel below solved and its updates scatter-added. Supernodes of one level are never ancestors of each other,
so their updates only reach later levels. The triangular solves run level by level the same way, twice per
application -- and the number of levels, not the work, is what an application costs on a GPU.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

__all__ = ["BlockCholeskyPlan", "block_cholesky_plan", "block_cholesky", "block_solve", "block_diagonal"]


class _Level(NamedTuple):
    """One level of supernodes, padded to ``w`` columns and ``r`` rows below (block counts)."""

    cols: jnp.ndarray  # (c, w) block columns of each supernode (p: padding)
    dslot: jnp.ndarray  # (c, w, w) slot of block (cols[a], cols[b]) for b <= a, else nb (zero)
    dpad: jnp.ndarray  # (c, w) bool, the padded columns (identity on the diagonal)
    rows: jnp.ndarray  # (c, r) block rows below (p: padding)
    pslot: jnp.ndarray  # (c, r, w) slot of block (rows[i], cols[b]), nb on padding
    tgt: jnp.ndarray  # (u,) update targets (a, b), pos(b) <= pos(a)
    src1: jnp.ndarray  # (u,) panel row (flat c * r + i) of a
    src2: jnp.ndarray  # (u,) panel row of b


class BlockCholeskyPlan(NamedTuple):
    """The symbolic factorization.

    Attributes:
        p: block rows.
        nb: stored blocks of ``L`` (diagonal, off-diagonal and fill).
        brow: ``(nb,)`` block row of each slot.
        bcol: ``(nb,)`` block column of each slot.
        levels: the supernodes by height in the supernodal tree, leaves first.
        order: ``(p,)`` the elimination order.
        factor: ``blocks -> L``, :func:`block_cholesky` on this plan, compiled once.
        solve: ``(L, b) -> x``, :func:`block_solve` on this plan, compiled once.
        nsuper: number of supernodes.
    """

    p: int
    nb: int
    brow: np.ndarray
    bcol: np.ndarray
    levels: tuple
    order: np.ndarray
    factor: Any = None
    solve: Any = None
    nsuper: int = 0

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

    # fundamental supernodes: j continues the supernode of the column before it when that column's only
    # parent is j, j has no other child, and the structures nest (struct(prev) = {j} + struct(j))
    sn_cols, sn_of = [], np.empty(p, np.int64)
    prev = -1
    for j in order.tolist():
        if prev >= 0 and parent[prev] == j and len(children[j]) == 1 and struct[prev] == struct[j] | {j}:
            sn_cols[-1].append(j)
        else:
            sn_cols.append([j])
        sn_of[j] = len(sn_cols) - 1
        prev = j
    ns = len(sn_cols)
    sn_rows = [sorted(struct[c[-1]], key=lambda a: pos[a]) for c in sn_cols]  # below the supernode
    sn_parent = np.array([sn_of[r[0]] if r else -1 for r in sn_rows], np.int64)
    height = np.zeros(ns, np.int64)
    for s in range(ns):  # supernodes are numbered in elimination order: children first
        if sn_parent[s] >= 0:
            height[sn_parent[s]] = max(height[sn_parent[s]], height[s] + 1)

    brow, bcol = list(range(p)), list(range(p))  # slot j: diagonal block (j, j)
    for j in range(p):
        for a in sorted(struct[j], key=lambda a: pos[a]):
            brow.append(a)
            bcol.append(j)
    brow, bcol = np.asarray(brow, np.int64), np.asarray(bcol, np.int64)
    plan = BlockCholeskyPlan(p, int(brow.size), brow, bcol, (), order, nsuper=ns)
    nb = plan.nb

    i32 = lambda a: jnp.asarray(np.asarray(a, np.int64), jnp.int32)  # noqa: E731
    levels = []
    for h in range(int(height.max(initial=-1)) + 1):
        sns = np.flatnonzero(height == h)
        c = sns.size
        w = max(len(sn_cols[s]) for s in sns)
        r = max(1, max(len(sn_rows[s]) for s in sns))
        cols = np.full((c, w), p, np.int64)
        rows = np.full((c, r), p, np.int64)
        for t, s in enumerate(sns):
            cols[t, : len(sn_cols[s])] = sn_cols[s]
            rows[t, : len(sn_rows[s])] = sn_rows[s]
        ca, cb = np.broadcast_to(cols[:, :, None], (c, w, w)), np.broadcast_to(cols[:, None, :], (c, w, w))
        lower = np.arange(w)[:, None] >= np.arange(w)[None, :]
        dslot = np.where(lower & (ca < p) & (cb < p), plan.slots(np.minimum(ca, p - 1), np.minimum(cb, p - 1)), nb)
        ra, rb = np.broadcast_to(rows[:, :, None], (c, r, w)), np.broadcast_to(cols[:, None, :], (c, r, w))
        pslot = np.where((ra < p) & (rb < p), plan.slots(np.minimum(ra, p - 1), np.minimum(rb, p - 1)), nb)
        tgt, s1, s2 = [], [], []
        for t, s in enumerate(sns):
            rs = sn_rows[s]
            for x, a in enumerate(rs):
                for y, b in enumerate(rs[: x + 1]):  # pos(b) <= pos(a): the lower triangle and the diagonal
                    tgt.append((a, b))
                    s1.append(t * r + x)
                    s2.append(t * r + y)
        tg = np.asarray(tgt, np.int64).reshape(-1, 2)
        tslot = plan.slots(tg[:, 0], tg[:, 1]) if tg.size else np.zeros(0, np.int64)
        real_d = dslot[(ca < p) & (cb < p) & lower]
        real_p = pslot[(ra < p) & (rb < p)]
        if np.any(real_d == nb) or np.any(real_p == nb) or np.any(tslot == nb):
            raise AssertionError("block_cholesky_plan: the fill pattern is not closed (a bug).")
        levels.append(
            _Level(i32(cols), i32(dslot), jnp.asarray(cols >= p), i32(rows), i32(pslot), i32(tslot), i32(s1), i32(s2))
        )
    plan = plan._replace(levels=tuple(levels))
    # One compiled program each, not one per level: run eagerly, every level's operations have their own shapes
    # and would each be compiled on first use. Inside a compiled solve they are inlined as usual.
    return plan._replace(
        factor=jax.jit(lambda b: block_cholesky(plan, b)), solve=jax.jit(lambda L, b: block_solve(plan, L, b))
    )


def _t(x):
    return jnp.swapaxes(x, -1, -2)


def _dense(blocks, slot):
    """``(..., m, n)`` blocks gathered at ``slot`` ``(c, a, b)`` into dense ``(c, a k, b k)`` panels."""
    c, a, b = slot.shape
    k = blocks.shape[-1]
    g = blocks[slot]  # (c, a, b, k, k)
    return jnp.swapaxes(g, 2, 3).reshape(c, a * k, b * k)


def _blocks(dense, a, b, k):
    """``(c, a k, b k)`` back to ``(c, a, b, k, k)``."""
    c = dense.shape[0]
    return jnp.swapaxes(dense.reshape(c, a, k, b, k), 2, 3)


def block_cholesky(plan: BlockCholeskyPlan, blocks):
    """Numeric phase (traceable): ``E``'s lower blocks ``(nb, k, k)`` in the plan's slots (fill slots zero) to the
    factor: per level, ``(LD, P)`` -- the INVERSE of each supernode's dense diagonal factor ``(c, w k, w k)`` and its
    panel below ``(c, r k, w k)``. An application is then batched matrix-vector products, where batched triangular
    solves are among the slowest kernels a GPU runs (as for the local solves in :mod:`schwarz`). A block that is not
    positive definite gives NaN -- the caller checks :func:`block_diagonal`."""
    k = blocks.shape[-1]
    Lb = jnp.concatenate([blocks, jnp.zeros((1, k, k), blocks.dtype)])  # slot nb: a zero block
    out = []
    for lev in plan.levels:
        c, w, _ = lev.dslot.shape
        r = lev.rows.shape[1]
        D = _dense(Lb, lev.dslot)
        D = jnp.tril(D) + _t(jnp.tril(D, -1))  # the stored lower blocks, made symmetric
        D = D + jnp.repeat(lev.dpad.astype(D.dtype), k, axis=1)[:, :, None] * jnp.eye(w * k, dtype=D.dtype)
        LD = jnp.linalg.cholesky(D)
        P = _t(solve_triangular(LD, _t(_dense(Lb, lev.pslot)), lower=True))  # A_RS L_D^-T
        out.append((solve_triangular(LD, jnp.broadcast_to(jnp.eye(w * k, dtype=LD.dtype), LD.shape), lower=True), P))
        Pr = P.reshape(c * r, k, w * k)  # panel rows, one block row each
        Lb = Lb.at[lev.tgt].add(-(Pr[lev.src1] @ _t(Pr[lev.src2])), mode="drop")
    return tuple(out)


def block_diagonal(plan: BlockCholeskyPlan, L):
    """``(p, k)`` the diagonal of ``L``, per block row, read off the stored inverses (its reciprocals): the
    pivots are its squares."""
    k = L[0][0].shape[-1] // plan.levels[0].dslot.shape[1]
    d = jnp.zeros((plan.p + 1, k), L[0][0].dtype)
    for lev, (LD, _P) in zip(plan.levels, L):
        c, w = lev.cols.shape
        d = d.at[lev.cols.reshape(-1)].set(1.0 / jnp.diagonal(LD, axis1=1, axis2=2).reshape(c * w, k), mode="drop")
    return d[: plan.p]


def block_solve(plan: BlockCholeskyPlan, L, b):
    """``x = (L L^T)^-1 b`` for ``b`` ``(p, k)`` (traceable): forward by levels, then backward in reverse."""
    k = b.shape[-1]
    x = jnp.concatenate([b, jnp.zeros((1, k), b.dtype)])  # row p: padding
    for lev, (LD, P) in zip(plan.levels, L):
        c, w = lev.cols.shape
        y = LD @ x[lev.cols].reshape(c, w * k, 1)  # (c, w k, 1)
        x = x.at[lev.cols].set(y.reshape(c, w, k))
        x = x.at[lev.rows].add(-(P @ y).reshape(c, -1, k))
        x = x.at[plan.p].set(0.0)  # what the padding collected
    for lev, (LD, P) in zip(reversed(plan.levels), reversed(L)):
        c, w = lev.cols.shape
        s = x[lev.cols].reshape(c, w * k, 1) - _t(P) @ x[lev.rows].reshape(c, -1, 1)
        x = x.at[lev.cols].set((_t(LD) @ s).reshape(c, w, k))
        x = x.at[plan.p].set(0.0)
    return x[: plan.p]
