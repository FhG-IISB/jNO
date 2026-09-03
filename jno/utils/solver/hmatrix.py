"""Hierarchical (H-matrix) compression of an integral-equation operator, by ACA.

:mod:`~jno.utils.solver.kernel` names two cases and implements both: elements on a **regular
lattice** are block-Toeplitz and apply by FFT in ``O(N log N)``; elements at **arbitrary positions**
have no structure at all and fall back to a chunked ``O(N^2)`` sum. This module fills that second
branch, which is what a locally-refined PEEC mesh needs -- a graded lattice is not translation
invariant, so the FFT does not apply to it and the dense path does not fit in memory.

The idea is the standard one. Split the elements into a binary **cluster tree**; a block coupling two
clusters that are far apart compared to their size is **admissible**, and such a block is numerically
low rank because ``1/r`` is smooth once the two clusters are separated. Admissible blocks are stored
as a rank-``k`` factorisation, everything near the diagonal densely, and the matvec becomes a sum of
small matmuls.

    Hackbusch, *A sparse matrix arithmetic based on H-matrices*, Computing 62:89, 1999 -- the
    hierarchical block structure and the admissibility condition.
    Bebendorf, *Approximation of boundary element matrices*, Numer. Math. 86:565, 2000 -- ACA, which
    finds the low-rank factors from sampled rows and columns of the matrix rather than from an
    analytic expansion of the kernel.

**Why ACA and not a fast multipole method.** FMM is asymptotically better but needs multipole and
local expansions *derived for each kernel*, plus the translation operators between them. ACA needs
only a function that evaluates an entry, so one implementation serves every kernel this codebase
already has -- the partial inductance, the welded cross blocks, and the magnetic potential and
coupling. That matters more here than the constant factor. Precedent for the combination in this
exact application: Kamon, Tsuk & White, *FASTHENRY: a multipole-accelerated 3-D inductance extraction
program*, IEEE Trans. MTT 42(9):1750, 1994.

**Structure on the host, values in jax.** ACA's pivot search is a data-dependent loop over residuals
-- it cannot be traced, and it must not be, because which rows it picks is a structural decision like
any other in this codebase. So the host pass chooses the pivots and the traced pass rebuilds the
factors from them, through the **skeleton** (cross) form

    A_block  ~=  A[:, J] @ pinv(A[I, J]) @ A[I, :]

for the row pivots ``I`` and column pivots ``J`` that ACA selected. Every term there is a gather of
kernel evaluations plus one small pseudo-inverse, so the whole apply differentiates with respect to
the element POSITIONS -- which `pair_matrix` already does, deliberately (see the NaN guard there),
and which an inverse design that moves metal depends on.

**STATUS: correct and verified, but NOT yet worth switching on. Measured.**

Compression of a bar lattice is real once the elements are partitioned by moment direction (below):
3.27x at 1,540 elements, 4.38x at 2,782, 5.76x at 4,589, accurate to 8.2e-09 at ``tol=1e-8``. An
earlier claim of 2.88x rising to 14.45x is **withdrawn** -- it measured blocks ACA had silently got
wrong, which the check in `_aca` now rejects.

But a whole welded solve is SLOWER with it, at every size tested, and uses more memory:

    elements        764     1,764    3,276
    exact          2.4 s    3.1 s    3.9 s     1.93 / 3.42 / 4.89 GB peak
    hierarchical   7.5 s   14.8 s   20.0 s     2.48 / 3.97 / 5.51 GB peak

The answers agree to the last digit, so this is economics, not correctness. Two reasons, both
structural rather than sloppy:

* **A dense matvec is one BLAS call; a compressed one is hundreds of small matmuls.** Measured at
  1,540 elements: 0.018 s dense against 0.186 s compressed -- 10x slower per apply, even though the
  compressed form stores 3x less. Removing redundancy does not pay until ``O(N^2)`` genuinely hurts,
  and at a few thousand elements it does not: a 1,540-square matvec is 2.4 Mflop.
* **The operator is not the memory bottleneck at these sizes.** At 3,276 elements the dense matrix is
  86 MB against a 4.9 GB peak, so compressing it 5x is invisible. The recorded pathology this was
  aimed at -- 2.7 GB allocated to produce a 46 MB block -- is at 12,000 elements, beyond what has
  been measured here.

**So: do not enable `jno.solve.hierarchical(...)` on the strength of the compression ratio.** Where
the crossover is has not been established, and the honest reading is that memory pays before time and
neither pays yet. What is established is that the machinery is correct, that ACA's failures are
caught rather than silently returned, and that the compression is genuine where it applies.

**Scope limits, up front.** ACA is for *asymptotically smooth* kernels: ``1/r`` and its derivatives
are fine, an oscillatory ``exp(ikr)/r`` at high ``kr`` is not, and this module makes no attempt to
detect that -- it reports the rank it needed, and a rank that grows to the block size is the signal.
The accuracy is a tolerance you choose, unlike the FFT which is exact to round-off. And the build is
host work that only pays for itself above a few thousand elements; below that the dense
`pair_matrix` is both exact and faster -- at 1,540 elements the ratio is under 3x and at 370 it is
1.38x, so this is not the small-network path and does not try to be.

The build cost is dominated by the per-block ACA kernel evaluations, at a near-constant 5-8 ms a
block; the block count is what grows. Caching the cluster boxes (below) removes an ``O(|cluster|)``
cost from every admissibility test but did **not** move the measured build time at these sizes,
because that was never the bottleneck -- it is kept as the algorithmically correct thing to do at
larger N, not as a measured improvement.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class HMatrix(NamedTuple):
    """The frozen STRUCTURE of a hierarchical operator -- indices only, no kernel values.

    Everything here is numpy and integral: which elements each block couples, and which rows and
    columns ACA chose for the admissible ones. The values are rebuilt from this by :func:`materialize`
    every time the geometry changes, which is what keeps the operator differentiable.

    Covers both shapes the welded path needs. ``square`` is one element set against itself -- the
    partial inductance of a conductor, whose diagonal is the self term. Not square is one set against
    a DIFFERENT one, which is `cross_block`: two welded parts coupling, with no diagonal at all.
    """

    shape: tuple  #: (n_row_elements, n_col_elements)
    nsub: tuple  #: sub-points per element on each side (uniform; see the guard in `build`)
    near: tuple  #: ((rows, cols), ...) inadmissible blocks, evaluated densely
    far: tuple  #: ((rows, cols, ipiv, jpiv), ...) admissible blocks, as skeletons
    ranks: tuple  #: the rank ACA needed for each far block, for reporting
    square: bool  #: whether row and column elements are the same set


def _uniform_groups(group, ne: int) -> int:
    """Sub-points per element, requiring it to be the SAME for every element.

    The same restriction `near_block` already imposes, and for the same reason: a uniform count is
    what lets the sub-point sum be one `einsum` instead of a ragged loop. Raised rather than worked
    around, because the alternative is a silent 100x in the innermost kernel evaluation.
    """
    counts = np.bincount(np.asarray(group), minlength=ne)
    if counts.size != ne or not np.all(counts == counts[0]):
        raise ValueError(
            "hmatrix: elements carry different numbers of sub-points "
            f"({int(counts.min())} to {int(counts.max())}), which the vectorised quadrature assumes. "
            "Discretise the parts with matching `quad`/`quad_t` -- `bar_filaments` uses quad*quad_t^2 "
            "and `line_filaments` uses quad*quad_t^2 as well, so they agree when both are given the "
            "same values."
        )
    return int(counts[0])


def _prep(what: str, pos, mom, group):
    """Sub-points reshaped per element, plus centroids. HOST ONLY -- a tracer raises here."""
    for name, arr in (("pos", pos), ("mom", mom)):
        if isinstance(arr, jax.core.Tracer):
            raise ValueError(
                f"{what}: {name} is a tracer. The cluster tree and the ACA pivots are structural -- "
                "they are chosen once from concrete geometry and then held fixed, the same split "
                "`.build()` uses for the PEEC discretisation and `precond.ams.build` for its "
                "auxiliaries. Build outside the trace and apply inside it."
            )
    pos = np.asarray(pos, dtype=float)
    mom = np.asarray(mom, dtype=float)
    if mom.ndim == 1:
        mom = mom[:, None]
    grp = np.asarray(group)
    ne = int(grp.max()) + 1
    nsub = _uniform_groups(grp, ne)
    order = np.argsort(grp, kind="stable")
    P = pos[order].reshape(ne, nsub, 3)
    M = mom[order].reshape(ne, nsub, -1)
    return P, M, P.mean(1)


def _families(M: np.ndarray, tol: float = 1e-9):
    """Element indices grouped by moment DIRECTION, or None when they do not group.

    A PEEC bar carries a moment along one axis, so a lattice has two or three families and the
    operator is block diagonal across them. Returns None when every element points a different way --
    a curved wire, say -- in which case there is nothing to partition and the general path applies.
    """
    v = M.sum(1)
    n = np.linalg.norm(v, axis=1)
    live = n > 0
    if not live.any():
        return None
    u = np.zeros_like(v)
    u[live] = v[live] / n[live, None]
    keys = np.round(np.abs(u) / (np.abs(u).max(1, keepdims=True) + 1e-300), 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    if len(uniq) > 8:  # not a family structure; do not pay for a partition that buys nothing
        return None
    return [np.flatnonzero(inv == k) for k in range(len(uniq))]


def _boxes(cen: np.ndarray, idx: np.ndarray):
    lo, hi = cen[idx].min(0), cen[idx].max(0)
    return lo, hi, float(np.linalg.norm(hi - lo))


def cluster(cen: np.ndarray, leaf: int):
    """A binary cluster tree over element centroids, by recursive bisection of the widest axis.

    Returns ``(nodes, children)`` where ``nodes[i]`` is the index array of node ``i`` and
    ``children[i]`` is ``(l, r)`` or ``None`` for a leaf. Geometric bisection rather than a k-d
    median split: it keeps the boxes compact, which is what the admissibility test measures, and a
    slightly unbalanced tree costs far less than an elongated cluster.
    """
    nodes: list = [np.arange(len(cen))]
    children: list = [None]
    stack = [0]
    while stack:
        n = stack.pop()
        idx = nodes[n]
        if len(idx) <= leaf:
            continue
        lo, hi, _ = _boxes(cen, idx)
        ax = int(np.argmax(hi - lo))
        mid = 0.5 * (lo[ax] + hi[ax])
        left = idx[cen[idx, ax] <= mid]
        right = idx[cen[idx, ax] > mid]
        if len(left) == 0 or len(right) == 0:  # degenerate spread: fall back to a median split
            order = idx[np.argsort(cen[idx, ax])]
            left, right = order[: len(order) // 2], order[len(order) // 2 :]
        children[n] = (len(nodes), len(nodes) + 1)
        nodes += [left, right]
        children += [None, None]
        stack += [children[n][0], children[n][1]]
    return nodes, children


def _admissible(box_a, box_b, eta: float) -> bool:
    """``min(diam A, diam B) <= eta * dist(A, B)`` -- the standard condition.

    Two clusters are far enough apart, relative to their own size, that the kernel is smooth across
    the block and a low-rank approximation converges. `eta` trades rank against block count.

    Takes PRECOMPUTED boxes. Recomputing them per test costs ``O(|cluster|)`` each and ``O(N)`` at
    the root, which turned the build into ``O(N^1.7)`` -- measured 4.0 s at 3,510 elements and
    extrapolating to ~18 minutes at 100,000, where the tree itself is ``O(N log N)``.
    """
    (la, ha, da), (lb, hb, db) = box_a, box_b
    gap = np.maximum(0.0, np.maximum(lb - ha, la - hb))
    dist = float(np.linalg.norm(gap))
    if dist <= 0.0:
        return False
    return min(da, db) <= eta * dist


def _aca(entry_rows, entry_cols, m: int, n: int, tol: float, maxrank: int):
    """Partially-pivoted ACA. Returns the chosen ``(row pivots, col pivots)``, not the factors.

    `entry_rows(i)` gives row `i` of the block, `entry_cols(j)` gives column `j`. The residual is
    carried as the running low-rank approximation, exactly as in Bebendorf sec. 4; what is kept is
    only which rows and columns were visited, because the traced pass rebuilds the factors from those
    through the skeleton form. Returning pivots instead of values is the whole reason this is
    differentiable.
    """
    ipiv: list = []
    jpiv: list = []
    us: list = []
    vs: list = []
    i = 0
    seen_i = set()
    nrm = 0.0
    for _ in range(min(maxrank, m, n)):
        while i in seen_i and len(seen_i) < m:
            i = (i + 1) % m
        row = np.array(entry_rows(i), dtype=float)
        for u, v in zip(us, vs):
            row = row - u[i] * v
        j = int(np.argmax(np.abs(row)))
        if abs(row[j]) < 1e-300:
            seen_i.add(i)
            i = (i + 1) % m
            if len(seen_i) >= m:
                break
            continue
        v = row / row[j]
        col = np.array(entry_cols(j), dtype=float)
        for u, vv in zip(us, vs):
            col = col - vv[j] * u
        us.append(col)
        vs.append(v)
        ipiv.append(i)
        jpiv.append(j)
        seen_i.add(i)
        # Frobenius growth, the usual stopping rule: the update's norm against the accumulated one.
        upd = float(np.linalg.norm(col) * np.linalg.norm(v))
        nrm = float(np.sqrt(max(nrm**2 + upd**2, 0.0)))
        if upd <= tol * nrm:
            break
        colr = np.abs(col)
        colr[np.array(ipiv, dtype=int)] = -1.0
        i = int(np.argmax(colr))

    # VERIFY on rows ACA never visited, instead of trusting its stopping rule.
    #
    # ACA's rule watches the size of its own updates, which says nothing about a part of the block it
    # never reached. On a matrix with ZERO SUB-BLOCKS the pivot chain can stay inside one part, drive
    # that part's residual small, stop, and leave the rest at full magnitude -- and the partial
    # inductance is exactly such a matrix, because `mom_a . mom_b` vanishes between perpendicular bar
    # families. Measured on a real bar lattice: one block 100 % wrong, 18 % overall, while random
    # scattered points stayed accurate to 1e-6 and every synthetic test passed.
    #
    # So a handful of unvisited rows are checked against the factorisation. Sampling rows rather than
    # the whole block keeps this ~m/8 of the cost of forming what ACA exists to avoid forming, and a
    # wholly missed sub-block cannot hide from it. Failing the check returns NO pivots, which the
    # caller reads as "store this densely".
    if not ipiv:
        return np.array([], dtype=int), np.array([], dtype=int)
    unseen = [t for t in range(m) if t not in seen_i]
    probe = (unseen or list(range(m)))[:: max(1, len(unseen or range(m)) // 8)][:8]
    for t in probe:
        exact = np.array(entry_rows(t), dtype=float)
        approx = np.zeros_like(exact)
        for u, v in zip(us, vs):
            approx = approx + u[t] * v
        scale = np.abs(exact).max()
        if scale > 0 and np.abs(approx - exact).max() > max(1e-8, 100.0 * tol) * scale:
            return np.array([], dtype=int), np.array([], dtype=int)
    return np.array(ipiv, dtype=int), np.array(jpiv, dtype=int)


def build(
    pos, mom, group, g: Callable, *, b=None, tol: float = 1e-6, leaf: int = 64, eta: float = 2.0, _only=None
) -> HMatrix:
    """Choose the block structure and the ACA pivots. HOST ONLY -- concrete geometry required.

    Args:
        pos: ``(N_sub, 3)`` sub-point positions, as `pair_matrix` takes them.
        mom: ``(N_sub, d)`` sub-point moments.
        group: ``(N_sub,)`` element index per sub-point.
        g: the kernel, a callable of scalar distance.
        b: ``(pos, mom, group)`` of a SECOND element set, for a rectangular block coupling two
            different discretisations -- which is what `cross_block` forms densely today. ``None``
            (the default) is one set against itself, and only that case carries a self diagonal.
        tol: relative accuracy asked of each admissible block.
        leaf: cluster size at which bisection stops.
        eta: admissibility constant; larger admits more blocks at higher rank.

    A traced argument raises rather than falling back, because the pivots ARE the structure and
    choosing them per trace would mean a different operator on every call.
    """
    other = None if b is None else _prep("hmatrix.build (the second set)", *b)
    A = _prep("hmatrix.build", pos, mom, group)
    B = A if other is None else other
    square = other is None
    P, M, cen = A
    Pb, Mb, cenb = B
    if _only is not None:  # one moment direction of a larger set; indices are remapped by the caller
        P, M, cen = P[_only], M[_only], cen[_only]
        Pb, Mb, cenb = P, M, cen

    def blk(rows, cols):
        """The exact element-element sub-block, on the host."""
        d = P[rows][:, None, :, None, :] - Pb[cols][None, :, None, :, :]
        r = np.sqrt((d * d).sum(-1))
        mm = np.einsum("apc,bqc->abpq", M[rows], Mb[cols])
        return (mm * g(r)).sum((2, 3))

    # PARTITION BY MOMENT DIRECTION before clustering.
    #
    # `mom_a . mom_b` vanishes exactly between perpendicular bar families, so the operator is block
    # diagonal by direction -- the same structure `lattice_apply` exploits and `bar_filaments`
    # documents. Clustering geometrically mixes the families into every block, and ACA cannot survive
    # the zero sub-blocks that creates: measured 1.00x compression, every block rejected. Splitting
    # first means each tree holds one direction, every moment in it is parallel, the structural zeros
    # are gone, and the cross-family blocks are not built at all because they are identically zero.
    fam = _families(M) if square else None
    if fam is not None and len(fam) > 1:
        near, far, ranks = [], [], []
        for sel in fam:
            sub = build(
                pos, mom, group, g, tol=tol, leaf=leaf, eta=eta, _only=sel
            )
            near += [(sel[r], sel[c]) for r, c in sub.near]
            far += [(sel[r], sel[c], i, j) for r, c, i, j in sub.far]
            ranks += list(sub.ranks)
        # every cross-family pair is exactly zero and is simply absent from the block list
        return HMatrix((len(cen), len(cenb)), (P.shape[1], Pb.shape[1]), tuple(near), tuple(far), tuple(ranks), True)

    nodes, children = cluster(cen, leaf)
    box = [_boxes(cen, idx) for idx in nodes]  # once per NODE, not once per block pair
    if square:
        nodes_b, children_b, box_b = nodes, children, box
    else:
        nodes_b, children_b = cluster(cenb, leaf)
        box_b = [_boxes(cenb, idx) for idx in nodes_b]
    near: list = []
    far: list = []
    ranks: list = []
    stack = [(0, 0)]
    while stack:
        a, bb = stack.pop()
        ia, ib = nodes[a], nodes_b[bb]
        if _admissible(box[a], box_b[bb], eta):
            rows = lambda i, ia=ia, ib=ib: blk(ia[i : i + 1], ib)[0]
            cols = lambda j, ia=ia, ib=ib: blk(ia, ib[j : j + 1])[:, 0]
            ip, jp = _aca(rows, cols, len(ia), len(ib), tol, maxrank=min(len(ia), len(ib)))
            # A block ACA could not compress is stored densely: a wrong answer is not the
            # alternative to a slow one, and the rank is reported so this is visible.
            if len(ip) == 0 or len(ip) >= min(len(ia), len(ib)) // 2:
                near.append((ia, ib))
            else:
                far.append((ia, ib, ip, jp))
                ranks.append(len(ip))
            continue
        ca, cb = children[a], children_b[bb]
        if ca is None and cb is None:
            near.append((ia, ib))
        elif ca is None:
            stack += [(a, cb[0]), (a, cb[1])]
        elif cb is None:
            stack += [(ca[0], bb), (ca[1], bb)]
        else:
            stack += [(ca[0], cb[0]), (ca[0], cb[1]), (ca[1], cb[0]), (ca[1], cb[1])]
    return HMatrix(
        (len(cen), len(cenb)), (P.shape[1], Pb.shape[1]), tuple(near), tuple(far), tuple(ranks), square
    )


def _pad_stack(idx_lists, width, fill=0):
    """Index lists of differing length into one ``(B, width)`` array, padded with `fill`."""
    out = np.full((len(idx_lists), width), fill, dtype=np.int64)
    for b, v in enumerate(idx_lists):
        out[b, : len(v)] = v
    return out


@partial(jax.jit, static_argnums=(6,))
def _blocks_chunk(P, M, Pb, Mb, R, C, g):
    """One chunk of element-element sub-blocks. JITTED: eagerly, each intermediate here is a fresh
    allocation XLA never gets to reuse, and the peak is the sum of them rather than the largest."""
    # The separation by EXPANSION, |a-b|^2 = |a|^2 + |b|^2 - 2 a.b, which never forms the
    # (b, m, n, nsub, nsub, 3) component tensor -- 11 GB at leaf 64. Splitting the sub-point axis to
    # shrink the remaining (b, m, n, nsub, nsub) further was tried and is WORSE (0.90 GB / 7.6 s
    # against 0.73 / 4.7): XLA fuses the single large einsum better than twelve accumulation steps.
    A, Bp = P[R], Pb[C]
    a2 = (A * A).sum(-1)
    b2 = (Bp * Bp).sum(-1)
    dot = jnp.einsum("bmpc,bnqc->bmnpq", A, Bp)
    r2 = a2[:, :, None, :, None] + b2[:, None, :, None, :] - 2.0 * dot
    mm = jnp.einsum("bmpc,bnqc->bmnpq", M[R], Mb[C])
    return (mm * g(jnp.sqrt(jnp.clip(r2, 1e-300)))).sum((3, 4))


@partial(jax.jit, static_argnums=(0,))
def _replay_group(k, C, R, ipos, jpos):
    """ACA's recursion at frozen pivots, for a whole rank group. JITTED for the same reason: the
    per-step functional updates are buffer reuse under XLA and fresh allocations without it."""
    Uc = jnp.zeros((C.shape[0], C.shape[1], k), C.dtype)
    Vc = jnp.zeros((C.shape[0], k, R.shape[2]), C.dtype)
    ar = jnp.arange(k)
    for t in range(k):
        past = (ar < t).astype(C.dtype)
        urow = jnp.take_along_axis(Uc, ipos[:, t : t + 1, None], axis=1)[:, 0, :]
        row = R[:, t, :] - jnp.einsum("bk,bkn->bn", urow * past, Vc)
        vcol = jnp.take_along_axis(Vc, jpos[:, None, t : t + 1], axis=2)[:, :, 0]
        col = C[:, :, t] - jnp.einsum("bk,bmk->bm", vcol * past, Uc)
        piv = jnp.take_along_axis(row, jpos[:, t : t + 1], axis=1)
        Uc = Uc.at[:, :, t].set(col)
        Vc = Vc.at[:, t, :].set(row / piv)
    return Uc, Vc


def _batched_blocks(P, M, Pb, Mb, rows, cols, g, chunk=16):
    """The exact element-element sub-blocks for a STACK of index pairs, in one jax pass.

    ``rows`` is ``(B, m)`` and ``cols`` is ``(B, n)``, both zero-padded; padded entries are masked to
    zero afterwards so they contribute nothing.

    Two things make this affordable. The separation is taken by EXPANSION, ``|a-b|^2 = |a|^2 + |b|^2
    - 2 a.b``, which avoids ever forming the ``(B, m, n, nsub, nsub, 3)`` component tensor -- that
    alone is 11 GB at leaf 64 -- and blocks are processed in chunks so the peak is bounded by
    ``chunk`` rather than by the block count.
    """
    # Every distinct ARRAY SHAPE compiles its own executable and jax keeps it, so a short final
    # chunk or a per-group width doubles the compilation count for nothing. Padding the last chunk to
    # full width means one compiled kernel per (width, depth) rather than one per call -- the padded
    # rows index element 0 and are masked to zero by the caller, so they cost arithmetic and never
    # correctness.
    nb = rows.shape[0]
    pad = (-nb) % chunk
    if pad:
        rows = np.concatenate([rows, np.zeros((pad, rows.shape[1]), rows.dtype)])
        cols = np.concatenate([cols, np.zeros((pad, cols.shape[1]), cols.dtype)])
    outs = []
    for lo in range(0, rows.shape[0], chunk):
        outs.append(_blocks_chunk(P, M, Pb, Mb, rows[lo : lo + chunk], cols[lo : lo + chunk], g))
    return jnp.concatenate(outs, axis=0)[:nb]


def materialize(h: HMatrix, pos, mom, self_g, g: Callable, scale=1.0, b=None, transpose=False):
    """``apply(x) -> K @ x`` in pure jax, rebuilding the block values from the frozen structure.

    Differentiable in ``pos`` and ``mom``: every value here is a kernel evaluation at frozen indices.

    **Everything is jax, and everything is batched.** The first version issued one small op per block
    -- about a thousand of them -- and paid for it twice: ~76 ms of dispatch each while building, and
    an unrolled graph of the same width when applying. Measured at 4,589 elements, that held 0.38 GB
    to store 0.025 GB of values and added 0.55 GB more on the first apply, for an operator whose
    DENSE form is 0.17 GB. Blocks are therefore padded and stacked, so the whole operator is a
    handful of arrays and the apply is four `einsum`s regardless of how many blocks there are.

    ``b`` is ``(pos, mom)`` of the second element set for a rectangular operator, and must be given
    exactly when the structure was built with one. ``self_g`` is then unused and may be ``None``.

    ``transpose`` gives ``K^T @ x`` from the SAME structure rather than a second build -- a welded
    network needs both directions of every cross block, and building twice could choose different
    pivots for the two, so the operator would stop being symmetric.
    """

    def _resh(p_, m_, ne, nsub):
        p_, m_ = jnp.asarray(p_), jnp.asarray(m_)
        if m_.ndim == 1:
            m_ = m_[:, None]
        return p_.reshape(ne, nsub, 3), m_.reshape(ne, nsub, -1)

    if (b is None) != h.square:
        raise ValueError(
            "hmatrix.materialize: the structure was built "
            f"{'for one element set' if h.square else 'for two'} but materialize was given "
            f"{'one' if b is None else 'two'}. The block indices address whichever sets built them, "
            "so mixing the two silently addresses the wrong elements."
        )
    P, M = _resh(pos, mom, h.shape[0], h.nsub[0])
    Pb, Mb = (P, M) if h.square else _resh(b[0], b[1], h.shape[1], h.nsub[1])

    # ---- near blocks: one padded stack -------------------------------------------------------
    # Width PER GROUP, not one width for everything. Blocks are not all leaf-sized: the ACA-rejection
    # path in `build` stores a whole admissible cluster pair densely, and those sit at any level of
    # the tree, so the largest near block can be thousands of elements wide. Padding every block to
    # that measured a 17.44 GB peak against 1.72 GB per-group -- the padding is quadratic and the
    # saving in compilations is not worth any of it.
    near = []
    if h.near:
        mw = max(len(r) for r, _c in h.near)
        nw = max(len(c) for _r, c in h.near)
        rows = _pad_stack([r for r, _c in h.near], mw)
        cols = _pad_stack([c for _r, c in h.near], nw)
        vals = _batched_blocks(P, M, Pb, Mb, rows, cols, g)
        live = jnp.asarray(
            (np.arange(mw)[None, :] < np.array([[len(r)] for r, _c in h.near]))[:, :, None]
            & (np.arange(nw)[None, :] < np.array([[len(c)] for _r, c in h.near]))[:, None, :]
        )
        vals = jnp.where(live, vals, 0.0)
        if h.square:  # an element never couples to itself here; the diagonal is the self term
            same = jnp.asarray(rows)[:, :, None] == jnp.asarray(cols)[:, None, :]
            vals = jnp.where(same & live, 0.0, vals)
        near = [(jnp.asarray(rows), jnp.asarray(cols), vals)]

    # ---- far blocks: one padded stack PER RANK, so the ACA replay batches exactly -------------
    far = []
    by_rank: dict = {}
    for (r, c, ip, jp), k in zip(h.far, h.ranks):
        by_rank.setdefault(int(k), []).append((r, c, ip, jp))
    for k, group in by_rank.items():
        mw = max(len(r) for r, _c, _i, _j in group)
        nw = max(len(c) for _r, c, _i, _j in group)
        rows = _pad_stack([r for r, _c, _i, _j in group], mw)
        cols = _pad_stack([c for _r, c, _i, _j in group], nw)
        # the pivot ROWS and COLUMNS, as element indices, and their positions within the block
        prow = _pad_stack([r[i] for r, _c, i, _j in group], k)
        pcol = _pad_stack([c[j] for _r, c, _i, j in group], k)
        ipos = jnp.asarray(_pad_stack([i for _r, _c, i, _j in group], k))
        jpos = jnp.asarray(_pad_stack([j for _r, _c, _i, j in group], k))
        C = _batched_blocks(P, M, Pb, Mb, rows, pcol, g)  # (B, m, k)
        R = _batched_blocks(P, M, Pb, Mb, prow, cols, g)  # (B, k, n)
        live_m = jnp.asarray(np.arange(mw)[None, :] < np.array([[len(r)] for r, _c, _i, _j in group]))
        live_n = jnp.asarray(np.arange(nw)[None, :] < np.array([[len(c)] for _r, c, _i, _j in group]))
        # ACA's recursion replayed at the frozen pivots -- k steps, batched across the whole group.
        # The skeleton form `A[:,J] pinv(A[I,J]) A[I,:]` is NOT equivalent: it needs `A[I,J]` well
        # conditioned, which fails on this operator's structural zeros. See `_aca`.
        #
        # The inner `sum over s < t` is a MATMUL against the factors built so far, not a loop over
        # them. Written as a loop it emits one op per (t, s) pair -- O(k^2), about a thousand ops per
        # rank group, which measured 2.66 GB and 45.9 s where the unbatched version took 0.38 GB.
        # As a masked matmul it is two ops per step, so O(k), and the factors are written into
        # preallocated arrays that are exactly the output.
        Uc, Vc = _replay_group(k, C, R, ipos, jpos)
        U = jnp.where(live_m[:, :, None], Uc, 0.0)  # (B, m, k)
        V = jnp.where(live_n[:, None, :], Vc, 0.0)  # (B, k, n)
        far.append((jnp.asarray(rows), jnp.asarray(cols), U, V))

    diag = (jnp.asarray(self_g) * (M.sum(1) ** 2).sum(-1)) if h.square else None
    nout = h.shape[1] if transpose else h.shape[0]

    def _one(x):
        y = diag * x if h.square else jnp.zeros(nout, x.dtype)
        for rows, cols, vals in near:
            if transpose:
                y = y.at[cols].add(jnp.einsum("bmn,bm->bn", vals, x[rows]))
            else:
                y = y.at[rows].add(jnp.einsum("bmn,bn->bm", vals, x[cols]))
        for rows, cols, U, V in far:
            if transpose:
                y = y.at[cols].add(jnp.einsum("bkn,bk->bn", V, jnp.einsum("bmk,bm->bk", U, x[rows])))
            else:
                y = y.at[rows].add(jnp.einsum("bmk,bk->bm", U, jnp.einsum("bkn,bn->bk", V, x[cols])))
        return y

    def apply(x):
        x = jnp.asarray(x)
        if jnp.iscomplexobj(x):
            return scale * (_one(jnp.real(x)) + 1j * _one(jnp.imag(x)))
        return scale * _one(x)

    return apply


class HierarchicalSpec:
    """What ``jno.solve.hierarchical(...)`` returns: how to compress, not what to compress.

    A value object, deliberately -- it carries no geometry, so one spec serves every block of a
    welded network and the same spec can be reused across a frequency sweep.
    """

    __slots__ = ("tol", "leaf", "eta", "floor")

    def __init__(self, tol: float, leaf: int, eta: float, floor: int):
        self.tol, self.leaf, self.eta, self.floor = float(tol), int(leaf), float(eta), int(floor)

    def __repr__(self):
        return f"hierarchical(tol={self.tol:g}, leaf={self.leaf}, eta={self.eta:g}, floor={self.floor})"

    def worth_it(self, na: int, nb: int) -> bool:
        """Whether a block this size is worth compressing at all.

        Below the floor the dense block is both exact and faster, so this returns False and the
        caller keeps the exact path. A compression slower than what it replaces is a defect, not a
        tuning choice.
        """
        return min(na, nb) >= self.floor
