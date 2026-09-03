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

**STATUS: does not yet compress a PEEC bar lattice. Read this before using it.**

The compression ratios first recorded here -- 2.88x at 1,540 elements rising to 14.45x at 16,705 --
were measured on blocks that were NUMERICALLY WRONG, and are withdrawn. ACA silently failed on them
and the storage of a wrong block means nothing. With the failures now detected and rejected
(see `_aca`), a bar lattice compresses **1.00x**: every block is stored densely.

The cause is structural, and it is specific to this operator. `mom_a . mom_b` vanishes *exactly*
between perpendicular bar families, so any block spanning both has a 2x2 structure with zero
off-diagonal parts. ACA's pivot chain stays inside one part, drives its residual small, stops, and
leaves the rest untouched. Clusters here are geometric and x- and y-bars are interleaved in space, so
essentially every block spans both families and essentially every block fails.

**The fix, not yet implemented:** build one hierarchical operator PER MOMENT DIRECTION and skip the
cross-family blocks entirely, since they are exactly zero. Within a family every moment is parallel,
the structural zeros are gone, and ACA has nothing to trip over. That is the same block-diagonal
structure `lattice_apply` already exploits and `bar_filaments` already documents -- "a bar's current
runs along ONE axis, and mom_i . mom_j vanishes between perpendicular bars, so the partial-inductance
operator is block diagonal by direction".

Until then this module is correct -- verified against `pair_matrix` to round-off on a real lattice --
and offers no speedup, so `jno.solve.hierarchical(...)` should not be switched on.

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
    pos, mom, group, g: Callable, *, b=None, tol: float = 1e-6, leaf: int = 64, eta: float = 2.0
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

    def blk(rows, cols):
        """The exact element-element sub-block, on the host."""
        d = P[rows][:, None, :, None, :] - Pb[cols][None, :, None, :, :]
        r = np.sqrt((d * d).sum(-1))
        mm = np.einsum("apc,bqc->abpq", M[rows], Mb[cols])
        return (mm * g(r)).sum((2, 3))

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


def materialize(h: HMatrix, pos, mom, self_g, g: Callable, scale=1.0, b=None, transpose=False):
    """``apply(x) -> K @ x`` in pure jax, rebuilding the block values from the frozen structure.

    Differentiable in ``pos`` and ``mom``: every value here is a kernel evaluation at frozen indices,
    and the only linear algebra is a small `pinv` per far block.

    ``b`` is ``(pos, mom)`` of the second element set for a rectangular operator, and must be given
    exactly when the structure was built with one. ``self_g`` is then unused and may be ``None``:
    two different parts have no diagonal to carry.

    ``transpose`` gives ``K^T @ x`` from the SAME structure rather than a second build. A welded
    network needs both directions of every cross block -- the coupling appears once in each part's
    row -- and the skeleton transposes exactly: ``(U R)^T = R^T U^T``. Building twice would cost
    twice and, worse, could choose different pivots for the two directions, so the operator would
    stop being symmetric.
    """

    def _resh(p, m, ne, nsub):
        p, m = jnp.asarray(p), jnp.asarray(m)
        if m.ndim == 1:
            m = m[:, None]
        return p.reshape(ne, nsub, 3), m.reshape(ne, nsub, -1)

    if (b is None) != h.square:
        raise ValueError(
            "hmatrix.materialize: the structure was built "
            f"{'for one element set' if h.square else 'for two'} but materialize was given "
            f"{'one' if b is None else 'two'}. The block indices address whichever sets built them, "
            "so mixing the two silently addresses the wrong elements."
        )
    P, M = _resh(pos, mom, h.shape[0], h.nsub[0])
    Pb, Mb = (P, M) if h.square else _resh(b[0], b[1], h.shape[1], h.nsub[1])

    # CONCRETE geometry evaluates the blocks in numpy; only a TRACED one needs jax.
    #
    # Not a micro-optimisation. There are three kernel evaluations per compressed block and one per
    # dense block -- about a thousand small arrays on a 1,540-element network -- and issuing each as
    # its own eager jax op costs ~76 ms of DISPATCH against microseconds of arithmetic: measured 35.6
    # s to materialize what took 1.75 s to choose. The same pathology this file's `cross_block` note
    # records for the PEEC assembly, where eager dispatch was 90 % of the solve. Values built on the
    # host arrive as jaxpr constants and the apply is then pure arithmetic.
    _host = not any(isinstance(x, jax.core.Tracer) for x in (P, M, Pb, Mb))
    xp = np if _host else jnp
    Pn, Mn, Pbn, Mbn = (np.asarray(P), np.asarray(M), np.asarray(Pb), np.asarray(Mb)) if _host else (P, M, Pb, Mb)

    def blk(rows, cols):
        d = Pn[rows][:, None, :, None, :] - Pbn[cols][None, :, None, :, :]
        r2 = (d * d).sum(-1)
        r = xp.sqrt(r2 if _host else jnp.clip(r2, 1e-300))
        mm = xp.einsum("apc,bqc->abpq", Mn[rows], Mbn[cols])
        return (mm * g(r)).sum((2, 3))

    # A near block straddling the diagonal contains entries where the row and column element are the
    # SAME one. `pair_matrix` zeroes those -- sub-point pairs inside one element do not contribute --
    # and carries the whole diagonal in `self_g` instead. Leaving them in would double-count the
    # element against itself with a singular kernel, which is a large wrong number, not a small one.
    dense = []
    for r, c in h.near:
        rj, cj = jnp.asarray(r), jnp.asarray(c)
        B = blk(r, c)
        if h.square:
            B = xp.where(np.asarray(r)[:, None] == np.asarray(c)[None, :], 0.0, B)
        dense.append((rj, cj, jnp.asarray(B)))
    low = []
    for r, c, ip, jp in h.far:
        C = blk(r, c[jp])  # (m, k) -- the pivot COLUMNS of the block
        R = blk(r[ip], c)  # (k, n) -- the pivot ROWS
        # REPLAY ACA's recursion at the frozen pivots, rather than reconstructing from a skeleton.
        #
        # The skeleton `A[:,J] pinv(A[I,J]) A[I,:]` looks equivalent and is not. It matches ACA only
        # where `A[I,J]` is well conditioned, which holds for a generic smooth kernel and FAILS on a
        # structurally sparse one -- and the partial inductance is structurally sparse, because
        # `mom_a . mom_b` vanishes exactly between perpendicular bar families. On a real bar lattice a
        # block's pivot rows can land on x-bars while its pivot columns land on y-bars, making
        # `A[I,J]` singular, `pinv` zero, and the whole block vanish: measured 100 % error on one
        # block and 18 % overall, while random scattered points stayed accurate to 1e-6.
        #
        # With the pivot SEQUENCE fixed, ACA is a deterministic chain of rank-1 updates, so replaying
        # it is pure arithmetic on `C` and `R` -- differentiable exactly as the skeleton was, and
        # equal to what the host chose rather than merely close to it.
        us, vs = [], []
        for t in range(len(ip)):
            row = R[t]
            col = C[:, t]
            for sdx in range(t):
                row = row - us[sdx][ip[t]] * vs[sdx]
                col = col - vs[sdx][jp[t]] * us[sdx]
            vs.append(row / row[jp[t]])
            us.append(col)
        U = xp.stack(us, axis=1)  # (m, k)
        V = xp.stack(vs, axis=0)  # (k, n)
        low.append((jnp.asarray(r), jnp.asarray(c), jnp.asarray(U), jnp.asarray(V)))
    # the diagonal is the self term alone: `pair_matrix` zeroes sub-point pairs within one element
    diag = jnp.asarray(np.asarray(self_g) * (Mn.sum(1) ** 2).sum(-1)) if (h.square and _host) else (
        (jnp.asarray(self_g) * (M.sum(1) ** 2).sum(-1)) if h.square else None
    )
    nout = h.shape[1] if transpose else h.shape[0]

    def _one(x):
        y = diag * x if h.square else jnp.zeros(nout, x.dtype)
        if transpose:  # the diagonal is symmetric, so only the off-diagonal blocks flip
            for r, c, B in dense:
                y = y.at[c].add(B.T @ x[r])
            for r, c, U, R in low:
                y = y.at[c].add(R.T @ (U.T @ x[r]))
            return y
        for r, c, B in dense:
            y = y.at[r].add(B @ x[c])
        for r, c, U, R in low:
            y = y.at[r].add(U @ (R @ x[c]))
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

        Below the floor the dense block is both exact and faster -- measured, the ratio is 1.38x at
        370 elements and under 3x at 1,540 -- so this returns False and the caller keeps the exact
        path. A compression that is slower than what it replaces is a defect, not a tuning choice.
        """
        return min(na, nb) >= self.floor
