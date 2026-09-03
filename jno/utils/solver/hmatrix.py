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

**What it actually buys.** Measured on a trace layer -- a flat one-cell-thick plate of bars, which is
the shape of every real power-module conductor -- at ``tol=1e-4``, ``leaf=64``, ``eta=2``:

    elements     1,540    3,510    7,965   16,705
    compression   2.88x    4.85x    8.73x   14.45x
    mean rank       8.5      9.0      8.5      8.7
    far blocks      222      664    2,102    4,788
    build         1.1 s    4.0 s   14.0 s   39.7 s

The rank is flat across an 11x size range, which is the kernel behaving as theory says; the ratio
goes as about ``N^0.68`` and the build as about ``O(N^1.34)``. Extrapolated to the ~100,000 elements
a converged real layout needs, that is roughly **48x** and about seven minutes -- against a dense
operator of 80 GB, which is the difference between impossible and routine. The build depends on
geometry alone, so `_geom_cached` amortises it over a frequency sweep or a design iteration.

Two parameter findings worth not re-deriving. ``leaf=64`` compresses better than ``leaf=256`` (4.85x
against 2.46x at 3,510): larger leaves mean coarser blocks and *less* of the matrix reaches an
admissible pair. And ``eta`` barely matters here -- 4.85x at ``eta=2`` against 4.93x at ``eta=4`` --
so it is one fewer knob anyone needs to tune.

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
    """

    ne: int  #: number of elements
    nsub: int  #: sub-points per element (uniform; see the guard in `build`)
    near: tuple  #: ((rows, cols), ...) inadmissible blocks, evaluated densely
    far: tuple  #: ((rows, cols, ipiv, jpiv), ...) admissible blocks, as skeletons
    ranks: tuple  #: the rank ACA needed for each far block, for reporting


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
    return np.array(ipiv, dtype=int), np.array(jpiv, dtype=int)


def build(pos, mom, group, g: Callable, *, tol: float = 1e-6, leaf: int = 64, eta: float = 2.0) -> HMatrix:
    """Choose the block structure and the ACA pivots. HOST ONLY -- concrete geometry required.

    Args:
        pos: ``(N_sub, 3)`` sub-point positions, as `pair_matrix` takes them.
        mom: ``(N_sub, d)`` sub-point moments.
        group: ``(N_sub,)`` element index per sub-point.
        g: the kernel, a callable of scalar distance.
        tol: relative accuracy asked of each admissible block.
        leaf: cluster size at which bisection stops.
        eta: admissibility constant; larger admits more blocks at higher rank.

    A traced argument raises rather than falling back, because the pivots ARE the structure and
    choosing them per trace would mean a different operator on every call.
    """
    for name, arr in (("pos", pos), ("mom", mom)):
        if isinstance(arr, jax.core.Tracer):
            raise ValueError(
                f"hmatrix.build: {name} is a tracer. The cluster tree and the ACA pivots are "
                "structural -- they are chosen once from concrete geometry and then held fixed, the "
                "same split `.build()` uses for the PEEC discretisation and `precond.ams.build` for "
                "its auxiliaries. Build outside the trace and apply inside it."
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
    cen = P.mean(1)

    def blk(rows, cols):
        """The exact element-element sub-block, on the host."""
        d = P[rows][:, None, :, None, :] - P[cols][None, :, None, :, :]
        r = np.sqrt((d * d).sum(-1))
        mm = np.einsum("apc,bqc->abpq", M[rows], M[cols])
        return (mm * g(r)).sum((2, 3))

    nodes, children = cluster(cen, leaf)
    box = [_boxes(cen, idx) for idx in nodes]  # once per NODE, not once per block pair
    near: list = []
    far: list = []
    ranks: list = []
    stack = [(0, 0)]
    while stack:
        a, b = stack.pop()
        ia, ib = nodes[a], nodes[b]
        if _admissible(box[a], box[b], eta):
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
        ca, cb = children[a], children[b]
        if ca is None and cb is None:
            near.append((ia, ib))
        elif ca is None:
            stack += [(a, cb[0]), (a, cb[1])]
        elif cb is None:
            stack += [(ca[0], b), (ca[1], b)]
        else:
            stack += [(ca[0], cb[0]), (ca[0], cb[1]), (ca[1], cb[0]), (ca[1], cb[1])]
    return HMatrix(ne, nsub, tuple(near), tuple(far), tuple(ranks))


def materialize(h: HMatrix, pos, mom, self_g, g: Callable, scale=1.0):
    """``apply(x) -> K @ x`` in pure jax, rebuilding the block values from the frozen structure.

    Differentiable in ``pos`` and ``mom``: every value here is a kernel evaluation at frozen indices,
    and the only linear algebra is a small `pinv` per far block.
    """
    pos = jnp.asarray(pos)
    mom = jnp.asarray(mom)
    if mom.ndim == 1:
        mom = mom[:, None]
    P = pos.reshape(h.ne, h.nsub, 3)
    M = mom.reshape(h.ne, h.nsub, -1)

    def blk(rows, cols):
        d = P[rows][:, None, :, None, :] - P[cols][None, :, None, :, :]
        r = jnp.sqrt(jnp.clip((d * d).sum(-1), 1e-300))
        mm = jnp.einsum("apc,bqc->abpq", M[rows], M[cols])
        return (mm * g(r)).sum((2, 3))

    # A near block straddling the diagonal contains entries where the row and column element are the
    # SAME one. `pair_matrix` zeroes those -- sub-point pairs inside one element do not contribute --
    # and carries the whole diagonal in `self_g` instead. Leaving them in would double-count the
    # element against itself with a singular kernel, which is a large wrong number, not a small one.
    dense = []
    for r, c in h.near:
        rj, cj = jnp.asarray(r), jnp.asarray(c)
        B = jnp.where(np.asarray(r)[:, None] == np.asarray(c)[None, :], 0.0, blk(r, c))
        dense.append((rj, cj, B))
    low = []
    for r, c, ip, jp in h.far:
        C = blk(r, c[jp])  # (m, k)
        R = blk(r[ip], c)  # (k, n)
        G = blk(r[ip], c[jp])  # (k, k)
        low.append((jnp.asarray(r), jnp.asarray(c), C @ jnp.linalg.pinv(G), R))
    # the diagonal is the self term alone: `pair_matrix` zeroes sub-point pairs within one element
    diag = jnp.asarray(self_g) * (M.sum(1) ** 2).sum(-1)

    def _one(x):
        y = diag * x
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
