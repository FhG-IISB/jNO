"""Coarse spaces and deflation: the correction ``Q r = Z (Z^T A Z)^{-1} Z^T r`` for a subspace ``Z``.

``Z`` is ``(n, k)`` vectors, optionally split over ``p`` parts: column ``a`` restricted to part ``i`` is its
own coarse vector, so the coarse space has ``p * k`` directions while only ``(n, k)`` and the part labels are
stored (never a dense ``n x (p k)``). A sparse ``Z`` keeps each group of columns on its own support, and the
groups may overlap. ``p = 1`` is a global space (Nicolaides, SIAM J. Numer. Anal. 24(2),
1987, for the per-part constants; Tang, Nabben, Vuik & Erlangga, J. Sci. Comput. 39 (2009) 340, for how the
correction combines with another preconditioner -- ``jno.precond.coarse`` and ``>>``).

Two phases, as for :mod:`schwarz`: the symbolic one (the space resolved to numbers, the partition, the
operator's entry tables) runs once on the host; the numeric one (``E = Z^T A Z`` and its inverse) is pure JAX
from the operator's current values, so it runs inside a compiled solve, a Newton loop or a time march.

``E`` is inverted through the whitened problem ``W^T E W`` with ``W = G^{-1/2}``, ``G = Z^T Z`` (block
diagonal, one ``k x k`` block per part). A direction with ``G ~ 0`` -- a part whose vectors vanish, e.g. a
part made only of eliminated Dirichlet rows -- carries nothing and is dropped. A direction with ``G`` well
away from zero but ``E ~ 0`` is a KERNEL of ``A`` inside the space: that is a singular problem, not a slow
one, and it is refused (in double precision; in single precision the cancellation that would expose it is
below the rounding, and it is dropped like the others).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "CoarsePattern",
    "operator_graph",
    "metis_parts",
    "coarse_pattern",
    "coarse_factor",
    "coarse_apply",
    "resolve_space",
]

#: A whitened direction whose |eigenvalue| is below this fraction of the largest is a kernel of ``A``.
#: Legitimate near-null directions sit far above it (measured in tests/test_precond_coarse.py).
KERNEL_RTOL = 1e-12
#: A direction of ``G = Z^T Z`` below this fraction of its part's largest carries no vector: dropped.
NULL_RTOL = 1e-12
#: Overlapping parts: a direction of the whitened Gram ``W^T Z^T Z W`` (its scale is 1) below this is a
#: combination of other parts' vectors -- no new direction, so it is not a kernel either.
DEP_RTOL = 1e-10


class CoarsePattern(NamedTuple):
    """The symbolic phase of a coarse space.

    The vectors are stored as PIECES: piece ``l`` is one unknown of one part, ``null[l]`` the part's ``k``
    vectors there. A part's coarse vectors are the columns of its pieces, zero elsewhere. Without ``row`` the
    pieces are the unknowns themselves, so the parts partition them; with ``row`` parts may overlap (a sparse
    space: every group of vectors kept on its own support).

    Attributes:
        n: system size.
        p: number of parts (1: a global space).
        k: vectors per part (a part with fewer has zero columns, dropped).
        part: ``(L,)`` part of each piece (``p``: outside every part -- an eliminated row).
        null: ``(L, k)`` the vectors on the pieces.
        W: ``(p, k, k)`` the whitening ``G^{-1/2}`` per part, zero columns for the dropped directions.
        dropped: directions with no vector (``G ~ 0``), padding included.
        row: ``(L,)`` the unknown of each piece, or ``None`` (piece ``l`` is unknown ``l``).
        pad: zero columns that fill parts with fewer than ``k`` vectors (counted in ``dropped``).
        dependent: directions that are combinations of other parts' vectors (overlapping parts only).
        kidx: ``(K,)`` the kept whitened directions (of ``p k``): the coarse problem is factored on these only,
            so padding and dropped directions cost nothing.
        cidx: ``(K0,)`` the coarse coefficients that carry a vector (``None``: all ``p k``; else padding is out).
        Gt: ``(K, K)`` the whitened Gram ``W^T Z^T Z W`` on the kept directions when parts overlap (``None``: they
            do not, and it is the identity).
        nnz: unique entries of ``A`` (0 when the pattern has no entry tables: a matrix-free global space).
        checksum: identifies ``A``'s pattern (``None`` without entry tables).
        erow, eent, epair: ``(T1,)`` one per (entry, piece on the entry's row): the piece, the entry, and the
            (part, column) pair it adds to -- ``Z^T A`` is summed over these.
        npair: number of (part, column) pairs.
        qpair, qcol, crow: ``(T2,)`` one per (pair, piece on the pair's column): the pair, the piece, and the
            coarse block ``i * p + j`` the product adds to.
    """

    n: int
    p: int
    k: int
    part: jnp.ndarray
    null: jnp.ndarray
    W: jnp.ndarray
    dropped: int
    row: Any = None
    pad: int = 0
    dependent: int = 0
    kidx: Any = None
    cidx: Any = None
    Gt: Any = None
    nnz: int = 0
    checksum: Any = None
    erow: Any = None
    eent: Any = None
    epair: Any = None
    npair: int = 0
    qpair: Any = None
    qcol: Any = None
    crow: Any = None


def operator_graph(A):
    """``(S, G, iso)`` of a CONCRETE operator: ``S`` its CSR (float64, duplicates summed), ``G`` the symmetric
    loop-free unit graph of ``|S| + |S|^T``, ``iso`` the unknowns with no neighbour (eliminated Dirichlet rows)."""
    from .amg import _to_scipy_csr

    S = _to_scipy_csr(A).astype(np.float64)
    S.sum_duplicates()
    n = S.shape[0]
    if S.shape != (n, n):
        raise ValueError(f"jno.precond: the operator must be square, got {S.shape}.")
    G = (abs(S) + abs(S).T).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    G.data[:] = 1.0
    return S, G, np.diff(G.indptr) == 0


def metis_parts(G, iso, parts: int) -> np.ndarray:
    """Part of every unknown: METIS k-way on the graph without its isolated unknowns (Karypis & Kumar, SIAM J.
    Sci. Comput. 20(1), 1998), which get part ``parts`` -- out of range everywhere downstream."""
    from .schwarz import _partition

    core = np.nonzero(~iso)[0]
    parts = int(min(max(1, parts), max(1, core.size)))
    part = np.full(G.shape[0], parts, np.int64)
    part[core] = _partition(G[core][:, core], parts)
    return part


def coarse_pattern(Z, part, p: int, A=None, S=None) -> CoarsePattern:
    """Symbolic phase. ``part`` ``(n,)`` in ``[0, p]`` (``p``: outside every part).

    ``Z`` dense ``(n, k)``: column ``a`` on part ``i`` is a coarse vector -- ``p * k`` directions from ``(n, k)``
    numbers and the labels. ``Z`` sparse ``(n, K)`` (``scipy.sparse``): kept by its nonzeros. Every column, split
    over the parts, joins the other columns with exactly its nonzero rows in one group, and the groups are the
    parts of the pattern (they may overlap). The grouping sets the cost, never the result.

    With a CONCRETE assembled ``A`` the entry tables are built, so ``E`` is summed from its entries; without them
    only a single part can be factored, from ``k`` products. ``S``: ``A``'s CSR from :func:`operator_graph`."""
    import scipy.sparse as sp

    part = np.asarray(part, np.int64)
    n = int(part.size)
    real = None  # (p, k) slots that carry a vector; None: all
    if sp.issparse(Z):
        row, part, null, p, real = _sparse_pieces(Z, part, int(p))
    else:
        Z = np.asarray(Z)
        if np.iscomplexobj(Z):
            raise ValueError("jno.precond.coarse: the space must be real (the operator may be complex).")
        Z = Z.astype(np.float64).reshape(Z.shape[0], -1)
        row = None
        null = np.where((part >= p)[:, None], 0.0, Z)
    k = null.shape[1]
    W, keep = _whitening(null, part, int(p))
    if real is not None:  # padding slots: no vector, so no weight (exactly, not to the eigensolver's rounding)
        W = np.where(real[:, :, None], W, 0.0)
    kidx = np.flatnonzero(keep)
    pad = 0 if real is None else int(real.size - real.sum())
    dropped = int(keep.size - kidx.size)
    dependent, Gt = 0, None
    if row is not None and np.bincount(row, minlength=n).max(initial=0) > 1:
        Gt = _whitened_gram(row, part, null, W, n, int(p))[np.ix_(kidx, kidx)]
        g = np.linalg.eigvalsh(Gt)
        dependent = int(np.sum(g <= DEP_RTOL * max(g.max(initial=0.0), 1.0)))
    head = dict(
        n=n,
        p=int(p),
        k=k,
        part=jnp.asarray(part, jnp.int32),
        null=jnp.asarray(null),
        W=jnp.asarray(W),
        dropped=dropped,
        row=None if row is None else jnp.asarray(row, jnp.int32),
        pad=pad,
        dependent=dependent,
        kidx=jnp.asarray(kidx, jnp.int32),
        cidx=None if real is None else jnp.asarray(np.flatnonzero(real), jnp.int32),
        Gt=None if Gt is None else jnp.asarray(Gt),
    )
    if A is None:
        return CoarsePattern(**head)
    from .fsai import _checksum

    if not jax.config.jax_enable_x64:
        raise ValueError(
            "jno.precond.coarse() needs x64 for an assembled operator: its entry keys row*n + col overflow int32."
        )
    if S is None:
        S = operator_graph(A)[0]
    coo = S.tocoo()
    keys = np.sort(np.asarray(coo.row, np.int64) * n + np.asarray(coo.col, np.int64))
    tables = _entry_tables(keys // n, keys % n, row, part, n, int(p))
    return CoarsePattern(**head, nnz=int(keys.size), checksum=tuple(int(v) for v in _checksum(keys, np)), **tables)


def _sparse_pieces(Z, part, p):
    """``(row, part, null, groups, real)`` of a sparse space: its columns split over the parts, grouped by
    their nonzero rows; one piece per (group, row)."""
    import scipy.sparse as sp

    Z = sp.csc_matrix(Z)
    if np.iscomplexobj(Z.data):
        raise ValueError("jno.precond.coarse: the space must be real (the operator may be complex).")
    Z.sum_duplicates()  # also sorts the rows of every column
    n, K = Z.shape
    r = Z.indices.astype(np.int64)
    c = np.repeat(np.arange(K, dtype=np.int64), np.diff(Z.indptr))
    v = Z.data.astype(np.float64)
    keep = part[r] < p
    r, c, v = r[keep], c[keep], v[keep]
    if r.size == 0:
        raise ValueError("jno.precond.coarse: the sparse space has no entry on the unknowns being solved for.")
    piece_of_col = c * (p + 1) + part[r]  # column c on part part[r]
    o = np.lexsort((r, piece_of_col))
    r, v, piece_of_col = r[o], v[o], piece_of_col[o]
    _cols, start, col = np.unique(piece_of_col, return_index=True, return_inverse=True)
    bounds = np.append(start, r.size)
    groups: dict = {}
    gid = np.empty(start.size, np.int64)
    slot = np.empty(start.size, np.int64)
    for j in range(start.size):  # one pass over the columns: equal row sets share a group
        g = groups.setdefault(r[bounds[j] : bounds[j + 1]].tobytes(), [len(groups), 0])
        gid[j], slot[j] = g
        g[1] += 1
    counts = np.bincount(gid)
    k = int(counts.max())
    keys, piece = np.unique(gid[col] * n + r, return_inverse=True)  # pieces sorted by group, then row
    null = np.zeros((keys.size, k))
    null[piece, slot[col]] = v
    real = np.arange(k)[None, :] < counts[:, None]  # (groups, k): the slots that hold a column
    return keys % n, keys // n, null, int(counts.size), real


def _part_bounds(part, p):
    order = np.argsort(part, kind="stable")
    return order, np.searchsorted(part[order], np.arange(p + 1))


def _whitening(null, part, p):
    """``W = G^{-1/2}`` per part, ``G = Z_i^T Z_i`` (host, once): a direction of ``G`` below ``NULL_RTOL`` of its
    part's largest carries no vector and gets a zero column. Returns ``W`` and the kept mask ``(p, k)``."""
    k = null.shape[1]
    G = np.zeros((p, k, k))
    order, bounds = _part_bounds(part, p)
    for i in range(p):  # Z_i^T Z_i per part: O(n k) memory, never the (n, k, k) outer products
        Zi = null[order[bounds[i] : bounds[i + 1]]]
        G[i] = Zi.T @ Zi
    g, V = np.linalg.eigh(G)
    gmax = np.max(g, axis=1, keepdims=True)
    keep = g > NULL_RTOL * np.maximum(gmax, np.finfo(float).tiny)
    W = V * np.where(keep, 1.0 / np.sqrt(np.where(keep, g, 1.0)), 0.0)[:, None, :]
    return W, keep


def _whitened_gram(row, part, null, W, n, p):
    """``W^T Z^T Z W`` ``(p k, p k)`` of overlapping parts (host, once; dense like ``E``). Its zero directions
    are the dropped ones and the combinations of other parts' vectors."""
    import scipy.sparse as sp

    k = null.shape[1]
    U = np.empty_like(null)
    order, bounds = _part_bounds(part, p)
    for i in range(p):
        sel = order[bounds[i] : bounds[i + 1]]
        U[sel] = null[sel] @ W[i]
    cols = (part[:, None] * k + np.arange(k)).ravel()
    Zw = sp.csr_matrix((U.ravel(), (np.repeat(row, k), cols)), shape=(n, p * k))
    return (Zw.T @ Zw).toarray()


def _entry_tables(r_e, c_e, row, part, n, p):
    """The tables that sum ``E = (Z^T A) Z`` from ``A``'s entries ``(r_e, c_e)`` (sorted keys): first
    ``Z^T A`` by (part, column), then times the pieces on that column. Partitioned (``row is None``) both have
    about ``nnz`` and ``n`` rows; overlapping parts multiply them by the pieces per unknown."""
    if row is None:
        cnt = np.ones(n, np.int64)
        order = np.arange(n)
    else:
        cnt = np.bincount(row, minlength=n)
        order = np.argsort(row, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(cnt)])

    def expand(unknowns):  # every (item, piece at the item's unknown)
        m = cnt[unknowns]
        item = np.repeat(np.arange(unknowns.size), m)
        off = np.arange(item.size) - np.repeat(np.cumsum(m) - m, m)
        return item, order[ptr[unknowns][item] + off]

    eent, erow = expand(r_e)
    inside = part[erow] < p
    eent, erow = eent[inside], erow[inside]
    pairs, epair = np.unique(part[erow] * n + c_e[eent], return_inverse=True)
    qpair, qcol = expand(pairs % n)
    inside = part[qcol] < p
    qpair, qcol = qpair[inside], qcol[inside]
    i32 = lambda a: jnp.asarray(a, jnp.int32)  # noqa: E731
    return dict(
        erow=i32(erow),
        eent=i32(eent),
        epair=i32(epair),
        npair=int(pairs.size),
        qpair=i32(qpair),
        qcol=i32(qcol),
        crow=i32((pairs // n)[qpair] * p + part[qcol]),
    )


def _dense_null(pat: CoarsePattern):
    """``(n, k)`` vectors of a single-part pattern."""
    if pat.row is None:
        return pat.null
    return jnp.zeros((pat.n, pat.k), pat.null.dtype).at[pat.row].add(pat.null)


def _galerkin(pat: CoarsePattern, A, mv):
    """``E = Z^T A Z`` as ``(p k, p k)``.

    A global space (``p = 1``) takes ``k`` products with the operator, ``Z^T (A Z)``: ``O(nnz k + n k^2)``, and it
    works matrix-free. Several parts are summed from the entries, one vector at a time in a loop the compiler
    does not unroll (``k^2`` unrolled sums made a 480-vector space take minutes to compile): row ``a`` of
    ``Z^T A`` by (part, column), then times the pieces on each column -- ``O(k (T1 + T2 k))`` work and
    ``O(T1 + T2 k)`` memory, ``T1 ~ nnz`` and ``T2 ~ n`` for a partition."""
    p, k = pat.p, pat.k
    if p == 1 and mv is not None:
        N = _dense_null(pat)
        AZ = jax.vmap(lambda z: mv(z.astype(N.dtype)), in_axes=1, out_axes=1)(N)
        return N.T.astype(AZ.dtype) @ AZ
    if pat.crow is not None and A is not None and hasattr(A, "indices"):
        from .fsai import FsaiPattern, _keys_and_values

        _keys, vals = _keys_and_values(FsaiPattern(pat.n, pat.nnz, pat.checksum, [], 1), A)
        N = pat.null.astype(jnp.result_type(vals.dtype, jnp.float32))
        ve = vals[pat.eent]
        Nq = N[pat.qcol]  # (T2, k)

        # E[(i,a), (j,b)] = sum_c (Z_ia^T A)_c Z_jb[c], with (Z_ia^T A)_c = sum_r Z_ia[r] A_rc
        def row(a):
            t = jax.ops.segment_sum(ve * N[pat.erow, a], pat.epair, num_segments=pat.npair)  # (npair,)
            return jax.ops.segment_sum(t[pat.qpair][:, None] * Nq, pat.crow, num_segments=p * p)  # (p*p, k)

        E = jax.lax.map(row, jnp.arange(k))  # (k, p*p, k)
        return E.reshape(k, p, p, k).transpose(1, 0, 2, 3).reshape(p * k, p * k)
    raise TypeError(
        "jno.precond.coarse: a partitioned or sparse coarse space is built from the ASSEMBLED operator's "
        "entries, and this solve only has a matvec. Use a dense global space (parts=None), which needs only "
        "products with the operator, or an assembled solve path."
    )


def _record_kernel(count, who):
    from .solver_api import _GATE_FAILURES

    _GATE_FAILURES.append(_kernel_message(int(count), who))


def _kernel_message(count, who):
    return (
        f"{who}: the coarse space contains {count} direction(s) the operator maps to zero -- a KERNEL of the "
        "system, not a slow mode. A singular problem is fixed in the problem, not the preconditioner: add the "
        "gauge term that removes it (pin a value, a mean-value constraint, a mass/epsilon term), or drop those "
        "vectors from the space."
    )


def coarse_factor(pat: CoarsePattern, A=None, *, mv=None, scale=None, check: bool = True, who: str = "jno.precond.coarse"):
    """Numeric phase (traceable): ``Einv = W (W^T E W)^+ W^T``, ``E = Z^T A Z``, on the coefficients that carry a
    vector (``cidx``; all ``p k`` without padding) and the kept whitened directions (``kidx``).

    ``scale`` is the operator's size (its largest diagonal entry). A right singular direction ``c`` of
    ``W^T E W`` is a kernel when its Rayleigh quotient ``s / |Z W c|^2`` is below ``KERNEL_RTOL * scale``
    while ``Z W c`` is not zero (with overlapping parts a combination of other parts' vectors has ``s ~ 0``
    too, and is no kernel). With ``check`` a kernel raises (concrete values) or is
    recorded for the solve's gate to raise (traced values) -- double precision only."""
    p, k = pat.p, pat.k
    E = _galerkin(pat, A, mv)
    rdt = jnp.finfo(E.dtype).dtype  # the real dtype of E's precision
    Wb = _block_diag(pat.W.astype(rdt))
    if pat.cidx is not None:
        E, Wb = E[pat.cidx][:, pat.cidx], Wb[pat.cidx]
    Wb = Wb[:, pat.kidx]
    Et = Wb.T @ E @ Wb
    Etinv, s, Vh = _coarse_pinv(Et, jnp.maximum(KERNEL_RTOL, 64 * jnp.finfo(rdt).eps * p * k))
    smax = jnp.max(s)
    Einv = Wb @ Etinv @ Wb.T
    if check and rdt == jnp.float64:
        ref = smax if scale is None else jnp.asarray(scale, rdt)
        if pat.Gt is None:  # parts partition the unknowns: Z W is orthonormal on the kept directions
            kernel = jnp.sum(s < KERNEL_RTOL * ref)
        else:  # |Z W c|^2 = c^H (W^T Z^T Z W) c
            g = jnp.real(jnp.sum((Vh @ pat.Gt.astype(rdt)) * Vh.conj(), axis=1))
            kernel = jnp.sum((g > DEP_RTOL) & (s < KERNEL_RTOL * ref * g))
        if isinstance(kernel, jax.core.Tracer):
            jax.lax.cond(kernel > 0, lambda c: jax.debug.callback(_record_kernel, c, who), lambda c: None, kernel)
        elif int(kernel) > 0:
            raise ValueError(_kernel_message(int(kernel), who))
    return Einv


def _coarse_pinv(Et, rtol):
    """``(Et^+, s, Vh)``: the pseudo-inverse of the whitened coarse matrix, dropping the directions below ``rtol``
    of the largest, its singular values and right singular vectors (rows of ``Vh``).

    A Hermitian ``Et`` (to ``sqrt(eps)``: the rounding of the assembly) is decomposed by ``eigh``
    (``s = |lambda|``), any other by the SVD -- on a GPU in float64 the SVD is 15-60x slower at a few hundred
    directions. ``lax.cond`` decides from the values, so a symmetric operator whose tangent turns
    non-symmetric later (Newton, a march) takes the SVD then, and never a symmetrized wrong inverse."""
    rdt = jnp.finfo(Et.dtype).dtype

    def inv(s):
        cut = rtol * jnp.max(s)
        return jnp.where(s > cut, 1.0 / jnp.where(s > cut, s, 1.0), 0.0)

    def by_eigh(M):
        lam, V = jnp.linalg.eigh(M)
        s = jnp.abs(lam)
        return (V * (inv(s) * jnp.sign(lam))[None, :]) @ V.conj().T, s, V.conj().T

    def by_svd(M):
        U, s, Vh = jnp.linalg.svd(M)
        return (Vh.conj().T * inv(s)[None, :]) @ U.conj().T, s, Vh

    hermitian = jnp.max(jnp.abs(Et - Et.conj().T)) <= jnp.sqrt(jnp.finfo(rdt).eps) * jnp.max(jnp.abs(Et))
    return jax.lax.cond(hermitian, by_eigh, by_svd, Et)


def _block_diag(W):
    """``(p k, p k)`` block diagonal of ``W`` ``(p, k, k)`` without a Python loop over parts."""
    p, k, _ = W.shape
    out = jnp.zeros((p, k, p, k), W.dtype)
    i = jnp.arange(p)
    out = out.at[i, :, i, :].set(W)
    return out.reshape(p * k, p * k)


def coarse_apply(pat: CoarsePattern, Einv):
    """``(fwd, t)``: ``Q r = Z Einv Z^T r`` and its transpose ``Z Einv^T Z^T r``."""
    p, k = pat.p, pat.k

    def through(M, r):
        N = pat.null.astype(jnp.finfo(r.dtype).dtype)  # the vectors in r's (real) precision
        rr = r if pat.row is None else r[pat.row]
        rc = jax.ops.segment_sum(N * rr[:, None], pat.part, num_segments=p).reshape(-1)  # Z^T r, (p*k,)
        if pat.cidx is not None:
            rc = rc[pat.cidx]
        xc = M.astype(jnp.result_type(M.dtype, r.dtype)) @ rc
        if pat.cidx is not None:
            xc = jnp.zeros(p * k, xc.dtype).at[pat.cidx].set(xc)
        xc = xc.reshape(p, k)
        y = jnp.sum(N * jnp.take(xc, pat.part, axis=0, mode="fill", fill_value=0), axis=1)
        return (y if pat.row is None else jax.ops.segment_sum(y, pat.row, num_segments=pat.n)).astype(r.dtype)

    return (lambda r: through(Einv, r)), (lambda r: through(Einv.T, r))


# -- resolving a space value to numbers -------------------------------------------------------------------
def _is_formula_space(space) -> bool:
    return isinstance(space, (list, dict))


def _field_index(fem, field) -> int:
    """Position of ``field`` (a trial symbol or an int) in the problem's per-field layout."""
    if isinstance(field, int):
        return field
    if hasattr(fem, "block_index"):
        return fem.block_index(field)
    unknowns = list(getattr(fem, "unknowns", None) or ())
    for i, u in enumerate(unknowns):
        if u is field or (
            getattr(u, "field_key", None) is not None
            and getattr(u, "field_key", None) == getattr(field, "field_key", object())
        ):
            return i
    raise KeyError(f"jno.precond.coarse: {getattr(field, 'name', field)!r} is not an unknown of this problem.")


def _require_nodal(fem, who):
    """Formulas are evaluated at nodes: refuse a field whose DOFs are not nodal Lagrange values."""
    from ..._fem import _trial_spaces

    cons = getattr(fem, "_constraints", None)
    if cons is None:
        return
    other = _trial_spaces(cons) - {"Lagrange"}
    if other:
        raise ValueError(
            f"{who}: a space written as formulas is evaluated at the DOF nodes, which needs nodal Lagrange fields; "
            f"this problem has {sorted(other)} (DOFs that are slopes, edge moments or enrichments, not values). "
            "Give the space as an (n, k) array instead."
        )


def element_triplets(fem, u=None, *, who: str = "jno.space.geneo"):
    """The operator's entries before they are summed, each tagged with its cell: ``(rows, cols, data, cell)``.

    Summing every triplet gives the tangent at ``u`` (zeros by default) before Dirichlet elimination;
    summing those of a SUBSET of cells gives the operator of that subset alone, with natural conditions where
    it is cut -- the local Neumann matrix of GenEO (Spillane et al., Numer. Math. 126, 2014). A boundary
    face's entries carry its parent cell. Rows and columns index the full finite-element space; the arrays
    stay on the device.

    Native Lagrange FEM, steady, real, unreduced only: anything else raises by name.
    """
    factory = getattr(fem, "_element_triplets_factory", None)
    if factory is None:
        raise ValueError(
            f"{who}: needs each cell's element matrix, which only the native Lagrange FEM assembly of a steady "
            "problem provides; this problem is "
            + ("not a jno.fem problem (FDM or an assembled matrix)" if not hasattr(fem, "_mode") else f"'{fem._mode}'")
            + " or uses non-nodal elements, or 1-D. Give the space as an (n, k) array instead."
        )
    _require_nodal(fem, who)
    if getattr(fem, "_mode", None) not in ("linear", "nonlinear"):
        raise ValueError(f"{who}: needs a steady, real problem; this one is '{fem._mode}'.")
    from .solver_api import _fem_reductions

    if _fem_reductions(fem):
        raise NotImplementedError(
            f"{who}: this problem is solved on a REDUCED space (periodic ties, slip or hanging nodes), whose "
            "per-cell matrices are not assembled yet. Give the space as an (n, k) array instead."
        )
    return factory(u)


def _eval_formula(expr, pts) -> np.ndarray:
    """A coordinate formula (or a number) at ``pts`` ``(N, dim)``, as ``(N,)``."""
    from ..._fem import _eval_value_node_at

    if isinstance(expr, (int, float, np.integer, np.floating)):
        return np.full(pts.shape[0], float(expr))
    with jax.ensure_compile_time_eval():
        v = np.asarray(_eval_value_node_at(expr, jnp.asarray(pts)), dtype=np.float64).reshape(-1)
    if v.size == 1:
        return np.full(pts.shape[0], float(v[0]))
    if v.size != pts.shape[0]:
        raise ValueError(
            f"jno.precond.coarse: a formula gave {v.size} values at {pts.shape[0]} nodes; give one value per point."
        )
    return v


def _formula_columns(fem, space, n_full, who) -> np.ndarray:
    """``(n_full, k)``: each column is one entry of the space, evaluated per field and component."""
    from ...precond import _dof_layout

    _require_nodal(fem, who)
    layout = _dof_layout(fem, n_full)
    if isinstance(space, dict):
        per_field = [(_field_index(fem, f), list(cols)) for f, cols in space.items()]
    else:
        if len(layout) != 1:
            raise ValueError(
                f"{who}: this problem has {len(layout)} fields, so a list of formulas is ambiguous. Say which field "
                "each belongs to: coarse({u: [...], p: [...]}) -- a field left out gets zeros."
            )
        per_field = [(0, list(space))]
    cols = []
    for i, entries in per_field:
        pts, idx = layout[i]
        pts = np.asarray(pts)
        ncomp = idx.shape[1]
        for entry in entries:
            comps = tuple(entry) if isinstance(entry, (tuple, list)) else (entry,) * ncomp
            if len(comps) != ncomp:
                raise ValueError(
                    f"{who}: an entry with {len(comps)} components on a field with {ncomp}; give a tuple of {ncomp} "
                    "formulas, or one formula for all of them."
                )
            col = np.zeros(n_full)
            for c, f in enumerate(comps):
                col[idx[:, c]] = _eval_formula(f, pts)
            cols.append(col)
    if not cols:
        raise ValueError(f"{who}: the space is empty.")
    return np.stack(cols, axis=1)


def _restrict(fem, Z_full, n, who) -> np.ndarray:
    """Carry full-space vectors onto the solved space (a periodic tie / slip / hanging nodes: ``P^T A P``),
    field by field with the kept-DOF gather (exact for a vector the reduced space contains)."""
    from .fem_utils import restrict_state
    from .solver_api import _field_layout

    slices, red = _field_layout(fem, n)
    if red is None:
        return Z_full
    out = np.zeros((int(n), Z_full.shape[1]))
    with jax.ensure_compile_time_eval():
        for full, sl, b in zip(fem.blocks, slices, red):
            for k in range(Z_full.shape[1]):
                out[sl, k] = np.asarray(restrict_state(b["P"], Z_full[full, k], b.get("kept"), b.get("vec", 1)))
    return out


def resolve_space(space, fem, n: int, ctx=None, *, A=None, sparse_ok: bool = False, who: str = "jno.precond.coarse"):
    """A space value as an ``(n, k)`` array on the solved space (size ``n``).

    ``space``: formulas (a list, or ``{field: [...]}``; needs the problem), an ``(n, k)`` array, a sparse
    ``(n, k)`` matrix (``scipy.sparse`` or a JAX ``BCOO``), an object with ``basis(ctx)`` (:mod:`jno.space`), or a
    callable ``ctx -> (n, k)``. A computed space or a callable sees ``ctx`` (or, when none is given, the
    problem, the size and the concrete operator ``A``). A sparse space comes back as a ``scipy.sparse`` CSC
    matrix with ``sparse_ok`` (a consumer that keeps it sparse), else as a dense array."""
    if _is_formula_space(space):
        if fem is None:
            raise ValueError(
                f"{who}: a space written as formulas is evaluated at the problem's DOFs, and this solve has no "
                "problem attached. Pass it through fem.solve(precond=...), or give an (n, k) array."
            )
        from .solver_api import _field_layout

        red = _field_layout(fem, n)[1] if getattr(fem, "blocks", None) else None
        if red is None:
            return _formula_columns(fem, space, int(n), who)
        return _restrict(fem, _formula_columns(fem, space, int(fem.blocks[-1].stop), who), n, who)
    if callable(getattr(space, "update", None)):
        raise NotImplementedError(
            f"{who}: this space has an `update` method, i.e. it changes from solve to solve (a recycled Krylov "
            "space). Spaces are resolved once, at setup, so it would silently stay frozen; that kind is not "
            "supported yet. Give a fixed space."
        )
    setup = ctx if ctx is not None else _SetupContext(fem, n, A)
    if hasattr(space, "basis") and callable(space.basis):
        Z = space.basis(setup)
    elif callable(space) and not hasattr(space, "shape"):
        Z = space(setup)
    else:
        Z = space
    if isinstance(Z, jax.core.Tracer) or isinstance(getattr(Z, "data", None), jax.core.Tracer):
        raise TypeError(f"{who}: the space must be concrete (it is set up once, outside the solve's trace).")
    Zs = _as_scipy_sparse(Z)
    if Zs is not None:
        if Zs.shape[0] != int(n):
            raise ValueError(_rows_message(who, Zs.shape, n))
        if not np.all(np.isfinite(Zs.data)):
            raise ValueError(f"{who}: the space has non-finite entries.")
        return Zs if sparse_ok else Zs.toarray()
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.ndim != 2 or Z.shape[0] != int(n):
        raise ValueError(_rows_message(who, Z.shape, n))
    if not np.all(np.isfinite(Z)):
        raise ValueError(f"{who}: the space has non-finite entries.")
    return Z


def _rows_message(who, shape, n):
    return (
        f"{who}: the space has shape {tuple(shape)}, but the operator being preconditioned has {int(n)} rows (a "
        "periodic tie / slip / hanging nodes solve on the REDUCED system P^T A P). Give one row per unknown of "
        "that system, or write the space as formulas."
    )


def _as_scipy_sparse(Z):
    """A ``scipy.sparse`` or JAX ``BCOO`` matrix as ``scipy.sparse`` CSC; ``None`` for anything else."""
    import scipy.sparse as sp

    if sp.issparse(Z):
        return sp.csc_matrix(Z)
    from jax.experimental import sparse as jsparse

    if isinstance(Z, jsparse.BCOO):
        if Z.ndim != 2 or Z.n_batch or Z.n_dense:
            raise ValueError("a sparse space must be a plain 2-D BCOO matrix (no batch or dense dimensions).")
        idx = np.asarray(Z.indices)
        return sp.csc_matrix((np.asarray(Z.data), (idx[:, 0], idx[:, 1])), shape=Z.shape)
    return None


class _SetupContext:
    """What a computed space or a space callable sees when resolved at setup: the problem, the size, the
    concrete operator (``None`` when the consumer was set up without one) and, for a space computed per
    subdomain, the consumer's Schwarz pattern (``None`` otherwise)."""

    def __init__(self, fem, n, A, schwarz=None):
        self.fem, self.n, self.A, self.schwarz = fem, int(n), A, schwarz
