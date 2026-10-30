"""GenEO coarse vectors: per overlapping subdomain, the eigenvectors of ``A_i^N v = lambda D_i A_ii D_i v`` below
``tau`` (Spillane, Dolean, Hauret, Nataf, Pechstein & Scheichl, Numer. Math. 126, 2014, sec. 3; Dolean, Jolivet &
Nataf, *An Introduction to Domain Decomposition Methods*, SIAM 2015, ch. 7).

``A_i^N`` is the subdomain's NEUMANN matrix: the operator assembled over the cells that lie inside the subdomain
only, so it has natural conditions where the subdomain is cut (a floating piece keeps its rigid-body modes). The
assembled operator cannot give it -- its rows on the cut hold the outside cells too -- so it is summed from the
per-cell element matrices (:func:`.deflation.element_triplets`). ``A_ii`` is the subdomain's block of the
operator, ``D_i`` the multiplicity partition of unity (``sum_i R_i^T D_i R_i = I``). The coarse vectors are
``R_i^T D_i v``, one column per kept eigenvector, supported on subdomain ``i``.

Two phases: the tables (which cells lie in which subdomain, where each entry goes) are host work on the
pattern, once; the local matrices and the eigenproblems are one batched JAX computation on the device --
a Cholesky of ``B_i = D_i A_ii D_i`` (symmetric positive definite: a principal block of an SPD operator,
scaled), then ``eigh`` of ``L^-1 A_i^N L^-T``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["neumann_tables", "geneo_space", "GeneoPlan", "geneo_plan", "geneo_values"]


def neumann_tables(pat, rows, cols, cells, held=None):
    """``(nrow, ncol, npos)``, each ``(p, L)``: every entry of every subdomain's Neumann matrix as a local
    (row, column) and its position in the triplet arrays (``len(rows)``: a zero slot, for padding).

    A cell lies inside subdomain ``i`` when all its free unknowns do (``pat.part < pat.p``; the eliminated
    Dirichlet rows belong to no subdomain, and their entries are left out; ``held``, ``(n,)`` bool: unknowns left
    out the same way, though they sit in a subdomain)."""
    import scipy.sparse as sp

    n, p, m = pat.n, pat.p, pat.m
    idx = np.asarray(pat.idx, np.int64)
    free = np.asarray(pat.part) < p
    if held is not None:
        free = free & ~held
    rows, cols, cells = (np.asarray(a, np.int64) for a in (rows, cols, cells))
    T = rows.size
    ncell = int(cells.max()) + 1 if T else 0
    fr = free[rows]
    C = sp.csr_matrix((np.ones(int(fr.sum())), (cells[fr], rows[fr])), shape=(ncell, n))
    C.sum_duplicates()
    C.data[:] = 1.0  # cell -> its free unknowns
    nfree = np.asarray(C.sum(axis=1)).ravel()
    valid = idx < n
    M = sp.csr_matrix((np.ones(int(valid.sum())), (idx[valid], np.nonzero(valid)[0])), shape=(n, p))
    cnt = (C @ M).tocoo()  # how many of a cell's free unknowns lie in each subdomain
    inside = cnt.data == nfree[cnt.row]
    in_cell, in_sub = cnt.row[inside].astype(np.int64), cnt.col[inside].astype(np.int64)
    o = np.argsort(in_sub, kind="stable")
    in_cell, in_sub = in_cell[o], in_sub[o]

    t = np.nonzero(free[rows] & free[cols])[0]
    t = t[np.argsort(cells[t], kind="stable")]
    tc = cells[t]
    start = np.searchsorted(tc, np.arange(ncell))
    lens = np.searchsorted(tc, np.arange(ncell), side="right")[in_cell] - start[in_cell]
    sub = np.repeat(in_sub, lens)  # sorted by subdomain
    first = np.cumsum(lens) - lens
    tid = t[np.repeat(start[in_cell] - first, lens) + np.arange(int(lens.sum()))]

    # local position of (subdomain, unknown): sorted keys i * n + unknown
    key = (np.arange(p, dtype=np.int64)[:, None] * n + idx)[valid]
    loc = np.broadcast_to(np.arange(m), (p, m))[valid]
    ko = np.argsort(key)
    key, loc = key[ko], loc[ko]
    lr = loc[np.searchsorted(key, sub * n + rows[tid])]
    lc = loc[np.searchsorted(key, sub * n + cols[tid])]

    counts = np.bincount(sub, minlength=p)
    L = max(int(counts.max(initial=0)), 1)
    slot = np.arange(sub.size) - np.repeat(np.cumsum(counts) - counts, counts)
    nrow = np.zeros((p, L), np.int64)
    ncol = np.zeros((p, L), np.int64)
    npos = np.full((p, L), T, np.int64)
    nrow[sub, slot], ncol[sub, slot], npos[sub, slot] = lr, lc, tid
    return jnp.asarray(nrow, jnp.int32), jnp.asarray(ncol, jnp.int32), jnp.asarray(npos, jnp.int32)


@functools.partial(jax.jit, static_argnames=("tau", "nev"))
def _local_eigs(blk, nrow, ncol, npos, vals, D, pad, *, tau, nev):
    """The ``nev`` lowest eigenpairs of ``A_i^N v = lambda D_i A_ii D_i v`` for every subdomain at once, and the
    coarse-vector entries ``D_i v``. Padded slots get ``A^N = B = `` a multiple of the identity with its
    eigenvalue above ``tau``, so they are never selected."""
    from jax.scipy.linalg import solve_triangular

    p, m, _ = blk.shape
    ext = jnp.concatenate([vals, jnp.zeros((1,), vals.dtype)])
    AN = jax.vmap(lambda r, c, q: jnp.zeros((m, m), vals.dtype).at[r, c].add(ext[q]))(nrow, ncol, npos)
    eye = jnp.eye(m, dtype=vals.dtype)
    keep = (1.0 - pad)[:, :, None] * (1.0 - pad)[:, None, :]  # a padded row and column couple to nothing
    AN = keep * AN + (2.0 * tau + 1.0) * pad[:, :, None] * eye
    AN = 0.5 * (AN + jnp.swapaxes(AN, 1, 2))
    B = keep * (D[:, :, None] * blk * D[:, None, :]) + pad[:, :, None] * eye
    B = 0.5 * (B + jnp.swapaxes(B, 1, 2))
    Lc = jnp.linalg.cholesky(B)
    X = solve_triangular(Lc, AN, lower=True)  # L^-1 A^N
    Cm = solve_triangular(Lc, jnp.swapaxes(X, 1, 2), lower=True)  # L^-1 A^N L^-T
    w, W = jnp.linalg.eigh(0.5 * (Cm + jnp.swapaxes(Cm, 1, 2)))
    V = solve_triangular(jnp.swapaxes(Lc, 1, 2), W[:, :, :nev], lower=False)  # L^-T w: B-orthonormal
    return w[:, :nev], D[:, :, None] * V


def geneo_space(fem, A, pat, tau: float, nev, *, who: str = "jno.space.geneo"):
    """The GenEO vectors of ``fem`` on the overlapping subdomains of a Schwarz pattern ``pat``.

    Args:
        fem: the problem (its per-cell element matrices are read).
        A: the concrete operator the pattern was built from (BCOO).
        pat: a :class:`.schwarz.SchwarzPattern`.
        tau: keep the eigenvalues below it.
        nev: at most this many vectors per subdomain (``None``: all below ``tau``).

    Returns:
        ``(Z, counts)``: a ``scipy.sparse`` CSC ``(n, K)``, each column supported on one subdomain, and the
        number kept per subdomain.
    """
    import scipy.sparse as sp

    from .deflation import element_triplets
    from .schwarz import local_blocks

    if fem is None:
        raise ValueError(f"{who}: needs the problem (its element matrices); use it through fem.solve(precond=...).")
    rows, cols, vals, cells = element_triplets(fem, who=who)
    n, p, m = pat.n, pat.p, pat.m
    if int(jnp.max(rows)) >= n or int(jnp.max(cols)) >= n:
        raise ValueError(f"{who}: the element matrices live on a larger space than the {n}-unknown operator.")
    nev = m if nev is None else min(int(nev), m)
    item = jnp.dtype(vals.dtype).itemsize
    need = 6 * p * m * m * item
    dev = jax.devices()[0]
    cap = (dev.memory_stats() or {}).get("bytes_limit") if hasattr(dev, "memory_stats") else None
    if cap and need > 0.5 * cap:
        raise ValueError(
            f"{who}: {p} local eigenproblems of size {m} need about {need / 2**30:.1f} GiB on the device, over half "
            "of its memory. Use more Schwarz parts (smaller subdomains) or overlap=1."
        )
    with jax.ensure_compile_time_eval():
        nrow, ncol, npos = neumann_tables(pat, rows, cols, cells)
        blk, _ = local_blocks(pat, A)
    idx = np.asarray(pat.idx)
    valid = idx < n
    mult = np.bincount(idx[valid], minlength=n).astype(np.float64)
    D = np.where(valid, 1.0 / np.maximum(mult[np.minimum(idx, n - 1)], 1.0), 1.0)
    w, V = _local_eigs(
        blk,
        nrow,
        ncol,
        npos,
        jnp.asarray(vals),
        jnp.asarray(D, blk.dtype),
        jnp.asarray(~valid, blk.dtype),
        tau=float(tau),
        nev=nev,
    )
    w, V = np.asarray(w), np.asarray(V)
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(V))):
        raise ValueError(
            f"{who}: a local eigenproblem failed (non-finite values). GenEO needs a symmetric positive definite "
            "operator: its subdomain blocks are factored by Cholesky."
        )
    sel = w < tau  # (p, nev)
    counts = sel.sum(axis=1)
    K = int(counts.sum())
    if K == 0:
        raise ValueError(f"{who}: no subdomain has an eigenvalue below tau={tau:g}, so the space is empty. Raise tau.")
    col = (np.cumsum(sel.ravel()) - 1).reshape(sel.shape)
    mask = valid[:, :, None] & sel[:, None, :]  # (p, m, nev)
    r = np.broadcast_to(idx[:, :, None], mask.shape)[mask]
    c = np.broadcast_to(col[:, None, :], mask.shape)[mask]
    Z = sp.csc_matrix((V[mask], (r, c)), shape=(n, K))
    logging.getLogger("jno").info(
        f"{who}: {K} coarse vectors on {p} subdomains ({counts.min()}-{counts.max()} per subdomain, mean "
        f"{counts.mean():.1f})."
    )
    return Z, counts


# --- a space with a STATIC structure: the same code sets it up once and follows the operator ---------------------
# Every subdomain contributes `nev` columns on its own free rows, whatever the eigenvalues turn out to be; the ones
# above `tau` are masked, never compacted away on the host. The structure -- the pieces, the coarse matrix's entry
# tables and its block-Cholesky plan -- is then fixed by the Schwarz pattern and `nev` alone, and only the values
# move, so they can be recomputed from the current operator inside a compiled solve (a Newton step, a load step).


class EdgeSplit(NamedTuple):
    """A symmetric diagonally dominant operator as pseudo-cells (:func:`edge_split`), ready for :func:`neumann_tables`.

    Attributes:
        n: unknowns.
        rows, cols, cells: the triplets' structure: four per edge, then one per unknown (its diagonal excess).
        keys: ``A``'s sorted keys.
        ei, ej: ``(e,)`` the edges' ends.
        pij, pji, dpos: positions of ``(i, j)``, ``(j, i)`` and the diagonal in ``keys`` (``keys.size``: not stored).
    """

    n: int
    rows: Any
    cols: Any
    cells: Any
    keys: Any
    ei: Any
    ej: Any
    pij: Any
    pji: Any
    dpos: Any


def edge_split(S):
    """The splitting of a symmetric diagonally dominant CSR ``S``, or ``None`` when it is not one (on the rows that
    are not eliminated; to rounding):

        ``A = sum_{i<j} |a_ij| (e_i + s_ij e_j)(e_i + s_ij e_j)^T + diag(d)``, ``d_i = a_ii - sum_j |a_ij| >= 0``.

    Each term is positive semi-definite, so the terms with both ends in a subdomain make a local matrix ``A_i`` with
    ``0 <= A_i <= A`` -- what GenEO's Neumann matrix is for -- from the operator alone: each edge is a "cell" of two
    unknowns. On a diffusion operator it measures the same as the cells of the mesh (iterations and space size,
    plans/deflation-probes/edge_split_geneo_probe.txt). The fully algebraic splitting of Al Daas & Jolivet (SISC 44(4),
    2022), for any SPD matrix, keeps 65 % of the unknowns on layered elasticity, so it is not used."""
    from .deflation import row_eliminated, symmetric_on_free

    if not symmetric_on_free(S):
        return None
    n = S.shape[0]
    coo = S.tocoo()
    held = row_eliminated(S)
    scale = float(np.abs(coo.data).max(initial=0.0))
    off = (coo.row != coo.col) & ~held[coo.row] & ~held[coo.col]
    excess = S.diagonal() - np.bincount(coo.row[off], np.abs(coo.data[off]), minlength=n)
    if np.any(excess[~held] < -1e-12 * scale):
        return None
    keys = np.sort(np.asarray(coo.row, np.int64) * n + np.asarray(coo.col, np.int64))
    up = coo.row < coo.col
    lo = coo.row > coo.col
    pairs = np.unique(
        np.concatenate(
            [np.asarray(coo.row[up], np.int64) * n + coo.col[up], np.asarray(coo.col[lo], np.int64) * n + coo.row[lo]]
        )
    )
    ei, ej = pairs // n, pairs % n

    def at(k):
        pos = np.minimum(np.searchsorted(keys, k), keys.size - 1)
        return np.where(keys[pos] == k, pos, keys.size)

    e = ei.size
    i32 = lambda a: jnp.asarray(a, jnp.int32)  # noqa: E731
    return EdgeSplit(
        n,
        np.concatenate([np.stack([ei, ej, ei, ej], axis=1).ravel(), np.arange(n)]),
        np.concatenate([np.stack([ei, ej, ej, ei], axis=1).ravel(), np.arange(n)]),
        np.concatenate([np.repeat(np.arange(e), 4), e + np.arange(n)]),
        jnp.asarray(keys),
        i32(ei),
        i32(ej),
        i32(at(ei * n + ej)),
        i32(at(ej * n + ei)),
        i32(at(np.arange(n, dtype=np.int64) * n + np.arange(n))),
    )


def edge_values(edges: EdgeSplit, A):
    """The pseudo-cells' values at the operator ``A`` (traceable), in the order of ``edges.rows``."""
    from .fsai import pattern_values

    vals = pattern_values(edges.keys, A, edges.n)
    ext = jnp.concatenate([vals, jnp.zeros((1,), vals.dtype)])
    a = 0.5 * (ext[edges.pij] + ext[edges.pji])
    aa = jnp.abs(a)
    d = ext[edges.dpos] - jax.ops.segment_sum(aa, edges.ei, num_segments=edges.n)
    d = d - jax.ops.segment_sum(aa, edges.ej, num_segments=edges.n)
    return jnp.concatenate([jnp.stack([aa, aa, a, a], axis=1).reshape(-1), d])


class GeneoPlan(NamedTuple):
    """The host part of a static-structure GenEO space (:func:`geneo_plan`).

    Attributes:
        tau: the eigenvalue threshold.
        nev: columns per subdomain (static).
        nrow, ncol, npos: ``(p, L)`` the Neumann tables (:func:`neumann_tables`).
        D: ``(p, m)`` the partition of unity on the subdomains' slots (1 on padding).
        pad: ``(p, m)`` 1 on the padded slots.
        valid: ``(p, m)`` numpy, the real slots: the pieces of the coarse pattern, subdomain by subdomain.
        n_triplets: entries of the RAW element-triplet arrays (a check).
        who: the name in messages.
        src, w: on a reduced system (periodic ties, slip, hanging nodes): the reduced triplets are
            ``vals[src] * w`` (:func:`~.deflation.triplet_map`); ``None`` otherwise.
        prolong: on a reduced system, ``(rows, cols, data, n_full)`` of ``P``: the full iterate is ``P u``.
        edges: no cells to sum: the operator's own edges (:class:`EdgeSplit`), read from ``A`` each time.
    """

    tau: float
    nev: int
    nrow: Any
    ncol: Any
    npos: Any
    D: Any
    pad: Any
    valid: Any
    n_triplets: int
    who: str
    src: Any = None
    w: Any = None
    prolong: Any = None
    edges: Any = None


def geneo_plan(fem, A, pat, tau: float, nev, *, headroom: bool = False, who: str = "jno.space.geneo"):
    """``(plan, w, V, vals)``: the host part of a static-structure GenEO space on the Schwarz pattern ``pat``, and
    the first eigenpairs ``(p, nev)`` / ``(p, m, nev)`` from the element triplets' values ``vals`` at zero.

    ``nev=None`` sizes the columns from these first eigenvalues: the largest count below ``tau`` -- doubled with
    ``headroom``, for a space that follows a changing operator and may need more later (a crack opening, a
    design becoming void). A given ``nev`` is the cap either way."""
    from .deflation import element_triplets, operator_graph, row_eliminated, solved_map, triplet_map
    from .schwarz import local_blocks

    n, p, m = pat.n, pat.p, pat.m
    edges = None
    try:
        rows, cols, vals, cells = element_triplets(fem, who=who, reduced_ok=True)
        P = solved_map(fem, n, who)
    except (ValueError, NotImplementedError) as no_cells:
        # No cells to sum: FDM, non-nodal elements, an assembled matrix. A symmetric diagonally dominant operator
        # still splits exactly, into its own edges (see `edge_split`).
        with jax.ensure_compile_time_eval():
            edges = edge_split(operator_graph(A)[0])
        if edges is None:
            raise ValueError(
                f"{no_cells} ({who} can also work from the operator alone, when it is symmetric and diagonally "
                "dominant; this one is not.)"
            ) from no_cells
        rows, cols, cells = edges.rows, edges.cols, edges.cells
        vals, P = edge_values(edges, A), None
    raw, n_raw = (None, 0) if edges is not None else (vals, int(rows.shape[0]))
    src = tw = prolong = None
    if P is not None:  # a reduced system: each cell's entries carried onto it, P^T K_e P
        with jax.ensure_compile_time_eval():
            r, c, cl = (np.asarray(a) for a in (rows, cols, cells))
        r, c, src_np, w_np = triplet_map(P, r, c)
        src, tw = jnp.asarray(src_np, jnp.int32), jnp.asarray(w_np, vals.dtype)
        rows, cols, cells, vals = jnp.asarray(r), jnp.asarray(c), jnp.asarray(cl[src_np]), vals[src] * tw
        Pc = P.tocoo()
        prolong = (jnp.asarray(Pc.row, jnp.int32), jnp.asarray(Pc.col, jnp.int32), jnp.asarray(Pc.data), P.shape[0])
    if int(jnp.max(rows)) >= n or int(jnp.max(cols)) >= n:
        raise ValueError(f"{who}: the element matrices live on a larger space than the {n}-unknown operator.")
    item = jnp.dtype(vals.dtype).itemsize
    need = 6 * p * m * m * item
    dev = jax.devices()[0]
    cap = (dev.memory_stats() or {}).get("bytes_limit") if hasattr(dev, "memory_stats") else None
    if cap and need > 0.5 * cap:
        raise ValueError(
            f"{who}: {p} local eigenproblems of size {m} need about {need / 2**30:.1f} GiB on the device, over half "
            "of its memory. Use more Schwarz parts (smaller subdomains) or overlap=1."
        )
    with jax.ensure_compile_time_eval():
        # a prescribed row of a march's tangent is eliminated by row only, its column stays: its increment is zero
        # whatever the column holds, so the unknown is left out of the eigenproblems and the coarse vectors
        held = row_eliminated(operator_graph(A)[0])
        nrow, ncol, npos = neumann_tables(pat, rows, cols, cells, held=held)
        blk, _ = local_blocks(pat, A)
    idx = np.asarray(pat.idx)
    valid = idx < n
    mult = np.bincount(idx[valid], minlength=n).astype(np.float64)
    D = jnp.asarray(np.where(valid, 1.0 / np.maximum(mult[np.minimum(idx, n - 1)], 1.0), 1.0), blk.dtype)
    pad = jnp.asarray(~valid | held[np.minimum(idx, n - 1)], blk.dtype)
    first = m if nev is None else min(int(nev), m)
    w, V = _local_eigs(blk, nrow, ncol, npos, jnp.asarray(vals), D, pad, tau=float(tau), nev=first)
    w = np.asarray(w)
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(np.asarray(V)))):
        raise ValueError(
            f"{who}: a local eigenproblem failed (non-finite values). GenEO needs a symmetric positive definite "
            "operator: its subdomain blocks are factored by Cholesky."
        )
    count = int((w < tau).sum(axis=1).max(initial=0))
    if count == 0:  # the local solves already reach everything (a short time step): no correction is needed
        logging.getLogger("jno").info(f"{who}: no subdomain has an eigenvalue below tau={tau:g}: the space is empty.")
    k = first if nev is not None else max(1, min(m, 2 * count if headroom else count))
    plan = GeneoPlan(float(tau), int(k), nrow, ncol, npos, D, pad, valid, n_raw, who, src, tw, prolong, edges)
    return plan, jnp.asarray(w[:, :k]), V[:, :, :k], raw


def geneo_pieces(plan: GeneoPlan, pat):
    """``(row, part)`` of the coarse pattern's pieces: every free slot of every subdomain, subdomain by subdomain."""
    idx = np.asarray(pat.idx)
    return idx[plan.valid], np.repeat(np.arange(pat.p), plan.valid.sum(axis=1))


def geneo_values(plan: GeneoPlan, pat, A, vals, frozen=None):
    """``(null, W, kmask, counts)`` of the space at the operator ``A`` and the element-triplet values ``vals``
    (traceable): the pieces' values ``(L, nev)``, the whitening ``(p, nev, nev)``, the kept directions
    ``(p, nev)`` and the eigenvalues below ``tau`` per subdomain.

    ``frozen`` ``(n,)`` bool: unknowns held fixed in this solve (the other fields of a staggered block step); they
    are padding to the eigenproblems, so no vector lives on them."""
    from .schwarz import local_blocks

    blk, _ = local_blocks(pat, A)
    pad = plan.pad
    if frozen is not None:
        fz = jnp.take(jnp.asarray(frozen), pat.idx, mode="fill", fill_value=True)
        pad = jnp.maximum(pad, fz.astype(pad.dtype))
    if plan.edges is not None:
        vals = edge_values(plan.edges, A)
    else:
        if int(vals.shape[0]) != plan.n_triplets:
            raise ValueError(f"{plan.who}: the element matrices changed shape since the space was set up.")
        if plan.src is not None:
            vals = vals[plan.src] * plan.w.astype(vals.dtype)
    w, V = _local_eigs(
        blk, plan.nrow, plan.ncol, plan.npos, vals, plan.D.astype(blk.dtype), pad, tau=plan.tau, nev=plan.nev
    )
    sel = w < plan.tau
    V = V * (sel[:, None, :] & (pad < 0.5)[:, :, None])
    null = V[plan.valid]  # (L, nev), subdomain by subdomain: the pieces of geneo_pieces
    G = jnp.einsum("pmk,pml->pkl", V, V)  # Z_i^T Z_i per subdomain
    g, U = jnp.linalg.eigh(0.5 * (G + jnp.swapaxes(G, 1, 2)))
    from .deflation import NULL_RTOL

    gmax = jnp.max(g, axis=1, keepdims=True)
    kmask = g > NULL_RTOL * jnp.maximum(gmax, jnp.finfo(g.dtype).tiny)
    W = U * jnp.where(kmask, 1.0 / jnp.sqrt(jnp.where(kmask, g, 1.0)), 0.0)[:, None, :]
    return null, W, kmask, jnp.sum(sel, axis=1)
