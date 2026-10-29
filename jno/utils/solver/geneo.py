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

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["neumann_tables", "geneo_space"]


def neumann_tables(pat, rows, cols, cells):
    """``(nrow, ncol, npos)``, each ``(p, L)``: every entry of every subdomain's Neumann matrix as a local
    (row, column) and its position in the triplet arrays (``len(rows)``: a zero slot, for padding).

    A cell lies inside subdomain ``i`` when all its free unknowns do (``pat.part < pat.p``; the eliminated
    Dirichlet rows belong to no subdomain, and their entries are left out)."""
    import scipy.sparse as sp

    n, p, m = pat.n, pat.p, pat.m
    idx = np.asarray(pat.idx, np.int64)
    free = np.asarray(pat.part) < p
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
    AN = AN + (2.0 * tau + 1.0) * pad[:, :, None] * eye
    AN = 0.5 * (AN + jnp.swapaxes(AN, 1, 2))
    B = D[:, :, None] * blk * D[:, None, :]
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
