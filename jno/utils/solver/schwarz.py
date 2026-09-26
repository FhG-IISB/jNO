"""Algebraic overlapping Schwarz preconditioners (one- and two-level) for any assembled sparse operator.

Toselli & Widlund, *Domain Decomposition Methods*, Springer 2005 (Ch. 3: additive Schwarz; Sec. 3.8 the
two-level method); Cai & Sarkis, SIAM J. Sci. Comput. 21(2), 1999 (restricted additive Schwarz, RAS);
Nicolaides, SIAM J. Numer. Anal. 24(2), 1987 (the piecewise-constant coarse space); Karypis & Kumar, SIAM J. Sci.
Comput. 20(1), 1998 (METIS, the partition).

Built purely from the operator ``A`` -- no mesh, no physics -- so it serves FEM of any element, FDM, multi-field
and the fused complex 2n block alike:

* the unknowns are split into ``p`` parts by METIS on ``A``'s graph (Karypis & Kumar 1998: balanced, compact parts
  with a minimal cut). Unknowns with no neighbours -- eliminated Dirichlet rows -- are not partitioned: each is its
  own 1x1 problem, solved by its diagonal;
* each part grows by ``overlap`` layers of graph neighbours; its local problem is ``A`` restricted to that set
  (Dirichlet on the grown boundary -- the algebraic Schwarz subproblem);
* all local problems are PADDED to one size and solved together (``vmap``): the same layout later shards
  across devices, one block of subdomains per device, which is what a compiled loop needs;
* the coarse space (two-level) is Nicolaides': one constant per part, ``A_c = Z^T A Z``, a dense ``p x p``.

Two phases, as for :mod:`fsai`: the symbolic one (partition, overlap, index tables) runs once on the host from a
concrete operator; the numeric one (local matrices, their inverses, the coarse matrix) is pure JAX from the
operator's current values -- inside a compiled solve, a Newton loop or a time march.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["SchwarzPattern", "schwarz_pattern", "schwarz_factor", "schwarz_apply"]


class SchwarzPattern(NamedTuple):
    """The symbolic phase (pattern only). ``idx``: each part's unknowns (padded with ``n``, out of range);
    ``own``: which of them the part OWNS (the non-overlapping partition); ``lrow``/``lcol``/``lpos``: every
    local-matrix entry as (row, col) in the part and its position in A's sorted unique entries (``nnz`` = a
    zero slot, ``nnz + 1`` = a one slot for the padded diagonal)."""

    n: int
    p: int
    m: int
    nnz: int
    checksum: tuple
    idx: jnp.ndarray  # (p, m) int32
    own: jnp.ndarray  # (p, m) bool
    lrow: jnp.ndarray  # (p, L) int32
    lcol: jnp.ndarray  # (p, L) int32
    lpos: jnp.ndarray  # (p, L) int32
    part: jnp.ndarray  # (n,) int32 owner of each unknown
    crow: jnp.ndarray  # (nnz,) int32 coarse index part[r] * p + part[c] of each unique entry
    erow: jnp.ndarray  # (nnz,) int32 row of each unique entry
    ecol: jnp.ndarray  # (nnz,) int32 column of each unique entry
    null: jnp.ndarray  # (n, k) the near-null-space vectors the coarse space carries per part (k = 1: Nicolaides)
    diso: jnp.ndarray  # (q,) int32 the unknowns with no neighbours (eliminated Dirichlet rows): solved by their diagonal
    dpos: jnp.ndarray  # (q,) int32 position of their diagonal in A's sorted unique entries


def _partition(G, parts: int) -> np.ndarray:
    """Part id of every node of the symmetric, loop-free graph ``G`` (no isolated nodes), by METIS k-way
    (Karypis & Kumar, SIAM J. Sci. Comput. 20(1), 1998): balanced parts with a minimal edge cut, i.e. compact
    subdomains with short interfaces. Contiguous parts are requested whenever the graph is connected (METIS
    cannot honour it otherwise). Seeded, so the same operator always gets the same partition."""
    if parts == 1:
        return np.zeros(G.shape[0], np.int64)
    import pymetis
    from scipy.sparse.csgraph import connected_components

    connected = connected_components(G, directed=False, return_labels=False) == 1
    adj = pymetis.CSRAdjacency(G.indptr.astype(np.int64), G.indices.astype(np.int64))
    cut = pymetis.part_graph(parts, adj, contiguous=connected or None, options=pymetis.Options(seed=0))
    return np.asarray(cut[1], np.int64)


def schwarz_pattern(A, *, parts: int, overlap: int = 1, nullspace=None) -> SchwarzPattern:
    """Symbolic phase from a CONCRETE operator."""
    import scipy.sparse as sp

    from .amg import _to_scipy_csr
    from .fsai import _checksum

    if not jax.config.jax_enable_x64:
        raise ValueError("jno.precond.schwarz() needs x64: its entry keys row*n + col overflow int32.")
    S = _to_scipy_csr(A).astype(np.float64)
    S.sum_duplicates()
    n = S.shape[0]
    if S.shape != (n, n):
        raise ValueError(f"jno.precond.schwarz(): the operator must be square, got {S.shape}.")
    G = (abs(S) + abs(S).T).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    G.data[:] = 1.0
    # Unknowns with no neighbours (an eliminated Dirichlet row: 400 of 12k on a square) are 1x1 problems of their
    # own, solved exactly by their diagonal. Partitioned, they formed whole subdomains of identity rows (2 of 16
    # on a 514-unknown square) -- blocks that cost as much as any other and did nothing.
    iso = np.diff(G.indptr) == 0
    core = np.nonzero(~iso)[0]
    parts = int(min(max(1, parts), max(1, core.size)))
    part = np.full(n, parts, np.int64)  # isolated unknowns: part id `parts`, out of range everywhere below
    part[core] = _partition(G[core][:, core], parts)

    member = sp.csr_matrix((np.ones(core.size), (core, part[core])), shape=(n, parts))
    for _ in range(int(overlap)):  # grow every part by one layer of graph neighbours
        member = ((G @ member) + member).tocsr()
        member.data[:] = 1.0
    memberT = member.T.tocsr()
    sets = [memberT.indices[memberT.indptr[i] : memberT.indptr[i + 1]] for i in range(parts)]
    m = max(s.size for s in sets)

    coo = S.tocoo()
    keys = np.sort(np.asarray(coo.row, np.int64) * n + np.asarray(coo.col, np.int64))
    idx = np.full((parts, m), n, np.int64)
    own = np.zeros((parts, m), bool)
    rows, cols, poss = [], [], []
    for i, s in enumerate(sets):
        s = np.sort(s)
        idx[i, : s.size] = s
        own[i, : s.size] = part[s] == i
        sub = S[s][:, s].tocoo()
        pos = np.searchsorted(keys, s[sub.row].astype(np.int64) * n + s[sub.col])
        pad = np.arange(s.size, m)  # identity on the padded slots
        rows.append(np.concatenate([sub.row, pad]))
        cols.append(np.concatenate([sub.col, pad]))
        poss.append(np.concatenate([pos, np.full(pad.size, keys.size + 1)]))
    L = max(r.size for r in rows)

    def stack(xs, fill):
        return np.stack([np.concatenate([x, np.full(L - x.size, fill)]) for x in xs])

    ku = keys // n, keys % n
    null = np.ones((n, 1)) if nullspace is None else np.asarray(nullspace, np.float64).reshape(n, -1)
    null = np.where(iso[:, None], 0.0, null)  # the coarse space lives on the partitioned unknowns only
    crow = np.where(iso[ku[0]] | iso[ku[1]], parts * parts, part[ku[0]] * parts + part[ku[1]])
    diso = np.nonzero(iso)[0]
    dpos = np.searchsorted(keys, diso.astype(np.int64) * n + diso)
    return SchwarzPattern(
        n,
        parts,
        m,
        int(keys.size),
        tuple(int(v) for v in _checksum(keys, np)),
        jnp.asarray(idx, jnp.int32),
        jnp.asarray(own),
        jnp.asarray(stack(rows, 0), jnp.int32),
        jnp.asarray(stack(cols, 0), jnp.int32),
        jnp.asarray(stack(poss, keys.size), jnp.int32),  # padding entries read the zero slot
        jnp.asarray(part, jnp.int32),
        jnp.asarray(crow, jnp.int32),
        jnp.asarray(ku[0], jnp.int32),
        jnp.asarray(ku[1], jnp.int32),
        jnp.asarray(null),
        jnp.asarray(diso, jnp.int32),
        jnp.asarray(dpos, jnp.int32),
    )


def schwarz_factor(pat: SchwarzPattern, A, *, coarse: bool, mesh=None):
    """Numeric phase (traceable): the inverse of every local matrix and, two-level, of the coarse matrix."""
    from .fsai import FsaiPattern, _keys_and_values

    _keys, vals = _keys_and_values(FsaiPattern(pat.n, pat.nnz, pat.checksum, [], 1), A)
    ext = jnp.concatenate([vals, jnp.zeros((1,), vals.dtype), jnp.ones((1,), vals.dtype)])
    blk = jnp.zeros((pat.p, pat.m, pat.m), vals.dtype)
    blk = jax.vmap(lambda b, r, c, q: b.at[r, c].add(ext[q]))(blk, pat.lrow, pat.lcol, pat.lpos)
    if mesh is not None:
        # One block of subdomains per device: each builds, inverts and stores only its own (p/devices of
        # them), and the application's scatter-add of their results becomes one all-reduce.
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from .sharding import SHARD_AXIS

        blk = jax.lax.with_sharding_constraint(blk, NamedSharding(mesh, P(SHARD_AXIS, None, None)))
    # The local INVERSES, not LU factors: applying them is then one batched dense matrix-vector product, where
    # batched triangular solves of small blocks are among the slowest kernels a GPU runs (measured ~1 ms per
    # application for 256 blocks of 102). Same memory (m^2 per part); the blocks are small principal
    # submatrices, as well conditioned as the operator allows.
    local = jnp.linalg.inv(blk)
    dinv = 1.0 / vals[pat.dpos]  # a zero diagonal on an isolated row is a singular operator: inf -> NaN, loudly
    if not coarse:
        return local, None, dinv
    # A_c = Z^T A Z with Z = the near-null-space vectors restricted to each part (one column per part and
    # vector). Entry by entry: A_c[(i,a), (j,b)] = sum over A's entries (r, c) with part(r) = i, part(c) = j of
    # N[r,a] A_rc N[c,b] -- k^2 streaming segment sums over the entries, no (nnz x k^2) intermediate.
    k = pat.null.shape[1]
    N = pat.null.astype(vals.dtype)
    Nr, Nc = N[pat.erow], N[pat.ecol]
    blocks = [
        [jax.ops.segment_sum(vals * Nr[:, a] * Nc[:, b], pat.crow, num_segments=pat.p * pat.p) for b in range(k)]
        for a in range(k)
    ]
    Ac = jnp.stack([jnp.stack(row, -1) for row in blocks], -2)  # (p*p, k, k)
    Ac = Ac.reshape(pat.p, pat.p, k, k).transpose(0, 2, 1, 3).reshape(pat.p * k, pat.p * k)
    # pinv: a part whose modes are pinned by Dirichlet rows (or that has fewer unknowns than modes) makes A_c
    # singular; the pseudo-inverse drops those directions instead of dividing by zero.
    return local, jnp.linalg.pinv(Ac, hermitian=not jnp.iscomplexobj(Ac)), dinv


def schwarz_apply(pat: SchwarzPattern, factors, *, restricted: bool, mv=None):
    """``v -> M^-1 v``: the local solves summed back (all overlap contributions for the symmetric additive
    method, only the owned ones for RAS) plus, two-level, the coarse correction ``Z A_c^-1 Z^T v`` -- applied
    MULTIPLICATIVELY after the coarse solve when ``mv`` (``v -> A v``) is given (the "balanced" hybrid, which
    converges faster than the purely additive sum)."""
    local, coarse, dinv = factors
    n, p = pat.n, pat.p

    def locals_(r):
        rl = jnp.take(r, pat.idx, mode="fill", fill_value=0)  # (p, m); padded slots read 0
        zl = jnp.einsum("pij,pj->pi", local, rl)
        w = pat.own if restricted else jnp.ones_like(pat.own)
        z = jnp.zeros((n,), r.dtype).at[pat.idx.reshape(-1)].add((zl * w).reshape(-1), mode="drop")
        return z.at[pat.diso].add(dinv.astype(r.dtype) * r[pat.diso])

    if coarse is None:
        return locals_

    N = pat.null
    k = N.shape[1]

    def coarse_(r):
        Nr = N.astype(r.dtype)
        rc = jax.ops.segment_sum(Nr * r[:, None], pat.part, num_segments=p).reshape(-1)  # Z^T r, (p*k,)
        xc = (coarse @ rc).reshape(p, k)
        return jnp.sum(Nr * jnp.take(xc, pat.part, axis=0, mode="fill", fill_value=0), axis=1)  # Z x_c

    if mv is None:
        return lambda r: coarse_(r) + locals_(r)

    def hybrid(r):  # coarse first, then the local solves on the remaining residual (Toselli & Widlund 2.4)
        z = coarse_(r)
        z = z + locals_(r - mv(z))
        return z + coarse_(r - mv(z)) if not restricted else z

    return hybrid
