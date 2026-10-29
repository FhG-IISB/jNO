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
* the coarse space (two-level) is any set of vectors split over the parts -- by default Nicolaides': one
  constant per part -- built, factored and applied by :mod:`deflation` (``Q = Z (Z^T A Z)^{-1} Z^T``).

Two phases, as for :mod:`fsai`: the symbolic one (partition, overlap, index tables) runs once on the host from a
concrete operator; the numeric one (local matrices, their inverses, the coarse matrix) is pure JAX from the
operator's current values -- inside a compiled solve, a Newton loop or a time march.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["SchwarzPattern", "schwarz_pattern", "schwarz_factor", "schwarz_apply"]


class SchwarzPattern(NamedTuple):
    """The symbolic phase (pattern only). ``idx``: each part's unknowns (padded with ``n``, out of range);
    ``own``: which of them the part OWNS (the non-overlapping partition); ``lrow``/``lcol``/``lpos``: every
    local-matrix entry as (row, col) in the part and its position in A's sorted unique entries (``nnz`` = a
    zero slot, ``nnz + 1`` = a one slot for the padded diagonal).

    Attributes:
        n: system size.
        p: number of parts.
        m: unknowns per part (padded).
        nnz: unique entries of ``A``.
        checksum: identifies ``A``'s pattern.
        idx: ``(p, m)`` each part's unknowns.
        own: ``(p, m)`` which of them the part owns.
        lrow: ``(p, L)`` local-matrix rows.
        lcol: ``(p, L)`` local-matrix columns.
        lpos: ``(p, L)`` their positions in ``A``'s unique entries.
        part: ``(n,)`` owner of each unknown.
        coarse: the two-level coarse space on these parts (a :class:`~.deflation.CoarsePattern`), or ``None``.
        diso: unknowns with no neighbours (eliminated rows), solved by their diagonal.
        dpos: position of their diagonal in ``A``'s unique entries.
    """

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
    coarse: Any  # the coarse space on these parts (deflation.CoarsePattern), or None for one level
    diso: jnp.ndarray  # (q,) int32 the unknowns with no neighbours (eliminated Dirichlet rows): solved by their diagonal
    dpos: jnp.ndarray  # (q,) int32 position of their diagonal in A's sorted unique entries


def _partition(G, parts: int) -> np.ndarray:
    """Part id of every node of the symmetric, loop-free graph ``G`` (no isolated nodes), by METIS k-way
    (Karypis & Kumar, SIAM J. Sci. Comput. 20(1), 1998): balanced parts with a minimal edge cut, i.e. compact
    subdomains with short interfaces. Contiguous parts are requested whenever the graph is connected (METIS
    cannot honour it otherwise). Seeded, so the same operator always gets the same partition."""
    if parts == 1:
        return np.zeros(G.shape[0], np.int64)
    try:
        import pymetis
    except ImportError as e:
        raise ImportError(
            "jno.precond.schwarz() partitions the operator with METIS, which is not installed. Install it with "
            'pip install "jax-numerical-operators[fem]" (or [fdm], or just [metis]).'
        ) from e
    from scipy.sparse.csgraph import connected_components

    connected = connected_components(G, directed=False, return_labels=False) == 1
    adj = pymetis.CSRAdjacency(G.indptr.astype(np.int64), G.indices.astype(np.int64))
    cut = pymetis.part_graph(parts, adj, contiguous=connected or None, options=pymetis.Options(seed=0))
    return np.asarray(cut[1], np.int64)


def schwarz_pattern(A, *, parts: int, overlap: int = 1, coarse_space=None) -> SchwarzPattern:
    """Symbolic phase from a CONCRETE operator. ``coarse_space``: ``(n, k)`` vectors split over the parts for the
    second level, or ``None`` for one level."""
    import scipy.sparse as sp

    from .deflation import coarse_pattern, metis_parts, operator_graph
    from .fsai import _checksum

    if not jax.config.jax_enable_x64:
        raise ValueError("jno.precond.schwarz() needs x64: its entry keys row*n + col overflow int32.")
    S, G, iso = operator_graph(A)
    n = S.shape[0]
    # Unknowns with no neighbours (an eliminated Dirichlet row: 400 of 12k on a square) are 1x1 problems of their
    # own, solved exactly by their diagonal. Partitioned, they formed whole subdomains of identity rows (2 of 16
    # on a 514-unknown square) -- blocks that cost as much as any other and did nothing.
    core = np.nonzero(~iso)[0]
    part = metis_parts(G, iso, parts)  # isolated unknowns: part id `parts`, out of range everywhere below
    parts = int(min(max(1, parts), max(1, core.size)))

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

    coarse = None if coarse_space is None else coarse_pattern(coarse_space, part, parts, A=A, S=S)
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
        coarse,
        jnp.asarray(diso, jnp.int32),
        jnp.asarray(dpos, jnp.int32),
    )


def local_blocks(pat: SchwarzPattern, A):
    """``(blk, vals)``: every part's local matrix ``R_i A R_i^T``, ``(p, m, m)`` with the identity on its padded
    slots, and ``A``'s sorted unique entries (traceable)."""
    from .fsai import FsaiPattern, _keys_and_values

    _keys, vals = _keys_and_values(FsaiPattern(pat.n, pat.nnz, pat.checksum, [], 1), A)
    ext = jnp.concatenate([vals, jnp.zeros((1,), vals.dtype), jnp.ones((1,), vals.dtype)])
    blk = jnp.zeros((pat.p, pat.m, pat.m), vals.dtype)
    return jax.vmap(lambda b, r, c, q: b.at[r, c].add(ext[q]))(blk, pat.lrow, pat.lcol, pat.lpos), vals


def schwarz_factor(pat: SchwarzPattern, A, *, coarse: bool, mesh=None):
    """Numeric phase (traceable): the inverse of every local matrix and, two-level, of the coarse matrix."""
    blk, vals = local_blocks(pat, A)
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
    if not coarse or pat.coarse is None:
        return local, None, dinv
    from .deflation import coarse_factor

    # Q = Z (Z^T A Z)^{-1} Z^T on the parts; a part whose vectors vanish (only Dirichlet rows) drops out, a kernel
    # of A inside the space is refused (deflation.coarse_factor).
    return local, coarse_factor(pat.coarse, A, scale=jnp.max(jnp.abs(vals)), who="jno.precond.schwarz(coarse=...)"), dinv


def schwarz_apply(pat: SchwarzPattern, factors, *, restricted: bool, mv=None):
    """``v -> M^-1 v``: the local solves summed back (all overlap contributions for the symmetric additive
    method, only the owned ones for RAS) plus, two-level, the coarse correction ``Z A_c^-1 Z^T v`` -- applied
    MULTIPLICATIVELY after the coarse solve when ``mv`` (``v -> A v``) is given (the "balanced" hybrid, which
    converges faster than the purely additive sum)."""
    local, coarse, dinv = factors
    n = pat.n

    def locals_(r):
        rl = jnp.take(r, pat.idx, mode="fill", fill_value=0)  # (p, m); padded slots read 0
        zl = jnp.einsum("pij,pj->pi", local, rl)
        w = pat.own if restricted else jnp.ones_like(pat.own)
        z = jnp.zeros((n,), r.dtype).at[pat.idx.reshape(-1)].add((zl * w).reshape(-1), mode="drop")
        return z.at[pat.diso].add(dinv.astype(r.dtype) * r[pat.diso])

    if coarse is None:
        return locals_
    from .deflation import coarse_apply

    coarse_ = coarse_apply(pat.coarse, coarse)[0]  # Z E^+ Z^T r

    if mv is None:
        return lambda r: coarse_(r) + locals_(r)

    def hybrid(r):  # coarse first, then the local solves on the remaining residual (Toselli & Widlund 2.4)
        z = coarse_(r)
        z = z + locals_(r - mv(z))
        return z + coarse_(r - mv(z)) if not restricted else z

    return hybrid
