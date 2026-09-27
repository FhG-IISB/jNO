"""Factored sparse approximate inverse (FSAI) for SPD operators: ``M^{-1} = G^T G ~ A^{-1}``.

Kolotilina & Yeremin, *Factorized sparse approximate inverse preconditionings I. Theory*, SIAM J. Matrix
Anal. Appl. 14(1):45-58, 1993 -- G is lower triangular on a prescribed pattern ``S`` (here the lower
triangle of ``A`` or of ``A^power``); row ``i`` solves the small SPD system ``A[J_i, J_i] y = e_i`` over
its pattern ``J_i`` and is scaled so that ``diag(G A G^T) = 1``: ``G[i, J_i] = y / sqrt(y_i)``.

Why it fits a GPU: *applying* it is two sparse products (``G v`` then ``G^T w``) -- no triangular solves
and no sequential sweep, unlike ILU / incomplete Cholesky -- and each row is computed independently, so
the setup is a batch of small dense Cholesky solves.

Two phases, so the setup composes with ``jit`` and with operators whose VALUES change every solve (a
Newton tangent, a time step): the **symbolic** phase (pattern, row groups, key table) depends only on the
sparsity pattern and runs once on the host; the **numeric** phase (gather the local blocks, solve, scale)
is pure JAX and re-runs from the current values at every materialisation.
"""

from __future__ import annotations

from typing import List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["FsaiPattern", "fsai_pattern", "fsai_factor", "fsai_apply", "g_indices"]

#: Local-block entries gathered per ``lax.map`` batch in the numeric phase: a MEMORY bound (8 bytes each,
#: 32 MB), not a speed knob -- it keeps the ``rows x w x w`` gather from growing with the mesh.
_ENTRIES_PER_BATCH = 1 << 22


class FsaiPattern(NamedTuple):
    """The symbolic phase: everything that depends only on the sparsity pattern.

    Deliberately SMALL: only G's own pattern is kept as arrays. A's key table is re-derived inside the
    trace from the operator's (traced) indices -- a closed-over table is baked into the compiled program
    as a constant, once per use, and at 300k unknowns that made a 3.9 GB executable the GPU refused to
    load. ``checksum`` lets the trace confirm the operator still has the pattern G was built for.
    """

    n: int
    nnz: int  # unique (row, col) positions of A
    checksum: tuple  # two modular sums of A's sorted keys
    groups: List[tuple]  # per distinct row width w: (rows (b,), J (b, w)) -- exact widths, no padding
    power: int


_MODS = (1_000_003, 998_244_353)


def _checksum(keys, xp):
    return tuple(xp.sum(keys % m) for m in _MODS)


def _scipy_csr(A):
    from .amg import _to_scipy_csr

    return _to_scipy_csr(A)


def fsai_pattern(A, *, power: int = 1) -> FsaiPattern:
    """Symbolic phase from a CONCRETE operator. Refuses a non-symmetric one (FSAI is an SPD method)."""
    import scipy.sparse as sp

    if not jax.config.jax_enable_x64:
        raise ValueError(
            "jno.precond.fsai() needs x64 (jax.config.update('jax_enable_x64', True)): its pattern keys are "
            "row*n + col, which overflow int32 beyond ~46k unknowns."
        )
    S = _scipy_csr(A).astype(np.float64)
    S.sum_duplicates()
    n = S.shape[0]
    if S.shape != (n, n):
        raise ValueError(f"jno.precond.fsai(): the operator must be square, got {S.shape}.")
    asym = abs(S - S.T).max() if S.nnz else 0.0
    scale = abs(S).max() if S.nnz else 1.0
    if asym > 1e-10 * scale:
        raise ValueError(
            f"jno.precond.fsai(): the operator is not symmetric (max |A - A^T| = {asym:.3e} against "
            f"max |A| = {scale:.3e}). FSAI approximates the inverse of an SPD matrix; for a non-symmetric "
            "operator use jno.precond.jacobi(), jno.precond.ilu() or jno.precond.amg()."
        )
    P = (abs(S) > 0).astype(np.float64)
    for _ in range(int(power) - 1):
        P = (P @ (abs(S) > 0).astype(np.float64)).tocsr()
    L = sp.tril(P, format="csr")
    L.sort_indices()
    widths = np.diff(L.indptr)
    if np.any(widths == 0) or np.any(L.indices[L.indptr[1:] - 1] != np.arange(n)):
        raise ValueError(
            "jno.precond.fsai(): a row of the operator has no diagonal entry, so its local system is empty. "
            "FSAI needs an SPD operator with a structurally non-zero diagonal."
        )
    groups = []
    for w in np.unique(widths):
        rows = np.nonzero(widths == w)[0]
        J = L.indices[L.indptr[rows][:, None] + np.arange(w)]  # (b, w), sorted, so the diagonal is last
        groups.append((jnp.asarray(rows, jnp.int32), jnp.asarray(J, jnp.int32)))
    coo = S.tocoo()
    keys = np.sort(np.asarray(coo.row, np.int64) * n + np.asarray(coo.col, np.int64))
    return FsaiPattern(n, int(keys.size), tuple(int(v) for v in _checksum(keys, np)), groups, int(power))


def _keys_and_values(pat: FsaiPattern, A):
    """A's sorted unique keys ``row*n + col`` and summed values, from the (traced) operator itself. If its
    pattern is not the one the factor was built for, the values are poisoned with NaN, so the solve fails
    loudly instead of using a factor for another matrix."""
    n = pat.n
    r, c = A.indices[:, 0].astype(jnp.int64), A.indices[:, 1].astype(jnp.int64)
    in_range = (r >= 0) & (r < n) & (c >= 0) & (c < n)  # sum_duplicates pads out of range with zeros
    sentinel = jnp.int64(n) * n
    tk = jnp.where(in_range, r * n + c, sentinel)
    keys, inv = jnp.unique(tk, return_inverse=True, size=pat.nnz + 1, fill_value=sentinel)
    vals = jax.ops.segment_sum(jnp.where(in_range, A.data, 0.0), inv.reshape(-1), num_segments=pat.nnz + 1)
    keys, vals = keys[: pat.nnz], vals[: pat.nnz]
    # Same pattern <=> same count (the extra slot is the sentinel, or unused) and the same key checksum.
    same = jnp.all(keys < sentinel)
    for got, want in zip(_checksum(keys, jnp), pat.checksum):
        same = same & (got == want)
    return keys, jnp.where(same, vals, jnp.nan)


def g_indices(pat: FsaiPattern):
    """G's (row, col) indices, in the order :func:`fsai_factor` emits its data."""
    rows = [jnp.repeat(r, J.shape[1]) for r, J in pat.groups]
    cols = [J.reshape(-1) for _r, J in pat.groups]
    return jnp.stack([jnp.concatenate(rows), jnp.concatenate(cols)], 1)


def fsai_factor(pat: FsaiPattern, A):
    """Numeric phase: the data of G (aligned with :func:`g_indices`) from A's current values -- traceable."""
    n = pat.n
    keys, vals = _keys_and_values(pat, A)
    out = []
    for _rows, J in pat.groups:
        w = J.shape[1]

        def rows_of(Jb, _w=w):
            kk = Jb[:, :, None].astype(jnp.int64) * n + Jb[:, None, :].astype(jnp.int64)  # (c, w, w)
            pos = jnp.clip(jnp.searchsorted(keys, kk), 0, keys.shape[0] - 1)
            loc = jnp.where(keys[pos] == kk, vals[pos], 0.0)  # structurally zero -> 0
            e = jnp.zeros((_w,), loc.dtype).at[-1].set(1.0)  # the diagonal is the last pattern entry
            chol = jnp.linalg.cholesky(loc)
            rhs = jnp.broadcast_to(e, Jb.shape)[..., None]
            y = jax.scipy.linalg.cho_solve((chol, True), rhs)[..., 0]
            return y / jnp.sqrt(y[:, -1:])

        batch = max(1, _ENTRIES_PER_BATCH // (w * w))
        g = jax.lax.map(lambda Jb: rows_of(Jb[None])[0], J, batch_size=min(batch, J.shape[0]))
        out.append(g.reshape(-1))
    return jnp.concatenate(out)


def fsai_apply(pat: FsaiPattern, g_data):
    """``v -> G^T (G v)`` in the storage measured fastest for G (CSR or split COO)."""
    import jax.experimental.sparse as jsp

    from .linear import sparse_matvec

    G = jsp.BCOO((g_data, g_indices(pat)), shape=(pat.n, pat.n))
    mv, mvt = sparse_matvec(G), sparse_matvec(G, transpose=True)
    return lambda v: mvt(mv(v))
