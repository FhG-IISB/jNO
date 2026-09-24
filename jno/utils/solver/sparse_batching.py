"""``vmap`` (batching) rules for JAX's sparse primitives that ship without one.

JAX defines JVP and transpose rules for ``csr_matvec`` / ``csr_matmat`` (the cuSPARSE-backed ``CSR @ x``)
and for ``spsolve`` (the sparse direct solve under every ``jno.solve.lu()`` / ``sparse_lu_solve``) but no
batching rule, so ``jax.vmap`` -- and everything built on it: ``jax.jacfwd``, ``jax.jacrev``,
per-sample gradients -- raises ``NotImplementedError: Batching rule for 'csr_matvec' not implemented``.
The rules here fill that gap without replacing anything JAX already defines, and keep every batched
product on cuSPARSE:

* **one matrix, many vectors** -- one SpMV per vector below :data:`SPMM_MIN_BATCH` vectors, one SpMM
  (``csr_matmat``) on the ``(n, B)`` block from there up. Measured on four real FEM operators
  (142k-515k DOF, RTX 3070, float64): SpMM costs 4.3-6.2x a loop of SpMVs at 2 vectors, 1.0-1.5x at 8,
  0.67-1.0x at 12 and 0.28-0.38x at 32 -- cuSPARSE's SpMM is slow for a handful of columns.
* **batched solves** (``spsolve``): the B systems become block-diagonal systems of up to
  :data:`SPSOLVE_BLOCK_NNZ` nonzeros each, one solve per block-diagonal system (see
  :func:`_spsolve_batch`).
* **batched matrices** (values, and optionally the pattern): the B matrices become ONE block-diagonal
  CSR -- column indices offset by ``b * ncols``, row pointers by ``b * nse`` -- applied in a single
  call. On a 303k-DOF operator with 16 value sets: 231 us per item (a single SpMV is 183 us), against
  372 for a ``lax.map`` over CSR and 523 for ``vmap`` over BCOO.

The registration is GLOBAL to the process: it changes what ``vmap`` does for these primitives in any
code, jNO or not. Only programs that raised before are affected. :func:`install` therefore announces
itself through the jNO logger when it registers anything.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental.sparse import csr as _csr
from jax.interpreters import batching

#: Below this many vectors a loop of cuSPARSE SpMVs beats one SpMM (see the module docstring).
SPMM_MIN_BATCH = 12

#: Largest block-diagonal system (in stored nonzeros) handed to ONE ``spsolve``; bigger batches are split
#: into chunks run by ``lax.map``. cuSolver's sparse QR allocates for its fill-in, not for ``nnz``:
#: measured on an 8 GB card, a 3-D 20k-DOF Laplacian failed to allocate at 32 copies (4.3M nnz) and
#: solved at 8 (1.07M), while a 2-D 20k-DOF operator solved at 32 copies (4.4M).
SPSOLVE_BLOCK_NNZ = 1_000_000

_UNBATCHED = None  # the batch dimension JAX passes for an operand that is not being vmapped


def _front(x, bd, size):
    """Move the batch axis to the front, broadcasting an unbatched operand to ``size``."""
    if bd is _UNBATCHED:
        return jnp.broadcast_to(x, (size,) + x.shape)
    return jnp.moveaxis(x, bd, 0)


def _batch_size(args, dims):
    return next(a.shape[d] for a, d in zip(args, dims) if d is not _UNBATCHED)


def _block_diagonal(data, indices, indptr, shape):
    """``(B, nse)`` data/indices and ``(B, nrows + 1)`` indptr -> one CSR of shape ``(B*nrows, B*ncols)``."""
    B, nse = data.shape
    nrows, ncols = shape
    if B * max(nse, nrows, ncols) >= jnp.iinfo(indices.dtype).max:
        raise NotImplementedError(
            f"vmap over a sparse matrix: the block-diagonal layout of {B} batched matrices (nse={nse}, "
            f"shape={shape}) overflows {jnp.dtype(indices.dtype).name} indices. Split the batch (e.g. "
            "jax.lax.map over chunks of it) or use int64 indices."
        )
    b = jnp.arange(B, dtype=indices.dtype)
    big_indices = (indices + (b * ncols)[:, None]).reshape(-1)
    big_indptr = jnp.concatenate([(indptr[:, :-1] + (b * nse)[:, None]).reshape(-1), jnp.full((1,), B * nse, indptr.dtype)])
    return data.reshape(-1), big_indices, big_indptr, (B * nrows, B * ncols)


def _matrix_unbatched(dims):
    return all(d is _UNBATCHED for d in dims[:3])


def _csr_matvec_batch(args, dims, *, shape, transpose):
    data, indices, indptr, v = args
    if _matrix_unbatched(dims):
        V = jnp.moveaxis(v, dims[3], 1)  # (n, B)
        if V.shape[1] < SPMM_MIN_BATCH:
            cols = [
                _csr._csr_matvec(data, indices, indptr, V[:, b], shape=shape, transpose=transpose)
                for b in range(V.shape[1])
            ]
            return jnp.stack(cols, 0), 0
        return _csr._csr_matmat(data, indices, indptr, V, shape=shape, transpose=transpose), 1
    size = _batch_size(args, dims)
    data, indices, indptr, v = (_front(a, d, size) for a, d in zip(args, dims))
    bd, bi, bp, bshape = _block_diagonal(data, indices, indptr, shape)
    out = _csr._csr_matvec(bd, bi, bp, v.reshape(-1), shape=bshape, transpose=transpose)
    return out.reshape(size, -1), 0


def _csr_matmat_batch(args, dims, *, shape, transpose):
    data, indices, indptr, X = args
    if _matrix_unbatched(dims):
        X = jnp.moveaxis(X, dims[3], 1)  # (n, batch, k): fold the batch into the columns
        n, size, k = X.shape
        out = _csr._csr_matmat(data, indices, indptr, X.reshape(n, size * k), shape=shape, transpose=transpose)
        return out.reshape(out.shape[0], size, k), 1
    size = _batch_size(args, dims)
    data, indices, indptr, X = (_front(a, d, size) for a, d in zip(args, dims))
    bd, bi, bp, bshape = _block_diagonal(data, indices, indptr, shape)
    out = _csr._csr_matmat(bd, bi, bp, X.reshape(-1, X.shape[-1]), shape=bshape, transpose=transpose)
    return out.reshape(size, -1, X.shape[-1]), 0


def _spsolve_batch(args, dims, *, tol, reorder):
    """B square systems -> block-diagonal systems, one ``spsolve`` each (cuSolver QR on GPU, SciPy on CPU).

    Up to :data:`SPSOLVE_BLOCK_NNZ` stored nonzeros go into one block-diagonal system; a larger batch is
    split into chunks of that size, run by ``lax.map`` (the last chunk padded with copies of a real
    system, never zeros, so padding cannot make it singular). Measured against a ``lax.map`` of single
    solves: 1.5-2.0x faster on 2-D operators, 1.07x on a 3-D one. A shared matrix with batched
    right-hand sides is replicated: the direct solver takes one right-hand side, so each block is
    factorised separately."""
    import jax
    from jax.experimental.sparse import linalg as _splinalg

    size = _batch_size(args, dims)
    data, indices, indptr, b = (_front(a, d, size) for a, d in zip(args, dims))
    n, nse = b.shape[1], data.shape[1]
    chunk = max(1, min(size, SPSOLVE_BLOCK_NNZ // max(nse, 1)))

    def solve_block(data, indices, indptr, b):
        bd, bi, bp, _ = _block_diagonal(data, indices, indptr, (n, n))
        return _splinalg.spsolve_p.bind(bd, bi, bp, b.reshape(-1), tol=tol, reorder=reorder).reshape(-1, n)

    if chunk >= size:
        return solve_block(data, indices, indptr, b), 0
    n_chunks = -(-size // chunk)
    pad = n_chunks * chunk - size
    padded = [
        jnp.concatenate([a, jnp.broadcast_to(a[:1], (pad,) + a.shape[1:])]) if pad else a
        for a in (data, indices, indptr, b)
    ]
    chunks = [a.reshape((n_chunks, chunk) + a.shape[1:]) for a in padded]
    out = jax.lax.map(lambda c: solve_block(*c), tuple(chunks))
    return out.reshape(n_chunks * chunk, n)[:size], 0


def _rules():
    from jax.experimental.sparse import linalg as _splinalg

    return [
        (_csr.csr_matvec_p, _csr_matvec_batch),
        (_csr.csr_matmat_p, _csr_matmat_batch),
        (_splinalg.spsolve_p, _spsolve_batch),
    ]


_INSTALLED: list = []


def install(log: bool = True) -> list:
    """Register the rules JAX lacks; never replace one JAX defines. Idempotent.

    Returns the names of the primitives registered by THIS call (empty when already installed or when
    JAX has since grown its own rules). When it registers anything it says so through the jNO logger,
    because the effect is process-wide.
    """
    new = []
    for prim, rule in _rules():
        # `primitive_batchers` is a write-only proxy onto `fancy_primitive_batchers` in current JAX
        if prim in batching.fancy_primitive_batchers:
            continue
        batching.primitive_batchers[prim] = rule
        new.append(prim.name)
    _INSTALLED.extend(new)
    if new and log:
        from ..logger import get_logger

        get_logger().warning(
            f"jNO registered vmap (batching) rules for the JAX primitives {', '.join(new)}, which JAX "
            "ships without one. The registration is process-wide: jax.vmap / jacfwd / jacrev through "
            "jax.experimental.sparse CSR products and spsolve now work in ANY code in this process, not "
            "only in jNO. "
            "Programs that worked before are unchanged -- these primitives previously raised "
            "NotImplementedError under vmap. See jno/utils/solver/sparse_batching.py."
        )
    return new
