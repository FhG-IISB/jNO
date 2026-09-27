"""``vmap`` (batching) rules for JAX's sparse primitives that ship without one.

JAX defines JVP and transpose rules for ``csr_matvec`` / ``csr_matmat`` (the cuSPARSE-backed ``CSR @ x``)
and for ``spsolve`` (the sparse direct solve under every ``jno.solve.lu()`` / ``sparse_lu_solve``) but no
batching rule, so ``jax.vmap`` -- and everything built on it: ``jax.jacfwd``, ``jax.jacrev``,
per-sample gradients -- raises ``NotImplementedError: Batching rule for 'csr_matvec' not implemented``.
The rules here fill that gap without replacing anything JAX already defines, and keep every batched
product on cuSPARSE:

* **one matrix, many vectors** -- either one SpMV per vector or one SpMM (``csr_matmat``) on the
  ``(n, B)`` block, whichever is faster ON THIS DEVICE: the choice is measured the first time a batch
  shape is traced (see :func:`_prefer_spmm`), never hard-coded. It matters: on one RTX 3070 SpMM cost
  4-6x a loop of SpMVs at 2 vectors and 0.3x at 32, and the crossover moves with the GPU, the cuSPARSE
  version and the dtype.
* **batched solves** (``spsolve``): one system per ``lax.map`` step by default -- the memory of an unbatched
  solve on any machine -- or ``jno.setup(lu_stack=k)`` systems per block-diagonal call (see
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

#: Per-device measured choices, keyed on (platform, device kind, shape, nnz, batch, dtype, transpose).
_SPMM_DECISIONS: dict = {}


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


def _prefer_spmm(shape, nse, batch, dtype, transpose) -> bool:
    """One SpMM, or one SpMV per vector? MEASURED on the default device the first time this batch shape
    is traced, then cached for the process -- the way XLA autotunes its GEMMs at compile time.

    The real sparsity pattern is a tracer here, so both variants are timed on a banded CSR of the SAME
    shape and nonzeros per row, generated on the device. If timing fails for any reason the loop is
    chosen -- it is never slower than ``batch`` separate products -- and the failure is logged rather
    than hidden.
    """
    import time

    import jax

    dev = jax.devices()[0]
    key = (dev.platform, dev.device_kind, tuple(shape), int(nse), int(batch), jnp.dtype(dtype).name, bool(transpose))
    if key in _SPMM_DECISIONS:
        return _SPMM_DECISIONS[key]
    try:
        with jax.ensure_compile_time_eval():
            # Same size as the real operator: a scaled-down stand-in moves the crossover (measured: a
            # 5k-row stand-in picked SpMM from 4 vectors where the 300k-row operator needs 8-12). The
            # pattern is a band generated ON the device -- sorted, in bounds, built in milliseconds.
            nrows, ncols = shape
            per_row = max(1, min(ncols, int(round(nse / max(nrows, 1)))))
            r = jnp.arange(nrows, dtype=jnp.int32)[:, None]
            start = (r.astype(jnp.int64) * (ncols - per_row) // max(nrows - 1, 1)).astype(jnp.int32)
            idx = (start + jnp.arange(per_row, dtype=jnp.int32)[None, :]).reshape(-1)
            ptr = jnp.arange(nrows + 1, dtype=jnp.int32) * per_row
            data = jnp.ones(idx.size, dtype)
            tshape = (nrows, ncols)
            V = jnp.ones((nrows if transpose else ncols, batch), dtype)

            spmm = jax.jit(lambda V: _csr._csr_matmat(data, idx, ptr, V, shape=tshape, transpose=transpose))
            loop = jax.jit(
                lambda V: jnp.stack(
                    [_csr._csr_matvec(data, idx, ptr, V[:, b], shape=tshape, transpose=transpose) for b in range(batch)]
                )
            )

            def best(f):
                jax.block_until_ready(f(V))
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    jax.block_until_ready(f(V))
                    ts.append(time.perf_counter() - t0)
                return min(ts)

            choice = best(spmm) < best(loop)
    except Exception as exc:  # noqa: BLE001 -- a failed measurement must not break the user's vmap
        from ..logger import get_logger

        get_logger().warning(
            f"jno sparse vmap: could not time SpMM against an SpMV loop for a batch of {batch} on "
            f"{dev.device_kind} ({type(exc).__name__}: {exc}); using the SpMV loop, which is never slower "
            f"than {batch} separate products."
        )
        choice = False
    _SPMM_DECISIONS[key] = choice
    return choice


def _matrix_unbatched(dims):
    return all(d is _UNBATCHED for d in dims[:3])


def _csr_matvec_batch(args, dims, *, shape, transpose):
    data, indices, indptr, v = args
    if _matrix_unbatched(dims):
        V = jnp.moveaxis(v, dims[3], 1)  # (n, B)
        if not _prefer_spmm(shape, data.shape[0], V.shape[1], data.dtype, transpose):
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


#: How many systems a vmapped ``spsolve`` (``jno.solve.lu()``, the default "device" backend) stacks into
#: ONE block-diagonal cuSolver call. Set it with ``jno.setup(lu_stack=k)`` (or ``[jno] lu_stack`` in
#: ``.jno.toml``); read when the vmap is traced.
_LU_STACK = 1


def set_lu_stack(k: int) -> None:
    """Set :data:`_LU_STACK` (``jno.setup(lu_stack=...)``). ``k`` must be a positive integer."""
    global _LU_STACK
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"jno.setup(lu_stack={k!r}): expected a positive integer number of systems per call.")
    if k != _LU_STACK:
        import jax

        _LU_STACK = k
        # The value is read while a vmap is TRACED, and JAX caches batched programs (a jitted solve's
        # batched jaxpr is reused across calls), so without this a change after the first vmapped
        # solve would be silently ignored -- verified: 8 set after 1 in one process still stacked 1.
        jax.clear_caches()


def _spsolve_batch(args, dims, *, tol, reorder):
    """B square systems through ``spsolve`` (cuSolver's sparse QR on GPU): ``lu_stack`` systems per call.

    The default, ONE system per ``lax.map`` step, costs exactly the memory of an unbatched solve on any
    machine. ``jno.setup(lu_stack=k)`` stacks ``k`` of them into one block-diagonal system instead --
    measured 1.07-2x faster -- but cuSolver's QR allocates for its fill-in, which depends on the sparsity
    pattern and on the device memory JAX leaves outside its pool, so how large ``k`` may be is the
    user's to choose for their machine and problem: too large fails loudly with a cuSolver allocation
    error, never with a wrong answer. (One 8 GB card ran a 3-D 20k-DOF Laplacian out of memory at 32
    stacked copies and not a 2-D operator of the same size.)

    cuSolver's QR takes one right-hand side, so even a SHARED matrix is factorised once per system here.
    To factor once and solve the whole batch, use a factor-once backend: ``jno.solve.lu(backend="host")``
    (see :func:`jno.utils.solver.linear.host_lu_solve`)."""
    import jax
    from jax.experimental.sparse import linalg as _splinalg

    size = _batch_size(args, dims)
    data, indices, indptr, b = (_front(a, d, size) for a, d in zip(args, dims))
    n = b.shape[1]
    chunk = max(1, min(size, _LU_STACK))

    def solve_stack(data, indices, indptr, b):
        if data.shape[0] == 1:
            return _splinalg.spsolve_p.bind(data[0], indices[0], indptr[0], b[0], tol=tol, reorder=reorder)[None]
        bd, bi, bp, _ = _block_diagonal(data, indices, indptr, (n, n))
        return _splinalg.spsolve_p.bind(bd, bi, bp, b.reshape(-1), tol=tol, reorder=reorder).reshape(-1, n)

    if chunk >= size:
        return solve_stack(data, indices, indptr, b), 0
    n_chunks = -(-size // chunk)
    pad = n_chunks * chunk - size  # padded with copies of a real system, never zeros (no singular pad)
    parts = [
        jnp.concatenate([a, jnp.broadcast_to(a[:1], (pad,) + a.shape[1:])]) if pad else a
        for a in (data, indices, indptr, b)
    ]
    stacked = tuple(a.reshape((n_chunks, chunk) + a.shape[1:]) for a in parts)
    out = jax.lax.map(lambda c: solve_stack(*c), stacked)
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
