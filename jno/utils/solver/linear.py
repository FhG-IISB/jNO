"""Linear-solve building blocks: a differentiable sparse-**direct** default and a diagonal (Jacobi)
preconditioner for the iterative path.

Both are pure JAX with **no external dependency** -- ``sparse_lu_solve`` wraps
``jax.experimental.sparse.linalg.spsolve`` (cuSolver on a CUDA GPU, a native LU on CPU), which is
``jit``-compatible and reverse-mode differentiable in both the matrix entries and the right-hand side
(verified against finite differences on non-symmetric systems). It therefore needs no hand-written
factorisation loop and no ``custom_linear_solve`` wrapper. A direct factorisation is robust on the
indefinite saddle-point systems (Stokes / Boussinesq / Biot) where unpreconditioned Krylov stalls.

Users always remain free to extract ``fem.operator`` and pass their own ``solve_fn`` to ``fem.solve``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import spsolve

__all__ = ["sparse_lu_solve", "jacobi", "matrix_diagonal"]

#: Host SuperLU factorizations, keyed on the CONTENT of the operator that produced them.
#:
#: The transient march compiles into ONE ``lax.scan``, so :func:`host_lu_solve` runs its Python body
#: once, at trace time, and the ``pure_callback`` is what executes per step. A cache outside the
#: callback would therefore never see a repeat. Inside it, the constant-operator path -- where
#: ``solver_api`` forms ``M + theta*dt*A`` once and then hands the SAME matrix to every step -- goes
#: from N factorizations to one.
#:
#: Keyed on content rather than identity for the same reason the facet cache is: under a trace there
#: is no stable object to key on, only the values arriving at the callback. Hashing costs a pass over
#: ``data + indices`` (~16 MB at 1M nonzeros, ~1-2% of a factorization of that size), which a
#: workload that never repeats -- a Newton loop, whose tangent changes every step -- pays for nothing.
#: That is the deliberate trade: a small tax on the case that cannot benefit, against removing all
#: but one factorization from the case that can.
#:
#: Bounded at 2 because the win is a repeated operator, not a diverse population of them, and a
#: sparse factorization is the biggest object either side of the solve (fill-in): holding a stale one
#: costs host memory for nothing. Two covers an alternating pair (a coupled two-field march).
_FACTOR_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_FACTOR_CACHE_MAX = 2


def matrix_diagonal(A):
    """Diagonal of ``A`` as a 1-D array, for a BCOO or dense operator (cheap: ``O(nnz)`` for BCOO)."""
    if hasattr(A, "indices"):  # BCOO
        idx = A.indices
        on_diag = idx[:, 0] == idx[:, 1]
        return jnp.zeros(A.shape[0], A.data.dtype).at[idx[:, 0]].add(jnp.where(on_diag, A.data, 0.0))
    return jnp.diagonal(jnp.asarray(A))


def jacobi(A):
    """Diagonal (Jacobi) preconditioner ``M^{-1} x ~ x / diag(A)`` for an iterative solver.

    Zero / near-zero diagonals (e.g. the pressure block of a saddle-point system) are left **unscaled**,
    so it never produces ``inf`` / ``NaN``. Note Jacobi accelerates diagonally-dominant (elliptic)
    systems -- heat, elasticity -- but does **not** rescue indefinite saddle systems; use a direct
    solver (``sparse_lu_solve``) there.
    """
    d = matrix_diagonal(A)
    safe = jnp.where(jnp.abs(d) > 1e-30, d, 1.0)
    inv = 1.0 / safe
    return lambda x: inv * x


@jax.jit
def _sparse_lu_solve(A, b):
    """The triplets-to-CSR conversion and the ``spsolve`` itself, as ONE compiled program.

    Left uncompiled, this ran as ~43 separately compiled single-primitive programs -- the lexsort, the
    gathers, the ``searchsorted``, each its own XLA module. On a Morley (C1) problem, whose 4th-order
    system needs the sparse-direct path, that was **1520 ms of the 1751 ms first solve**: 87% of the
    solve spent compiling fragments of a conversion. It is not a Morley problem -- every
    ``jno.solve.lu()``, the 1-D default and every non-nodal solve goes through here.

    Module level and array-only, so ``jax.jit`` keys on shapes and hits across calls and across
    problems of the same size; under an enclosing trace it simply inlines.
    """
    b = jnp.asarray(b).reshape(-1)
    n = b.shape[0]
    A = A.sum_duplicates(nse=A.nse)  # static nse -> jit-safe; merges any repeated (i, j) entries
    idx = A.indices
    order = jnp.lexsort((idx[:, 1], idx[:, 0]))  # CSR ordering: by row, then column
    rows = idx[order, 0]
    cols = idx[order, 1].astype(jnp.int32)
    data = A.data[order]
    indptr = jnp.searchsorted(rows, jnp.arange(n + 1, dtype=rows.dtype)).astype(jnp.int32)
    return spsolve(data, cols, indptr, b)


def sparse_lu_solve(A, b):
    """Differentiable sparse-direct solve ``A x = b`` via JAX's ``spsolve`` (cuSolver GPU / native CPU).

    ``A`` is a BCOO (or dense) operator. ``jit``-compatible and reverse-mode differentiable in both
    ``A``'s stored values and ``b`` -- no external dependency, no hand-written factorisation. Returns a
    solution containing ``NaN`` (rather than raising) for a singular ``A``; wrap it in a residual check
    if you need a hard failure. Drop-in for the ``(A, b) -> u`` solver contract of ``fem.solve``.

    The densification of a non-sparse ``A`` stays out here: it needs a concrete ``nse``, so it cannot
    live inside the compiled body (see :func:`_sparse_lu_solve`).
    """
    import jax.experimental.sparse as jsp

    if not (hasattr(A, "indices") and hasattr(A, "sum_duplicates")):
        A = jsp.BCOO.fromdense(jnp.asarray(A))
    return _sparse_lu_solve(A, b)


def host_lu_solve(A, b):
    """Sparse-direct solve factored on the HOST (SuperLU), driven from the device.

    Same contract as :func:`sparse_lu_solve` -- ``(A, b) -> x``, jit-compatible, reverse-mode
    differentiable -- but the factorisation runs in host memory instead of on the GPU. That is the
    whole point: cuSolver's sparse LU is the ceiling on jNO's two hardest cases (an H(curl) eddy
    problem stops near 20k complex DOFs, and a Taylor-Hood Stokes system returns "Singular matrix"
    at a mesh the CPU factors without complaint), while host SuperLU was measured reaching 57,746
    DOFs on the same problem -- 3.1x.

    **Why moving THIS across PCIe is affordable when moving a Krylov iteration is not.** A direct
    solve factorises once: the operator crosses once (nnz x 12 bytes -- ~12 MB, ~0.5 ms, against a
    multi-second factorisation) and thereafter only ``b`` and ``x`` move. Streaming a Krylov vector
    every iteration would instead pay the full PCIe-vs-device bandwidth penalty (~25 GB/s against
    448 GB/s on this card, ~18x) on every one of them. Offload what is touched once per solve;
    never what is touched per iteration.

    Differentiability is preserved by wrapping the host solve in ``lax.custom_linear_solve``, which
    needs a transpose solve -- supplied by the same factorisation via SuperLU's ``trans="T"``. So
    ``jax.grad`` flows to the operator entries and the right-hand side exactly as for the GPU path.

    **The factorization is reused when the operator repeats** (see :data:`_FACTOR_CACHE`). A
    constant-operator transient forms ``M + theta*dt*A`` once and solves against it at every step, so
    this turns N factorizations into one -- measured on a 200-step heat march, and the reuse extends
    to the TRANSPOSE solve, so the adjoint pass costs no factorization at all. What it does NOT help
    is a Newton loop: the tangent's VALUES change every iteration, so every call legitimately misses,
    and reusing only the symbolic analysis (the part that does not change) is not something SuperLU
    exposes through scipy -- that is the gap a phase-separated solver like cuDSS would close.

    Limitations, all inherited from ``pure_callback``: no ``vmap`` batching rule, and the callback
    is forward-only, so this cannot appear inside a transformation that needs to differentiate
    *through* the callback itself (the ``custom_linear_solve`` firewall means it does not have to).
    """
    import jax
    import jax.experimental.sparse as jsp

    if not (hasattr(A, "indices") and hasattr(A, "sum_duplicates")):
        A = jsp.BCOO.fromdense(jnp.asarray(A))
    # deliberately NOT A.sum_duplicates(): it needs a concrete ``nse`` and so cannot run inside a
    # jitted body. scipy sums duplicate (i, j) entries when building the matrix anyway.
    n = int(A.shape[0])
    shape = tuple(int(s) for s in A.shape)

    def _host_solve(data, indices, rhs, transpose):
        import hashlib

        import numpy as _np
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _spla

        rhs = _np.asarray(rhs)
        dat = _np.ascontiguousarray(data)
        idx = _np.ascontiguousarray(_np.asarray(indices))

        # One hasher over both arrays: together they ARE the matrix, so this identifies the
        # factorization exactly -- a changed coefficient misses and re-factors, which is the whole
        # correctness requirement. `transpose` is deliberately NOT in the key: one factorization
        # serves both directions via SuperLU's trans="T", so the adjoint reuses the forward's.
        h = hashlib.blake2b(digest_size=16)
        h.update(dat.view(_np.uint8))
        h.update(idx.view(_np.uint8))
        key = (h.digest(), shape, dat.dtype.str)

        lu = _FACTOR_CACHE.get(key)
        if lu is None:
            mat = _sp.csc_matrix((dat, (idx[:, 0], idx[:, 1])), shape=shape)
            lu = _spla.splu(mat)
            _FACTOR_CACHE[key] = lu
            if len(_FACTOR_CACHE) > _FACTOR_CACHE_MAX:
                _FACTOR_CACHE.popitem(last=False)
        else:
            _FACTOR_CACHE.move_to_end(key)
        return _np.asarray(lu.solve(rhs, trans="T" if transpose else "N"), dtype=rhs.dtype)

    def _call(rhs, transpose):
        return jax.pure_callback(
            lambda d, i, r: _host_solve(d, i, r, transpose),
            jax.ShapeDtypeStruct((n,), rhs.dtype),
            A.data,
            A.indices,
            rhs,
        )

    return jax.lax.custom_linear_solve(
        lambda x: A @ x,
        jnp.asarray(b).reshape(-1),
        lambda _matvec, rhs: _call(rhs, False),
        transpose_solve=lambda _matvec, rhs: _call(rhs, True),
        symmetric=False,
    )
