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

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import spsolve

__all__ = ["sparse_lu_solve", "jacobi", "matrix_diagonal"]


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

    Limitations, all inherited from ``pure_callback``: no ``vmap`` batching rule, and the callback
    is forward-only, so this cannot appear inside a transformation that needs to differentiate
    *through* the callback itself (the ``custom_linear_solve`` firewall means it does not have to).
    The matrix is re-factored on every call -- correct for a one-shot solve, wasteful if you are
    solving repeatedly with a fixed operator, where a frozen preconditioner is the better tool.
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
        import numpy as _np
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _spla

        rhs = _np.asarray(rhs)
        idx = _np.asarray(indices)
        mat = _sp.csc_matrix((_np.asarray(data), (idx[:, 0], idx[:, 1])), shape=shape)
        return _np.asarray(_spla.splu(mat).solve(rhs, trans="T" if transpose else "N"), dtype=rhs.dtype)

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
