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


def sparse_lu_solve(A, b):
    """Differentiable sparse-direct solve ``A x = b`` via JAX's ``spsolve`` (cuSolver GPU / native CPU).

    ``A`` is a BCOO (or dense) operator. ``jit``-compatible and reverse-mode differentiable in both
    ``A``'s stored values and ``b`` -- no external dependency, no hand-written factorisation. Returns a
    solution containing ``NaN`` (rather than raising) for a singular ``A``; wrap it in a residual check
    if you need a hard failure. Drop-in for the ``(A, b) -> u`` solver contract of ``fem.solve``.
    """
    import jax.experimental.sparse as jsp

    b = jnp.asarray(b).reshape(-1)
    n = b.shape[0]
    if not (hasattr(A, "indices") and hasattr(A, "sum_duplicates")):
        A = jsp.BCOO.fromdense(jnp.asarray(A))
    A = A.sum_duplicates(nse=A.nse)  # static nse -> jit-safe; merges any repeated (i, j) entries
    idx = A.indices
    order = jnp.lexsort((idx[:, 1], idx[:, 0]))  # CSR ordering: by row, then column
    rows = idx[order, 0]
    cols = idx[order, 1].astype(jnp.int32)
    data = A.data[order]
    indptr = jnp.searchsorted(rows, jnp.arange(n + 1, dtype=rows.dtype)).astype(jnp.int32)
    return spsolve(data, cols, indptr, b)
