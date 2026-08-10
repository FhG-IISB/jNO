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

#: cuDSS solvers, keyed on the operator's **SPARSITY** rather than its full content.
#:
#: This is the difference that makes cuDSS worth having, and it is why this cache is separate from
#: :data:`_FACTOR_CACHE`. A cuDSS solver owns two things: a *symbolic plan* (reordering + symbolic
#: factorization), which depends only on the sparsity pattern, and a *numeric factorization*, which
#: depends on the values. A Newton loop changes the values every iteration and holds the pattern
#: fixed -- so keying on the pattern lets the expensive plan survive and only the numeric phase
#: repeat. Measured on lap3d 40^3 (n=64,000, RTX 3070, fp64): a full solve is 564 ms, a subsequent
#: Newton step reusing the plan is **192 ms**, against scipy SuperLU's 12,395 ms -- 64.7x per step,
#: with ONE plan serving six solves. This is precisely the case :func:`host_lu_solve` documents that
#: it cannot help.
#:
#: Each entry additionally remembers the value hash, so an operator that repeats EXACTLY (a
#: constant-operator transient) skips the numeric factorization too and goes straight to the solve.
#:
#: Bounded small and evicted with ``solver.free()``: an entry pins a GPU copy of the matrix *and* its
#: factors, which are the largest objects in the solve (fill-in measured 69x-218x the operator's nnz
#: at lap3d 20^3-40^3), so an unbounded cache would exhaust device memory. Verified: 10 distinct
#: matrices leave 4 live and 6 evicted, and clearing returns the pool to its baseline exactly.
_CUDSS_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_CUDSS_CACHE_MAX = 4


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


def _cudss_available() -> bool:
    """True when the optional cuDSS stack (nvmath-python + cuDSS + a GPU array library) imports."""
    try:
        import cupy  # noqa: F401
        import nvmath.sparse.advanced  # noqa: F401
    except Exception:
        return False
    return True


def _cudss_host_solve(data, indices, rhs, shape, transpose):
    """One cuDSS solve, with the plan/factorization cache described on :data:`_CUDSS_CACHE`.

    Runs inside a ``pure_callback``, so it sees concrete numpy arrays. The transpose direction is
    obtained by swapping the COO row/column arrays -- cuDSS exposes no ``trans`` flag on ``solve``
    (unlike SuperLU), so the adjoint gets its own plan and factorization under its own cache key.
    """
    import hashlib

    import cupy as cp
    import cupyx.scipy.sparse as csp
    import numpy as _np
    import nvmath.sparse.advanced as nsa
    import scipy.sparse as _sp

    def _h(*arrs):
        d = hashlib.blake2b(digest_size=16)
        for a in arrs:
            a = _np.ascontiguousarray(a)
            d.update(_np.asarray(a.shape, dtype=_np.int64).view(_np.uint8))
            d.update(a.view(_np.uint8))
        return d.digest()

    data = _np.ascontiguousarray(data)
    idx = _np.ascontiguousarray(_np.asarray(indices))
    rhs = _np.ascontiguousarray(rhs)
    rows, cols = (idx[:, 1], idx[:, 0]) if transpose else (idx[:, 0], idx[:, 1])

    skey = (_h(rows, cols), shape, data.dtype.str, bool(transpose))  # the SPARSITY: what a plan depends on
    dhash = _h(data)  # the VALUES: what a numeric factorization depends on
    entry = _CUDSS_CACHE.get(skey)

    if entry is None:
        csr = _sp.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
        csr.sum_duplicates()
        Ag = csp.csr_matrix(csr)
        bg = cp.asarray(rhs.reshape(-1))
        solver = nsa.DirectSolver(Ag, bg)
        solver.plan()
        solver.factorize()
        _CUDSS_CACHE[skey] = entry = {"solver": solver, "Ag": Ag, "bg": bg, "dhash": dhash, "rows": rows, "cols": cols}
        while len(_CUDSS_CACHE) > _CUDSS_CACHE_MAX:
            _, old = _CUDSS_CACHE.popitem(last=False)
            old["solver"].free()  # WITHOUT this an evicted factorization leaks device memory
    else:
        _CUDSS_CACHE.move_to_end(skey)
        if entry["dhash"] != dhash:
            # same sparsity, new values -> refresh the CSR values IN PLACE and re-factorize; the plan
            # (the expensive half) survives. Getting this test wrong would silently reuse a STALE
            # factorization, so it hashes the values rather than trusting the caller.
            csr = _sp.coo_matrix((data, (entry["rows"], entry["cols"])), shape=shape).tocsr()
            csr.sum_duplicates()
            entry["Ag"].data[:] = cp.asarray(csr.data)
            entry["solver"].factorize()
            entry["dhash"] = dhash

    entry["bg"][:] = cp.asarray(rhs.reshape(-1))
    entry["solver"].reset_operands(b=entry["bg"])
    return _np.asarray(cp.asnumpy(entry["solver"].solve()).reshape(-1), dtype=rhs.dtype)


def cudss_lu_solve(A, b):
    """Sparse-direct solve factored on the GPU by **NVIDIA cuDSS**, driven from the device.

    Same contract as :func:`sparse_lu_solve` -- ``(A, b) -> x``, jit-compatible, reverse-mode
    differentiable -- and the same ``pure_callback`` + ``lax.custom_linear_solve`` structure as
    :func:`host_lu_solve`. What differs is the cache: cuDSS separates the symbolic plan from the
    numeric factorization, so :data:`_CUDSS_CACHE` keys on the SPARSITY and a Newton loop re-uses the
    plan (measured 64.7x per step against scipy at n=64,000; see that cache's note for the full
    figures and for why cuSolver is not the comparison).

    Measured against host SuperLU on an RTX 3070 (fp64 at 1/64 rate -- i.e. the *unfavourable* card):
    factorization 3.4 ms vs 79.9 ms on a Taylor-Hood Stokes saddle, 576 ms vs 64,856 ms on lap3d
    50^3. It also factors systems cuSolver refuses outright (that same Stokes saddle returns
    "Singular matrix" there), and its residuals are consistently smaller.

    **Not a substitute for a preconditioner.** Fill-in still governs 3-D: measured 69x -> 141x -> 218x
    growth in nonzeros at lap3d 20^3/30^3/40^3. cuDSS moves the ceiling and makes device memory the
    binding constraint; it does not change the asymptotics.

    Requires the optional stack (``nvmath-python``, ``cudss``, ``cupy``) and a GPU; raises a clear
    ``ImportError`` otherwise. Limitations inherited from ``pure_callback``: no ``vmap`` batching
    rule, and the callback is forward-only (the ``custom_linear_solve`` firewall means it need not be
    differentiable itself).
    """
    import jax
    import jax.experimental.sparse as jsp

    if not _cudss_available():
        raise ImportError(
            "jno.solve.lu(backend='cudss') needs the optional cuDSS stack. Install it with "
            "`pip install nvmath-python[cu12] cupy-cuda12x` (NVIDIA ships cuDSS as a wheel; no source "
            "build and no MPI is required), and note that it needs a CUDA GPU at run time. Use "
            "backend='host' for the CPU SuperLU path instead."
        )
    if not (hasattr(A, "indices") and hasattr(A, "sum_duplicates")):
        A = jsp.BCOO.fromdense(jnp.asarray(A))
    n = int(A.shape[0])
    shape = tuple(int(s) for s in A.shape)

    def _call(rhs, transpose):
        return jax.pure_callback(
            lambda d, i, r: _cudss_host_solve(d, i, r, shape, transpose),
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
    )


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
    exposes through scipy. That gap is now closed by :func:`cudss_lu_solve` (``lu(backend="cudss")``),
    which keys its cache on the sparsity and re-uses the plan across a Newton loop -- measured 64.7x
    per step against this function at n=64,000.

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
