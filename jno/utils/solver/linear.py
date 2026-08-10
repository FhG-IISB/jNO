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

#: MKL PARDISO solvers, keyed on the operator's SPARSITY -- the same idea as :data:`_CUDSS_CACHE`,
#: because PARDISO splits into the same phases: 11 (symbolic analysis, sparsity only), 22 (numeric
#: factorization, values), 33 (solve). Driving those phases directly is the entire point: pypardiso's
#: own reuse is keyed on the WHOLE matrix, so a Newton step whose values changed would redo the
#: analysis, exactly the gap :func:`host_lu_solve` has.
#:
#: Measured on this machine (20 cores, MKL chose 14 threads) against single-threaded scipy SuperLU:
#: lap3d 50^3 (n=125,000) factorization **298 ms vs 65,212 ms**, and a Newton re-factorization
#: reusing the analysis is **296 ms -- 220x**. For comparison cuDSS on the same problem factors in
#: 576 ms and re-factors at 115x, so PARDISO is the faster FACTORIZATION; cuDSS keeps the faster
#: repeated SOLVE (3.5 ms against 40 ms), which is what makes the two complementary rather than
#: redundant.
#:
#: Unlike cuDSS there is **no transpose entry**: PARDISO solves ``A^T x = b`` from the SAME
#: factorization via ``iparm[12] = 2`` (verified to 2e-16), so the adjoint pass costs a solve rather
#: than a whole second factorization.
_PARDISO_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_PARDISO_CACHE_MAX = 4


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


def _cudss_sym_perms(rows, cols):
    """Orderings of the COO entries by ``(row, col)`` and by ``(col, row)``, plus whether they MATCH.

    If they match, entry ``k`` of the first ordering and entry ``k`` of the second are the transposed
    pair ``(i,j)`` / ``(j,i)`` -- so comparing the two permuted value arrays is the whole symmetry
    test. Both permutations are properties of the SPARSITY, so they are computed once per plan and
    cached; re-testing a Newton step's new values then costs a gather and a compare (measured 1.7 ms
    at 438k nonzeros, against a 125 ms factorization) instead of two sorts (18 ms).
    """
    import numpy as _np

    order = _np.lexsort((cols, rows))
    orderT = _np.lexsort((rows, cols))
    pat = _np.array_equal(rows[order], cols[orderT]) and _np.array_equal(cols[order], rows[orderT])
    return order, orderT, pat


def _cudss_matrix_kind(values, order, orderT, pat) -> str:
    """``"symmetric"`` / ``"hermitian"`` / ``"general"`` -- which factorization cuDSS may use.

    Worth detecting: ``SYMMETRIC`` (LDL^T) measured 1.41x faster than general LU with **1.38x less
    peak device memory** on lap3d 40^3, and memory is what bounds a sparse direct solve in 3-D.

    Tested by EXACT (bitwise) equality on purpose. A symmetric factorization reads one triangle, so
    accepting a matrix that is symmetric only to ~1e-15 would quietly factor ``(A+Aᵀ)/2`` instead of
    ``A`` -- an error of order ``cond(A) * 1e-15``, negligible on a well-conditioned system and ~1e-4
    at cond 1e11. Falling back to general LU costs 1.41x; a quietly different answer on an
    ill-conditioned system is not a trade jNO makes.

    **SPD is never inferred.** It requires definiteness, which no cheap test establishes, and guessing
    it wrong returns NaN (measured on an indefinite saddle). ``SYMMETRIC`` is valid for ANY symmetric
    matrix -- indefinite Stokes/Biot saddles included -- and captures 1.41x of SPD's 1.74x while
    saving the same memory, so the safe inference gets essentially all of the win.
    """
    import numpy as _np

    if not pat:
        return "general"
    lower, upper = values[order], values[orderT]
    if _np.array_equal(lower, upper):
        return "symmetric"
    if _np.iscomplexobj(values) and _np.array_equal(lower, _np.conj(upper)):
        return "hermitian"
    return "general"


def _cudss_check_factorization(solver, Ag, bg, cp, shape):
    """Raise if cuDSS silently factored a singular operator.

    cuDSS does NOT report singularity through an exception, and -- unlike ``spsolve``, which returns
    ``NaN`` -- it returns a FINITE, plausible-looking vector. Measured on ``diag(1, 2, 0, 4)``: it
    returns ``1e+13`` in the null component with relative residual 1.0 and ``info == 0``. A wrong
    number that looks right is the worst failure mode jNO can have, so this path checks.

    ``npivots`` is the signal: it counts pivots cuDSS had to REPLACE, so a nonzero value means the
    factorization is of a perturbed matrix, not the one asked for. Measured 0 on lap3d 30^3, on an
    indefinite Stokes-shaped saddle, and on a cond-1e11 system -- i.e. it does not fire on
    merely-hard problems -- and 1 on both a zero-pivot and a rank-deficient matrix.

    The residual is only computed WHEN ``npivots`` fires, so the common path pays a host-side
    attribute read and no SpMV. A perturbed pivot that still yields a good answer passes.
    """
    try:
        npivots = int(solver.factorization_info.npivots)
    except Exception:  # an older cuDSS that does not expose it -- do not fail the solve over this
        return
    if npivots == 0:
        return
    x = solver.solve()
    denom = cp.linalg.norm(bg)
    rel = float(cp.linalg.norm(Ag @ x - bg) / cp.where(denom > 0, denom, 1.0))
    if rel > 1e-6:
        raise RuntimeError(
            f"cuDSS factorized a SINGULAR operator ({shape[0]}x{shape[1]}): it replaced {npivots} "
            f"pivot(s) and the solution has relative residual {rel:.2e}. cuDSS reports this through "
            f"neither an exception nor a NaN, so jNO checks it -- the returned vector would have been "
            f"finite and wrong. Check for an unconstrained mode (a pure-Neumann problem with no gauge "
            f"term, a floating region, or a saddle system whose constraint block has an empty row)."
        )


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

    _KIND = {
        "symmetric": nsa.DirectSolverMatrixType.SYMMETRIC,
        "hermitian": nsa.DirectSolverMatrixType.HERMITIAN,
        "general": nsa.DirectSolverMatrixType.GENERAL,
    }

    def _to_device_rhs(r):
        # cuDSS requires a COL-MAJOR multi-RHS block and raises on a C-ordered one.
        return cp.asfortranarray(cp.asarray(r)) if r.ndim > 1 else cp.asarray(r)

    data = _np.ascontiguousarray(data)
    idx = _np.ascontiguousarray(_np.asarray(indices))
    rhs = _np.ascontiguousarray(rhs)
    rows, cols = (idx[:, 1], idx[:, 0]) if transpose else (idx[:, 0], idx[:, 1])
    nrhs = 1 if rhs.ndim == 1 else int(rhs.shape[1])

    # nrhs is in the key because a cuDSS solver is planned against a specific operand shape; a Newton
    # loop is always 1 and an eigen block always m, so this does not fragment the cache in practice.
    skey = (_h(rows, cols), shape, data.dtype.str, bool(transpose), nrhs)
    dhash = _h(data)  # the VALUES: what a numeric factorization depends on
    entry = _CUDSS_CACHE.get(skey)

    if entry is not None and entry["dhash"] != dhash:
        # The values changed. They can change the matrix TYPE too (a Newton tangent that starts
        # symmetric need not stay symmetric), and the type is baked into the plan -- so re-detect
        # first and throw the plan away if it no longer applies. Only reached when the values
        # actually differ, so a repeat solve still costs no detection.
        if _cudss_matrix_kind(data, entry["order"], entry["orderT"], entry["pat"]) != entry["kind"]:
            _CUDSS_CACHE.pop(skey)["solver"].free()
            entry = None

    if entry is None:
        order, orderT, pat = _cudss_sym_perms(rows, cols)
        kind = _cudss_matrix_kind(data, order, orderT, pat)
        mtype = _KIND[kind]
        csr = _sp.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
        csr.sum_duplicates()
        Ag = csp.csr_matrix(csr)
        bg = _to_device_rhs(rhs)
        opts = {"sparse_system_type": mtype, "sparse_system_view": nsa.DirectSolverMatrixViewType.FULL}
        solver = nsa.DirectSolver(Ag, bg, options=opts)
        solver.plan()
        solver.factorize()
        try:
            _cudss_check_factorization(solver, Ag, bg, cp, shape)
        except Exception:
            solver.free()  # never cached, so nothing else will ever free it
            raise
        _CUDSS_CACHE[skey] = entry = {
            "solver": solver,
            "Ag": Ag,
            "bg": bg,
            "dhash": dhash,
            "rows": rows,
            "cols": cols,
            "order": order,
            "orderT": orderT,
            "pat": pat,
            "kind": kind,
        }
        while len(_CUDSS_CACHE) > _CUDSS_CACHE_MAX:
            _, old = _CUDSS_CACHE.popitem(last=False)
            old["solver"].free()  # WITHOUT this an evicted factorization leaks device memory
    else:
        _CUDSS_CACHE.move_to_end(skey)
        if entry["dhash"] != dhash:
            # same sparsity AND same type, new values -> refresh the CSR values IN PLACE and
            # re-factorize; the plan (the expensive half) survives. Getting this test wrong would
            # silently reuse a STALE factorization, so it hashes the values rather than trusting
            # the caller.
            csr = _sp.coo_matrix((data, (entry["rows"], entry["cols"])), shape=shape).tocsr()
            csr.sum_duplicates()
            entry["Ag"].data[:] = cp.asarray(csr.data)
            entry["solver"].factorize()
            _cudss_check_factorization(entry["solver"], entry["Ag"], entry["bg"], cp, shape)
            entry["dhash"] = dhash

    entry["bg"][...] = _to_device_rhs(rhs)
    entry["solver"].reset_operands(b=entry["bg"])
    out = cp.asnumpy(entry["solver"].solve())
    return _np.asarray(out.reshape(rhs.shape), dtype=rhs.dtype)


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

    Two things happen automatically. A **block right-hand side** (``b`` of shape ``(n, k)``) is
    solved in one call -- 1.9x at k=4 to 5.5x at k=32 over the same factorization solved column by
    column. And the operator's **matrix type is detected**: an exactly symmetric operator is factored
    as ``SYMMETRIC`` (LDL^T) rather than general LU, measured 1.41x faster with 1.38x less peak device
    memory on lap3d 40^3. Symmetry is tested bitwise and SPD is never inferred -- see
    ``_cudss_matrix_kind`` for why both of those are deliberate.

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
    b = jnp.asarray(b)
    # A BLOCK right-hand side is solved in ONE cuDSS call rather than column by column: measured
    # 1.9x at 4 columns rising to 5.5x at 32 against the same factorization solved sequentially.
    # That is the shift-invert eigensolver's inner apply (see eigen._apply_C), which needs the k+guard
    # columns of a subspace-iteration block every sweep.
    bshape = (n,) if b.ndim == 1 else (n, int(b.shape[1]))

    def _call(rhs, transpose):
        return jax.pure_callback(
            lambda d, i, r: _cudss_host_solve(d, i, r, shape, transpose),
            jax.ShapeDtypeStruct(bshape, rhs.dtype),
            A.data,
            A.indices,
            rhs,
        )

    return jax.lax.custom_linear_solve(
        lambda x: A @ x,
        b.reshape(bshape),
        lambda _matvec, rhs: _call(rhs, False),
        transpose_solve=lambda _matvec, rhs: _call(rhs, True),
    )


def _pardiso_available() -> bool:
    """True when pypardiso imports AND exposes the private phase hooks this backend drives."""
    try:
        from pypardiso import PyPardisoSolver
    except Exception:
        return False
    return all(hasattr(PyPardisoSolver, m) for m in ("_check_A", "_call_pardiso", "set_phase", "set_iparm"))


#: real/complex x general/symmetric/hermitian -> PARDISO ``mtype``. The symmetric entries are the
#: INDEFINITE ones (-2, -4) rather than the definite ones (2, 4) on purpose: they cost the same, they
#: cover indefinite saddles, and claiming definiteness that does not hold is a way to be wrong.
_PARDISO_MTYPE = {
    (False, "general"): 11,
    (False, "symmetric"): -2,
    (False, "hermitian"): -2,
    (True, "general"): 13,
    (True, "symmetric"): 6,
    (True, "hermitian"): -4,
}


def _pardiso_upper_with_diag(A, sp, np):
    """Upper triangle with the diagonal stored EXPLICITLY -- what PARDISO's symmetric modes require.

    The explicit diagonal is not a detail. A saddle system's constraint block sits entirely in the
    LOWER triangle, so its rows would come back empty and PARDISO would reject the matrix as
    structurally singular -- which is precisely the class of system (Stokes, Biot) this backend is
    most wanted for. Storing a zero diagonal keeps those rows present.
    """
    C = A.tocoo()
    keep = C.col >= C.row
    d = np.arange(A.shape[0])
    rows = np.concatenate([C.row[keep], d])
    cols = np.concatenate([C.col[keep], d])
    vals = np.concatenate([C.data[keep], np.zeros(A.shape[0], dtype=C.data.dtype)])
    return sp.coo_matrix((vals, (rows, cols)), shape=A.shape).tocsr()


def _pardiso_host_solve(data, indices, rhs, shape, transpose):
    """One MKL PARDISO solve, with the phase-separated cache described on :data:`_PARDISO_CACHE`.

    Runs inside a ``pure_callback``, so it sees concrete numpy arrays.

    Driving the phases means reaching for ``PyPardisoSolver._check_A`` / ``._call_pardiso``, which are
    private. That is deliberate and unavoidable -- the public ``solve``/``factorize`` decide phases
    from their own whole-matrix hash, which is the behaviour this cache exists to replace -- but it
    does mean the backend is coupled to pypardiso internals, so :func:`_pardiso_available` checks for
    them and the import error says so.
    """
    import hashlib

    import numpy as _np
    import scipy.sparse as _sp
    from pypardiso import PyPardisoSolver

    def _h(*arrs):
        d = hashlib.blake2b(digest_size=16)
        for a in arrs:
            a = _np.ascontiguousarray(a)
            d.update(_np.asarray(a.shape, dtype=_np.int64).view(_np.uint8))
            d.update(a.view(_np.uint8))
        return d.digest()

    data = _np.ascontiguousarray(data)
    idx = _np.ascontiguousarray(_np.asarray(indices))
    rhs2 = _np.asarray(rhs).reshape(rhs.shape[0], -1)
    rows, cols = idx[:, 0], idx[:, 1]

    order, orderT, pat = _cudss_sym_perms(rows, cols)
    kind = _cudss_matrix_kind(data, order, orderT, pat)
    mtype = _PARDISO_MTYPE[(bool(_np.iscomplexobj(data)), kind)]

    # NO transpose in the key: one factorization serves both directions (iparm[12]).
    skey = (_h(rows, cols), shape, data.dtype.str, mtype)
    dhash = _h(data)
    entry = _PARDISO_CACHE.get(skey)

    def _build(values):
        A = _sp.coo_matrix((values, (rows, cols)), shape=shape).tocsr()
        A.sum_duplicates()
        return _pardiso_upper_with_diag(A, _sp, _np) if mtype in (-2, 6, -4, 2, 4) else A

    b = _np.asfortranarray(rhs2.astype(data.dtype if _np.iscomplexobj(data) else rhs2.dtype))

    if entry is None:
        A = _build(data)
        solver = PyPardisoSolver(mtype=mtype)
        solver._check_A(A)
        solver.set_phase(11)  # symbolic analysis: the half that survives a value change
        solver._call_pardiso(A, b)
        solver.set_phase(22)
        solver._call_pardiso(A, b)
        _PARDISO_CACHE[skey] = entry = {"solver": solver, "A": A, "dhash": dhash}
        while len(_PARDISO_CACHE) > _PARDISO_CACHE_MAX:
            _, old = _PARDISO_CACHE.popitem(last=False)
            old["solver"].free_memory()  # PARDISO holds its factors in MKL-owned memory
    else:
        _PARDISO_CACHE.move_to_end(skey)
        if entry["dhash"] != dhash:
            entry["A"] = _build(data)  # same sparsity -> phase 22 only, analysis reused
            entry["solver"].set_phase(22)
            entry["solver"]._call_pardiso(entry["A"], b)
            entry["dhash"] = dhash

    solver, A = entry["solver"], entry["A"]
    _pardiso_check_factorization(solver, A, b, _np, shape)
    solver.set_iparm(12, 2 if transpose else 0)  # A^T x = b from the SAME factorization
    solver.set_phase(33)
    try:
        x = solver._call_pardiso(A, b)
    finally:
        solver.set_iparm(12, 0)
    return _np.asarray(x.reshape(rhs.shape), dtype=rhs.dtype)


def _pardiso_check_factorization(solver, A, b, np, shape):
    """Raise if PARDISO silently factored a singular operator.

    PARDISO behaves exactly like cuDSS here and exactly unlike ``spsolve``: measured on a rank-3
    4x4 matrix it returned ``[-1e13, 5e12, 0.2, 0.4]`` -- finite, plausible, residual 0.5, no
    exception. ``iparm[13]`` (perturbed pivots) is the tell, the direct analogue of cuDSS's
    ``npivots``; it read 1 on that matrix. As there, the residual is only computed once the pivot
    count fires, so a healthy factorization pays a single integer read.

    pypardiso separately rejects a STRUCTURALLY singular matrix (an empty row or column) up front in
    ``_check_A`` with its own clear message; this covers the numerically singular case it cannot see.
    """
    try:
        perturbed = int(solver.get_iparm(14))
    except Exception:  # a pypardiso that does not expose it -- do not fail a working solve over this
        return
    if perturbed == 0:
        return
    solver.set_phase(33)
    x = solver._call_pardiso(A, b)
    denom = np.linalg.norm(b)
    rel = float(np.linalg.norm(A @ x - b) / (denom if denom > 0 else 1.0))
    if rel > 1e-6:
        raise RuntimeError(
            f"MKL PARDISO factorized a SINGULAR operator ({shape[0]}x{shape[1]}): it perturbed "
            f"{perturbed} pivot(s) and the solution has relative residual {rel:.2e}. PARDISO reports "
            f"this through neither an exception nor a NaN, so jNO checks it -- the returned vector "
            f"would have been finite and wrong. Check for an unconstrained mode (a pure-Neumann "
            f"problem with no gauge term, a floating region, or a constraint block with an empty row)."
        )


def pardiso_lu_solve(A, b):
    """Sparse-direct solve factored on the CPU by **Intel MKL PARDISO**, driven from the device.

    Same contract as :func:`sparse_lu_solve` -- ``(A, b) -> x``, jit-compatible, reverse-mode
    differentiable -- and the same structure as :func:`cudss_lu_solve`, including a cache keyed on the
    SPARSITY so a Newton loop reuses the symbolic analysis (see :data:`_PARDISO_CACHE`).

    **Fastest factorization of the three backends, measured.** lap3d 50^3 (n=125,000): factorization
    298 ms against cuDSS's 576 ms and host SuperLU's 65,212 ms; the Newton re-factorization is 296 ms,
    i.e. **220x** SuperLU where cuDSS reaches 115x. It runs on the CPU, so it is also the answer when
    a factorization does not fit in device memory. cuDSS keeps the faster repeated SOLVE (3.5 ms vs
    40 ms), which is what a shift-invert eigensolve or a constant-operator transient is made of -- so
    pick by which phase your problem repeats.

    An exactly symmetric operator is factored as symmetric-indefinite (LDL^T), which additionally
    needs only the upper triangle. Measured 1.9x on the factorization at lap3d 50^3 and 13x on a
    saddle. Symmetry is detected exactly, by the same :func:`_cudss_matrix_kind` the cuDSS path uses.

    A **block right-hand side is not a win here** and is simply passed through: measured 0.32x at 4
    columns and 1.12x at 32 against the same factorization solved column by column, PARDISO's
    single-RHS solve already being threaded. Only cuDSS advertises ``multi_rhs``.

    Requires ``pypardiso`` (which bundles MKL); raises a clear ``ImportError`` otherwise. Limitations
    inherited from ``pure_callback``: no ``vmap`` batching rule, and the callback is forward-only.
    """
    import jax
    import jax.experimental.sparse as jsp

    if not _pardiso_available():
        raise ImportError(
            "jno.solve.lu(backend='pardiso') needs MKL PARDISO. Install it with "
            "`pip install jax-numerical-operators[pardiso]` (or `pip install pypardiso`, which "
            "bundles the MKL runtime; x86-64 only). If it IS installed, this jNO build drives "
            "PARDISO's phases directly and needs PyPardisoSolver._check_A / ._call_pardiso, which "
            "this pypardiso version does not expose -- pin pypardiso>=0.4.4. Use backend='host' for "
            "the dependency-free CPU path."
        )
    if not (hasattr(A, "indices") and hasattr(A, "sum_duplicates")):
        A = jsp.BCOO.fromdense(jnp.asarray(A))
    n = int(A.shape[0])
    shape = tuple(int(s) for s in A.shape)
    b = jnp.asarray(b)
    bshape = (n,) if b.ndim == 1 else (n, int(b.shape[1]))

    def _call(rhs, transpose):
        return jax.pure_callback(
            lambda d, i, r: _pardiso_host_solve(d, i, r, shape, transpose),
            jax.ShapeDtypeStruct(bshape, rhs.dtype),
            A.data,
            A.indices,
            rhs,
        )

    return jax.lax.custom_linear_solve(
        lambda x: A @ x,
        b.reshape(bshape),
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
