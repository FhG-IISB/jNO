"""The built-in differentiable linear-solve defaults: sparse-direct ``sparse_lu_solve`` (JAX
``spsolve``, no external dependency) and the diagonal ``jacobi`` preconditioner.

Pins: correctness vs a dense solve, ``jit``-compatibility, reverse-mode differentiability in BOTH the
right-hand side and the matrix entries (vs finite differences, on a NON-symmetric system so a transpose
bug cannot hide), robustness on an indefinite saddle-point matrix (where Jacobi's ``1/diag`` is
degenerate), and that the ``fem.solve`` steady-linear default uses it end-to-end.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsp  # noqa: E402
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.linear import jacobi, matrix_diagonal, sparse_lu_solve


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _nonsym(n, seed=0):
    rng = np.random.default_rng(seed)
    import scipy.sparse as sp

    A = (sp.random(n, n, density=0.3, rng=rng) + sp.eye(n) * n).tocsr()
    return jsp.BCOO.fromdense(jnp.asarray(A.toarray())), jnp.asarray(rng.standard_normal(n)), np.asarray(A.toarray())


def test_sparse_lu_matches_dense_and_jits():
    A, b, Ad = _nonsym(20)
    x = sparse_lu_solve(A, b)
    assert np.allclose(np.asarray(x), np.asarray(jnp.linalg.solve(jnp.asarray(Ad), b)), atol=1e-10)
    xj = jax.jit(sparse_lu_solve)(A, b)  # jit-compatible (no custom loop, no host callback)
    assert np.allclose(np.asarray(xj), np.asarray(x), atol=1e-10)


def test_sparse_lu_differentiable_nonsymmetric():
    """grad wrt b AND wrt the matrix data, vs finite differences, on a non-symmetric system."""
    A, b, Ad = _nonsym(12, seed=3)
    assert np.abs(Ad - Ad.T).max() > 0.1, "test matrix must be non-symmetric"
    L = lambda data, bb: jnp.sum(sparse_lu_solve(jsp.BCOO((data, A.indices), shape=A.shape), bb) ** 2)
    eps = 1e-6
    gb = np.asarray(jax.grad(L, 1)(A.data, b))
    gb_fd = np.array(
        [(float(L(A.data, b.at[i].add(eps))) - float(L(A.data, b.at[i].add(-eps)))) / (2 * eps) for i in range(b.size)]
    )
    assert np.abs(gb - gb_fd).max() / (np.abs(gb_fd).max() + 1e-12) < 1e-5, "grad wrt b wrong"
    gd = np.asarray(jax.grad(L, 0)(A.data, b))
    gd_fd = np.array(
        [(float(L(A.data.at[i].add(eps), b)) - float(L(A.data.at[i].add(-eps), b))) / (2 * eps) for i in range(A.data.size)]
    )
    assert np.abs(gd - gd_fd).max() / (np.abs(gd_fd).max() + 1e-12) < 1e-5, "grad wrt matrix data wrong (transpose path)"


def test_sparse_lu_solves_indefinite_saddle():
    """A saddle-point [[K, B],[B^T, 0]] (zero pressure-diagonal -> Jacobi degenerate) -- direct handles it."""
    rng = np.random.default_rng(0)
    nu, npr = 24, 8
    K = rng.standard_normal((nu, nu))
    K = K @ K.T + nu * np.eye(nu)
    B = rng.standard_normal((nu, npr))
    S = np.block([[K, B], [B.T, np.zeros((npr, npr))]])
    A = jsp.BCOO.fromdense(jnp.asarray(S))
    b = jnp.asarray(rng.standard_normal(nu + npr))
    x = sparse_lu_solve(A, b)
    assert float(jnp.linalg.norm(jnp.asarray(S) @ x - b)) < 1e-9


def test_jacobi_guards_zero_diagonal():
    """Jacobi must never produce inf/NaN even when the diagonal has zeros (saddle pressure block)."""
    S = np.diag([2.0, 3.0, 0.0, 5.0]).astype(float)  # a zero on the diagonal
    A = jsp.BCOO.fromdense(jnp.asarray(S))
    M = jacobi(A)
    out = np.asarray(M(jnp.ones(4)))
    assert np.all(np.isfinite(out)), "Jacobi produced inf/NaN on a zero diagonal"
    assert np.allclose(np.asarray(matrix_diagonal(A)), [2.0, 3.0, 0.0, 5.0])
    assert np.allclose(out, [0.5, 1.0 / 3.0, 1.0, 0.2])  # zero-diag entry left unscaled (1.0)


def test_fem_steady_linear_default_is_sparse_direct():
    """The fem.solve steady-linear default (sparse-direct) matches a dense solve on a real Poisson."""
    pytest.importorskip("shapely")
    from shapely.geometry import box

    import jno

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.08)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    x = np.asarray(fem.solve())  # default = sparse_lu_solve
    A0 = fem.operator[0]
    A = jnp.asarray(A0.todense() if hasattr(A0, "todense") else A0)
    b = jnp.asarray(fem.operator[1]).reshape(-1)
    assert np.allclose(x, np.asarray(jnp.linalg.solve(A, b)), atol=1e-9)


def test_sparse_lu_is_one_compiled_program():
    """The triplets-to-CSR conversion in front of ``spsolve`` -- a lexsort, two gathers and a
    ``searchsorted`` -- used to run uncompiled, so each primitive became its own XLA module: ~43 of
    them. On a Morley (C1) problem, whose 4th-order system needs the sparse-direct path, that was
    1520 ms of a 1751 ms first solve, 87% of it spent compiling fragments of an index conversion. It
    was never Morley-specific: every ``jno.solve.lu()``, the 1-D default and every non-nodal solve
    goes through here.

    Pins the property rather than the timing: one program on first call, none on the second, and the
    conversion is shape-keyed so a same-shaped problem reuses it.
    """
    import jax._src.compiler as _compiler

    from jno.utils.solver.linear import sparse_lu_solve

    rng = np.random.default_rng(0)
    n = 60
    Ad = rng.standard_normal((n, n)) + n * np.eye(n)
    A = jsp.BCOO.fromdense(jnp.asarray(Ad))
    b = jnp.asarray(rng.standard_normal(n))

    count = {"n": 0}
    orig = _compiler.compile_or_get_cached

    def counting(*a, **kw):
        count["n"] += 1
        return orig(*a, **kw)

    _compiler.compile_or_get_cached = counting
    try:
        x1 = np.asarray(sparse_lu_solve(A, b))
        first = count["n"]
        x2 = np.asarray(sparse_lu_solve(A, b))
        second = count["n"] - first
        # a DIFFERENT operator of the same shape must reuse the compilation
        A2 = jsp.BCOO.fromdense(jnp.asarray(rng.standard_normal((n, n)) + n * np.eye(n)))
        np.asarray(sparse_lu_solve(A2, b))
        third = count["n"] - first - second
    finally:
        _compiler.compile_or_get_cached = orig

    assert first <= 2, f"the conversion compiled {first} programs -- it is meant to be one"
    assert second == 0, "a repeated solve recompiled"
    assert third == 0, "a same-shaped operator recompiled -- the conversion is not shape-keyed"
    assert np.allclose(x1, np.linalg.solve(Ad, np.asarray(b)), atol=1e-8)
    assert np.allclose(x1, x2)


# --------------------------------------------------------------------------------------------
# host-factored direct solve: same answer, same gradients, past cuSolver's ceiling
# --------------------------------------------------------------------------------------------
def _sparse_nonsym(n=48, seed=5):
    """A sparse, non-symmetric, diagonally-dominant system and its dense twin."""
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(n, n))
    dense[np.abs(dense) < 1.4] = 0.0
    dense += np.eye(n) * 6.0
    return jsp.BCOO.fromdense(jnp.asarray(dense)), jnp.asarray(dense), jnp.asarray(rng.normal(size=n))


def test_host_lu_matches_the_dense_reference():
    from jno.utils.solver.linear import host_lu_solve

    A, dense, b = _sparse_nonsym()
    got, want = host_lu_solve(A, b), jnp.linalg.solve(dense, b)
    assert float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want)) < 1e-10


def test_host_lu_matches_the_device_path():
    """The point of the option is WHERE it factors, not WHAT it computes."""
    from jno.utils.solver.linear import host_lu_solve, sparse_lu_solve

    A, _dense, b = _sparse_nonsym()
    got, want = host_lu_solve(A, b), sparse_lu_solve(A, b)
    assert float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want)) < 1e-10


def test_host_lu_runs_under_jit():
    """It is a pure_callback underneath, so this is the property that could silently break."""
    from jno.utils.solver.linear import host_lu_solve

    A, dense, b = _sparse_nonsym()
    got = jax.jit(host_lu_solve)(A, b)
    assert float(jnp.linalg.norm(got - jnp.linalg.solve(dense, b)) / jnp.linalg.norm(b)) < 1e-10


@pytest.mark.parametrize("argnum", [0, 1])
def test_host_lu_gradients_match_the_device_path(argnum):
    """A pure_callback is forward-only; gradients survive only because the solve is wrapped in
    lax.custom_linear_solve, whose transpose solve reuses the same factorisation (SuperLU trans=T).
    Checks the matrix-entry gradient as well as the right-hand-side one."""
    from jno.utils.solver.linear import host_lu_solve, sparse_lu_solve

    A, _dense, b = _sparse_nonsym()
    loss = lambda data, rhs, f: jnp.sum(f(jsp.BCOO((data, A.indices), shape=A.shape), rhs) ** 2)  # noqa: E731
    g_host = jax.grad(loss, argnum)(A.data, b, host_lu_solve)
    g_dev = jax.grad(loss, argnum)(A.data, b, sparse_lu_solve)
    assert float(jnp.linalg.norm(g_host - g_dev) / jnp.linalg.norm(g_dev)) < 1e-8


def test_host_lu_solves_an_indefinite_saddle_point():
    """The case the option exists for: cuSolver reports 'Singular matrix' on these at modest size."""
    from jno.utils.solver.linear import host_lu_solve

    rng = np.random.default_rng(0)
    nu, npr = 24, 8
    K = rng.normal(size=(nu, nu))
    K = K @ K.T + np.eye(nu) * nu
    B = rng.normal(size=(nu, npr))
    dense = np.block([[K, B], [B.T, np.zeros((npr, npr))]])  # zero pressure block
    A = jsp.BCOO.fromdense(jnp.asarray(dense))
    b = jnp.asarray(rng.normal(size=nu + npr))
    got = host_lu_solve(A, b)
    assert float(jnp.linalg.norm(jnp.asarray(dense) @ got - b) / jnp.linalg.norm(b)) < 1e-10


def test_lu_host_spec_is_distinguishable_from_the_device_spec():
    """Two specs that factor in different memories must not be reported as the same solver."""
    import jno

    assert jno.solve.lu().name != jno.solve.lu(host=True).name


# --------------------------------------------------------------------------------------------
# factorization reuse: the SAME operator must factor ONCE, a CHANGED one must never go stale
# --------------------------------------------------------------------------------------------
@pytest.fixture
def count_factorizations(monkeypatch):
    """Count ``splu`` calls and start from a cold cache."""
    import scipy.sparse.linalg as spla

    from jno.utils.solver.linear import _FACTOR_CACHE

    _FACTOR_CACHE.clear()
    calls = []
    original = spla.splu
    monkeypatch.setattr(spla, "splu", lambda *a, **kw: calls.append(1) or original(*a, **kw))
    yield calls
    _FACTOR_CACHE.clear()


def test_repeating_the_same_operator_factors_once(count_factorizations):
    from jno.utils.solver.linear import host_lu_solve

    A, dense, b = _sparse_nonsym()
    for _ in range(5):
        got = host_lu_solve(A, b)
    assert len(count_factorizations) == 1, f"re-factored {len(count_factorizations)}x for one operator"
    assert float(jnp.linalg.norm(got - jnp.linalg.solve(dense, b)) / jnp.linalg.norm(b)) < 1e-10


def test_a_changed_operator_is_never_served_a_stale_factorization(count_factorizations):
    """The correctness requirement. A cache that misses is slow; a cache that hits wrongly is a
    silently wrong answer, so this checks the VALUE, not just the factorization count."""
    from jno.utils.solver.linear import host_lu_solve

    A, dense, b = _sparse_nonsym()
    first = host_lu_solve(A, b)

    scaled = jsp.BCOO((A.data * 2.7, A.indices), shape=A.shape)  # same pattern, different values
    second = host_lu_solve(scaled, b)
    assert len(count_factorizations) == 2, "a changed coefficient reused the old factorization"
    np.testing.assert_allclose(np.asarray(second), np.asarray(jnp.linalg.solve(dense * 2.7, b)), rtol=1e-9)
    assert not np.allclose(np.asarray(first), np.asarray(second)), "the two answers should differ"


def test_a_different_pattern_is_not_confused_with_the_same_values(count_factorizations):
    """Both arrays are hashed: identical data on a DIFFERENT sparsity pattern is a different matrix."""
    from jno.utils.solver.linear import host_lu_solve

    d = jnp.asarray([4.0, 1.0, 1.0, 4.0])
    idx_a = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    idx_b = jnp.asarray([[0, 0], [1, 0], [0, 1], [1, 1]])  # same values, transposed placement
    rhs = jnp.asarray([1.0, 2.0])
    xa = host_lu_solve(jsp.BCOO((d, idx_a), shape=(2, 2)), rhs)
    xb = host_lu_solve(jsp.BCOO((d, idx_b), shape=(2, 2)), rhs)
    assert len(count_factorizations) == 2
    np.testing.assert_allclose(
        np.asarray(xa), np.asarray(jnp.linalg.solve(jnp.asarray([[4.0, 1.0], [1.0, 4.0]]), rhs)), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(xb), np.asarray(jnp.linalg.solve(jnp.asarray([[4.0, 1.0], [1.0, 4.0]]).T, rhs)), rtol=1e-10
    )


def test_the_transpose_solve_reuses_the_forward_factorization(count_factorizations):
    """``transpose`` is not in the cache key, so the adjoint pass costs no factorization at all."""
    from jno.utils.solver.linear import host_lu_solve

    A, _dense, b = _sparse_nonsym()
    loss = lambda data: jnp.sum(host_lu_solve(jsp.BCOO((data, A.indices), shape=A.shape), b) ** 2)  # noqa: E731
    g = jax.grad(loss)(A.data)
    assert np.all(np.isfinite(np.asarray(g)))
    assert len(count_factorizations) == 1, f"forward + adjoint took {len(count_factorizations)} factorizations"


def test_the_factorization_cache_is_bounded(count_factorizations):
    """A sparse factorization is the biggest object around the solve (fill-in) -- an unbounded cache
    would hold every stale one in host memory."""
    from jno.utils.solver.linear import _FACTOR_CACHE, _FACTOR_CACHE_MAX, host_lu_solve

    A, _dense, b = _sparse_nonsym()
    for s in range(_FACTOR_CACHE_MAX + 3):
        host_lu_solve(jsp.BCOO((A.data * (1.0 + s), A.indices), shape=A.shape), b)
    assert len(_FACTOR_CACHE) <= _FACTOR_CACHE_MAX


# --------------------------------------------------------------------------------------------------
# Symmetry detection for the sparse-DIRECT backends.
#
# cuDSS and PARDISO factor an exactly symmetric operator as LDLᵀ instead of a general LU — measured
# 1.41x with 1.38x less peak device memory (cuDSS) and up to 1.9x (PARDISO). Symmetry was tested
# BITWISE, which no vector or coupled FEM tangent passes: the element block contracts components in a
# different order for (a,i),(b,j) than for (b,j),(a,i), so the two triangles differ by a fraction of
# an ulp. Every such problem was therefore factored as a general LU.
#
# `_symmetrized_kind_and_values` averages the two triangles when they agree to within a few ulps. The
# tests below pin both halves of that: it must fire on assembly round-off, and it must NEVER fire on a
# genuinely non-symmetric operator.
# --------------------------------------------------------------------------------------------------
def _perm_and_kinds(rows, cols, vals):
    from jno.utils.solver.linear import _cudss_matrix_kind, _cudss_sym_perms, _symmetrized_kind_and_values

    order, orderT, pat = _cudss_sym_perms(np.asarray(rows), np.asarray(cols))
    v = np.asarray(vals, dtype=float)
    strict = _cudss_matrix_kind(v, order, orderT, pat)
    kind, out = _symmetrized_kind_and_values(v, order, orderT, pat)
    return strict, kind, out, (order, orderT)


def _tridiag(n, off=-1.0, asym=0.0):
    """A symmetric tridiagonal, optionally with one off-diagonal entry nudged to break symmetry."""
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i), cols.append(i), vals.append(2.0)
        if i + 1 < n:
            rows += [i, i + 1]
            cols += [i + 1, i]
            vals += [off + asym, off]
    return np.array(rows), np.array(cols), np.array(vals)


def test_roundoff_asymmetry_is_treated_as_symmetric_and_averaged():
    eps = float(np.finfo(np.float64).eps)
    n = 40
    rows, cols, vals = _tridiag(n, asym=8.0 * eps)  # ~4 ulps against |A| = 2
    strict, kind, out, (order, orderT) = _perm_and_kinds(rows, cols, vals)
    assert strict == "general", "the strict bitwise test must still reject this — else the test is vacuous"
    assert kind == "symmetric", "a few-ulp asymmetry must be accepted as symmetric"
    assert np.array_equal(out[order], out[orderT]), "the returned values must be BITWISE symmetric"
    # The correction is bounded by the asymmetry it removes — it adds no error the assembly had not made.
    assert np.abs(out - vals).max() <= 0.5 * np.abs(np.asarray(vals)[order] - np.asarray(vals)[orderT]).max() + 1e-300


def test_a_genuinely_nonsymmetric_matrix_is_never_averaged():
    rows, cols, vals = _tridiag(40, asym=0.5)  # a real asymmetry, not round-off
    strict, kind, out, _ = _perm_and_kinds(rows, cols, vals)
    assert strict == "general" and kind == "general", "a genuinely non-symmetric matrix must stay general"
    assert np.array_equal(out, vals), "its values must be returned untouched"


def test_the_gate_rejects_asymmetry_well_below_a_physically_meaningful_one():
    """Measured on the real thing: a vector FEM tangent sits at ~0.25 ulps, while an advection term so
    weak it is physically meaningless (coefficient 1e-12) already reaches ~191 ulps. The gate must sit
    between those two populations, not on a continuum."""
    eps = float(np.finfo(np.float64).eps)
    _s, kind_lo, _o, _p = _perm_and_kinds(*_tridiag(40, asym=0.5 * eps))  # ~0.25 ulps: round-off
    _s, kind_hi, _o, _p = _perm_and_kinds(*_tridiag(40, asym=400.0 * eps))  # ~200 ulps: genuine
    assert kind_lo == "symmetric"
    assert kind_hi == "general"


def test_a_vector_fem_tangent_now_qualifies_as_symmetric():
    """The case this exists for — a 3-D elasticity tangent, which no bitwise test accepts."""
    nn = jno.np
    inner, grad, symg, trace, ident = nn.inner, nn.grad, nn.sym, nn.trace, nn.identity
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.4).domain()
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co, cb = d.variable("interior", split=True), d.variable("bot", split=True)
    X, I3 = [co[0], co[1], co[2]], ident(3)
    u, phi = d.fem_symbols(value_shape=(3,))
    e = symg(grad(u, X))
    sig = 80.0 * trace(e) * I3 + 120.0 * e + 200.0 * inner(e, e, 2) * e  # nonlinear -> residual path
    zh = nn.asarray([0.0, 0.0, 1.0])
    fem = jno.fem(
        [inner(sig, symg(grad(phi, X)), 2) - 4.0 * inner(zh, phi, 1)] + [u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)]
    )
    J = fem._op.jacobian(jnp.asarray(fem.solve()))
    strict, kind, _out, _p = _perm_and_kinds(np.asarray(J.indices[:, 0]), np.asarray(J.indices[:, 1]), np.asarray(J.data))
    assert strict == "general", "if this ever becomes bitwise symmetric the change is moot — check why"
    assert kind == "symmetric", "a vector FEM tangent must reach the LDLt path"


def test_pardiso_agrees_with_the_iterative_default_on_a_symmetric_tangent():
    """End to end: the symmetric factorization must give the same answer as the matrix-free default."""
    pytest.importorskip("pypardiso")
    from jno.utils.solver.linear import _pardiso_available

    if not _pardiso_available():
        pytest.skip("pypardiso present but without the private phase hooks this backend drives")
    nn = jno.np
    inner, grad, symg, trace, ident = nn.inner, nn.grad, nn.sym, nn.trace, nn.identity
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.45).domain()
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co, cb = d.variable("interior", split=True), d.variable("bot", split=True)
    X, I3 = [co[0], co[1], co[2]], ident(3)
    u, phi = d.fem_symbols(value_shape=(3,))
    e = symg(grad(u, X))
    sig = 80.0 * trace(e) * I3 + 120.0 * e + 200.0 * inner(e, e, 2) * e
    zh = nn.asarray([0.0, 0.0, 1.0])
    fem = jno.fem(
        [inner(sig, symg(grad(phi, X)), 2) - 4.0 * inner(zh, phi, 1)] + [u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)]
    )
    ref = np.asarray(fem.solve())
    got = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True), linear=jno.solve.lu(backend="pardiso")))
    assert np.abs(ref).max() > 1e-6
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8, "the LDLt factorization changed the answer"
