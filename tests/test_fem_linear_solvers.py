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
