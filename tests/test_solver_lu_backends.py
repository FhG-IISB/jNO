"""``jno.solve.lu(backend=...)``: the backend selector, and the cuDSS path's cache contract.

The cache is the whole reason cuDSS is worth having (the symbolic plan survives a change of values),
and it is also the only place this path can go silently WRONG -- a stale factorization returns a
plausible number for the previous matrix. So the cuDSS tests below check staleness explicitly, not
just that a solve is accurate.

Everything touching cuDSS skips unless the optional stack is installed; the selector and the
missing-dependency error are tested unconditionally.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsp

import jno
from jno.utils.solver.linear import _cudss_available, cudss_lu_solve

requires_cudss = pytest.mark.skipif(not _cudss_available(), reason="optional cuDSS stack not installed")


@pytest.fixture(autouse=True)
def _x64():
    """A direct solver is an x64 path -- and an FD gradient check is meaningless in float32.

    Set and restored per test rather than at import, so this file cannot dictate the precision of
    whatever runs after it in the same session.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _spd(n, seed=0, shift=0.0):
    """A well-conditioned non-symmetric sparse operator (tridiagonal + a corner), as BCOO."""
    rng = np.random.default_rng(seed)
    d = 4.0 + shift + rng.uniform(0.0, 0.5, n)
    rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n), [0]])
    cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1), [n - 1]])
    vals = np.concatenate([d, -np.ones(n - 1), -1.3 * np.ones(n - 1), [0.7]])
    idx = jnp.asarray(np.stack([rows, cols], axis=1))
    return jsp.BCOO((jnp.asarray(vals), idx), shape=(n, n))


# --------------------------------------------------------------------------------------
# the selector — no cuDSS needed
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend,name", [("device", "lu"), ("host", "lu-host"), ("cudss", "lu-cudss")])
def test_backend_names_are_distinct(backend, name):
    """Each placement reports its own name, so two specs are never cached as the same solver."""
    spec = jno.solve.lu(backend=backend)
    assert spec.name == name
    assert spec.direct is True
    assert spec.traits["vmap"] == "no"


def test_host_kwarg_still_selects_the_host_backend():
    """The deprecated spelling keeps working -- existing scripts must not break."""
    assert jno.solve.lu(host=True).name == jno.solve.lu(backend="host").name
    assert jno.solve.lu(host=False).name == jno.solve.lu(backend="device").name


def test_unknown_backend_names_the_valid_ones():
    with pytest.raises(ValueError, match="not a known backend"):
        jno.solve.lu(backend="cuDSS")  # right solver, wrong case
    with pytest.raises(ValueError, match="not a known backend"):
        jno.solve.lu(backend="pardiso")


def test_backend_and_host_together_is_an_error():
    """Silently letting one win would make the other argument a no-op."""
    with pytest.raises(ValueError, match="deprecated spelling"):
        jno.solve.lu(backend="cudss", host=True)


@pytest.mark.skipif(_cudss_available(), reason="cuDSS installed, so the missing-dependency path cannot run")
def test_missing_cudss_says_how_to_install_and_what_to_use_instead():
    """An optional backend must fail loud with the fix, not fall back to a different solver."""
    with pytest.raises(ImportError, match="nvmath-python"):
        cudss_lu_solve(_spd(8), jnp.ones(8))


# --------------------------------------------------------------------------------------
# the cuDSS path itself
# --------------------------------------------------------------------------------------


@requires_cudss
def test_solves_and_is_accurate():
    A, b = _spd(200), jnp.asarray(np.random.default_rng(1).normal(size=200))
    x = cudss_lu_solve(A, b)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-10


@requires_cudss
def test_new_values_on_the_same_sparsity_are_not_stale():
    """THE failure mode this cache can have: reusing a factorization after the values changed.

    Solving A1 then A2 (same pattern, different values) must give A2's answer, not A1's.
    """
    n = 150
    A1, A2 = _spd(n, shift=0.0), _spd(n, shift=3.0)
    b = jnp.asarray(np.random.default_rng(2).normal(size=n))
    cudss_lu_solve(A1, b)  # populates the cache under this sparsity
    x2 = cudss_lu_solve(A2, b)
    assert np.linalg.norm(A2 @ x2 - b) / np.linalg.norm(b) < 1e-10
    assert np.linalg.norm(A1 @ x2 - b) / np.linalg.norm(b) > 1e-3  # genuinely A2's answer


@requires_cudss
def test_repeated_operator_reuses_the_plan_and_the_cache_stays_bounded():
    from jno.utils.solver.linear import _CUDSS_CACHE, _CUDSS_CACHE_MAX

    _CUDSS_CACHE.clear()
    n = 120
    b = jnp.ones(n)
    for shift in range(6):  # same sparsity throughout -> ONE plan
        cudss_lu_solve(_spd(n, shift=float(shift)), b)
    assert len(_CUDSS_CACHE) == 1, "same sparsity must not create a second plan"

    for k in range(_CUDSS_CACHE_MAX + 4):  # distinct sparsities -> LRU must evict
        cudss_lu_solve(_spd(20 + k), jnp.ones(20 + k))
    assert len(_CUDSS_CACHE) <= _CUDSS_CACHE_MAX
    _CUDSS_CACHE.clear()


@requires_cudss
def test_gradients_match_finite_differences_on_a_non_symmetric_system():
    """``custom_linear_solve``'s transpose rule -- the adjoint needs its own cuDSS factorization."""
    n = 40
    A = _spd(n, seed=3)
    b = jnp.asarray(np.random.default_rng(4).normal(size=n))

    def loss(vals):
        x = cudss_lu_solve(jsp.BCOO((vals, A.indices), shape=A.shape), b)
        return jnp.sum(x**2)

    g = jax.grad(loss)(A.data)
    eps, k = 1e-6, 5
    fd = np.array([(loss(A.data.at[i].add(eps)) - loss(A.data.at[i].add(-eps))) / (2 * eps) for i in range(k)])
    np.testing.assert_allclose(np.asarray(g[:k]), fd, rtol=1e-5, atol=1e-8)


@requires_cudss
def test_composes_as_a_fem_linear_slot():
    """It must be a drop-in for the `linear=` slot, not a standalone function."""
    d = jno.Shape.rectangle(1.0, 1.0).domain(resolution=8)
    u, phi = d.fem_symbols()
    fem = jno.fem([jno.np.inner(jno.np.grad(u, d.coords), jno.np.grad(phi, d.coords)) - phi, u("boundary")])
    sol = fem.solve(linear=jno.solve.lu(backend="cudss"))
    ref = fem.solve(linear=jno.solve.lu(backend="host"))
    np.testing.assert_allclose(np.asarray(sol), np.asarray(ref), rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------------------
# the JAX wiring, exercised WITHOUT cuDSS
# --------------------------------------------------------------------------------------
#
# `cudss_lu_solve` is two separable halves: a host kernel that calls cuDSS, and the
# `pure_callback` + `custom_linear_solve` wiring that makes it jit-able and differentiable. The
# second half is where a shape, a dtype, or the adjoint's row/column swap can be wrong, and it does
# not need a GPU to be wrong. Substituting a scipy kernel with the SAME signature runs that wiring
# everywhere, so these paths are covered on machines where the tests above skip.


def _scipy_stand_in(data, indices, rhs, shape, transpose):
    """Byte-for-byte the signature of ``_cudss_host_solve``, backed by SuperLU instead."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    idx = np.asarray(indices)
    rows, cols = (idx[:, 1], idx[:, 0]) if transpose else (idx[:, 0], idx[:, 1])
    A = sp.coo_matrix((np.asarray(data), (rows, cols)), shape=shape).tocsc()
    return np.asarray(spla.spsolve(A, np.asarray(rhs)), dtype=rhs.dtype)


@pytest.fixture
def wired(monkeypatch):
    """`cudss_lu_solve` with its cuDSS kernel swapped out -- the wiring, none of the CUDA."""
    import jno.utils.solver.linear as lin

    monkeypatch.setattr(lin, "_cudss_available", lambda: True)
    monkeypatch.setattr(lin, "_cudss_host_solve", _scipy_stand_in)
    return lin.cudss_lu_solve


def test_wiring_solves_under_jit(wired):
    n = 60
    A = _spd(n, seed=5)
    b = jnp.asarray(np.random.default_rng(6).normal(size=n))
    x = jax.jit(wired)(A, b)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-10


def test_wiring_transpose_solve_uses_the_swapped_indices(wired):
    """The adjoint of a NON-symmetric operator is only right if rows/cols actually swap.

    d/db of sum(A^-1 b) is A^-T 1, so a forgotten swap silently returns A^-1 1 instead -- which for a
    symmetric test matrix would look correct. Hence the deliberately non-symmetric `_spd`.
    """
    n = 30
    A = _spd(n, seed=7)
    g = jax.grad(lambda b: jnp.sum(wired(A, b)))(jnp.zeros(n))
    dense = np.asarray(A.todense())
    np.testing.assert_allclose(np.asarray(g), np.linalg.solve(dense.T, np.ones(n)), rtol=1e-8, atol=1e-9)
    assert not np.allclose(np.linalg.solve(dense.T, np.ones(n)), np.linalg.solve(dense, np.ones(n)))


def test_wiring_gradient_flows_to_the_operator_entries(wired):
    n = 25
    A = _spd(n, seed=8)
    b = jnp.asarray(np.random.default_rng(9).normal(size=n))

    def loss(vals):
        return jnp.sum(wired(jsp.BCOO((vals, A.indices), shape=A.shape), b) ** 2)

    g = jax.grad(loss)(A.data)
    eps = 1e-6
    fd = np.array([(loss(A.data.at[i].add(eps)) - loss(A.data.at[i].add(-eps))) / (2 * eps) for i in range(5)])
    np.testing.assert_allclose(np.asarray(g[:5]), fd, rtol=1e-5, atol=1e-8)


def test_wiring_accepts_a_dense_operator(wired):
    n = 20
    A = _spd(n, seed=10)
    b = jnp.asarray(np.random.default_rng(11).normal(size=n))
    np.testing.assert_allclose(np.asarray(wired(A.todense(), b)), np.asarray(wired(A, b)), rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------------------
# the singular-operator guard
# --------------------------------------------------------------------------------------
#
# cuDSS reports a singular factorization through NEITHER an exception NOR a NaN -- measured, it
# returns a finite, plausible vector (`diag(1,2,0,4)` -> 1e+13 in the null slot, relative residual
# 1.0, `info == 0`). `_cudss_check_factorization` catches that. It takes its array module as an
# argument, so the decision logic is testable with fakes on any machine; the @requires_cudss test
# below covers the real thing.


class _FakeSolver:
    def __init__(self, npivots, x):
        self.factorization_info = type("I", (), {"npivots": npivots})()
        self._x = x

    def solve(self):
        return self._x


def _guard(npivots, x, A, b):
    from jno.utils.solver.linear import _cudss_check_factorization

    return _cudss_check_factorization(_FakeSolver(npivots, jnp.asarray(x)), jnp.asarray(A), jnp.asarray(b), jnp, (4, 4))


def test_guard_is_silent_when_no_pivot_was_replaced():
    """The common path must cost a host-side attribute read and NO SpMV."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    assert _guard(0, [99.0, 99.0, 99.0, 99.0], A, np.ones(4)) is None  # garbage x, but npivots==0


def test_guard_raises_on_a_replaced_pivot_with_a_bad_residual():
    A = np.diag([1.0, 2.0, 0.0, 4.0])
    with pytest.raises(RuntimeError, match="SINGULAR"):
        _guard(1, [1.0, 0.5, 1e13, 0.25], A, np.ones(4))


def test_guard_allows_a_replaced_pivot_that_still_solved():
    """A perturbed pivot is only a *suspicion*; a good residual clears it, so no false failure."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    assert _guard(1, [1.0, 0.5, 1 / 3, 0.25], A, np.ones(4)) is None


def test_guard_tolerates_a_cudss_that_does_not_expose_npivots():
    """An older cuDSS must not turn a working solve into an error."""

    class _Old:
        @property
        def factorization_info(self):
            raise AttributeError("npivots")

    from jno.utils.solver.linear import _cudss_check_factorization

    assert _cudss_check_factorization(_Old(), jnp.eye(4), jnp.ones(4), jnp, (4, 4)) is None


def test_guard_handles_a_zero_rhs_without_dividing_by_zero():
    A = np.diag([1.0, 2.0, 0.0, 4.0])
    assert _guard(1, [0.0, 0.0, 0.0, 0.0], A, np.zeros(4)) is None  # 0 residual on a 0 rhs is correct


@requires_cudss
def test_singular_operator_raises_instead_of_returning_finite_garbage():
    """The real thing: cuDSS would return a finite wrong answer here."""
    idx = jnp.asarray(np.stack([np.arange(4), np.arange(4)], axis=1))
    A = jsp.BCOO((jnp.asarray([1.0, 2.0, 0.0, 4.0]), idx), shape=(4, 4))
    with pytest.raises(RuntimeError, match="SINGULAR"):
        cudss_lu_solve(A, jnp.ones(4))
