"""The native FEM assembler builds the global system matrix as a sparse ``BCOO`` via COO
triplets — it never materialises the dense ``(n, n)`` array (``O(nnz)`` storage, GPU-able at
large ``N``). These tests pin that the assembled operator is BCOO, has ``O(n)`` nonzeros (not
``O(n^2)``), reproduces the dense reference exactly (``todense`` / matvec), keeps Dirichlet
elimination correct, and stays differentiable in the assembled values.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_native import assemble_fem_native  # noqa: E402

inner, grad = jno.np.inner, jno.np.grad
PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(M):
    return np.asarray(M.todense() if hasattr(M, "todense") else M)


def _poisson(d):
    u, w = d.fem_symbols(names=("u", "w"))
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    vv = w.bind(x=xi, y=yi)
    f = 2 * PI**2 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    return [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - f * vv, u(xb, yb) - 0.0]


def test_steady_operator_is_bcoo_and_sparse():
    """The steady-linear operator is a BCOO whose nonzeros scale like ``O(n)`` (≈ nnz/row · n),
    far below the dense ``n^2`` — this is the never-densify memory win."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.15)
    op, mode, _ = assemble_fem_native(d, *_classify(d), vec=1, quad_degree=2)
    assert mode == "linear"
    A = op[0]
    assert hasattr(A, "indices")  # BCOO, not a dense array
    n = A.shape[0]
    assert A.shape == (n, n)
    # genuine sparsity: even unsummed (per-element duplicates) the triplet count is O(n), not O(n^2)
    assert int(A.nse) < n * n
    assert int(A.nse) < 64 * n


def test_steady_bcoo_todense_matches_route():
    """``op.todense()`` reproduces the dense matrix the ``jno.fem`` route exposes (Dirichlet
    symmetric elimination included) — entry-for-entry."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2)
    op, _mode, _ = assemble_fem_native(d, *_classify(d), vec=1, quad_degree=2)
    A_sparse = _dense(op[0])
    fem = jno.fem(_poisson(jno.domain(box(0, 0, 1, 1), mesh_size=0.2)), quad_degree=2)
    A_route = np.asarray(fem.A)
    assert A_sparse.shape == A_route.shape
    assert np.abs(A_sparse - A_route).max() < 1e-11


def test_steady_solve_matches_analytic():
    """End-to-end: the never-densify operator solves to the manufactured ``sin(πx)sin(πy)``."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.06)
    fem = jno.fem(_poisson(d), quad_degree=2)
    u = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points)
    exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    assert np.abs(u - exact).max() < 5e-3


def test_transient_block_M_A_are_bcoo_matvec_exact():
    """The transient semidiscrete block carries BCOO ``M`` / ``A``; matvec equals the dense matvec,
    and the block integrates to a finite, bounded field."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2, time=(0.0, 0.05, 5))
    u, w = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb = d.variable("boundary", split=True)[:2]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
    weak = ui.t * vi + (ui.x * vi.x + ui.y * vi.y)
    icf = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [ci[0], ci[1]])
    fem = jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - icf])

    block = fem.operator
    assert hasattr(block.M, "indices") and hasattr(block.A, "indices")  # both BCOO
    n = block.M.shape[0]
    v = jnp.asarray(np.cos(np.arange(n, dtype=np.float64)))
    assert np.allclose(np.asarray(block.M @ v), _dense(block.M) @ np.asarray(v))
    assert np.allclose(np.asarray(block.A @ v), _dense(block.A) @ np.asarray(v))
    # the BCOO block integrates through the real backward-Euler scan (M @ w / A @ w matvecs)
    from jno.utils.solver.backend_blocks import _block_time_grid, _default_transient_integrate

    ts = np.asarray(_block_time_grid(block))
    ys = np.asarray(_default_transient_integrate(block, {}, ts))
    assert np.isfinite(ys).all()
    assert ys.max() <= 1.0 + 1e-6  # sin IC, heat decays


# Differentiability of the BCOO assembly (grad flows through the assembled ``data``; indices are
# static) is exercised end-to-end by the parametric inverse oracle ``tests/test_fem_inverse.py``.


# ---------------------------------------------------------------------------
# helper: classify a fresh Poisson constraint list into the assembler's inputs
# ---------------------------------------------------------------------------
def _classify(d):
    from jno._fem import (
        _bare,
        _contains,
        _dirichlet_spec,
        _field_key_of,
        _region_and_support,
        _retag_coords_for_quadrature,
    )
    from jno.trace import TestFunction, TrialFunction

    vt: list = []
    bt: dict = {}
    dr: list = []
    for c in _poisson(d):
        if _contains(c, TestFunction):
            support, region = _region_and_support(c, d)
            _retag_coords_for_quadrature(c, support, region)
            bare = _bare(c)
            (vt if support == "volume" else bt.setdefault(region, [])).append(bare)
        elif _contains(c, TrialFunction):
            _, region = _region_and_support(c, d)
            comp, value, value_node = _dirichlet_spec(_bare(c))
            dr.append((_field_key_of(c), region, comp, value, value_node))
    return vt, bt, dr, []


# ---------------------------------------------------------------------------------------------
# compress_plan memoization
# ---------------------------------------------------------------------------------------------
#
# The duplicate-collapse plan is a pure function of the triplet pattern, and the pattern is fixed by
# mesh and terms — so rebuilding the same problem recomputed an identical plan. Measured on 3-D
# Poisson at 27,833 nodes that was 0.68 s of a 1.5 s build. The cache is keyed on the pattern's
# CONTENT, which is what makes it safe: a remeshed domain changes the content and simply misses.


def _plan_cache():
    from jno.utils.solver import fem_utils

    return fem_utils._PLAN_CACHE


def test_compress_plan_is_reused_for_an_identical_pattern():
    from jno.utils.solver.fem_utils import compress_plan

    _plan_cache().clear()
    idx = jnp.asarray(np.array([[0, 0], [0, 1], [1, 1], [0, 0], [2, 2]], dtype=np.int32))
    first = compress_plan(idx)
    assert len(_plan_cache()) == 1
    # a DIFFERENT array object holding the same values must hit: the key is content, not identity
    second = compress_plan(jnp.asarray(np.asarray(idx)))
    assert len(_plan_cache()) == 1, "identical content must not create a second entry"
    assert second is first
    _plan_cache().clear()


def test_a_different_pattern_gets_its_own_plan():
    from jno.utils.solver.fem_utils import compress_plan

    _plan_cache().clear()
    a = compress_plan(jnp.asarray(np.array([[0, 0], [1, 1], [0, 0]], dtype=np.int32)))
    b = compress_plan(jnp.asarray(np.array([[0, 0], [1, 1], [1, 1]], dtype=np.int32)))
    assert len(_plan_cache()) == 2
    assert a[2] == 2 and b[2] == 2
    # the INVERSE differs even though both collapse to 2 slots — a shared plan would be wrong
    assert not np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
    _plan_cache().clear()


def test_the_plan_cache_is_bounded():
    from jno.utils.solver.fem_utils import _PLAN_CACHE_MAX, compress_plan

    _plan_cache().clear()
    for k in range(_PLAN_CACHE_MAX + 3):
        compress_plan(jnp.asarray(np.array([[0, 0], [1, 1], [k % 7, k]], dtype=np.int32)))
    assert len(_plan_cache()) <= _PLAN_CACHE_MAX
    _plan_cache().clear()


def test_memoized_plan_matches_an_uncached_recomputation():
    """The oracle: what the cache returns must equal what the algorithm computes from scratch."""
    from jno.utils.solver.fem_utils import compress_plan

    rng = np.random.default_rng(0)
    idx = jnp.asarray(np.stack([rng.integers(0, 12, 400), rng.integers(0, 12, 400)], axis=1).astype(np.int32))
    _plan_cache().clear()
    fresh = compress_plan(idx)
    cached = compress_plan(idx)
    _plan_cache().clear()
    recomputed = compress_plan(idx)
    for got, ref in ((cached, fresh), (recomputed, fresh)):
        np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(ref[0]))
        np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(ref[1]))
        assert got[2] == ref[2]
    _plan_cache().clear()


def test_a_remeshed_domain_does_not_reuse_a_stale_plan():
    """The one way a plan cache could be silently wrong. Two different meshes of the same problem
    must produce different operators, not the first mesh's sparsity applied to the second."""
    _plan_cache().clear()
    sols, nnzs = [], []
    for res in (6, 9):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=1.0 / res)
        u, v = d.fem_symbols()
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
        A, _b = fem.operator
        nnzs.append(int(np.asarray(A.data).size))
        sols.append(np.asarray(fem.solve()).reshape(-1))
    assert nnzs[0] != nnzs[1], "the two meshes must not share a sparsity pattern"
    assert sols[0].size != sols[1].size
    for s in sols:
        assert np.all(np.isfinite(s)) and np.max(s) > 0.0
    _plan_cache().clear()


# --------------------------------------------------------------------------------------------------
# `bcoo_identity_rows` — replace rows by identity from a TRACED boolean mask. The index-array helpers
# beside it cannot serve a min-map's active set, which is a function of the current iterate.
# --------------------------------------------------------------------------------------------------
def _dense_identity_rows(A, mask):
    """The oracle, written out: rows where `mask` holds become rows of the identity."""
    out = np.array(A, dtype=float, copy=True)
    out[mask, :] = 0.0
    out[mask, mask] = 1.0
    return out


@pytest.mark.parametrize(
    "mask",
    [
        [False, False, False, False],  # none active — must be an exact no-op
        [True, True, True, True],  # ALL active — the result is the identity
        [True, False, True, False],  # the ordinary mixed case
        [False, False, False, True],  # only the last row
    ],
)
def test_bcoo_identity_rows_matches_the_dense_oracle(mask):
    from jax.experimental import sparse as jsparse

    from jno.utils.solver.fem_utils import bcoo_identity_rows

    rng = np.random.default_rng(0)
    dense = rng.normal(size=(4, 4))
    dense[1, 3] = 0.0  # a structural zero, so the pattern is not trivially full
    A = jsparse.BCOO.fromdense(jnp.asarray(dense))
    m = np.asarray(mask)
    got = np.asarray(bcoo_identity_rows(A, jnp.asarray(m)).todense())
    want = _dense_identity_rows(dense, m)
    assert np.allclose(got, want), f"\ngot\n{got}\nwant\n{want}"


def test_bcoo_identity_rows_keeps_a_static_shape_under_jit():
    """The whole reason it takes a mask: the active set changes every iterate, so this runs inside a
    traced Newton loop and may not produce a data-dependent nnz."""
    from jax.experimental import sparse as jsparse

    from jno.utils.solver.fem_utils import bcoo_identity_rows

    A = jsparse.BCOO.fromdense(jnp.eye(3) * 2.0 + 1.0)

    @jax.jit
    def go(mask):
        return bcoo_identity_rows(A, mask).todense()

    a = np.asarray(go(jnp.asarray([True, False, False])))
    b = np.asarray(go(jnp.asarray([False, True, True])))
    assert np.allclose(a, _dense_identity_rows(np.asarray(A.todense()), np.array([True, False, False])))
    assert np.allclose(b, _dense_identity_rows(np.asarray(A.todense()), np.array([False, True, True])))
