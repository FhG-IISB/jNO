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
