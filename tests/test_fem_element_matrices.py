"""The operator's per-cell element matrices (``deflation.element_triplets``): the entries before they are summed.

Oracles: summed over every cell they ARE the assembled operator (off the eliminated Dirichlet rows); summed
over a subset of cells they are that subset's own operator with natural conditions on its cut, so a rigid-body
mode lies in its kernel (elasticity) and so does the constant (Laplacian). That subset operator is the local
Neumann matrix of GenEO (Spillane et al., Numer. Math. 126, 2014). Anything the accessor cannot serve raises.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import scipy.sparse as sp

import jno
from jno.utils.solver.deflation import element_triplets

R = jno.shape.rect


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _summed(fem, cells=None):
    """The triplets summed into a dense matrix on the full space, optionally over ``cells`` only."""
    r, c, v, cell = (np.asarray(a) for a in element_triplets(fem))
    if cells is not None:
        keep = np.isin(cell, cells)
        r, c, v = r[keep], c[keep], v[keep]
    n = int(fem._op[0].shape[0])
    return sp.coo_matrix((v, (r, c)), shape=(n, n)).toarray()


def _assert_is_the_operator(fem):
    A = np.asarray(fem._op[0].todense())
    off = A - np.diag(np.diag(A))
    free = np.any(off != 0, axis=1)  # the eliminated Dirichlet rows keep only their diagonal
    K = _summed(fem)
    np.testing.assert_allclose(K[np.ix_(free, free)], A[np.ix_(free, free)], rtol=1e-10, atol=1e-12 * np.abs(A).max())


def _elasticity(order, *, size=0.25):
    d = R(0.0, 0.0, 2.0, 1.0, size=size).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=order)
    ε = lambda w: jno.np.symgrad(w.bind(x=x, y=y), [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace
    fem = jno.fem([2.0 * ddot(ε(u), ε(v)) + 1.5 * tr(ε(u)) * tr(ε(v)) + 0.1 * v.bind(x=x, y=y)[1], u(xl, yl) - (0.0, 0.0)])
    return fem, d


@pytest.mark.parametrize("order", [1, 2])
def test_summed_cells_are_the_operator_and_a_cell_subset_holds_the_rigid_modes(order):
    fem, d = _elasticity(order)
    _assert_is_the_operator(fem)
    # a block of cells, cut out of the clamped body: its own operator is floating, so the three rigid-body
    # modes are its kernel even though the clamp makes the whole operator definite
    cells = np.asarray(d._fem_native_assembly_cells)
    pts = np.asarray(fem.points)
    centre = pts[cells[:, :3]].mean(axis=1)
    sub = np.flatnonzero((centre[:, 0] > 1.0) & (centre[:, 0] < 1.6))
    K = _summed(fem, sub)
    xs, ys = np.repeat(pts[:, 0], 2), np.repeat(pts[:, 1], 2)
    ex, ey = np.tile([1.0, 0.0], len(pts)), np.tile([0.0, 1.0], len(pts))
    rigid = np.column_stack([ex, ey, -ys * ex + xs * ey])
    assert np.abs(K @ rigid).max() < 1e-10 * np.abs(K).max()
    assert np.abs(np.diag(K)).max() > 0  # and it is not empty


def test_three_d_and_a_robin_term():
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.4).domain()
    x, y, z, _ = d.variable("interior", split=True)
    xr, yr, zr, _ = d.variable("boundary", split=True)
    u, w = d.fem_symbols()
    ui, wi = u.bind(x=x, y=y, z=z), w.bind(x=x, y=y, z=z)
    fem = jno.fem([ui.x * wi.x + ui.y * wi.y + ui.z * wi.z - wi, 2.0 * u.bind(x=xr, y=yr, z=zr) * w.bind(x=xr, y=yr, z=zr)])
    _assert_is_the_operator(fem)
    # the Robin term sits on the boundary faces: summing every cell keeps it, so the constant is NOT a kernel
    one = np.ones(fem._op[0].shape[0])
    assert np.abs(_summed(fem) @ one).max() > 1e-3


def test_two_fields():
    d = R(0.0, 0.0, 2.0, 1.0, size=0.3).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    gu, gv = jno.np.grad(u, [x, y]), jno.np.grad(v, [x, y])
    pp, qq = p.bind(x=x, y=y), q.bind(x=x, y=y)
    fem = jno.fem(
        [
            jno.np.inner(gu, gv, n_contract=2) - pp * jno.np.trace(gv) - qq * jno.np.trace(gu) + 1e-3 * pp * qq,
            u(xb, yb) - (0.0, 0.0),
        ]
    )
    _assert_is_the_operator(fem)


def test_what_it_cannot_serve_raises():
    with pytest.raises(ValueError, match="element matrix"):
        element_triplets(object())

    d = R(0.0, 0.0, 1.0, 1.0).structured(n=4).domain()  # periodic in x: solved on the reduced space
    d.tag("left", lambda x, y: x < 1e-9)
    d.tag("right", lambda x, y: x > 1 - 1e-9)
    d.tag("bot", lambda x, y: (y < 1e-9) & (x > 1e-9))
    x, y = d.variable("interior", split=True)[:2]
    u, w = d.fem_symbols()
    ui, wi = u.bind(x=x, y=y), w.bind(x=x, y=y)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    periodic = jno.fem([ui.x * wi.x + ui.y * wi.y - wi, u(*d.variable("bot", split=True)) - 0.0, u(xl, yl) - u(xr, yr)])
    with pytest.raises(NotImplementedError, match="REDUCED"):
        element_triplets(periodic)
