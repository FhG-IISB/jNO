"""Lagrange Q_k (k > 2) on quadrilaterals sits on the Gauss-Lobatto-Legendre nodes: the spectral-element basis.

Checks: the promoted mesh nodes along an edge are the GLL points; Poisson on a coarse quad mesh converges
exponentially in the degree (p-refinement), with the error at the nodes falling to ~1e-9 by Q8.
"""

import jax
import jno.numpy as jnn
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(order, n=2):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).quad().domain()
    u, v = d.fem_symbols(order=order)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    prob = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])
    sol = np.asarray(prob.solve()).reshape(-1)
    P = np.asarray(prob.points)[:, :2]
    return sol, P


def test_edge_nodes_are_gll_points():
    k = 5
    _sol, P = _poisson(k, n=1)
    on_bottom = np.unique(np.round(P[np.isclose(P[:, 1], 0.0), 0], 14))
    cells = (on_bottom.size - 1) // k
    ref = 0.5 * (1 + np.sort(np.concatenate([[-1.0, 1.0], np.polynomial.legendre.Legendre.basis(k).deriv().roots()])))
    expect = np.unique(np.round(np.concatenate([(c + ref) / cells for c in range(cells)]), 14))
    assert on_bottom.size == cells * k + 1
    assert np.allclose(on_bottom, expect, atol=1e-12)


def test_p_refinement_converges_exponentially():
    errs = []
    for k in (3, 5, 8):
        sol, P = _poisson(k)
        errs.append(np.abs(sol - np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])).max())
    assert errs[1] < 1e-2 * errs[0] and errs[2] < 1e-8, errs  # the floor is the iterative solve's tolerance


def _mass(order, quadrature):
    d = jno.shape.rect(0.0, 0.0, 2.0, 1.0).structured(n=2).quad().domain()
    u, v = d.fem_symbols(order=order)
    x, y, _ = d.variable("interior", split=True)
    A = jno.fem([u.bind(x=x, y=y) * v.bind(x=x, y=y)], quadrature=quadrature).A
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


@pytest.mark.parametrize("order", [3, 5])
def test_gll_quadrature_makes_the_mass_matrix_diagonal(order):
    M = _mass(order, "gll")
    off = M - np.diag(np.diag(M))
    assert np.abs(off).max() < 1e-14 * np.abs(M).max()
    assert abs(M.sum() - 2.0) < 1e-13  # the area: total mass is integrated exactly
    G = _mass(order, None)  # the Gauss rule: the consistent mass, not diagonal
    assert np.abs(G - np.diag(np.diag(G))).max() > 1e-3 * np.abs(G).max()


def test_gll_quadrature_keeps_poisson_spectral():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=2).quad().domain()
    u, v = d.fem_symbols(order=8)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    prob = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quadrature="gll")
    sol = np.asarray(prob.solve()).reshape(-1)
    P = np.asarray(prob.points)[:, :2]
    assert np.abs(sol - np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])).max() < 1e-7


def test_quadrature_name_is_checked():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=1).quad().domain()
    u, v = d.fem_symbols(order=2)
    x, y, _ = d.variable("interior", split=True)
    with pytest.raises(ValueError, match="quadrature="):
        jno.fem([u.bind(x=x, y=y) * v.bind(x=x, y=y)], quadrature="lobatto")


# ------------------------------------------------------------------------------------------------ hexahedra


def _poisson_hex(order, n=2):
    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=n).quad().domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols(order=order)
    x, y, z, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    f = 3 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y) * jnn.sin(np.pi * z)
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - f * vi, u(xb, yb, zb) - 0.0])


def test_p_refinement_converges_exponentially_on_hexahedra():
    errs = []
    for k in (2, 4, 6):
        prob = _poisson_hex(k)
        P = np.asarray(prob.points)
        sol = np.linalg.solve(np.asarray(prob.A), np.asarray(prob.b))
        errs.append(np.abs(sol - np.prod(np.sin(np.pi * P), axis=1)).max())
    assert errs[1] < 1e-2 * errs[0] and errs[2] < 1e-2 * errs[1] and errs[2] < 1e-6, errs


# ------------------------------------------------------------------------------ off a box: unstructured quads


def _off_box(make, order, **kw):
    d = make()
    u, v = d.fem_symbols(order=order)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = (4.0 - 0.25) * jnn.sin(2.0 * x + 0.3) * jnn.exp(0.5 * y)
    g = jnn.sin(2.0 * xb + 0.3) * jnn.exp(0.5 * yb)
    prob = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - g], quadrature="gll", **kw)
    sol = np.asarray(prob.solve(linear=jno.solve.cg(tol=1e-13), precond=jno.precond.jacobi())).reshape(-1)
    P = np.asarray(prob.points)
    return np.abs(sol - np.sin(2.0 * P[:, 0] + 0.3) * np.exp(0.5 * P[:, 1])).max()


_SHAPES = {
    "l-shape": lambda: (
        (jno.shape.rect(0, 0, 2, 2).sized(0.5) - jno.shape.rect(1, 1, 2, 2).sized(0.5))
        .structured()
        .quad()
        .domain(compute_mesh_connectivity=False)
    ),
    "polygon": lambda: (
        jno.shape.polygon([(0, 0), (2, 0.3), (2.4, 1.6), (0.9, 2.1), (-0.3, 1.2)]).sized(0.6).quad().domain()
    ),
}


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_spectral_elements_converge_off_a_box(shape):
    """The spectral-element route to geometry ``jno.spectral`` does not take: unstructured (non-affine)
    quadrilaterals, GLL nodes and quadrature, the operator applied matrix-free by sum factorisation."""
    errs = [_off_box(_SHAPES[shape], k, matrix_free=True) for k in (3, 5, 8)]
    assert errs[1] < 2e-2 * errs[0] and errs[2] < 1e-3 * errs[1] and errs[2] < 1e-9, errs
