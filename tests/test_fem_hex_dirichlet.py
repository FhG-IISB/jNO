"""A Dirichlet condition on a hexahedral mesh pins every boundary node, at any order."""

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


def _poisson_hex(order, n=2):
    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=n).quad().domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols(order=order)
    x, y, z, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    f = 3 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y) * jnn.sin(np.pi * z)
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - f * vi, u(xb, yb, zb) - 0.0])


@pytest.mark.parametrize("order", [2, 3, 4])
def test_a_dirichlet_condition_pins_every_boundary_node_of_a_hexahedral_mesh(order):
    """A hexahedron's face nodes were collected by walking its four vertices as a perimeter, but basix numbers
    them (0,0), (1,0), (0,1), (1,1): two "edges" were diagonals, and face and edge nodes went missing -- left
    free under ``u(boundary) - g`` (54 of 386 at Q4, and some already at Q2)."""
    prob = _poisson_hex(order)
    A = np.asarray(prob.A)
    P = np.asarray(prob.points)
    pinned = np.abs(A - np.diag(np.diag(A))).sum(axis=1) < 1e-12
    on_boundary = (np.isclose(P, 0.0) | np.isclose(P, 1.0)).any(axis=1)
    assert np.array_equal(pinned, on_boundary)


def test_the_second_order_hexahedral_poisson_error_is_the_interpolation_error():
    """With face and edge nodes left free the Q2 error on 2^3 cells was 4.5e-2; pinned, it is 2.3e-2."""
    prob = _poisson_hex(2)
    P = np.asarray(prob.points)
    sol = np.linalg.solve(np.asarray(prob.A), np.asarray(prob.b))
    assert np.abs(sol - np.prod(np.sin(np.pi * P), axis=1)).max() < 3e-2
