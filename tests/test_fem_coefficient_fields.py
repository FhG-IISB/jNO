"""Coefficient fields on any element space: a known field ``e.bind(...).freeze(values)`` or a field parameter
``jno.np.parameter(e)`` of an N1E / N2E / RT / DG / P_k symbol, inside a non-nodal form.

The coefficient is evaluated with its OWN space's basis and Piola map, so the oracles are that space's own
matrices: a load ``(E, v)`` with ``E`` in the unknown's space is ``M h`` (M the mass matrix, h the DOF values),
``(curl E, curl v)`` is ``K h``. Independently of jNO's own matrices: a constant vector interpolated into N1E_1
(edge circulations) is the constant again, and a P_2 field of a linear function is that function.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_dofmap import build_dofmap  # noqa: E402
from jno.utils.solver.fem_topology import BASIX_TET_EDGES, build_edge_topology  # noqa: E402

inner = jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def box():
    return jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.5).domain()


def _xyz(d):
    c = d.variable("interior", split=True)
    return c[0], c[1], c[2]


def _space(d, space, order=1, value_shape=(3,), names=("a", "b"), xyz=None):
    x, y, z = xyz or _xyz(d)
    u, v = d.fem_symbols(value_shape=value_shape, names=names, space=space, order=order)
    return u, v, u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)


def _Ab(op, args=None):
    out = op.evaluate(args) if hasattr(op, "evaluate") else op
    A, b = out[0], out[1]
    return np.asarray(A.todense()), np.asarray(b)


def _n_dofs(d, fam, k):
    cells = np.asarray(d.mesh.cells_dict["tetra"])
    return int(build_dofmap(cells, fam, k, n_verts=len(d.mesh.points)).n_dofs)


@pytest.mark.parametrize("space,order", [("N1E", 1), ("N1E", 2), ("RT", 1)])
def test_frozen_field_load_is_the_mass_times_its_dofs(box, space, order):
    d = box
    u, v, A, V = _space(d, space, order, names=("u", "v"))
    e, _f, _E, _F = _space(d, space, order, names=("e", "f"))
    n = _n_dofs(d, space, order)
    h = np.random.default_rng(0).normal(size=n)
    x, y, z = _xyz(d)
    E = e.bind(x=x, y=y, z=z).freeze(h)
    M, _ = _Ab(jno.fem([inner(A, V)]).operator)
    _, b = _Ab(jno.fem([inner(A, V) - inner(E, V)]).operator)
    np.testing.assert_allclose(b, M @ h, rtol=1e-10, atol=1e-12 * np.abs(M @ h).max())


def test_frozen_n1e_field_of_a_constant_is_the_constant(box):
    """edge circulations of c (DOF e = c . (P[hi] - P[lo]) for the edge lo -> hi) give back c: the load equals
    that of the coordinate-function coefficient c"""
    d = box
    u, v, A, V = _space(d, "N1E", names=("u", "v"))
    e, _f, _E, _F = _space(d, "N1E", names=("e", "f"))
    P = np.asarray(d.mesh.points)
    ev = np.asarray(build_edge_topology(np.asarray(d.mesh.cells_dict["tetra"]), BASIX_TET_EDGES).edge_vertices)
    c = np.array([0.3, -1.2, 2.0])
    h = (P[ev[:, 1]] - P[ev[:, 0]]) @ c
    x, y, z = _xyz(d)
    E = e.bind(x=x, y=y, z=z).freeze(h)
    C = jno.np.vector(c[0] + 0.0 * x, c[1] + 0.0 * x, c[2] + 0.0 * x)
    _, b_field = _Ab(jno.fem([inner(A, V) - inner(E, V)]).operator)
    _, b_fn = _Ab(jno.fem([inner(A, V) - inner(C, V)]).operator)
    np.testing.assert_allclose(b_field, b_fn, rtol=1e-10, atol=1e-12 * np.abs(b_fn).max())


def test_curl_of_a_frozen_n1e_field_is_the_curl_curl_matrix_times_its_dofs(box):
    d = box
    u, v, A, V = _space(d, "N1E", names=("u", "v"))
    e, _f, _E, _F = _space(d, "N1E", names=("e", "f"))
    x, y, z = _xyz(d)
    h = np.random.default_rng(1).normal(size=_n_dofs(d, "N1E", 1))
    cV = v.vector.curl(x, y, z)
    K, _ = _Ab(jno.fem([inner(u.vector.curl(x, y, z), cV)]).operator)
    cE = e.bind(x=x, y=y, z=z).freeze(h).curl(x, y, z)
    _, b = _Ab(jno.fem([inner(u.vector.curl(x, y, z), cV) - inner(cE, cV)]).operator)
    np.testing.assert_allclose(b, K @ h, rtol=1e-9, atol=1e-11 * np.abs(K @ h).max())


def test_n1e_field_parameter_load_and_gradient(box):
    """the parameter route: the same load as the frozen field, linear in the DOFs, gradient M^T w"""
    d = box
    u, v, A, V = _space(d, "N1E", names=("u", "v"))
    e, _f, _E, _F = _space(d, "N1E", names=("e", "f"))
    n = _n_dofs(d, "N1E", 1)
    Ep = jno.np.parameter(e, name="Ep")
    Ep.dtype(jnp.float64)
    op = jno.fem([inner(A, V) - inner(Ep, V)]).operator
    M, _ = _Ab(jno.fem([inner(A, V)]).operator)
    h = jnp.asarray(np.random.default_rng(2).normal(size=n))
    _, b = _Ab(op, {"Ep": h})
    np.testing.assert_allclose(b, M @ np.asarray(h), rtol=1e-10, atol=1e-12 * np.abs(b).max())
    w = jnp.asarray(np.random.default_rng(3).normal(size=n))
    g = jax.grad(lambda t: w @ op.evaluate({"Ep": t})[1])(h)
    np.testing.assert_allclose(np.asarray(g), M.T @ np.asarray(w), rtol=1e-10, atol=1e-12 * np.abs(g).max())
    with pytest.raises(ValueError, match="one per DOF"):
        op.evaluate({"Ep": jnp.ones(len(d.mesh.points))})


def test_p2_scalar_coefficient_of_a_linear_function(box):
    """k = 1 + 2x - z as P_2 DOF values (vertex values, then edge midpoints) scales the N1E mass exactly as the
    coordinate function does, at the same quadrature"""
    d = box
    x, y, z = _xyz(d)
    u, v, A, V = _space(d, "N1E", names=("u", "v"), xyz=(x, y, z))
    k_sym, _q = d.fem_symbols(names=("k", "q"), order=2)
    P = np.asarray(d.mesh.points)
    cells = np.asarray(d.mesh.cells_dict["tetra"])
    ev = np.asarray(build_edge_topology(cells, BASIX_TET_EDGES).edge_vertices)
    dm = build_dofmap(
        cells, "Lagrange", 2, n_verts=len(P), edges=(build_edge_topology(cells, BASIX_TET_EDGES).cell_edges, ev)
    )
    X = np.concatenate([P, 0.5 * (P[ev[:, 0]] + P[ev[:, 1]])])
    assert dm.n_dofs == len(X)
    f = lambda x_, y_, z_: 1.0 + 2.0 * x_ - z_  # noqa: E731
    kP2 = k_sym.bind(x=x, y=y, z=z).freeze(f(X[:, 0], X[:, 1], X[:, 2]))
    K_field, _ = _Ab(jno.fem([kP2 * inner(A, V)], quad_degree=6).operator)
    K_fn, _ = _Ab(jno.fem([f(x, y, z) * inner(A, V)], quad_degree=6).operator)
    np.testing.assert_allclose(K_field, K_fn, rtol=1e-10, atol=1e-12 * np.abs(K_fn).max())


def test_space_field_parameter_refused_by_the_nodal_assembler(box):
    d = box
    u, v = d.fem_symbols(names=("u", "v"))
    e, _f = d.fem_symbols(value_shape=(3,), names=("e", "f"), space="N1E")
    Ep = jno.np.parameter(e, name="Ep")
    x, y, z = _xyz(d)
    U, Vv = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    with pytest.raises(NotImplementedError, match="element space"):
        jno.fem([U * Vv + Ep[0] * Vv]).operator
