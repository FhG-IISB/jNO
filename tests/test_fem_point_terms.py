"""Point loads ``c * v(px, py, pz)`` and point values ``u(px, py, pz)`` in jno.fem.

A test function called with POINT coordinates (numbers, arrays, ``jno.np.parameter``s) is a point load:
``sum_i c_i v(p_i)``, assembled through the shape functions of the element holding each point. The
oracles: the shape functions reproduce every linear field exactly (so they are the element's own
interpolation) and are one-hot at a vertex; the heat a point source puts in is exactly what leaves
through the cooled face (partition of unity); the solution is linear in the load; and the derivative
in the point's position equals a central difference. A point evaluation that cannot be honoured
raises by name -- it is never dropped or integrated as if it were a region.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.trace import BinaryOp, PointEval, _is_point_coord
from jno.utils.solver.fem_points import PointShape

jax.config.update("jax_enable_x64", True)


def box(hexes=True):
    s = jno.shape.box(0, 0, 0, 10.0, 8.0, 2.0).structured(n=(10, 8, 4))
    return (s.quad() if hexes else s).domain(compute_mesh_connectivity=False)


def cooled(d, *load):
    """k grad T . grad v + h T v on the bottom (ambient 0), plus the given point terms."""
    T, v = d.fem_symbols()
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xs, ys, zs = d.variable("bottom", split=True)[:3]
    Ti, vi = T.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    Ts, vs = T.bind(x=xs, y=ys, z=zs), v.bind(x=xs, y=ys, z=zs)
    terms = [1.0 * (Ti.x * vi.x + Ti.y * vi.y + Ti.z * vi.z), 0.05 * Ts * vs]
    return terms, v, Ts


@pytest.mark.parametrize("hexes", [True, False])
def test_the_shape_functions_are_the_elements_own_interpolation(hexes):
    d = box(hexes)
    kind = "hexahedron" if hexes else "tetra"
    from jno.utils.solver.fem_native import _basix_ordered

    pts = np.asarray(d.mesh.points)
    cells = np.asarray(d.mesh.cells_dict[kind])
    ps = PointShape(kind, pts, cells, _basix_ordered(cells, kind), 3)
    rng = np.random.default_rng(0)
    q = rng.uniform([0.01, 0.01, 0.01], [9.99, 7.99, 1.99], (50, 3))
    cell, N = ps(jnp.asarray(q))
    nodes = np.asarray(ps.cells_basix)[np.asarray(cell)]
    f = lambda x: 1.5 + 0.3 * x[..., 0] - 0.7 * x[..., 1] + 2.0 * x[..., 2]  # noqa: E731
    assert np.allclose(np.asarray(N).sum(1), 1.0, atol=1e-12)
    assert np.allclose((np.asarray(N) * f(pts[nodes])).sum(1), f(q), atol=1e-10)
    # at a vertex: one-hot
    v0 = pts[cells[7, 0]][None]
    _c, Nv = ps(jnp.asarray(v0))
    assert np.isclose(np.asarray(Nv).max(), 1.0, atol=1e-10) and np.isclose(np.asarray(Nv).sum(), 1.0)


@pytest.mark.parametrize("hexes", [True, False])
def test_the_heat_a_point_source_puts_in_leaves_through_the_cooled_face(hexes):
    d = box(hexes)
    terms, v, Ts = cooled(d)
    P = 1.25
    fem = jno.fem(terms + [-P * v(np.array([3.3, 6.1]), np.array([4.7, 2.2]), 1.7)])
    assert "point x1" in fem.classification
    u = np.asarray(fem.solve(linear=jno.solve.lu()))
    assert float(fem.eval(0.05 * Ts, u)) == pytest.approx(2 * P, rel=1e-10)


def test_the_solution_is_linear_in_the_load():
    d = box()
    sols = []
    for P in (1.0, 3.0):
        terms, v, _ = cooled(d)
        sols.append(np.asarray(jno.fem(terms + [-P * v(4.5, 3.5, 1.0)]).solve(linear=jno.solve.lu())))
    assert np.allclose(sols[1], 3.0 * sols[0], rtol=1e-10, atol=1e-13)


def test_the_derivative_in_the_point_matches_a_central_difference():
    d = box()
    terms, v, _ = cooled(d)
    init = lambda val: lambda key, shape, dtype=jnp.float64: jnp.full(shape, val, dtype)  # noqa: E731
    P = jno.np.parameter((1,), name="P").initialize(init(1.0))
    px = jno.np.parameter((1,), name="px").initialize(init(3.3))
    fem = jno.fem(terms + [-P * v(px, 4.7, 1.7)])
    u = fem.solve()
    assert type(u).__name__ == "FemSolutionCall"
    shape_fn, cd0 = fem._op._point_space()

    def T_at(a, where):
        uv = fem._op.solve(values={"P": jnp.array([1.0]), "px": jnp.reshape(a, (1,))})
        cell, N = shape_fn(jnp.reshape(where(a), (1, 3)))
        return jnp.sum(N * uv[cd0[cell]])

    for where in (lambda a: jnp.array([5.0, 4.0, 0.0]), lambda a: jnp.stack([a, 4.7, 1.7])):
        g = float(jax.grad(T_at)(jnp.array(3.3), where))
        h = 1e-4
        fd = (float(T_at(jnp.array(3.3 + h), where)) - float(T_at(jnp.array(3.3 - h), where))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-7)


def test_the_solution_node_reads_points_and_composes():
    d = box()
    terms, v, _ = cooled(d)
    init = lambda val: lambda key, shape, dtype=jnp.float64: jnp.full(shape, val, dtype)  # noqa: E731
    P = jno.np.parameter((1,), name="P").initialize(init(1.0))
    u = jno.fem(terms + [-P * v(3.3, 4.7, 1.7)]).solve()
    a, b = u(np.array([1.0, 2.0]), 3.0, 0.5), u(9.0, 7.0, 1.5)
    assert isinstance(a - b, BinaryOp)
    uv = jnp.arange(float(np.asarray(d.mesh.points).shape[0]))
    # the readout's own function, on a known nodal vector that is linear in x: value = x-coordinate
    pts = np.asarray(d.mesh.points)
    got = a.fn(jnp.asarray(pts[:, 0]), jnp.array([1.0, 2.0]), 3.0, 0.5)
    assert np.allclose(np.asarray(got), [1.0, 2.0])
    del uv


def test_a_point_outside_the_mesh_raises_naming_it():
    d = box()
    terms, v, _ = cooled(d)
    with pytest.raises(Exception, match="outside the mesh"):
        jno.fem(terms + [-1.0 * v(12.0, 4.0, 1.0)]).solve()


def test_what_is_not_a_point_load_is_refused_by_name():
    d = box()
    terms, v, _ = cooled(d)
    T, _v = d.fem_symbols()
    with pytest.raises(NotImplementedError, match="TRIAL field at points"):
        jno.fem(terms + [T(1.0, 1.0, 1.0) * v(1.0, 1.0, 1.0)])
    xi, yi, zi = d.variable("interior", split=True)[:3]
    with pytest.raises(ValueError):
        jno.fem(terms + [xi * v(1.0, 1.0, 1.0)])


def test_a_region_binding_keeps_its_meaning():
    """Only numbers, arrays and parameters (and arithmetic of them) name points; anything built on a
    region variable -- a shifted coordinate -- is still today's region binding."""
    d = box()
    xi, yi, zi = d.variable("interior", split=True)[:3]
    assert _is_point_coord(1.0) and _is_point_coord(np.array([1.0, 2.0]))
    assert not _is_point_coord(xi) and not _is_point_coord(xi + 1.0)
    _T, v = d.fem_symbols()
    assert not isinstance(v(xi, yi, zi), PointEval)
    assert isinstance(v(1.0, 2.0, 0.5), PointEval)


def test_points_and_loads_given_as_parameter_arrays():
    """Sixteen points from (16,) parameters: arrays on no FE space, which the point assembly reads whole
    (the volume kernels would refuse them -- a parameter array there has no element to belong to)."""
    d = box()
    terms, v, Ts = cooled(d)
    n = 16
    PX, PY, PP = (jno.np.parameter((n,), name=nm) for nm in ("PX", "PY", "PP"))
    fem = jno.fem(terms + [-PP * v(PX, PY, 1.5)])
    rng = np.random.default_rng(1)
    vals = {"PX": rng.uniform(0.5, 9.5, n), "PY": rng.uniform(0.5, 7.5, n), "PP": rng.uniform(0.1, 1.0, n)}
    u = fem._op.solve(values={k: jnp.asarray(x) for k, x in vals.items()})
    assert float(fem.eval(0.05 * Ts, np.asarray(u))) == pytest.approx(vals["PP"].sum(), rel=1e-7)
