"""Boundary-facet extraction and Dirichlet-pair construction.

Both were per-element Python loops that dominated assembly (7.9 s of a 14.1 s build on a
424k-tet mesh). These tests pin the properties the vectorised versions have to preserve:
which facets are on the boundary, their ORIENTATION, and the Dirichlet values that come out.
"""

import numpy as np
import pytest
from shapely.geometry import box

import jno
from jno.utils.solver.fem_facets import build_facet_connectivity


def test_single_triangle_is_all_boundary():
    conn = build_facet_connectivity(np.array([[0, 1, 2]]), "triangle")
    assert conn.n_bfaces == 3
    assert {tuple(sorted(f)) for f in conn.face_nodes.tolist()} == {(0, 1), (1, 2), (0, 2)}
    assert set(conn.parent_cell.tolist()) == {0}
    assert sorted(conn.local_face.tolist()) == [0, 1, 2]


def test_single_tet_is_all_boundary():
    conn = build_facet_connectivity(np.array([[0, 1, 2, 3]]), "tetrahedron")
    assert conn.n_bfaces == 4
    assert {tuple(sorted(f)) for f in conn.face_nodes.tolist()} == {(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)}


def test_shared_face_is_interior():
    """Two tets glued on (1,2,3): 8 faces total, the shared one dropped from both."""
    conn = build_facet_connectivity(np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), "tetrahedron")
    assert conn.n_bfaces == 6
    assert (1, 2, 3) not in {tuple(sorted(f)) for f in conn.face_nodes.tolist()}


def test_empty_mesh_gives_empty_arrays():
    conn = build_facet_connectivity(np.zeros((0, 3), dtype=int), "triangle")
    assert conn.n_bfaces == 0
    assert conn.face_nodes.shape == (0, 2)
    assert conn.parent_cell.shape == (0,)


def test_extra_columns_are_ignored():
    """P2 connectivity (tetra10) must give the same facets as its P1 vertex columns."""
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    p2 = np.hstack([tets, tets[:, :6] + 100])  # midside nodes the facet code must not read
    a, b = build_facet_connectivity(tets, "tetrahedron"), build_facet_connectivity(p2, "tetrahedron")
    assert np.array_equal(a.face_nodes, b.face_nodes)
    assert np.array_equal(a.parent_cell, b.parent_cell)


def test_face_nodes_keep_local_orientation_not_sorted_order():
    """Normals are computed from these, so the stored order must be the local face's, not sorted."""
    # local tet face 0 is (1, 2, 3); relabel so that ordering is NOT ascending
    conn = build_facet_connectivity(np.array([[0, 3, 2, 1]]), "tetrahedron")
    faces = [tuple(f) for f in conn.face_nodes.tolist()]
    assert (3, 2, 1) in faces, f"face 0 lost its local orientation: {faces}"


@pytest.mark.parametrize("cell_type", ["triangle", "tetrahedron"])
def test_boundary_closes_on_a_real_mesh(cell_type):
    """Every facet appears once and each is used by exactly one cell."""
    if cell_type == "triangle":
        mesh = jno.domain(box(0, 0, 1, 1), mesh_size=0.15).built_mesh
        cells = np.asarray(mesh.cells_dict["triangle"])
    else:
        mesh = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.25).domain().built_mesh
        cells = np.asarray(mesh.cells_dict["tetra"])
    conn = build_facet_connectivity(cells, cell_type)
    assert conn.n_bfaces > 0
    keys = {tuple(sorted(f)) for f in conn.face_nodes.tolist()}
    assert len(keys) == conn.n_bfaces, "a facet was reported twice"
    assert conn.parent_cell.max() < len(cells)


def test_unsupported_cell_type_is_refused():
    with pytest.raises(NotImplementedError, match="cell_type"):
        build_facet_connectivity(np.zeros((1, 8), dtype=int), "hexahedron")


# --------------------------------------------------------------------------------------------
# Dirichlet values: the batched evaluation must reproduce the per-node one
# --------------------------------------------------------------------------------------------
def _solved(fem):
    return np.asarray(fem.solve()).reshape(-1)


def test_constant_dirichlet_value_is_applied():
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 2.5], quad_degree=3)
    sol = _solved(fem)
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    on_b = np.any((nodes < 1e-9) | (nodes > 1 - 1e-9), axis=1)
    assert np.allclose(sol[on_b], 2.5), "constant Dirichlet value not applied on the boundary"


def test_spatially_varying_dirichlet_value_is_per_node():
    """g(x, y) = 1 + 2x + 3y must land node-by-node -- the case a broadcast would silently flatten."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 0.0 * vi, u(xb, yb) - (1.0 + 2.0 * xb + 3.0 * yb)], quad_degree=3)
    sol = _solved(fem)
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    on_b = np.any((nodes < 1e-9) | (nodes > 1 - 1e-9), axis=1)
    want = 1.0 + 2.0 * nodes[on_b, 0] + 3.0 * nodes[on_b, 1]
    assert np.allclose(sol[on_b], want), "varying Dirichlet profile was not evaluated per node"


def test_vector_field_constant_dirichlet_is_applied():
    """A vector field whose constant BC has as many components as there could be nodes -- the shape
    coincidence that makes 'did this scale with the batch?' undecidable from shape alone."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.25)
    u, v = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    body = jno.np.inner(jno.np.grad(ui, [xi, yi]), jno.np.grad(vi, [xi, yi]), n_contract=2) - 1.0 * vi[1]
    fem = jno.fem([body, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0], quad_degree=3)
    sol = np.asarray(fem.solve()).reshape(-1, 2)
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    on_b = np.any((nodes < 1e-9) | (nodes > 1 - 1e-9), axis=1)
    assert np.allclose(sol[on_b], 0.0), "vector Dirichlet components not pinned"
