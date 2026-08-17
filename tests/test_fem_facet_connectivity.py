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


def test_tensor_product_cells_are_supported():
    """Quadrilaterals and hexahedra go through the same sort+unique as the simplices.

    This assertion used to be the opposite one -- ``"hexahedron"`` was in the refusal list. The
    boundary detection itself never needed changing (a facet is on the boundary if its sorted node
    ids occur once, whatever the cell), only the local-facet table it is handed.
    """
    quad = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)  # two quads sharing edge (1, 4)
    conn = build_facet_connectivity(quad, "quadrilateral")
    assert conn.n_bfaces == 6 and conn.face_nodes.shape[1] == 2  # 8 edges - 2 shared
    assert {tuple(sorted(f)) for f in conn.face_nodes.tolist()}.isdisjoint({(1, 4)})

    hexa = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)  # a single hexahedron
    conn = build_facet_connectivity(hexa, "hexahedron")
    assert conn.n_bfaces == 6 and conn.face_nodes.shape[1] == 4


def test_unsupported_cell_type_is_refused():
    """A cell with no facet table still refuses by name rather than guessing one."""
    with pytest.raises(NotImplementedError, match="cell_type"):
        build_facet_connectivity(np.zeros((1, 6), dtype=int), "prism")


# --------------------------------------------------------------------------------------------
# the shared computation: domain build and assembly must agree, and pay for it once
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cells,cell_type",
    [
        (np.array([[0, 1, 2]]), "triangle"),
        (np.zeros((0, 3), dtype=int), "triangle"),
        (np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), "tetra"),
    ],
)
def test_shared_boundary_set_matches_the_independent_implementation(cells, cell_type):
    from jno.domain.mesh_utils import MeshUtils

    shared = MeshUtils._get_boundary_elements(cells, cell_type)
    reference = MeshUtils._get_boundary_elements_reference(cells, cell_type)
    assert np.array_equal(shared, reference), "shared path diverged from the sort+unique oracle"


def test_shared_boundary_set_matches_on_a_real_mesh():
    from jno.domain.mesh_utils import MeshUtils

    cells = np.asarray(jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain().built_mesh.cells_dict["tetra"])
    assert np.array_equal(
        MeshUtils._get_boundary_elements(cells, "tetra"),
        MeshUtils._get_boundary_elements_reference(cells, "tetra"),
    )


def test_the_face_computation_is_shared_between_equal_but_distinct_arrays():
    """The domain build and the assembler reach this with different array OBJECTS holding the same
    connectivity, so an identity-keyed cache scored 0% and both paid. Content keying is the fix."""
    from jno.utils.solver import fem_facets

    cells = np.asarray(jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain().built_mesh.cells_dict["tetra"])
    fem_facets._FACET_CACHE.clear()
    fem_facets.boundary_face_set(cells, "tetra")
    fem_facets.build_facet_connectivity(cells.copy(), "tetrahedron")  # a DIFFERENT object
    assert len(fem_facets._FACET_CACHE) == 1, "equal connectivity was computed twice"


def test_the_cache_does_not_confuse_different_meshes():
    from jno.utils.solver import fem_facets

    fem_facets._FACET_CACHE.clear()
    one = fem_facets.boundary_face_set(np.array([[0, 1, 2, 3]]), "tetra")
    two = fem_facets.boundary_face_set(np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), "tetra")
    assert len(one) == 4 and len(two) == 6, "a different mesh got a stale cached answer"


# --------------------------------------------------------------------------------------------
# packed face keys: the shared unique must be row-for-row what np.unique(axis=0) gives
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cells,dim,order",
    [
        (None, 2, 1),  # real 2-D mesh, P1
        (None, 2, 2),  # real 2-D mesh, P2 (the facet-DOF gather)
        (None, 3, 1),  # real 3-D mesh
        (np.array([[0, 1, 2]]), 2, 1),  # single cell
        (np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), 3, 1),  # shared interior face
    ],
)
def test_boundary_facets_match_the_row_wise_unique(cells, dim, order):
    """``pack_face_keys`` must reproduce ``np.unique(axis=0)`` EXACTLY -- same rows, same order --
    not merely the same set: callers index ``allf`` by the first-occurrence index it returns."""
    from jno._fem import _boundary_facets
    from jno.utils.solver.fem_facets import pack_face_keys

    if cells is None:
        if dim == 2:
            mesh = jno.domain(box(0, 0, 1, 1), mesh_size=0.15).built_mesh
            pts, cells = np.asarray(mesh.points), np.asarray(mesh.cells_dict["triangle"])
        else:
            mesh = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain().built_mesh
            pts, cells = np.asarray(mesh.points), np.asarray(mesh.cells_dict["tetra"])
        if order >= 2:  # P2 connectivity: vertices first, then one midpoint per edge
            e = np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
            uniq, inv = np.unique(e[:, 0] * (cells.max() + 1) + e[:, 1], return_inverse=True)
            mids = len(pts) + inv.reshape(3, -1).T
            pts = np.vstack([pts, np.zeros((len(uniq), pts.shape[1]))])
            cells = np.hstack([cells, mids])
    else:
        pts = np.zeros((int(cells.max()) + 1, dim))

    verts = np.asarray(cells)[:, : dim + 1]
    combos = [(0, 1), (1, 2), (2, 0)] if dim == 2 else [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    allf = np.concatenate([verts[:, list(c)] for c in combos], axis=0)
    canonical = np.sort(allf, axis=1)

    keys = pack_face_keys(canonical)
    _u, i_packed, c_packed = np.unique(keys, return_index=True, return_counts=True)
    _u, i_rows, c_rows = np.unique(canonical, axis=0, return_index=True, return_counts=True)
    assert np.array_equal(i_packed[c_packed == 1], i_rows[c_rows == 1]), "packed key reordered the facets"

    got = _boundary_facets(pts, cells, dim, order)
    assert got is not None and got.shape[0] == int((c_rows == 1).sum())
    assert np.array_equal(got[:, :dim], allf[i_rows[c_rows == 1]])  # a facet has `dim` vertices


def test_pack_face_keys_declines_when_the_key_would_overflow():
    """Ids too large to pack must return None so the caller falls back, not silently collide."""
    from jno.utils.solver.fem_facets import pack_face_keys

    assert pack_face_keys(np.array([[0, 1, 2]], dtype=np.int64)) is not None
    assert pack_face_keys(np.array([[0, 0, 10**7]], dtype=np.int64)) is None  # (1e7)^3 = 1e21 > 2^62
    assert pack_face_keys(np.zeros((0, 3), dtype=np.int64)).shape == (0,)


# --------------------------------------------------------------------------------------------
# outward face normals: the array expression must reproduce the per-face loop it replaced
# --------------------------------------------------------------------------------------------
def _face_normals_reference(points, conn, cells, cell_type):
    """The per-face Python loop ``compute_face_normals`` used to be, kept as the oracle."""
    from jno.utils.solver.fem_facets import _LOCAL_FACES_TET, _LOCAL_FACES_TRI

    points, cells = np.asarray(points), np.asarray(cells)
    local_faces, n_face_nodes, dim = (_LOCAL_FACES_TRI, 2, 2) if cell_type == "triangle" else (_LOCAL_FACES_TET, 3, 3)
    normals = np.empty((conn.n_bfaces, dim))
    for i in range(conn.n_bfaces):
        entry = local_faces[int(conn.local_face[i])]
        c = int(conn.parent_cell[i])
        verts = points[[int(cells[c, j]) for j in entry[:n_face_nodes]], :dim]
        opp = points[int(cells[c, entry[n_face_nodes]]), :dim]
        if dim == 2:
            t = verts[1] - verts[0]
            n = np.array([t[1], -t[0]])
        else:
            n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        if np.dot(n, np.mean(verts, axis=0) - opp) < 0:
            n = -n
        normals[i] = n / np.linalg.norm(n)
    return normals


@pytest.mark.parametrize(
    "shape,cell_key,cell_type",
    [
        (jno.Shape.rect(0, 0, 1, 1, size=0.12), "triangle", "triangle"),
        (jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3), "tetra", "tetrahedron"),
    ],
)
def test_face_normals_match_the_per_face_loop(shape, cell_key, cell_type):
    from jno.utils.solver.fem_facets import compute_face_normals

    mesh = shape.domain().built_mesh
    pts, cells = np.asarray(mesh.points), np.asarray(mesh.cells_dict[cell_key])
    conn = build_facet_connectivity(cells, cell_type)
    assert conn.n_bfaces > 0
    np.testing.assert_allclose(
        compute_face_normals(pts, conn, cells, cell_type),
        _face_normals_reference(pts, conn, cells, cell_type),
        atol=1e-14,
    )


def test_face_normals_on_a_concave_domain_still_point_out():
    """An L-shape: orientation comes from the owning cell's apex, so the reentrant corner is
    handled by construction -- a centroid-based rule would flip normals there."""
    from shapely.geometry import Polygon

    from jno.utils.solver.fem_facets import compute_face_normals

    poly = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    mesh = jno.domain(poly, mesh_size=0.15).built_mesh
    pts, cells = np.asarray(mesh.points), np.asarray(mesh.cells_dict["triangle"])
    conn = build_facet_connectivity(cells, "triangle")
    n = compute_face_normals(pts, conn, cells, "triangle")
    np.testing.assert_allclose(n, _face_normals_reference(pts, conn, cells, "triangle"), atol=1e-14)
    # every normal points away from its own cell's interior, concave corner included
    apex = np.array([pts[cells[int(conn.parent_cell[i])]].mean(axis=0)[:2] for i in range(conn.n_bfaces)])
    mid = pts[conn.face_nodes][:, :, :2].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", n, mid - apex) > 0)


@pytest.mark.parametrize("cell_type,dim", [("triangle", 2), ("tetrahedron", 3)])
def test_face_normals_of_an_empty_boundary_are_empty(cell_type, dim):
    from jno.utils.solver.fem_facets import compute_face_normals

    conn = build_facet_connectivity(np.zeros((0, dim + 1), dtype=int), cell_type)
    out = compute_face_normals(np.zeros((0, dim)), conn, np.zeros((0, dim + 1), dtype=int), cell_type)
    assert out.shape == (0, dim) and out.dtype == np.float64


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
