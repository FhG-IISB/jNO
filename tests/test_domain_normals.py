import numpy as np
import pytest
from shapely.geometry import box

import jno
from jno.domain.mesh_utils import MeshUtils


def test_compute_normals_from_boundary_faces_cube_outward_unit():
    # Unit cube vertices
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 12 boundary triangles (orientation may be arbitrary)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )

    normals, boundary_indices = MeshUtils._compute_normals_from_boundary_faces(points, faces)

    assert normals.shape == (8, 3)
    assert boundary_indices.shape == (8,)

    # Unit-length normals
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-7)

    # Outward check: dot with (vertex - centroid) should be positive.
    centroid = np.mean(points, axis=0)
    radial = points[boundary_indices] - centroid
    dots = np.sum(normals * radial, axis=1)
    assert np.all(dots > 0.0)


@pytest.fixture(scope="module")
def annulus_normals():
    """PCA normals on a unit square with a central square hole."""
    mesh = jno.domain(box(0.0, 0.0, 1.0, 1.0).difference(box(0.4, 0.4, 0.6, 0.6)), mesh_size=0.05).mesh
    boundary_indices = np.unique(MeshUtils._get_boundary_elements(mesh.cells_dict["triangle"], "triangle"))
    normals, idx = MeshUtils._compute_normals_pca(mesh.points, boundary_indices, 2, mesh=mesh)
    return normals, mesh.points[idx, :2]


def test_pca_normals_are_unit_length(annulus_normals):
    normals, _ = annulus_normals
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_pca_normals_point_out_of_the_material_on_both_boundaries(annulus_normals):
    """The hole is what makes this a real test.

    A normal points OUT of the material, so on the hole it points INWARD, toward the domain
    centre -- the opposite of the outer boundary. The centroid heuristic alone gets that backwards;
    only the point-in-polygon test distinguishes the two boundaries.
    """
    normals, xy = annulus_normals
    on_hole = np.all((xy > 0.39) & (xy < 0.61), axis=1)
    assert on_hole.sum() > 0 and (~on_hole).sum() > 0

    radial = np.einsum("ij,ij->i", normals, xy - 0.5)
    assert np.all(radial[~on_hole] > 0.0)  # outer boundary: away from the centre
    assert np.all(radial[on_hole] < 0.0)  # hole boundary: toward the centre


# --------------------------------------------------------------------------------------------
# the face-normal accumulation is vectorised: it must still reproduce the per-face loop
# --------------------------------------------------------------------------------------------
def _boundary_face_normals_reference(points, faces, apex_points=None):
    """The per-face Python loop ``_compute_normals_from_boundary_faces`` used to be."""
    pts = np.asarray(points[:, :3], dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0, 3)), np.array([], dtype=np.int64)
    apex = None if apex_points is None else np.asarray(apex_points[:, :3], dtype=np.float64)
    centroid, vnorm, eps = np.mean(pts, axis=0), np.zeros_like(pts), 1e-20
    for k, f in enumerate(faces):
        p0, p1, p2 = pts[int(f[0])], pts[int(f[1])], pts[int(f[2])]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < eps:
            continue
        ref = centroid if apex is None else apex[k]
        if np.dot(n, (p0 + p1 + p2) / 3.0 - ref) < 0.0:
            n = -n
        for i in (0, 1, 2):
            vnorm[int(f[i])] += n
    bidx = np.unique(faces.ravel())
    out = vnorm[bidx]
    lens = np.linalg.norm(out, axis=1, keepdims=True)
    bad = lens[:, 0] < eps
    if np.any(bad):
        radial = pts[bidx[bad]] - centroid
        rlen = np.linalg.norm(radial, axis=1, keepdims=True)
        rlen[rlen < eps] = 1.0
        out[bad] = radial / rlen
        lens[bad] = 1.0
    return out / lens, bidx


@pytest.mark.parametrize("with_apex", [False, True])
def test_boundary_face_normals_match_the_per_face_loop(with_apex):
    """A real tet mesh, with and without the apex orientation (the concave-safe path)."""
    from jno.utils.solver.fem_facets import _LOCAL_FACES_TET, _boundary_faces

    mesh = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.25).domain().built_mesh
    pts, cells = np.asarray(mesh.points), np.asarray(mesh.cells_dict["tetra"], dtype=np.int64)
    flat, sel, n_local = _boundary_faces(cells, _LOCAL_FACES_TET, 3)
    faces = flat[sel]
    apex = None
    if with_apex:
        apex_local = np.asarray([e[3] for e in _LOCAL_FACES_TET], dtype=np.int64)
        apex = pts[cells[sel // n_local, apex_local[sel % n_local]]]

    got_n, got_i = MeshUtils._compute_normals_from_boundary_faces(pts, faces, apex_points=apex)
    ref_n, ref_i = _boundary_face_normals_reference(pts, faces, apex_points=apex)
    assert np.array_equal(got_i, ref_i)
    np.testing.assert_allclose(got_n, ref_n, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(got_n, axis=1), 1.0, atol=1e-12)


def test_boundary_face_normals_skip_a_degenerate_face():
    """A zero-area face contributes nothing -- the vectorised form masks where the loop `continue`d."""
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1.0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 1]], dtype=np.int64)  # last is degenerate
    got, gi = MeshUtils._compute_normals_from_boundary_faces(pts, faces)
    ref, ri = _boundary_face_normals_reference(pts, faces)
    assert np.array_equal(gi, ri)
    np.testing.assert_allclose(got, ref, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-12)


def test_boundary_face_normals_of_an_empty_mesh_are_empty():
    n, idx = MeshUtils._compute_normals_from_boundary_faces(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert n.shape == (0, 3) and idx.shape == (0,)


def test_coordinate_tag_gets_per_point_normals_like_a_facet_tag():
    """A boundary named by an ordinary COORDINATE predicate must carry per-point normals, exactly as
    one named by a facet predicate does.

    It did not: ``normals_by_tag`` was populated only by the mesh cell-set path and by
    ``_tag_by_facet``, so a coordinate-tagged boundary had a region but no normals — and every reader
    of that dict saw a *silently missing* entry rather than an error. This is what broke RCWA's
    superstrate/substrate detection once a tag was re-derived from a predicate (a remesh, or
    ``_domain_from_arrays``) instead of coming from the mesh's cell sets."""
    d = jno.domain(jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 2.0, size=0.5).domain())
    e = 1e-6
    d.tag("zbot", lambda x, y, z: z < e)
    d.tag("ztop", lambda x, y, z: z > 2.0 - e)
    d.tag("xlo", lambda x, y, z: x < e)

    for tag, expect in (("zbot", [0, 0, -1]), ("ztop", [0, 0, 1]), ("xlo", [-1, 0, 0])):
        assert tag in d.normals_by_tag, f"{tag}: a coordinate-tagged boundary must carry normals"
        n = np.asarray(d.normals_by_tag[tag])
        assert n.shape == (np.asarray(d._boundary_regions[tag].points).shape[0], 3), "one normal per region point"
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9), "normals must be unit"
        assert np.allclose(n.mean(0), expect, atol=1e-9), f"{tag}: outward normal should be {expect}, got {n.mean(0)}"


def test_coordinate_tag_normals_survive_a_rebuild_from_arrays():
    """The exact regression path: ``_domain_from_arrays`` captures the template's named regions as
    predicates and re-derives them on the new mesh. Those re-derived tags must still carry normals —
    they previously did not, which is how a face became invisible to anything reading them."""
    from jno.utils.solver.fem_adapt import _domain_from_arrays

    src = jno.domain(jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 2.0, size=0.5).domain())
    pts = np.asarray(src.mesh.points)
    tets = np.asarray(src.mesh.cells_dict["tetra"])
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    uq, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    bf = uq[cnt == 1]

    d = _domain_from_arrays(src, pts, tets, bf, copy=True)
    d.tag("ztop", lambda x, y, z: z > 2.0 - 1e-6)
    assert "ztop" in d.normals_by_tag, "a tag on a rebuilt domain must carry normals"
    n = np.asarray(d.normals_by_tag["ztop"])
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
    assert np.allclose(n.mean(0), [0, 0, 1], atol=1e-9), f"outward +z expected, got {n.mean(0)}"


@pytest.mark.parametrize(
    "shape_of",
    [
        lambda: jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1 / 16),
        lambda: jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=1 / 24),
        lambda: jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=1 / 8),
    ],
)
def test_the_lattice_boundary_hint_changes_nothing(shape_of):
    """A lattice knows its boundary nodes in closed form, and `_boundary_faces` uses that to prune the
    faces it counts -- a boundary face's nodes all lie on the boundary, so a face with any other node is
    interior. The pruning must be invisible: same faces, same order, same normals, same node ids.

    It is what makes the count linear in the PERIMETER instead of in the cells: 25M faces become 8k at
    4.2M nodes, and a 3-D structured build went from 6.92 s to 3.07 s at 2.1M nodes."""
    from jno.domain.mesh_utils import p1_cells_dict
    from jno.utils.solver.fem_facets import _FACET_CACHE, _LOCAL_FACES_TET, _LOCAL_FACES_TRI, _boundary_faces

    d = shape_of().structured().domain()
    points = np.asarray(d.mesh.points)[:, : d.dimension]
    hint = d._lattice_boundary_nodes(points)
    assert hint is not None, "a full box lattice must know its boundary nodes"
    # the hint is exactly the boundary of the bounding box, so it may only ever over-cover
    assert hint.sum() < len(points)

    cells_dict = p1_cells_dict(d.mesh)
    table, n_face_nodes, key = (_LOCAL_FACES_TRI, 2, "triangle") if d.dimension == 2 else (_LOCAL_FACES_TET, 3, "tetra")
    cells = np.asarray(cells_dict[key], dtype=np.int64)

    _FACET_CACHE.clear()
    flat_a, sel_a, n_a = _boundary_faces(cells, table, n_face_nodes)
    _FACET_CACHE.clear()
    flat_b, sel_b, n_b = _boundary_faces(cells, table, n_face_nodes, hint)
    assert n_a == n_b and np.array_equal(flat_a, flat_b) and np.array_equal(sel_a, sel_b)

    _FACET_CACHE.clear()
    normals_a, idx_a = MeshUtils.get_boundary_normals(d.mesh)
    _FACET_CACHE.clear()
    normals_b, idx_b = MeshUtils.get_boundary_normals(d.mesh, boundary_nodes=hint)
    assert np.array_equal(idx_a, idx_b)
    np.testing.assert_array_equal(np.asarray(normals_a), np.asarray(normals_b))


def test_a_mesh_that_is_not_a_lattice_gets_no_boundary_hint():
    """The hint is only sound where the grid FILLS its bounding box: a lattice masked to some other shape
    has boundary nodes strictly inside the box, and a predicate that missed them would drop real boundary
    faces. An unstructured mesh has no grid at all, so it declines."""
    d = jno.shape.disk(0.5, 0.5, 0.4, size=0.05).domain()
    points = np.asarray(d.mesh.points)[:, : d.dimension]
    assert d._lattice_boundary_nodes(points) is None
    with pytest.raises(NotImplementedError, match="single axis-aligned rect/box"):
        jno.shape.disk(0.5, 0.5, 0.4, size=0.05).structured().domain()
