"""Tests for jno.domain loading from existing mesh files.

Key finding: Abaqus ``.inp`` is the only mesh format (without optional heavy
dependencies such as h5py) that round-trips ``cell_sets`` through meshio —
VTK/VTU convert them to ``cell_data`` and discard tag names.  All
tag-extraction and geometry tests therefore use ``.inp``.

These tests verify that passing a file path to ``jno.domain(constructor=path)``
correctly:

* reads the mesh via meshio
* infers spatial dimension from point coordinates
* populates ``avaiable_mesh_tags`` and ``_mesh_pool``
* stores points with the correct shape and within expected bounds
* raises on a missing file path
* populates ``context`` with shape ``(B=1, T=1, N, D=2)`` after ``variable()``
* returns Variable objects for each spatial + temporal dimension
* correctly sub-samples on request
"""

import numpy as np
import pytest
import meshio

import jno


# ---------------------------------------------------------------------------
# Shared mesh factory (pure numpy, no pygmsh/gmsh required)
# ---------------------------------------------------------------------------


def _make_2d_mesh(nx: int = 4, ny: int = 4, x_range=(0.0, 1.0), y_range=(0.0, 1.0)) -> meshio.Mesh:
    """Structured triangular mesh with named boundary ``cell_sets``.

    Mirrors ``Geometries.equi_distant_rect`` so the resulting ``.inp`` file
    exercises the same code paths as a gmsh-generated mesh.

    Tags created: ``interior``, ``boundary``, ``bottom``, ``top``,
    ``left``, ``right``.
    """
    x0, x1 = x_range
    y0, y1 = y_range

    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros((nx + 1) * (ny + 1))])

    def idx(i, j):
        return i * (ny + 1) + j

    triangles = []
    for i in range(nx):
        for j in range(ny):
            p0, p1 = idx(i, j), idx(i + 1, j)
            p2, p3 = idx(i + 1, j + 1), idx(i, j + 1)
            triangles += [[p0, p1, p2], [p0, p2, p3]]
    triangles = np.array(triangles)

    bot = np.array([[idx(i, 0), idx(i + 1, 0)] for i in range(nx)])
    top = np.array([[idx(i, ny), idx(i + 1, ny)] for i in range(nx)])
    lft = np.array([[idx(0, j), idx(0, j + 1)] for j in range(ny)])
    rgt = np.array([[idx(nx, j), idx(nx, j + 1)] for j in range(ny)])
    all_edges = np.vstack([bot, top, lft, rgt])

    nb, nt, nl, nr = len(bot), len(top), len(lft), len(rgt)
    cell_sets = {
        "interior": [np.arange(len(triangles)), np.array([], dtype=np.int64)],
        "boundary": [np.array([], dtype=np.int64), np.arange(len(all_edges))],
        "bottom": [np.array([], dtype=np.int64), np.arange(0, nb)],
        "top": [np.array([], dtype=np.int64), np.arange(nb, nb + nt)],
        "left": [np.array([], dtype=np.int64), np.arange(nb + nt, nb + nt + nl)],
        "right": [np.array([], dtype=np.int64), np.arange(nb + nt + nl, nb + nt + nl + nr)],
    }
    return meshio.Mesh(
        points=pts,
        cells=[("triangle", triangles), ("line", all_edges)],
        cell_sets=cell_sets,
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inp_2d_path(tmp_path_factory):
    """Write a small 2-D Abaqus .inp mesh and return its path."""
    path = str(tmp_path_factory.mktemp("meshes") / "rect_4x4.inp")
    meshio.write(path, _make_2d_mesh(nx=4, ny=4))
    return path


@pytest.fixture(scope="module")
def inp_2d_domain(inp_2d_path):
    """``jno.domain`` loaded from a 2-D .inp file (no mesh connectivity)."""
    return jno.domain(constructor=inp_2d_path, compute_mesh_connectivity=False)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestLoadErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            jno.domain(
                constructor=str(tmp_path / "does_not_exist.inp"),
                compute_mesh_connectivity=False,
            )

    def test_invalid_constructor_type_raises(self):
        with pytest.raises((ValueError, TypeError)):
            jno.domain(constructor=42, compute_mesh_connectivity=False)  # type: ignore


# ---------------------------------------------------------------------------
# Basic mesh properties after loading
# ---------------------------------------------------------------------------


class TestLoadedMeshProperties:
    def test_mesh_object_is_set(self, inp_2d_domain):
        assert inp_2d_domain.mesh is not None

    def test_dimension_is_2(self, inp_2d_domain):
        """z-column with all zeros → dimension inferred as 2."""
        assert inp_2d_domain.dimension == 2

    def test_total_mesh_point_count(self, inp_2d_domain):
        """4 × 4 grid has (4+1) × (4+1) = 25 vertices."""
        assert len(inp_2d_domain.mesh.points) == 25

    def test_points_attribute_is_2d(self, inp_2d_domain):
        pts = inp_2d_domain.points
        assert pts.ndim == 2
        assert pts.shape[1] == 2  # spatial only, z-column dropped

    def test_points_within_unit_square(self, inp_2d_domain):
        pts = inp_2d_domain.points
        assert np.all(pts[:, 0] >= -1e-10) and np.all(pts[:, 0] <= 1.0 + 1e-10)
        assert np.all(pts[:, 1] >= -1e-10) and np.all(pts[:, 1] <= 1.0 + 1e-10)

    def test_stationary_time_context_present(self, inp_2d_domain):
        assert "__time__" in inp_2d_domain.context


# ---------------------------------------------------------------------------
# Tag discovery
# ---------------------------------------------------------------------------


class TestTagDiscovery:
    def test_available_mesh_tags_non_empty(self, inp_2d_domain):
        assert len(inp_2d_domain.avaiable_mesh_tags) > 0

    def test_interior_tag_found(self, inp_2d_domain):
        assert "interior" in inp_2d_domain.avaiable_mesh_tags

    def test_all_boundary_tags_found(self, inp_2d_domain):
        expected = {"bottom", "top", "left", "right", "boundary"}
        found = set(inp_2d_domain.avaiable_mesh_tags)
        assert expected.issubset(found)

    def test_mesh_pool_keys_match_available_tags(self, inp_2d_domain):
        for tag in inp_2d_domain.avaiable_mesh_tags:
            assert tag in inp_2d_domain._mesh_pool, f"'{tag}' missing from _mesh_pool"

    def test_no_gmsh_internal_tags(self, inp_2d_domain):
        """No 'gmsh:…' tags should leak into the public tag list."""
        for tag in inp_2d_domain.avaiable_mesh_tags:
            assert not tag.startswith("gmsh:")


# ---------------------------------------------------------------------------
# _mesh_pool shapes and geometry
# ---------------------------------------------------------------------------


class TestMeshPoolShapes:
    def test_interior_pool_ndim_is_2(self, inp_2d_domain):
        assert inp_2d_domain._mesh_pool["interior"].ndim == 2

    def test_interior_pool_has_2_spatial_cols(self, inp_2d_domain):
        assert inp_2d_domain._mesh_pool["interior"].shape[1] == 2

    def test_interior_pool_non_empty(self, inp_2d_domain):
        assert inp_2d_domain._mesh_pool["interior"].shape[0] > 0

    def test_all_tagged_pools_have_2_cols(self, inp_2d_domain):
        for tag in inp_2d_domain.avaiable_mesh_tags:
            pts = inp_2d_domain._mesh_pool[tag]
            assert pts.shape[1] == 2, f"'{tag}' pool has {pts.shape[1]} cols"

    def test_boundary_pool_points_on_perimeter(self, inp_2d_domain):
        """Every point in the 'boundary' pool must lie on the unit-square perimeter."""
        pts = inp_2d_domain._mesh_pool["boundary"]
        on_edge = np.isclose(pts[:, 0], 0.0, atol=1e-10) | np.isclose(pts[:, 0], 1.0, atol=1e-10) | np.isclose(pts[:, 1], 0.0, atol=1e-10) | np.isclose(pts[:, 1], 1.0, atol=1e-10)
        assert np.all(on_edge), f"{np.sum(~on_edge)} / {len(on_edge)} boundary points are NOT on the perimeter"

    def test_left_pool_x_equals_zero(self, inp_2d_domain):
        pts = inp_2d_domain._mesh_pool["left"]
        assert np.allclose(pts[:, 0], 0.0, atol=1e-10)

    def test_right_pool_x_equals_one(self, inp_2d_domain):
        pts = inp_2d_domain._mesh_pool["right"]
        assert np.allclose(pts[:, 0], 1.0, atol=1e-10)

    def test_bottom_pool_y_equals_zero(self, inp_2d_domain):
        pts = inp_2d_domain._mesh_pool["bottom"]
        assert np.allclose(pts[:, 1], 0.0, atol=1e-10)

    def test_top_pool_y_equals_one(self, inp_2d_domain):
        pts = inp_2d_domain._mesh_pool["top"]
        assert np.allclose(pts[:, 1], 1.0, atol=1e-10)

    def test_boundary_has_fewer_points_than_interior(self, inp_2d_domain):
        n_int = inp_2d_domain._mesh_pool["interior"].shape[0]
        n_bnd = inp_2d_domain._mesh_pool["boundary"].shape[0]
        assert n_bnd < n_int


# ---------------------------------------------------------------------------
# variable() — context shapes and content
# ---------------------------------------------------------------------------


class TestVariable:
    def test_variable_returns_three_objects_for_2d(self, tmp_path):
        """x, y, t — three Variable placeholders for a 2-D mesh."""
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        result = dom.variable("interior")
        assert len(result) == 3

    def test_variable_populates_context(self, tmp_path):
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        dom.variable("interior")
        assert "interior" in dom.context

    def test_context_interior_shape_is_B1_T1_N_D2(self, tmp_path):
        """After variable(), context['interior'] must be (1, 1, N, 2)."""
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=4, ny=4))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        dom.variable("interior")
        arr = dom.context["interior"]
        assert arr.ndim == 4, f"Expected 4-D (B,T,N,D), got {arr.shape}"
        B, T, N, D = arr.shape
        assert B == 1
        assert T == 1
        assert D == 2

    def test_context_boundary_shape_is_B1_T1_N_D2(self, tmp_path):
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=4, ny=4))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        dom.variable("boundary")
        B, T, N, D = dom.context["boundary"].shape
        assert B == 1 and T == 1 and D == 2

    def test_context_n_matches_mesh_pool_n(self, tmp_path):
        """N in context must match the number of points in _mesh_pool."""
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=6, ny=6))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        dom.variable("interior")
        n_pool = dom._mesh_pool["interior"].shape[0]
        n_ctx = dom.context["interior"].shape[2]
        assert n_ctx == n_pool

    def test_interior_coord_range(self, tmp_path):
        """Coordinates from context must lie inside the unit square."""
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=4, ny=4))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        dom.variable("interior")
        pts = np.asarray(dom.context["interior"][0, 0, :, :])  # (N, 2)
        assert np.all(pts[:, 0] >= -1e-10) and np.all(pts[:, 0] <= 1.0 + 1e-10)
        assert np.all(pts[:, 1] >= -1e-10) and np.all(pts[:, 1] <= 1.0 + 1e-10)

    def test_boundary_points_on_perimeter_after_variable(self, tmp_path):
        """Boundary context points must lie on the unit-square perimeter."""
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=4, ny=4))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        dom.variable("boundary")
        pts = np.asarray(dom.context["boundary"][0, 0, :, :])
        on_edge = np.isclose(pts[:, 0], 0.0, atol=1e-10) | np.isclose(pts[:, 0], 1.0, atol=1e-10) | np.isclose(pts[:, 1], 0.0, atol=1e-10) | np.isclose(pts[:, 1], 1.0, atol=1e-10)
        assert np.all(on_edge)

    def test_subsampling_respects_requested_count(self, tmp_path):
        """Requesting fewer samples than the pool gives exactly that many."""
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=6, ny=6))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        n_pool = dom._mesh_pool["interior"].shape[0]
        n_req = max(1, n_pool // 2)
        dom.variable("interior", sample=(n_req, None))
        assert dom.context["interior"].shape[2] == n_req


# ---------------------------------------------------------------------------
# Custom coordinate range
# ---------------------------------------------------------------------------


class TestCustomCoordinateRange:
    def test_dimension_is_2_for_non_unit_square(self, tmp_path):
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3, x_range=(-1.0, 2.0), y_range=(0.5, 3.5)))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        assert dom.dimension == 2

    def test_points_lie_in_custom_range(self, tmp_path):
        x0, x1, y0, y1 = -1.0, 2.0, 0.5, 3.5
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3, x_range=(x0, x1), y_range=(y0, y1)))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        pts = dom.points
        assert np.all(pts[:, 0] >= x0 - 1e-10) and np.all(pts[:, 0] <= x1 + 1e-10)
        assert np.all(pts[:, 1] >= y0 - 1e-10) and np.all(pts[:, 1] <= y1 + 1e-10)

    def test_boundary_pool_in_custom_range(self, tmp_path):
        x0, x1, y0, y1 = -1.0, 2.0, 0.5, 3.5
        path = str(tmp_path / "m.inp")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3, x_range=(x0, x1), y_range=(y0, y1)))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        pts = dom._mesh_pool["boundary"]
        on_edge = np.isclose(pts[:, 0], x0, atol=1e-10) | np.isclose(pts[:, 0], x1, atol=1e-10) | np.isclose(pts[:, 1], y0, atol=1e-10) | np.isclose(pts[:, 1], y1, atol=1e-10)
        assert np.all(on_edge)


# ---------------------------------------------------------------------------
# File format support
# ---------------------------------------------------------------------------


class TestFileFormats:
    def test_abaqus_inp_loads_with_tags(self, tmp_path):
        """Abaqus .inp round-trips cell_sets → tags are extracted correctly."""
        path = str(tmp_path / "mesh.inp")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        assert dom.dimension == 2
        assert "interior" in dom.avaiable_mesh_tags
        assert "boundary" in dom.avaiable_mesh_tags

    def test_vtk_loads_basic_geometry_no_tags(self, tmp_path):
        """VTK does not preserve cell_sets; basic geometry still loads."""
        path = str(tmp_path / "mesh.vtk")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        assert dom.dimension == 2
        assert len(dom.mesh.points) == 16  # (3+1)*(3+1)

    def test_vtu_loads_basic_geometry_no_tags(self, tmp_path):
        """VTU does not preserve cell_sets; basic geometry still loads."""
        path = str(tmp_path / "mesh.vtu")
        meshio.write(path, _make_2d_mesh(nx=3, ny=3))
        dom = jno.domain(constructor=path, compute_mesh_connectivity=False)
        assert dom.dimension == 2
        assert len(dom.mesh.points) == 16
