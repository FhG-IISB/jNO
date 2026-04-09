"""Tests for jno.domain with built-in Geometries constructors.

Covers 1-D, 2-D, and 3-D geometry construction both with and without a time
dimension, always using ``compute_mesh_connectivity=True`` so that the full
preprocessing pipeline (connectivity, normals, etc.) is exercised.

Mesh sizes are kept deliberately coarse to keep the suite fast.
"""

import pytest
import jno
import inspect


def test_geometry_shortcut_returns_domain_instance():
    dom = jno.domain.rect(mesh_size=0.3, compute_mesh_connectivity=True)

    assert isinstance(dom, jno.domain)
    assert dom.dimension == 2
    assert dom.mesh is not None


def test_geometry_shortcut_routes_domain_kwargs():
    dom = jno.domain.line(mesh_size=0.2, time=(0, 1, 5), compute_mesh_connectivity=False)

    assert isinstance(dom, jno.domain)
    assert dom._is_time_dependent is True
    assert dom.mesh_connectivity is None


@pytest.mark.parametrize(
    ("shape_name", "expected_parameters"),
    [
        ("line", {"x_range", "mesh_size", "algorithm", "time", "compute_mesh_connectivity"}),
        ("rect", {"x_range", "y_range", "mesh_size", "algorithm", "time", "compute_mesh_connectivity"}),
        ("equi_distant_rect", {"x_range", "y_range", "nx", "ny", "algorithm", "time", "compute_mesh_connectivity"}),
        ("poseidon", {"nx", "ny", "algorithm", "time", "compute_mesh_connectivity"}),
        ("cube", {"x_range", "y_range", "z_range", "mesh_size", "algorithm", "time", "compute_mesh_connectivity"}),
        ("disk", {"center", "radius", "mesh_size", "num_points", "algorithm", "time", "compute_mesh_connectivity"}),
        ("triangle", {"vertices", "mesh_size", "algorithm", "time", "compute_mesh_connectivity"}),
        ("polygon", {"vertices", "mesh_size", "algorithm", "time", "compute_mesh_connectivity"}),
        ("l_shape", {"size", "mesh_size", "separate_boundary", "algorithm", "time", "compute_mesh_connectivity"}),
        ("rectangle_with_hole", {"outer_size", "hole_size", "mesh_size", "separate_boundary", "algorithm", "time", "compute_mesh_connectivity"}),
        ("rect_pml", {"x_range", "y_range", "mesh_size", "pml_thickness_top", "pml_thickness_bottom", "algorithm", "time", "compute_mesh_connectivity"}),
        ("rectangle_with_holes", {"outer_size", "holes", "mesh_size", "separate_boundary", "algorithm", "time", "compute_mesh_connectivity"}),
    ],
)
def test_geometry_shortcuts_expose_explicit_signatures(shape_name, expected_parameters):
    signature = inspect.signature(getattr(jno.domain, shape_name))

    assert expected_parameters.issubset(signature.parameters)
    assert "constructor" not in signature.parameters


# ---------------------------------------------------------------------------
# 1-D line – steady state
# ---------------------------------------------------------------------------


class TestLine1DStationary:
    """``Geometries.line`` with no time dependence."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(
            constructor=jno.domain.line(mesh_size=0.2),
            compute_mesh_connectivity=True,
        )

    def test_no_exception_on_creation(self, dom):
        assert dom is not None

    def test_dimension_is_1(self, dom):
        assert dom.dimension == 1

    def test_mesh_is_set(self, dom):
        assert dom.mesh is not None

    def test_is_not_time_dependent(self, dom):
        assert dom._is_time_dependent is False

    def test_interior_tag_present(self, dom):
        assert "interior" in dom.avaiable_mesh_tags

    def test_boundary_tags_present(self, dom):
        assert "left" in dom.avaiable_mesh_tags
        assert "right" in dom.avaiable_mesh_tags

    def test_mesh_pool_has_interior(self, dom):
        assert "interior" in dom._mesh_pool
        pts = dom._mesh_pool["interior"]
        assert pts.ndim == 2
        assert pts.shape[1] == 1  # 1-D spatial coords

    def test_stationary_time_in_context(self, dom):
        assert "__time__" in dom.context


# ---------------------------------------------------------------------------
# 1-D line – time dependent
# ---------------------------------------------------------------------------


class TestLine1DTimeDep:
    """``Geometries.line`` with a time dimension ``(0, 1, 5)``."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(
            constructor=jno.domain.line(mesh_size=0.2),
            time=(0, 1, 5),
            compute_mesh_connectivity=True,
        )

    def test_no_exception_on_creation(self, dom):
        assert dom is not None

    def test_dimension_is_1(self, dom):
        assert dom.dimension == 1

    def test_is_time_dependent(self, dom):
        assert dom._is_time_dependent is True

    def test_interior_tag_present(self, dom):
        assert "interior" in dom.avaiable_mesh_tags

    def test_boundary_pools_are_1d(self, dom):
        for tag in ("left", "right"):
            if tag in dom._mesh_pool:
                assert dom._mesh_pool[tag].shape[1] == 1


# ---------------------------------------------------------------------------
# 2-D rect (pygmsh) – steady state
# ---------------------------------------------------------------------------


class TestRect2DStationary:
    """``Geometries.rect`` – unstructured pygmsh mesh, no time."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            compute_mesh_connectivity=True,
        )

    def test_no_exception_on_creation(self, dom):
        assert dom is not None

    def test_dimension_is_2(self, dom):
        assert dom.dimension == 2

    def test_mesh_is_set(self, dom):
        assert dom.mesh is not None

    def test_is_not_time_dependent(self, dom):
        assert dom._is_time_dependent is False

    def test_interior_tag_present(self, dom):
        assert "interior" in dom.avaiable_mesh_tags

    def test_boundary_tag_present(self, dom):
        assert "boundary" in dom.avaiable_mesh_tags

    def test_interior_pool_shape(self, dom):
        pts = dom._mesh_pool["interior"]
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_side_tags_present(self, dom):
        for tag in ("one", "two", "three", "four", "top", "right", "bottom", "left"):
            assert tag in dom.avaiable_mesh_tags, f"Expected tag '{tag}' in avaiable_mesh_tags"

    def test_stationary_time_in_context(self, dom):
        assert "__time__" in dom.context


# ---------------------------------------------------------------------------
# 2-D rect (pygmsh) – time dependent
# ---------------------------------------------------------------------------


class TestRect2DTimeDep:
    """``Geometries.rect`` with a time dimension ``(0, 2, 4)``."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            time=(0, 2, 4),
            compute_mesh_connectivity=True,
        )

    def test_no_exception_on_creation(self, dom):
        assert dom is not None

    def test_dimension_is_2(self, dom):
        assert dom.dimension == 2

    def test_is_time_dependent(self, dom):
        assert dom._is_time_dependent is True

    def test_interior_pool_has_2_cols(self, dom):
        # _mesh_pool is (T, N, D) for time-dep domains; check last axis is D=2
        assert dom._mesh_pool["interior"].shape[-1] == 2

    def test_boundary_pool_non_empty(self, dom):
        assert dom._mesh_pool["boundary"].shape[0] > 0


# ---------------------------------------------------------------------------
# 2-D equi_distant_rect – steady state (structured, no pygmsh needed)
# ---------------------------------------------------------------------------


class TestEquiDistantRect2DStationary:
    """``Geometries.equi_distant_rect`` – pure-numpy structured mesh, no time."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(
            constructor=jno.domain.equi_distant_rect(nx=5, ny=5),
            compute_mesh_connectivity=True,
        )

    def test_no_exception_on_creation(self, dom):
        assert dom is not None

    def test_dimension_is_2(self, dom):
        assert dom.dimension == 2

    def test_interior_tag_present(self, dom):
        assert "interior" in dom.avaiable_mesh_tags

    def test_all_side_tags_present(self, dom):
        for tag in ("bottom", "top", "left", "right", "boundary"):
            assert tag in dom.avaiable_mesh_tags

    def test_interior_pool_size(self, dom):
        # 6×6 = 36 vertices total
        assert dom._mesh_pool["interior"].shape == (36, 2)

    def test_is_not_time_dependent(self, dom):
        assert dom._is_time_dependent is False


# ---------------------------------------------------------------------------
# 3-D cube (pygmsh) – steady state
# ---------------------------------------------------------------------------


class TestCube3DStationary:
    """``Geometries.cube`` – 3-D unstructured pygmsh mesh, no time."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(
            constructor=jno.domain.cube(mesh_size=0.5),
            compute_mesh_connectivity=True,
        )

    def test_no_exception_on_creation(self, dom):
        assert dom is not None

    def test_dimension_is_3(self, dom):
        assert dom.dimension == 3

    def test_mesh_is_set(self, dom):
        assert dom.mesh is not None

    def test_is_not_time_dependent(self, dom):
        assert dom._is_time_dependent is False

    def test_interior_tag_present(self, dom):
        assert "interior" in dom.avaiable_mesh_tags

    def test_interior_pool_has_3_cols(self, dom):
        pts = dom._mesh_pool["interior"]
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_mesh_pool_non_empty(self, dom):
        assert dom._mesh_pool["interior"].shape[0] > 0

    def test_stationary_time_in_context(self, dom):
        assert "__time__" in dom.context


# ---------------------------------------------------------------------------
# 3-D cube – time dependent
# ---------------------------------------------------------------------------


class TestCube3DTimeDep:
    """``Geometries.cube`` with a time dimension ``(0, 1, 3)``."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(
            constructor=jno.domain.cube(mesh_size=0.5),
            time=(0, 1, 3),
            compute_mesh_connectivity=True,
        )

    def test_no_exception_on_creation(self, dom):
        assert dom is not None

    def test_dimension_is_3(self, dom):
        assert dom.dimension == 3

    def test_is_time_dependent(self, dom):
        assert dom._is_time_dependent is True

    def test_interior_pool_has_3_cols(self, dom):
        # _mesh_pool is (T, N, D) for time-dep domains; check last axis is D=3
        assert dom._mesh_pool["interior"].shape[-1] == 3


# ---------------------------------------------------------------------------
# Triangle geometry
# ---------------------------------------------------------------------------


class TestTriangle:
    """Basic triangle domain from custom vertices."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain.triangle(
            vertices=((0, 0), (2, 0), (1, 1)),
            mesh_size=0.3,
            compute_mesh_connectivity=True,
        )

    def test_dimension_is_2(self, dom):
        assert dom.dimension == 2

    def test_interior_tag_present(self, dom):
        assert "interior" in dom._mesh_pool

    def test_boundary_tag_present(self, dom):
        assert "boundary" in dom._mesh_pool

    def test_interior_has_2_cols(self, dom):
        assert dom._mesh_pool["interior"].shape[-1] == 2


# ---------------------------------------------------------------------------
# Polygon geometry
# ---------------------------------------------------------------------------


class TestPolygon:
    """Generic polygon domain with auto-orientation and boundary labels."""

    @pytest.fixture(scope="class")
    def dom(self):
        # Pentagon (given in CW order to test auto-reorientation)
        verts = [(0, 0), (0, 2), (1, 3), (2, 2), (2, 0)]
        return jno.domain.polygon(vertices=verts, mesh_size=0.5, compute_mesh_connectivity=True)

    def test_dimension_is_2(self, dom):
        assert dom.dimension == 2

    def test_interior_and_boundary_present(self, dom):
        assert "interior" in dom._mesh_pool
        assert "boundary" in dom._mesh_pool

    def test_five_boundary_labels(self, dom):
        tags = set(dom._mesh_pool.keys())
        for name in ("one", "two", "three", "four", "five"):
            assert name in tags, f"Missing boundary label '{name}'"

    def test_rect_has_four_boundary_labels(self):
        dom = jno.domain.rect(mesh_size=0.4)
        tags = set(dom._mesh_pool.keys())
        for name in ("one", "two", "three", "four"):
            assert name in tags

    def test_triangle_has_three_boundary_labels(self):
        dom = jno.domain.triangle(mesh_size=0.4)
        tags = set(dom._mesh_pool.keys())
        for name in ("one", "two", "three"):
            assert name in tags


# ---------------------------------------------------------------------------
# Multi-geometry domain stacking via + operator
# ---------------------------------------------------------------------------


class TestDomainStacking:
    """Verify that combining domains via ``+`` correctly stacks batches."""

    def test_two_geometries_batch_shape(self):
        dom = 3 * jno.domain.rect(mesh_size=0.3)
        dom += 2 * jno.domain.disk(mesh_size=0.3)
        x, y, _ = dom.variable("interior", (10, None))
        ctx = dom.context["interior"]
        assert ctx.shape[0] == 5  # 3 rect + 2 disk
        assert ctx.shape[2] == 10
        assert ctx.shape[3] == 2

    def test_three_geometries_batch_shape(self):
        dom = 4 * jno.domain.rect(mesh_size=0.3)
        dom += 3 * jno.domain.disk(mesh_size=0.3)
        dom += 2 * jno.domain.l_shape(mesh_size=0.3)
        x, y, _ = dom.variable("interior", (8, None))
        ctx = dom.context["interior"]
        assert ctx.shape[0] == 9  # 4 + 3 + 2
        assert ctx.shape[2] == 8

    def test_boundary_also_stacks(self):
        dom = 2 * jno.domain.rect(mesh_size=0.3)
        dom += 3 * jno.domain.triangle(mesh_size=0.3)
        x, y, _ = dom.variable("boundary")
        ctx = dom.context["boundary"]
        assert ctx.shape[0] == 5  # 2 + 3

    def test_total_samples_updated(self):
        dom = 5 * jno.domain.rect(mesh_size=0.3)
        dom += 3 * jno.domain.disk(mesh_size=0.3)
        assert dom.total_samples == 8
