"""Tests for jno.domain construction.

Covers 1-D, 2-D, and 3-D domain construction both with and without a time
dimension, always using ``compute_mesh_connectivity=True`` so that the full
preprocessing pipeline (connectivity, normals, etc.) is exercised. 1-D lines and
structured grids come from the ``jno.domain.line`` / ``equi_distant_rect`` /
``poseidon`` classmethods; 2-D/3-D geometries come from ``jno.Shape`` (the CSG
build-plan) realized via ``Shape(...).domain()``.

Mesh sizes are kept deliberately coarse to keep the suite fast.
"""

import inspect

import jax.numpy as jnp
import pytest

import jno


def test_shape_domain_returns_domain_instance():
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.3).domain(compute_mesh_connectivity=True)

    assert isinstance(dom, jno.domain)
    assert dom.dimension == 2
    assert dom.mesh is not None


def test_geometry_shortcut_routes_domain_kwargs():
    dom = jno.domain.line(mesh_size=0.2, time=(0, 1, 5), compute_mesh_connectivity=False)

    assert isinstance(dom, jno.domain)
    assert dom._is_time_dependent is True
    assert dom.mesh_connectivity is None


def test_line_domain_exposes_named_endpoints():
    """jno.domain.line must resolve left/right/boundary at the interval ends.

    Regression: the pygmsh 1-D path emitted a malformed single-block mesh (scalar
    point cell_sets, no vertex block), so variable('left'/'right'/'boundary') raised.
    """
    import numpy as np

    dom = jno.domain(constructor=jno.domain.line(x_range=(0.0, 1.0), mesh_size=0.1))
    assert dom.dimension == 1
    for tag in ("interior", "left", "right", "boundary"):
        dom.variable(tag)  # must not raise
    assert np.allclose(np.asarray(dom._mesh_pool["left"]).reshape(-1), [0.0])
    assert np.allclose(np.asarray(dom._mesh_pool["right"]).reshape(-1), [1.0])
    assert np.allclose(np.sort(np.asarray(dom._mesh_pool["boundary"]).reshape(-1)), [0.0, 1.0])


@pytest.mark.parametrize(
    ("shape_name", "expected_parameters"),
    [
        (
            "line",
            {"x_range", "mesh_size", "algorithm", "time", "compute_mesh_connectivity"},
        ),
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
    """``Shape.rect`` – unstructured gmsh mesh, no time."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.Shape.rect(0, 0, 1, 1, size=0.3).domain(compute_mesh_connectivity=True)

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
        for tag in ("top", "right", "bottom", "left"):
            assert tag in dom.avaiable_mesh_tags, f"Expected tag '{tag}' in avaiable_mesh_tags"

    def test_stationary_time_in_context(self, dom):
        assert "__time__" in dom.context


# ---------------------------------------------------------------------------
# 2-D rect (pygmsh) – time dependent
# ---------------------------------------------------------------------------


class TestRect2DTimeDep:
    """``Shape.rect`` with a time dimension ``(0, 2, 4)``."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.Shape.rect(0, 0, 1, 1, size=0.3).domain(
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
        return jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=5).domain(compute_mesh_connectivity=True)

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
    """``Shape.box`` – 3-D unstructured gmsh mesh, no time."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain(compute_mesh_connectivity=True)

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
    """``Shape.box`` with a time dimension ``(0, 1, 3)``."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain(
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
    """Basic triangle domain from custom vertices (a 3-vertex polygon)."""

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.Shape.polygon(((0, 0), (2, 0), (1, 1)), size=0.3).domain(
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
    """Generic polygon domain with per-edge boundary labels (``e0..e{n-1}``)."""

    @pytest.fixture(scope="class")
    def dom(self):
        verts = [(0, 0), (0, 2), (1, 3), (2, 2), (2, 0)]
        return jno.Shape.polygon(verts, size=0.5).domain(compute_mesh_connectivity=True)

    def test_dimension_is_2(self, dom):
        assert dom.dimension == 2

    def test_interior_and_boundary_present(self, dom):
        assert "interior" in dom._mesh_pool
        assert "boundary" in dom._mesh_pool

    def test_five_boundary_labels(self, dom):
        tags = set(dom._mesh_pool.keys())
        for name in ("e0", "e1", "e2", "e3", "e4"):
            assert name in tags, f"Missing boundary label '{name}'"

    def test_rect_has_named_side_labels(self):
        dom = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
        tags = set(dom._mesh_pool.keys())
        for name in ("left", "right", "top", "bottom"):
            assert name in tags

    def test_triangle_has_three_boundary_labels(self):
        dom = jno.Shape.polygon(((0, 0), (1, 0), (0, 1)), size=0.4).domain()
        tags = set(dom._mesh_pool.keys())
        for name in ("e0", "e1", "e2"):
            assert name in tags


# ---------------------------------------------------------------------------
# Multi-geometry domain stacking via + operator
# ---------------------------------------------------------------------------


class TestDomainStacking:
    """Verify that combining domains via ``+`` correctly stacks batches."""

    def test_two_geometries_batch_shape(self):
        dom = 3 * jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
        dom += 2 * jno.Shape.disk(0, 0, 1, size=0.3).domain()
        x, y, _ = dom.variable("interior", (10, None))
        ctx = dom.context["interior"]
        assert ctx.shape[0] == 5  # 3 rect + 2 disk
        assert ctx.shape[2] == 10
        assert ctx.shape[3] == 2

    def test_three_geometries_batch_shape(self):
        dom = 4 * jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
        dom += 3 * jno.Shape.disk(0, 0, 1, size=0.3).domain()
        dom += 2 * jno.Shape.polygon([(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)], size=0.3).domain()
        x, y, _ = dom.variable("interior", (8, None))
        ctx = dom.context["interior"]
        assert ctx.shape[0] == 9  # 4 + 3 + 2
        assert ctx.shape[2] == 8

    def test_boundary_also_stacks(self):
        dom = 2 * jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
        dom += 3 * jno.Shape.polygon(((0, 0), (1, 0), (0, 1)), size=0.3).domain()
        x, y, _ = dom.variable("boundary")
        ctx = dom.context["boundary"]
        assert ctx.shape[0] == 5  # 2 + 3

    def test_total_samples_updated(self):
        dom = 5 * jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
        dom += 3 * jno.Shape.disk(0, 0, 1, size=0.3).domain()
        assert dom.total_samples == 8


# ---------------------------------------------------------------------------
# distance_function
# ---------------------------------------------------------------------------


class TestDistanceFunction:
    """Verify domain.distance_function() returns correct boundary distances."""

    def test_distances_nonnegative(self):
        import jno

        dom = jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
        dom.variable("interior")  # populate context
        d = dom.distance_function("interior")
        # Variable tag should exist in context
        assert d.tag in dom.context
        dist_arr = dom.context[d.tag]
        assert float(jnp.min(jnp.array(dist_arr))) >= 0.0

    def test_boundary_points_have_zero_distance(self):
        """Points ON the boundary should have distance ≈ 0."""
        import numpy as np

        import jno

        dom = jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
        dom.variable("boundary")
        dom.variable("interior")
        d_var = dom.distance_function("boundary", boundary_tags=["interior"])
        dist_arr = np.array(dom.context[d_var.tag])  # (1, 1, N, 1)
        assert dist_arr.min() < 0.05  # at least some boundary pts very close to interior

    def test_interior_distances_positive(self):
        """Interior points that are not on the boundary should have d > 0."""
        import numpy as np

        import jno

        dom = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain()
        dom.variable("interior")
        d_var = dom.distance_function("interior")
        dist_arr = np.array(dom.context[d_var.tag])
        # The mean distance should be meaningfully positive (> mesh_size/2)
        assert float(dist_arr.mean()) > 0.05

    def test_custom_name(self):
        import jno

        dom = jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
        dom.variable("interior")
        d_var = dom.distance_function("interior", name="my_dist")
        assert d_var.tag == "my_dist"
        assert "my_dist" in dom.context


# Tensor-tag attachment via domain.variable(sample=array)
# ---------------------------------------------------------------------------


class TestVariableTensorAttach:
    """Attach arrays of varying leading-dim length via ``variable(sample=...)``.

    The compiler routes by ``shape[0]`` (see ``jno/trace_compiler.py``):

      * ``shape[0] == B`` → per-batch (vmapped)
      * ``shape[0] == 1`` → broadcast across the batch
      * ``shape[0]`` anything else → shared, full array exposed every step

    All three paths must accept the attached tensor without warning.
    """

    def _dom(self):
        return jno.domain(constructor=jno.domain.line(mesh_size=0.2))

    def test_per_batch_shape(self):
        import numpy as np

        dom = self._dom()
        B = dom._effective_batch_count()
        arr = np.zeros((B, 3), dtype=np.float32)
        dom.variable("per_batch", sample=arr)
        assert dom.context["per_batch"].shape == (B, 3)
        assert "per_batch" in dom._param_tags

    def test_broadcast_shape(self):
        import numpy as np

        dom = self._dom()
        arr = np.zeros((1, 3), dtype=np.float32)
        dom.variable("bcast", sample=arr)
        assert dom.context["bcast"].shape == (1, 3)
        assert "bcast" in dom._param_tags

    def test_shared_length_mismatch_no_warning(self, caplog):
        """Length-mismatched tensor stores cleanly and emits no warning."""
        import logging

        import numpy as np

        dom = self._dom()
        arr = np.zeros((16, 3), dtype=np.float32)
        with caplog.at_level(logging.WARNING):
            dom.variable("u_labels", sample=arr)
        assert dom.context["u_labels"].shape == (16, 3)
        assert "u_labels" in dom._param_tags
        assert not any("Was this intended" in r.message for r in caplog.records)

    def test_returned_object_is_tensor_tag(self):
        import numpy as np

        from jno.trace import TensorTag

        dom = self._dom()
        result = dom.variable("coeff", sample=np.zeros((1, 1), dtype=np.float32))
        assert isinstance(result, TensorTag)
