from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

import jno


SQUARE_A = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
SQUARE_B = [(0.5, 0.0), (1.5, 0.0), (1.5, 1.0), (0.5, 1.0)]


def test_polygon_domain_is_domain_and_lazy():
    dom = jno.PolygonDomain(SQUARE_A, name="a")

    assert isinstance(dom, jno.domain)
    assert dom.mesh is None
    assert dom.dimension == 2
    assert "interior" in dom.avaiable_mesh_tags
    assert "boundary_a_0" in dom.boundary_tags()
    assert "interior" not in dom.context


def test_polygon_domain_accepts_constant_z_vertices():
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    dom = jno.PolygonDomain(verts, name="z")

    assert dom.dimension == 2
    assert "boundary_z_0" in dom.boundary_tags()


def test_domain_poly_factory_returns_polygon_domain_without_changing_mesh_polygon():
    dom = jno.domain.poly(SQUARE_A, name="a")
    mesh_dom = jno.domain.polygon(SQUARE_A, mesh_size=0.5, compute_mesh_connectivity=False)

    assert isinstance(dom, jno.PolygonDomain)
    assert dom.mesh is None
    assert mesh_dom.mesh is not None


def test_lazy_interior_sampling_has_exact_context_shape_and_points_inside():
    np.random.seed(0)
    dom = jno.PolygonDomain(SQUARE_A, name="a")

    x, y, t = dom.variable("interior", sample=(128, None))
    pts = dom.context[x.tag][0, 0]

    assert x.tag == "interior"
    assert y.tag == "interior"
    assert t.tag == "__time__"
    assert dom.context["interior"].shape == (1, 1, 128, 2)
    assert np.all((pts[:, 0] > 0.0) & (pts[:, 0] < 1.0))
    assert np.all((pts[:, 1] > 0.0) & (pts[:, 1] < 1.0))


def test_boundary_edge_sampling_uses_input_order_and_exact_normals():
    np.random.seed(1)
    dom = jno.PolygonDomain(SQUARE_A, name="a")

    xb, yb, tb, nx, ny = dom.variable("boundary_a_0", sample=(64, None), normals=True)
    pts = dom.context[xb.tag][0, 0]
    normals = dom.context[nx.tag][0, 0]

    assert xb.tag == "boundary_a_0"
    assert tb.tag == "__time__"
    assert pts.shape == (64, 2)
    assert np.allclose(pts[:, 1], 0.0, atol=1e-12)
    assert np.all(pts[:, 0] > 0.0)
    assert np.all(pts[:, 0] < 1.0)
    assert np.allclose(normals, np.array([0.0, -1.0]), atol=1e-7)
    assert ny.tag == nx.tag


def test_union_operator_is_true_csg_not_batch_stacking():
    np.random.seed(2)
    a = jno.PolygonDomain(SQUARE_A, name="a")
    b = jno.PolygonDomain(SQUARE_B, name="b")

    c = a + b
    c_alt = a | b
    x, y, t = c.variable("interior", sample=(200, None))
    pts = c.context[x.tag][0, 0]

    assert isinstance(c, jno.PolygonDomain)
    assert c.total_samples == 1
    assert c.context[x.tag].shape == (1, 1, 200, 2)
    assert c._active_geometry.area == pytest.approx(1.5)
    assert c_alt._active_geometry.area == pytest.approx(c._active_geometry.area)
    assert all(c._active_geometry.contains(Point(float(px), float(py))) for px, py in pts)


def test_intersection_samples_only_overlap():
    np.random.seed(3)
    a = jno.PolygonDomain(SQUARE_A, name="a")
    b = jno.PolygonDomain(SQUARE_B, name="b")

    c = a & b
    x, y, t = c.variable("interior", sample=(100, None))
    pts = c.context[x.tag][0, 0]

    assert c._active_geometry.area == pytest.approx(0.5)
    assert np.all((pts[:, 0] > 0.5) & (pts[:, 0] < 1.0))
    assert np.all((pts[:, 1] > 0.0) & (pts[:, 1] < 1.0))


def test_difference_exposes_subtrahend_boundary_as_hole_boundary():
    np.random.seed(4)
    outer = jno.PolygonDomain(SQUARE_A, name="outer")
    inner = jno.PolygonDomain([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)], name="hole")

    dom = outer - inner
    xb, yb, tb, nx, ny = dom.variable("boundary_hole_0", sample=(64, None), normals=True)
    pts = dom.context[xb.tag][0, 0]
    normals = dom.context[nx.tag][0, 0]

    assert dom._active_geometry.area == pytest.approx(0.96)
    assert np.allclose(pts[:, 1], 0.4, atol=1e-12)
    assert np.allclose(normals, np.array([0.0, 1.0]), atol=1e-7)


def test_from_polygons_registers_named_region_and_edge_tags():
    dom = jno.PolygonDomain.from_polygons(
        {
            "Air": SQUARE_A,
            "Wall": [(1.2, 0.0), (1.5, 0.0), (1.5, 1.0), (1.2, 1.0)],
        }
    )

    assert "interior_Air" in dom.avaiable_mesh_tags
    assert "interior_Wall" in dom.avaiable_mesh_tags
    assert "boundary_Air_0" in dom.boundary_tags()
    assert "boundary_Wall_0" in dom.boundary_tags()


def test_from_regions_uses_preprocessed_air_geometry_for_interior_sampling():
    from shapely.geometry import Polygon

    scene_box = Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)])
    left_solid = Polygon([(0.0, 0.4), (0.4, 0.4), (0.4, 0.6), (0.0, 0.6)])
    right_solid = Polygon([(2.6, 0.4), (3.0, 0.4), (3.0, 0.6), (2.6, 0.6)])
    explicit_air = Polygon([(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)])

    dom = jno.PolygonDomain.from_regions(
        {
            "Air": scene_box.difference(left_solid.union(right_solid)),
            "LeftSolid": left_solid,
            "RightSolid": right_solid,
            "LegacyAir": explicit_air,
        }
    )

    np.random.seed(7)
    x, y, _ = dom.variable("interior_Air", sample=(256, None))
    pts = dom.context[x.tag][0, 0]

    assert pts.shape == (256, 2)
    assert np.any((pts[:, 0] > 1.0) & (pts[:, 0] < 2.0))
    assert not np.any((pts[:, 0] < 0.4) & (pts[:, 1] > 0.4) & (pts[:, 1] < 0.6))
    assert not np.any((pts[:, 0] > 2.6) & (pts[:, 1] > 0.4) & (pts[:, 1] < 0.6))


def test_enclosure_view_factor_uses_cross_tag_blocks_only():
    np.random.seed(5)
    dom = jno.PolygonDomain.from_polygons(
        {
            "Air": [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)],
            "LeftSolid": [(-0.2, 0.25), (0.0, 0.25), (0.0, 0.75), (-0.2, 0.75)],
            "RightSolid": [(3.0, 0.25), (3.2, 0.25), (3.2, 0.75), (3.0, 0.75)],
        }
    )
    dom.add_boundary_segments("radiation_left", [[(0.0, 0.25), (0.0, 0.75)]], normal_geometry="LeftSolid")
    dom.add_boundary_segments("radiation_right", [[(3.0, 0.75), (3.0, 0.25)]], normal_geometry="RightSolid")

    for tag in ["radiation_left", "radiation_right"]:
        dom.variable(tag, sample=(32, None), normals=True)
    dom.compute_enclosure_view_factor(["radiation_left", "radiation_right"], medium_tags=["Air"])

    assert np.sum(dom.context["v_radiation_left__radiation_left"]) == pytest.approx(0.0)
    assert np.sum(dom.context["f_radiation_left__radiation_left"]) == pytest.approx(0.0)
    assert np.sum(dom.context["v_radiation_right__radiation_right"]) == pytest.approx(0.0)
    assert np.sum(dom.context["f_radiation_right__radiation_right"]) == pytest.approx(0.0)
    assert float(np.sum(dom.context["v_radiation_left__radiation_right"])) > 0.0
    assert float(np.sum(dom.context["f_radiation_left__radiation_right"])) > 0.0


def test_enclosure_visibility_filters_rays_outside_solid_subtracted_medium():
    np.random.seed(6)
    dom = jno.PolygonDomain.from_polygons(
        {
            "Air": [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)],
            "Block": [(1.25, 0.2), (1.75, 0.2), (1.75, 0.8), (1.25, 0.8)],
        }
    )
    dom.add_boundary_segments("radiation_block_left", [[(1.25, 0.2), (1.25, 0.8)]], normal_geometry="Block")
    dom.add_boundary_segments("radiation_block_right", [[(1.75, 0.8), (1.75, 0.2)]], normal_geometry="Block")

    for tag in ["radiation_block_left", "radiation_block_right"]:
        dom.variable(tag, sample=(32, None), normals=True)
    dom.compute_enclosure_view_factor(["radiation_block_left", "radiation_block_right"], medium_tags=["Air"])

    assert np.sum(dom.context["v_radiation_block_left__radiation_block_right"]) == pytest.approx(0.0)
    assert np.sum(dom.context["v_radiation_block_right__radiation_block_left"]) == pytest.approx(0.0)


def test_air_medium_is_inferred_as_scene_box_void_not_explicit_air_polygon():
    dom = jno.PolygonDomain.from_polygons(
        {
            "Air": [(0.0, 0.0), (0.25, 0.0), (0.25, 0.25), (0.0, 0.25)],
            "LeftSolid": [(0.0, 0.4), (0.2, 0.4), (0.2, 0.6), (0.0, 0.6)],
            "RightSolid": [(2.8, 0.4), (3.0, 0.4), (3.0, 0.6), (2.8, 0.6)],
        }
    )

    medium = dom._medium_geometry(["Air"])

    assert medium is not None
    assert medium.contains(Point(1.5, 0.5))
    assert not medium.contains(Point(0.1, 0.5))
    assert not medium.contains(Point(2.9, 0.5))


def test_visibility_filter_requires_positive_cosines_at_both_endpoints():
    dom = jno.PolygonDomain(SQUARE_A, name="box")

    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    visible = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    facing_normals = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )
    filtered_facing = dom._filter_visibility_by_normals(points, facing_normals, visible)
    assert filtered_facing[0, 1] == pytest.approx(1.0)
    assert filtered_facing[1, 0] == pytest.approx(1.0)

    away_normals = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    filtered_away = dom._filter_visibility_by_normals(points, away_normals, visible)
    assert filtered_away[0, 1] == pytest.approx(0.0)
    assert filtered_away[1, 0] == pytest.approx(0.0)

    perpendicular_normals = np.array(
        [
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )
    filtered_perpendicular = dom._filter_visibility_by_normals(points, perpendicular_normals, visible)
    assert filtered_perpendicular[0, 1] == pytest.approx(0.0)
    assert filtered_perpendicular[1, 0] == pytest.approx(0.0)
