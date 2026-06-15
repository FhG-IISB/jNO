from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

import jno

SQUARE_A = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
SQUARE_B = [(0.5, 0.0), (1.5, 0.0), (1.5, 1.0), (0.5, 1.0)]


def test_polygon_domain_is_domain_and_lazy():
    dom = jno.domain.csg(SQUARE_A, name="a")

    assert isinstance(dom, jno.domain)
    assert dom.mesh is None
    assert dom.dimension == 2
    assert "interior" in dom.avaiable_mesh_tags
    assert "boundary_a_0" in dom.boundary_tags()
    assert "interior" not in dom.context


def test_polygon_domain_accepts_constant_z_vertices():
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    dom = jno.domain.csg(verts, name="z")

    assert dom.dimension == 2
    assert "boundary_z_0" in dom.boundary_tags()


def test_domain_poly_factory_returns_polygon_domain_without_changing_mesh_polygon():
    dom = jno.domain.poly(SQUARE_A, name="a")
    mesh_dom = jno.domain.polygon(SQUARE_A, mesh_size=0.5, compute_mesh_connectivity=False)

    assert isinstance(dom, jno.domain.csg)
    assert dom.mesh is None
    assert mesh_dom.mesh is not None


def test_lazy_interior_sampling_has_exact_context_shape_and_points_inside():
    np.random.seed(0)
    dom = jno.domain.csg(SQUARE_A, name="a")

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
    dom = jno.domain.csg(SQUARE_A, name="a")

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
    a = jno.domain.csg(SQUARE_A, name="a")
    b = jno.domain.csg(SQUARE_B, name="b")

    c = a + b
    c_alt = a | b
    x, y, t = c.variable("interior", sample=(200, None))
    pts = c.context[x.tag][0, 0]

    assert isinstance(c, jno.domain.csg)
    assert c.total_samples == 1
    assert c.context[x.tag].shape == (1, 1, 200, 2)
    assert c._active_geometry.area == pytest.approx(1.5)
    assert c_alt._active_geometry.area == pytest.approx(c._active_geometry.area)
    assert all(c._active_geometry.contains(Point(float(px), float(py))) for px, py in pts)


def test_intersection_samples_only_overlap():
    np.random.seed(3)
    a = jno.domain.csg(SQUARE_A, name="a")
    b = jno.domain.csg(SQUARE_B, name="b")

    c = a & b
    x, y, t = c.variable("interior", sample=(100, None))
    pts = c.context[x.tag][0, 0]

    assert c._active_geometry.area == pytest.approx(0.5)
    assert np.all((pts[:, 0] > 0.5) & (pts[:, 0] < 1.0))
    assert np.all((pts[:, 1] > 0.0) & (pts[:, 1] < 1.0))


def test_difference_exposes_subtrahend_boundary_as_hole_boundary():
    np.random.seed(4)
    outer = jno.domain.csg(SQUARE_A, name="outer")
    inner = jno.domain.csg([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)], name="hole")

    dom = outer - inner
    xb, yb, tb, nx, ny = dom.variable("boundary_hole_0", sample=(64, None), normals=True)
    pts = dom.context[xb.tag][0, 0]
    normals = dom.context[nx.tag][0, 0]

    assert dom._active_geometry.area == pytest.approx(0.96)
    assert np.allclose(pts[:, 1], 0.4, atol=1e-12)
    assert np.allclose(normals, np.array([0.0, 1.0]), atol=1e-7)


def test_from_polygons_registers_named_region_and_edge_tags():
    dom = jno.domain.csg.from_polygons(
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

    dom = jno.domain.csg.from_regions(
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
    dom = jno.domain.csg.from_polygons(
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
    dom = jno.domain.csg.from_polygons(
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
    dom = jno.domain.csg.from_polygons(
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
    dom = jno.domain.csg(SQUARE_A, name="box")

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


# ---------------------------------------------------------------------------
# Numbered boundary sub-tag existence and geometry
# ---------------------------------------------------------------------------


def test_from_regions_produces_per_edge_subtags():
    """from_regions must decompose each polygon into per-edge sub-tags, not one combined tag."""
    from shapely.geometry import Polygon

    dom = jno.domain.csg.from_regions({"WallO": Polygon(SQUARE_A)})
    tags = dom.boundary_tags()

    for i in range(4):
        assert f"boundary_WallO_{i}" in tags, f"boundary_WallO_{i} missing — from_regions returned only: " + str(
            [t for t in tags if "WallO" in t]
        )


def test_all_numbered_boundary_subtags_present_for_square():
    """A 4-sided polygon must expose boundary_{name}_0 … _3 and the combined tag."""
    dom = jno.domain.csg(SQUARE_A, name="wall")
    tags = dom.boundary_tags()

    for i in range(4):
        assert f"boundary_wall_{i}" in tags, f"boundary_wall_{i} missing from {tags}"
    assert "boundary_wall" in tags


def test_numbered_boundary_subtags_are_geometrically_distinct():
    """Each boundary_{name}_{i} sub-tag must sample from a different edge of the square.

    For an axis-aligned unit square every edge has one coordinate that is constant
    (either x=0, x=1, y=0, or y=1).  The four sub-tags must cover four distinct
    constant-coordinate values, confirming they are separate faces.
    """
    np.random.seed(42)
    dom = jno.domain.csg(SQUARE_A, name="wall")

    constant_coords = set()
    for i in range(4):
        tag = f"boundary_wall_{i}"
        xb, yb, _ = dom.variable(tag, sample=(64, None))
        pts = dom.context[tag][0, 0]  # (64, 2)

        x_spread = pts[:, 0].max() - pts[:, 0].min()
        y_spread = pts[:, 1].max() - pts[:, 1].min()

        # One coordinate must be (near-)constant — it's an axis-aligned edge
        assert min(x_spread, y_spread) < 1e-10, (
            f"{tag}: neither x_spread={x_spread:.2e} nor y_spread={y_spread:.2e} is constant — not an axis-aligned edge"
        )

        # Record which (axis, value) this edge sits on
        if x_spread < 1e-10:
            constant_coords.add(("x", round(float(pts[0, 0]), 6)))
        else:
            constant_coords.add(("y", round(float(pts[0, 1]), 6)))

    # Four distinct faces ↔ four distinct (axis, value) pairs
    assert len(constant_coords) == 4, f"Expected 4 distinct edges, got {len(constant_coords)}: {constant_coords}"


# ---------------------------------------------------------------------------
# Auto-default sampling (no explicit sample count)
# ---------------------------------------------------------------------------


def test_variable_no_sample_defaults_to_one_point():
    """Calling variable() without a sample count materializes exactly 1 point."""
    dom = jno.domain.csg(SQUARE_A, name="a")
    x, y, _ = dom.variable("interior")
    pts = dom.context[x.tag][0, 0]
    assert pts.shape == (1, 2), f"expected (1, 2), got {pts.shape}"
    # Point must lie inside the unit square
    assert 0.0 < pts[0, 0] < 1.0
    assert 0.0 < pts[0, 1] < 1.0


def test_variable_no_sample_attaches_per_step_resampling():
    """Auto-default path attaches RandomResampling(resample_every=1, fraction=1.0)."""
    from jno.utils.adaptive.resampling import RandomResampling

    dom = jno.domain.csg(SQUARE_A, name="a")
    x, y, _ = dom.variable("interior")
    tag = x.tag
    assert tag in dom._resampling_strategies, "no resampling strategy was registered"
    strat = dom._resampling_strategies[tag]
    assert isinstance(strat, RandomResampling)
    assert strat.resample_every == 1
    assert strat.resample_fraction == 1.0


def test_variable_explicit_count_unchanged():
    """variable('interior', (64, None)) still works, uses the given count, no auto-resample."""
    np.random.seed(0)
    dom = jno.domain.csg(SQUARE_A, name="a")
    x, y, _ = dom.variable("interior", (64, None))
    pts = dom.context[x.tag][0, 0]
    assert pts.shape == (64, 2)
    # Explicit count → no auto-resampling strategy attached
    assert x.tag not in dom._resampling_strategies


def test_variable_no_sample_explicit_resampling_strategy_respected():
    """If the user explicitly passes a resampling_strategy, it is not overridden."""
    from jno.utils.adaptive.resampling import RandomResampling

    custom = RandomResampling(resample_every=50, resample_fraction=0.5)
    dom = jno.domain.csg(SQUARE_A, name="a")
    x, y, _ = dom.variable("interior", resampling_strategy=custom)
    tag = x.tag
    assert dom._resampling_strategies[tag] is custom
    assert dom._resampling_strategies[tag].resample_every == 50


@pytest.mark.integration
def test_variable_no_sample_drives_crux_solve_end_to_end():
    """Auto-default + crux.solve must run: 1 point + per-step random resample."""
    import foundax
    import jax
    import optax

    dom = jno.domain.csg(SQUARE_A, name="a")
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))
    u = net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)
    pde = u.laplacian(x, y)
    crux = jno.core([pde.mse], domain=dom)
    stats = crux.solve(3)
    # End-to-end smoke: solve runs without crashing and the loop exercised the
    # resample → step pipeline at every epoch (resample_every=1) — that's what
    # the auto-default + RandomResampling combination is supposed to deliver.
    assert stats is not None
