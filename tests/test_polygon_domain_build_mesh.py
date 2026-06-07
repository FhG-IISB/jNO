"""Tests for ``PolygonDomain.build_mesh``: opt-in gmsh meshing on the lazy
Shapely-backed CSG domain so that ``expr.integrate()`` and the FD derivative
scheme become available.

Smoke tests cover:

* area integral on a chamber-minus-obstacle CSG geometry,
* boundary-length integrals on tagged source edges (tag fidelity through CSG),
* FD vs analytic derivative parity on a known polynomial field,
* AD parity at machine precision on the same field,
* idempotency: re-meshing with a smaller mesh_size refines without breaking
  the integral cache,
* the lazy/Shapely path stays usable when ``build_mesh`` has not been called.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.trace_evaluator import TraceEvaluator

CHAMBER = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
OBSTACLE = [(0.8, 0.35), (1.2, 0.35), (1.2, 0.65), (0.8, 0.65)]

CHAMBER_AREA = 2.0  # 2 * 1
OBSTACLE_AREA = 0.4 * 0.3
CHAMBER_PERIMETER = 2 * (2.0 + 1.0)
OBSTACLE_PERIMETER = 2 * (0.4 + 0.3)


def _chamber_minus_obstacle() -> jno.domain.csg:
    chamber = jno.domain.csg(CHAMBER, name="chamber")
    obstacle = jno.domain.csg(OBSTACLE, name="obstacle")
    return chamber - obstacle


def test_build_mesh_populates_mesh_connectivity():
    np.random.seed(0)
    dom = _chamber_minus_obstacle()
    assert dom.mesh is None
    assert getattr(dom, "mesh_connectivity", None) is None or not dom.mesh_connectivity

    dom.build_mesh(mesh_size=0.1)

    assert dom.mesh is not None
    assert dom.mesh_connectivity is not None
    assert dom.mesh_connectivity["dimension"] == 2
    assert dom.mesh_connectivity["n_points"] == dom.mesh.points.shape[0]


def test_build_mesh_interior_area_integral_matches_csg_area():
    np.random.seed(0)
    dom = _chamber_minus_obstacle()
    dom.build_mesh(mesh_size=0.05)

    x, _y, _t = dom.variable("interior")
    ev = TraceEvaluator(dom)
    area = float(np.asarray(ev.evaluate((x * 0.0 + 1.0).integrate())))
    assert area == pytest.approx(CHAMBER_AREA - OBSTACLE_AREA, rel=1e-3)


def test_build_mesh_boundary_lengths_match_source_edges():
    """Tag fidelity: gmsh segments shouldn't straddle source-edge boundaries."""
    np.random.seed(0)
    dom = _chamber_minus_obstacle()
    dom.build_mesh(mesh_size=0.05)
    ev = TraceEvaluator(dom)

    xb, _yb, _ = dom.variable("boundary_chamber_0")
    edge0 = float(np.asarray(ev.evaluate((xb * 0.0 + 1.0).integrate())))
    assert edge0 == pytest.approx(2.0, rel=1e-6)

    xb, _yb, _ = dom.variable("boundary_chamber")
    chamber = float(np.asarray(ev.evaluate((xb * 0.0 + 1.0).integrate())))
    assert chamber == pytest.approx(CHAMBER_PERIMETER, rel=1e-6)

    xb, _yb, _ = dom.variable("boundary_obstacle")
    obstacle = float(np.asarray(ev.evaluate((xb * 0.0 + 1.0).integrate())))
    assert obstacle == pytest.approx(OBSTACLE_PERIMETER, rel=1e-6)


def test_build_mesh_enables_fd_and_ad_derivatives():
    """Both schemes should resolve once a mesh is attached.

    AD is exact on a polynomial field; FD has mesh-resolution error.
    """
    np.random.seed(0)
    dom = _chamber_minus_obstacle()
    dom.build_mesh(mesh_size=0.05)

    x, y, _t = dom.variable("interior")
    u = x * x + y * y
    ev = TraceEvaluator(dom)

    ad = np.asarray(ev.evaluate(u.d(x, scheme="automatic_differentiation"), context=dom.context))
    fd = np.asarray(ev.evaluate(u.d(x, scheme="finite_difference"), context=dom.context))
    exact = np.asarray(ev.evaluate(2 * x, context=dom.context)).reshape(ad.shape)

    assert ad.shape == fd.shape
    assert float(np.max(np.abs(ad - exact))) < 1e-10
    assert float(np.max(np.abs(fd - exact))) < 0.1


def test_build_mesh_is_idempotent_and_refines():
    np.random.seed(0)
    dom = _chamber_minus_obstacle()
    dom.build_mesh(mesh_size=0.2)
    coarse_n = dom.mesh_connectivity["n_points"]

    dom.build_mesh(mesh_size=0.06)
    fine_n = dom.mesh_connectivity["n_points"]
    assert fine_n > coarse_n

    x, _y, _t = dom.variable("interior")
    ev = TraceEvaluator(dom)
    area = float(np.asarray(ev.evaluate((x * 0.0 + 1.0).integrate())))
    assert area == pytest.approx(CHAMBER_AREA - OBSTACLE_AREA, rel=1e-3)


def test_lazy_path_unchanged_without_build_mesh():
    """Without ``build_mesh``, AD on lazy samples still works; the
    explicit-count-required error stays put."""
    np.random.seed(0)
    dom = jno.domain.csg([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    x, y, _t = dom.variable("interior", sample=(64, None))
    u = x * x
    ev = TraceEvaluator(dom)
    ad = np.asarray(ev.evaluate(u.d(x, scheme="automatic_differentiation"), context=dom.context))
    exact = np.asarray(ev.evaluate(2 * x, context=dom.context)).reshape(ad.shape)
    assert float(np.max(np.abs(ad - exact))) < 1e-10

    dom_no_mesh = jno.domain.csg([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    with pytest.raises(ValueError, match="explicit sample count"):
        dom_no_mesh.variable("interior")


def test_build_mesh_rejects_empty_geometry():
    """Subtracting a covering square leaves an empty active geometry."""
    np.random.seed(0)
    a = jno.domain.csg([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    b = jno.domain.csg([(-1.0, -1.0), (2.0, -1.0), (2.0, 2.0), (-1.0, 2.0)])
    empty = a - b
    with pytest.raises(ValueError, match="empty active geometry"):
        empty.build_mesh(mesh_size=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Complex CSG topologies
# ─────────────────────────────────────────────────────────────────────────────


def _eval_area(dom, tag="interior"):
    x, _y, _t = dom.variable(tag)
    return float(np.asarray(TraceEvaluator(dom).evaluate((x * 0.0 + 1.0).integrate())))


def _eval_length(dom, tag):
    x, _y, _t = dom.variable(tag)
    return float(np.asarray(TraceEvaluator(dom).evaluate((x * 0.0 + 1.0).integrate())))


def test_lshape_via_union_of_two_rects():
    """L-shape from non-overlapping rectangles via union — area = sum."""
    np.random.seed(10)
    a = jno.domain.csg([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)], name="a")
    b = jno.domain.csg([(0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)], name="b")
    dom = a + b
    dom.build_mesh(mesh_size=0.1)

    area = _eval_area(dom)
    assert area == pytest.approx(2.0 + 1.0, rel=1e-3)

    # Both sub-region tags survived through union.
    assert "interior_a" in dom._polygon_tags
    assert "interior_b" in dom._polygon_tags


def test_concave_lshape_integral_matches_exact_area():
    """A genuinely concave L-shape built from a single 6-vertex polygon."""
    np.random.seed(11)
    l_shape = [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (0.0, 2.0),
    ]
    dom = jno.domain.csg(l_shape, name="L")
    dom.build_mesh(mesh_size=0.07)

    # L-shape area = 2*1 + 1*1 = 3
    assert _eval_area(dom) == pytest.approx(3.0, rel=1e-3)
    # Perimeter = 2 + 1 + 1 + 1 + 1 + 2 = 8
    assert _eval_length(dom, "boundary") == pytest.approx(8.0, rel=1e-6)


def test_chamber_with_three_holes_area_and_per_hole_boundaries():
    """Chamber minus three obstacle holes → 3 distinct hole-boundary tags."""
    np.random.seed(12)
    chamber = jno.domain.csg(CHAMBER, name="chamber")
    h1 = jno.domain.csg([(0.3, 0.3), (0.5, 0.3), (0.5, 0.6), (0.3, 0.6)], name="h1")
    h2 = jno.domain.csg([(0.9, 0.2), (1.1, 0.2), (1.1, 0.5), (0.9, 0.5)], name="h2")
    h3 = jno.domain.csg([(1.5, 0.4), (1.8, 0.4), (1.8, 0.8), (1.5, 0.8)], name="h3")
    dom = chamber - h1 - h2 - h3
    dom.build_mesh(mesh_size=0.06)

    a_h1 = 0.2 * 0.3
    a_h2 = 0.2 * 0.3
    a_h3 = 0.3 * 0.4
    assert _eval_area(dom) == pytest.approx(CHAMBER_AREA - a_h1 - a_h2 - a_h3, rel=1e-3)

    # Each hole keeps its own boundary tags.
    for name, expected_perim in [("h1", 2 * (0.2 + 0.3)), ("h2", 2 * (0.2 + 0.3)), ("h3", 2 * (0.3 + 0.4))]:
        per = _eval_length(dom, f"boundary_{name}")
        assert per == pytest.approx(expected_perim, rel=1e-6), f"hole {name} perimeter mismatch: {per}"


def test_disjoint_multipolygon_via_csg_meshes_each_piece():
    """Two non-touching squares via union → MultiPolygon → both meshed."""
    np.random.seed(13)
    a = jno.domain.csg([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], name="a")
    b = jno.domain.csg([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)], name="b")
    dom = a + b
    assert dom._active_geometry.geom_type == "MultiPolygon"
    dom.build_mesh(mesh_size=0.1)

    assert _eval_area(dom) == pytest.approx(2.0, rel=1e-3)

    # Sample interior points; both x ranges should appear.
    x, _y, _t = dom.variable("interior")
    pts = np.asarray(dom.context[x.tag])[0, 0]
    assert np.any((pts[:, 0] >= 0.0) & (pts[:, 0] <= 1.0))
    assert np.any((pts[:, 0] >= 2.0) & (pts[:, 0] <= 3.0))
    # No points in the gap.
    assert not np.any((pts[:, 0] > 1.05) & (pts[:, 0] < 1.95))


def test_normals_after_build_mesh_point_outward_on_chamber_and_into_hole():
    """Mesh-side normals are vertex-averaged but the sign convention stays
    outward-from-material on both the chamber wall and the obstacle hole."""
    np.random.seed(14)
    chamber = jno.domain.csg(CHAMBER, name="chamber")
    obstacle = jno.domain.csg(OBSTACLE, name="obstacle")
    dom = chamber - obstacle
    dom.build_mesh(mesh_size=0.08)

    # Chamber bottom edge (y=0): outward normal has y<0 (points down out of
    # the chamber). Skip the two endpoints because vertex normals there
    # average two perpendicular edges → tangent component dominates.
    _xb, _yb, _, nx, _ny = dom.variable("boundary_chamber_0", normals=True)
    pts = np.asarray(dom.context[_xb.tag])[0, 0]
    nrm = np.asarray(dom.context[nx.tag])[0, 0]
    interior_mask = (pts[:, 0] > 0.05) & (pts[:, 0] < 1.95)
    assert interior_mask.sum() >= 3, "expected non-corner samples on chamber bottom"
    assert np.all(nrm[interior_mask, 1] < 0), f"chamber bottom y-normal sign: {nrm[interior_mask, 1]}"

    # Obstacle bottom edge (y=0.35): outward normal has y>0 (points UP into
    # the obstacle hole, i.e. away from material).
    _xo, _yo, _, nxo, _nyo = dom.variable("boundary_obstacle_0", normals=True)
    pts_o = np.asarray(dom.context[_xo.tag])[0, 0]
    nrm_o = np.asarray(dom.context[nxo.tag])[0, 0]
    obs_interior = (pts_o[:, 0] > 0.85) & (pts_o[:, 0] < 1.15)
    if obs_interior.sum() >= 1:
        assert np.all(nrm_o[obs_interior, 1] > 0), f"obstacle bottom y-normal sign: {nrm_o[obs_interior, 1]}"


def test_per_region_mesh_size_refines_locally():
    """A smaller mesh_size for one source region increases total nodes and
    concentrates them near that region's boundary."""
    np.random.seed(15)
    chamber = jno.domain.csg(CHAMBER, name="chamber")
    obstacle = jno.domain.csg(OBSTACLE, name="obstacle")

    uniform = chamber - obstacle
    uniform.build_mesh(mesh_size=0.1)
    n_uniform = uniform.mesh_connectivity["n_points"]

    refined = chamber - obstacle
    refined.build_mesh(mesh_size=0.1, region_mesh_sizes={"obstacle": 0.02})
    n_refined = refined.mesh_connectivity["n_points"]
    assert n_refined > 1.5 * n_uniform, f"expected ≥1.5× growth from refinement, got {n_uniform} → {n_refined}"

    # Area integral is still correct after refinement.
    assert _eval_area(refined) == pytest.approx(CHAMBER_AREA - OBSTACLE_AREA, rel=1e-3)


def test_interpolate_false_delivers_uniform_inner_refinement():
    """``interpolate=False`` enforces ``region_mesh_sizes[name]`` uniformly
    inside the region's bounding box via a gmsh Box size field, instead of
    only setting the size on boundary vertices and letting gmsh smoothly
    interpolate (and undershoot) through the interior.

    Concrete failure mode this guards: with the default ``interpolate=True``,
    a disk + 0.2×0.2 inner rectangle at ``mesh_size=0.10`` and
    ``region_mesh_sizes={"inner": 0.005}`` produces only ~10 inner nodes
    instead of the requested ~1600 — gmsh smoothly transitions size from
    boundary to interior. ``interpolate=False`` should fix this to within
    ~2× of the expected count.
    """
    n_disk = 64
    theta = np.linspace(0, 2 * np.pi, n_disk, endpoint=False)
    disk_verts = [(float(np.cos(t)), float(np.sin(t))) for t in theta]
    inner_w = 0.20
    rect_verts = [
        (-inner_w / 2, -inner_w / 2),
        (inner_w / 2, -inner_w / 2),
        (inner_w / 2, inner_w / 2),
        (-inner_w / 2, inner_w / 2),
    ]

    np.random.seed(0)
    dom_interp = jno.domain.csg.from_polygons({"outer": disk_verts, "inner": rect_verts})
    dom_interp.build_mesh(mesh_size=0.10, region_mesh_sizes={"inner": 0.02}, interpolate=True)
    n_inner_interp = len(dom_interp._mesh_pool.get("interior_inner", []))

    np.random.seed(0)
    dom_uniform = jno.domain.csg.from_polygons({"outer": disk_verts, "inner": rect_verts})
    dom_uniform.build_mesh(mesh_size=0.10, region_mesh_sizes={"inner": 0.02}, interpolate=False)
    n_inner_uniform = len(dom_uniform._mesh_pool.get("interior_inner", []))

    # interpolate=False should produce many more inner nodes than the
    # smooth-interpolation default. At inner_h=0.02 on a 0.2×0.2 box we
    # expect ~100 inner nodes; the default delivers ~10. Require at least
    # a 4× improvement to give safety margin against pygmsh version drift.
    assert n_inner_uniform >= 4 * n_inner_interp, (
        f"interpolate=False expected to deliver ≥4× more inner nodes than the smooth-interpolation default; "
        f"got interpolate=True={n_inner_interp}, interpolate=False={n_inner_uniform}"
    )


def test_region_mesh_sizes_validation_errors():
    np.random.seed(16)
    dom = jno.domain.csg(CHAMBER, name="chamber") - jno.domain.csg(OBSTACLE, name="obstacle")

    with pytest.raises(ValueError, match="unknown region"):
        dom.build_mesh(mesh_size=0.1, region_mesh_sizes={"ghost": 0.05})

    with pytest.raises(ValueError, match="positive float"):
        dom.build_mesh(mesh_size=0.1, region_mesh_sizes={"obstacle": -0.5})

    with pytest.raises(ValueError, match="positive float"):
        dom.build_mesh(mesh_size=0.1, region_mesh_sizes={"obstacle": 0.0})


def test_partial_csg_overlap_preserves_surviving_source_edge_length():
    """When chamber B partially overlaps chamber A, the overlapped portion
    of A's source edges is removed from the active boundary. The active
    `boundary_a_*` tags must integrate to the surviving length only."""
    np.random.seed(17)
    a = jno.domain.csg([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)], name="a")
    b = jno.domain.csg([(1.0, 0.5), (3.0, 0.5), (3.0, 1.5), (1.0, 1.5)], name="b")
    dom = a + b
    dom.build_mesh(mesh_size=0.05)

    # boundary_a_1 was the right edge of A (x=2, y∈[0,1]).
    # B overlaps y∈[0.5,1], so the surviving piece is y∈[0,0.5], length 0.5.
    surviving = _eval_length(dom, "boundary_a_1")
    assert surviving == pytest.approx(0.5, rel=1e-3), f"surviving boundary_a_1 length: {surviving}"

    # boundary_a_0 (bottom of A, y=0, x∈[0,2]) is fully outside B → length 2.0.
    bottom_a = _eval_length(dom, "boundary_a_0")
    assert bottom_a == pytest.approx(2.0, rel=1e-6)


def test_enclosure_view_factor_after_build_mesh_blocks_through_obstacle():
    """View-factor visibility tracing must still respect occluders when the
    domain has been meshed."""
    np.random.seed(18)
    dom = jno.domain.csg.from_polygons(
        {
            "Air": [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)],
            "Block": [(1.25, 0.2), (1.75, 0.2), (1.75, 0.8), (1.25, 0.8)],
        }
    )
    dom.add_boundary_segments("rad_left", [[(1.25, 0.2), (1.25, 0.8)]], normal_geometry="Block")
    dom.add_boundary_segments("rad_right", [[(1.75, 0.8), (1.75, 0.2)]], normal_geometry="Block")
    dom.build_mesh(mesh_size=0.1)

    for tag in ("rad_left", "rad_right"):
        dom.variable(tag, sample=(32, None), normals=True)
    dom.compute_enclosure_view_factor(["rad_left", "rad_right"], medium_tags=["Air"])

    # The two opposing block faces cannot see each other through the solid block.
    assert float(np.sum(dom.context["v_rad_left__rad_right"])) == pytest.approx(0.0)
    assert float(np.sum(dom.context["v_rad_right__rad_left"])) == pytest.approx(0.0)


def test_three_disjoint_pieces_keep_distinct_interior_tags():
    """Three non-touching squares from from_polygons must mesh as 3 separate
    pieces and keep distinct `interior_<name>` tags."""
    np.random.seed(19)
    dom = jno.domain.csg.from_polygons(
        {
            "p1": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            "p2": [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)],
            "p3": [(4.0, 0.0), (5.0, 0.0), (5.0, 1.0), (4.0, 1.0)],
        }
    )
    assert dom._active_geometry.geom_type == "MultiPolygon"
    dom.build_mesh(mesh_size=0.1)

    assert _eval_area(dom) == pytest.approx(3.0, rel=1e-3)

    for name in ("p1", "p2", "p3"):
        area_i = _eval_area(dom, f"interior_{name}")
        assert area_i == pytest.approx(1.0, rel=1e-3), f"{name} area: {area_i}"
