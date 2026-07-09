"""Tests for the gmsh-OCC ``Shape`` geometry layer (jno/geometry).

Covers the pure naming/selection algebra (no gmsh), the mesh + ``cell_sets`` contract
that ``jno.domain`` consumes, per-shape graded sizing, and -- the assertion that drove
the shared adjacent-element normal fix -- correct outward normals on a concave arc.
"""

import math

import numpy as np
import pytest

from jno.geometry import Path, Shape

gmsh = pytest.importorskip("gmsh", reason="gmsh-OCC required for the Shape mesher")


# --------------------------------------------------------------------------- pure
def test_selection_algebra_no_gmsh():
    """edge()/edges_from()/| compose without touching a mesher."""
    strip = Shape.rect(0, 0, 4, 1)
    roll = Shape.disk(2, 1, 0.5)
    solid = (strip - roll).extrude(0.7)

    # every primitive is reachable; keys are identity-stable through the plan
    assert len(solid.leaves()) == 2
    assert solid.keys() == strip.keys() | roll.keys()

    merged = solid.edge("top") | solid.edges_from(roll)
    assert merged.matches(next(iter(roll.keys())), "arc")  # the disk's arc
    assert merged.matches(12345, "top")  # any 'top' edge, regardless of provenance
    assert not merged.matches(999, "bottom")


def test_extrude_requires_2d():
    with pytest.raises(ValueError):
        Shape.box(0, 0, 0, 1, 1, 1).extrude(1.0)


# --------------------------------------------------------------------------- mesh
def test_rect_minus_disk_2d_regions():
    """rect - disk: flat 'top' splits into two segments; the dip becomes 'arc'."""
    mesh, dim, ds = (Shape.rect(0, 0, 4, 1) - Shape.disk(2, 1, 0.5)).build()
    assert dim == 2
    types = {b.type for b in mesh.cells}
    assert types == {"triangle", "line"}

    def n(tag):
        return sum(len(a) for a in mesh.cell_sets[tag])

    for tag in ("interior", "boundary", "left", "right", "bottom", "top", "arc"):
        assert tag in mesh.cell_sets, tag
    # named boundary regions partition 'boundary' exactly
    assert n("left") + n("right") + n("bottom") + n("top") + n("arc") == n("boundary")
    assert n("interior") == len(mesh.cells[0].data)
    assert n("arc") > 0 and n("top") > 0  # arc is separate from the flat top (mergeable later)


def test_extrude_3d_mesh_and_caps():
    """Extrusion yields a tetra volume, one lateral face per base edge, plus front/back caps."""
    mesh, dim, ds = (Shape.rect(0, 0, 4, 1) - Shape.disk(2, 1, 0.5)).extrude(0.7).build()
    assert dim == 3
    assert mesh.cells[0].type == "tetra" and mesh.cells[1].type == "triangle"

    def n(tag):
        return sum(len(a) for a in mesh.cell_sets[tag])

    for tag in ("interior", "boundary", "front", "back", "left", "right", "bottom", "top", "arc"):
        assert tag in mesh.cell_sets, tag
    assert n("front") > 0 and n("back") > 0
    lateral = n("left") + n("right") + n("bottom") + n("top") + n("arc")
    assert lateral + n("front") + n("back") == n("boundary")

    # every boundary triangle is a face of some tetra (well-formed surface)
    tets = np.asarray(mesh.cells[0].data)
    tris = np.asarray(mesh.cells[1].data)
    tet_faces = set()
    for t in tets:
        for f in ((t[0], t[1], t[2]), (t[0], t[1], t[3]), (t[0], t[2], t[3]), (t[1], t[2], t[3])):
            tet_faces.add(tuple(sorted(map(int, f))))
    assert all(tuple(sorted(map(int, tr))) in tet_faces for tr in tris)


def test_per_shape_size_grades_mesh():
    """A finer `size` on one shape refines the mesh near its boundary, not globally."""
    coarse = Shape.rect(0, 0, 4, 1, size=0.2)
    fine_disk = Shape.disk(2, 1, 0.5, size=0.03)
    mesh, _dim, ds = (coarse - fine_disk).build()
    pts = mesh.points[:, :2]
    lines = np.asarray(mesh.cells[1].data)

    def mean_edge_len(tag):
        idx = mesh.cell_sets[tag][1]
        seg = lines[idx]
        return float(np.linalg.norm(pts[seg[:, 0]] - pts[seg[:, 1]], axis=1).mean())

    assert mean_edge_len("arc") < 0.5 * mean_edge_len("bottom")  # arc clearly finer
    assert ds == pytest.approx(0.03)


# ------------------------------------------------------------------- domain bridge
def test_domain_regions_and_outward_arc_normals():
    """jno.domain(Shape) exposes every region and gives OUTWARD normals on the concave arc."""
    import jno

    cx, cy, r = 2.0, 1.0, 0.4
    d = jno.domain((Shape.rect(0, 0, 4, 1, size=0.2) - Shape.disk(cx, cy, r, size=0.08)).extrude(0.6))
    assert d.dimension == 3
    assert {"interior", "boundary", "arc", "top", "left", "right", "bottom", "front", "back"} <= set(d.avaiable_mesh_tags)

    # arc normals must point OUT of the material -> toward the (removed) disk centre in xy
    n = np.asarray(d.normals_by_tag["arc"])
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    arc_pts = d.points[d.tag_indices["arc"]]
    to_center = np.zeros_like(n)
    to_center[:, 0] = cx - arc_pts[:, 0]
    to_center[:, 1] = cy - arc_pts[:, 1]
    to_center /= np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-30
    # outward (toward centre of the carved disk) for every arc node -- no inward flips
    assert np.min(np.sum(n * to_center, axis=1)) > 0.0


# --------------------------------------------------------------- more primitives
def _sets(mesh):
    return {k: sum(len(a) for a in v) for k, v in mesh.cell_sets.items() if sum(len(a) for a in v)}


def test_polygon_edges_auto_named():
    """An L-shaped polygon meshes with one auto-named region per segment (e0..e5)."""
    mesh, dim, _ds = Shape.polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]).build()
    assert dim == 2
    s = _sets(mesh)
    edges = [f"e{i}" for i in range(6)]
    assert all(e in s for e in edges)
    assert sum(s[e] for e in edges) == s["boundary"]


def test_cylinder_arbitrary_axis_faces():
    """A cylinder on the x-axis names lateral 'side' and the two caps 'bottom'/'top'."""
    mesh, dim, _ds = Shape.cylinder(0, 0, 0, 5, 0, 0, 1.0).build()
    assert dim == 3 and mesh.cells[0].type == "tetra"
    s = _sets(mesh)
    assert {"side", "top", "bottom"} <= set(s)
    assert s["side"] + s["top"] + s["bottom"] == s["boundary"]


def test_box_minus_cylinder_drilled_hole():
    """A through-hole keeps the six box faces and adds the cylinder wall 'side'."""
    mesh, dim, _ds = (Shape.box(0, 0, 0, 4, 4, 1) - Shape.cylinder(2, 2, -0.5, 0, 0, 2, 0.5)).build()
    s = _sets(mesh)
    assert {"left", "right", "front", "back", "top", "bottom", "side"} <= set(s)
    assert s["side"] > 0  # the drilled wall


def test_sphere_concave_dimple_outward_normals():
    """box - sphere (a spherical dimple): every 'surface' normal points out of the material."""
    import jno

    c = np.array([1.0, 1.0, 2.0])
    d = jno.domain(Shape.box(0, 0, 0, 2, 2, 2) - Shape.sphere(c[0], c[1], c[2], 0.7, size=0.15))
    n = np.asarray(d.normals_by_tag["surface"])
    pts = d.points[d.tag_indices["surface"]]
    toward_center = c - pts
    toward_center /= np.linalg.norm(toward_center, axis=1, keepdims=True) + 1e-30
    # material is the box outside the sphere -> outward points toward the carved sphere centre
    assert np.min(np.sum(n * toward_center, axis=1)) > 0.0


# ------------------------------------------------------------------------ revolve
_Y = ((0, 0, 0), (0, 1, 0))  # y-axis through the origin


def test_half_donut_tube_and_caps():
    """A disk revolved 180deg about the y-axis: tube 'arc' + two flat end caps 'back'/'front'."""
    mesh, dim, _ds = Shape.disk(2.0, 0.0, 0.6, size=0.25).revolve(*_Y, angle=math.pi).build()
    assert dim == 3
    s = _sets(mesh)
    assert {"arc", "back", "front"} <= set(s)
    assert s["arc"] + s["back"] + s["front"] == s["boundary"]
    assert s["back"] > 0 and s["front"] > 0


def test_full_torus_meshes_via_split_fallback():
    """A full (2pi) detached solid of revolution meshes (two-halves fallback) with no caps."""
    mesh, dim, _ds = Shape.disk(2.0, 0.0, 0.6).revolve(*_Y, angle=2 * math.pi).build()
    assert dim == 3
    s = _sets(mesh)
    assert s["arc"] == s["boundary"]  # closed tube, no end caps
    assert "back" not in s and "front" not in s


def test_cone_from_revolved_triangle():
    """A triangle touching the axis revolved 2pi -> a cone (single-sweep, no split needed)."""
    mesh, dim, _ds = Shape.polygon([(0, 0), (1, 0), (0, 2)]).revolve(*_Y, angle=2 * math.pi).build()
    assert dim == 3
    s = _sets(mesh)
    assert s.get("e0", 0) > 0 and s.get("e1", 0) > 0  # base + slant


def test_revolve_requires_2d_and_supported_axis():
    with pytest.raises(ValueError):
        Shape.box(0, 0, 0, 1, 1, 1).revolve(*_Y, angle=math.pi)
    with pytest.raises(NotImplementedError):  # z-axis not supported
        Shape.disk(2, 0, 0.5).revolve((0, 0, 0), (0, 0, 1), math.pi).build()


# ---------------------------------------------------------------- Path (contours)
def test_path_line_and_arc_face():
    """A half-disk from a diameter (line) + a semicircular arc; segments name the boundary."""
    half = Path(0, -1).line_to(0, 1, name="diameter").arc_to(0, -1, through=(1, 0), name="dome").face()
    mesh, dim, _ds = half.build()
    assert dim == 2
    s = _sets(mesh)
    assert {"diameter", "dome"} <= set(s)
    assert s["diameter"] + s["dome"] == s["boundary"]


def test_path_extrude_named_segments():
    """Contour segment names flow onto the swept lateral faces of an extruded D-prism."""
    dshape = Path(-1, 0).line_to(1, 0, name="flat").arc_to(-1, 0, through=(0, 1), name="round").face()
    mesh, dim, _ds = dshape.extrude(0.5).build()
    assert dim == 3
    s = _sets(mesh)
    assert {"flat", "round", "front", "back"} <= set(s)
    assert s["flat"] + s["round"] + s["front"] + s["back"] == s["boundary"]


def test_path_revolve_makes_sphere():
    """A half-disk (diameter on the axis) revolved 2pi is a sphere with the 'dome' arc named."""
    import jno

    half = Path(0, -1).line_to(0, 1, name="diameter").arc_to(0, -1, through=(1, 0), name="dome").face()
    d = jno.domain(half.revolve((0, 0, 0), (0, 1, 0), 2 * math.pi))
    assert d.dimension == 3
    assert "dome" in d.avaiable_mesh_tags
    n = np.asarray(d.normals_by_tag["dome"])
    pts = d.points[d.tag_indices["dome"]]
    radial = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-30)  # outward = radial at origin
    assert np.min(np.sum(n * radial, axis=1)) > 0.0  # all dome normals point outward


# --------------------------------------------------------- transforms (rotate/translate)
def test_translate_preserves_names_and_position():
    """A translated box keeps its six face-names; the +x face moves with it."""
    mesh, dim, _ds = Shape.box(0, 0, 0, 1, 1, 1).translate((5, 0, 0)).build()
    s = _sets(mesh)
    assert {"left", "right", "top", "bottom", "front", "back"} <= set(s)
    tri, pts = np.asarray(mesh.cells[1].data), mesh.points
    rx = pts[tri[mesh.cell_sets["right"][1]]].mean(1)[:, 0].mean()
    assert abs(rx - 6.0) < 1e-6  # the +x face moved from x=1 to x=6


def test_rotate_preserves_face_names():
    """A 90deg-rotated box is no longer axis-aligned -- names survive only via the
    transform-aware classifier (un-rotating the query point back into the box frame)."""
    mesh, dim, _ds = Shape.box(0, 0, 0, 2, 1, 1).rotate((0, 0, 0), (0, 0, 1), math.pi / 2).build()
    s = _sets(mesh)
    assert {"left", "right", "top", "bottom", "front", "back"} <= set(s)


def test_transform_composes_with_boolean_and_normals():
    """Names + outward normals survive a boolean followed by a translate."""
    import jno

    d = jno.domain((Shape.box(0, 0, 0, 4, 4, 1) - Shape.cylinder(2, 2, -1, 0, 0, 3, 0.5)).translate((10, 10, 0)))
    assert {"side", "top", "left"} <= set(d.avaiable_mesh_tags)  # hole wall + faces survive the move
    assert np.asarray(d.normals_by_tag["top"]).mean(0)[2] > 0.5  # top face still points +z


# ---------------------------------------------------------------------- fillet
def test_fillet_all_edges_keeps_flat_faces():
    """Rounding all edges keeps the six flat faces named; blend faces fall into 'boundary'."""
    mesh, dim, _ds = Shape.box(0, 0, 0, 2, 2, 1).fillet(0.3).build()
    assert dim == 3 and mesh.cells[0].data.shape[0] > 0
    s = _sets(mesh)
    flat = sum(s.get(f, 0) for f in ("left", "right", "top", "bottom", "front", "back"))
    assert {"left", "right", "top", "bottom", "front", "back"} <= set(s)
    assert 0 < flat < s["boundary"]  # blend faces exist, sitting only in 'boundary'


def test_fillet_predicate_and_after_boolean():
    """A predicate rounds a subset of edges, and fillet composes after a boolean (mid-build sync)."""
    mesh, dim, _ds = (
        (Shape.box(0, 0, 0, 4, 4, 1) - Shape.cylinder(2, 2, -1, 0, 0, 3, 0.5))
        .fillet(0.15, where=lambda x, y, z: z > 0.9)
        .build()
    )
    assert dim == 3 and mesh.cells[0].data.shape[0] > 0
    assert "side" in _sets(mesh)  # the drilled hole wall survives the fillet


def test_fillet_outward_normals_via_domain():
    import jno

    d = jno.domain(Shape.box(0, 0, 0, 2, 2, 2).fillet(0.4))
    assert np.asarray(d.normals_by_tag["top"]).mean(0)[2] > 0.9  # flat top still points +z


# ------------------------------------------------------------------------- sweep
def test_sweep_vertical_line_dispatches_to_extrude():
    """A straight vertical sweep IS an extrude -- it reuses the rich naming (caps + lateral 'arc')."""
    mesh, dim, _ds = Shape.disk(0, 0, 0.3).sweep(Path(0, 0, 0).line_to(0, 0, 3)).build()
    assert dim == 3 and mesh.cells[0].data.shape[0] > 0
    assert {"front", "back", "arc"} <= set(_sets(mesh))  # extrude-quality names, not just boundary


def test_sweep_arc_makes_a_bent_pipe():
    """Sweeping along a smooth arc makes a bent pipe (meshes without hanging)."""
    bent = Shape.disk(0, 0, 0.3).sweep(Path(0, 0, 0).arc_to(2, 0, 2, through=(0.6, 0, 1.4)))
    mesh, dim, _ds = bent.build()
    assert dim == 3 and mesh.cells[0].data.shape[0] > 0


def test_sweep_sharp_corner_is_rejected():
    """A sharp line->line corner self-intersects the swept profile -- reject up front, never hang."""
    with pytest.raises(ValueError):
        Shape.disk(0, 0, 0.3).sweep(Path(0, 0, 0).line_to(0, 0, 3).line_to(3, 0, 3))


# ------------------------------------------------------------------------- array
def test_array_linear():
    """A linear array fuses n copies spaced by `step` (pure translate/fuse composition)."""
    mesh, dim, _ds = Shape.disk(0, 0, 0.2).array(3, step=(1, 0, 0)).build()
    assert dim == 2
    tri = np.asarray(mesh.cells[0].data)
    cx = mesh.points[tri].mean(1)[:, 0]
    assert cx.min() < 0.3 and cx.max() > 1.7  # cells span all three disks (x ~ 0, 1, 2)


def test_array_polar_bolt_circle():
    """A polar array makes a ring of holes: plate - disk.array(n, about=axis)."""
    holes = Shape.disk(3, 0, 0.3).array(6, about=((0, 0, 0), (0, 0, 1)))
    mesh, dim, _ds = (Shape.rect(-5, -5, 5, 5) - holes).extrude(0.4).build()
    assert dim == 3 and mesh.cells[0].data.shape[0] > 0
    assert "arc" in _sets(mesh)  # the six hole walls


def test_array_requires_one_mode():
    with pytest.raises(ValueError):
        Shape.disk(0, 0, 0.2).array(3)  # neither step nor about
    with pytest.raises(ValueError):
        Shape.disk(0, 0, 0.2).array(3, step=(1, 0, 0), about=((0, 0, 0), (0, 0, 1)))  # both


# ---------------------------------------------- richer d.tag(f(x, n, name)) predicate
def test_tag_facet_predicate_normal_name_and_backcompat():
    """d.tag(name, f(x, n, name)) selects boundary facets by coords + outward normal + current name;
    the classic f(x, y, z) predicate is unaffected."""
    import jno

    d = jno.domain(Shape.box(0, 0, 0, 2, 2, 1))
    d.tag("east", lambda x, n, name: n[:, 0] > 0.9)  # by outward normal -> the +x face
    assert "east" in d.avaiable_mesh_tags
    assert np.asarray(d.normals_by_tag["east"]).mean(0)[0] > 0.9  # points +x

    # inclusion + exclusion in one predicate: vertical side walls (not top/bottom/boundary)
    d.tag("sides", lambda x, n, name: (name != "top") & (name != "bottom") & (name != "boundary"))
    assert np.abs(np.asarray(d.normals_by_tag["sides"])[:, 2]).mean() < 0.1  # walls point sideways

    d.tag("plain", lambda x, y, z: x < 0.1)  # classic coord predicate still routes the old way
    assert "plain" in d.avaiable_mesh_tags


# ---------------------------------------------------------------- mesh-size control
def test_size_callable_grades_by_position():
    """A callable size f(x,y,z) refines by position (denser where it is smaller)."""
    graded = (Shape.rect(0, 0, 4, 1) - Shape.disk(2, 1, 0.4)).extrude(0.6).sized(lambda x, y, z: 0.03 + 0.12 * x)
    mesh, _dim, _ds = graded.build()
    cells = np.asarray(mesh.cells[0].data)
    cx = mesh.points[cells].mean(1)[:, 0]

    def mean_edge(sub):
        return float(np.linalg.norm(mesh.points[sub[:, 0]] - mesh.points[sub[:, 1]], axis=1).mean())

    assert mean_edge(cells[cx < 1]) < 0.5 * mean_edge(cells[cx > 3])  # left (small size) clearly denser


def test_size_scalar_on_composite_caps_globally():
    """.sized(scalar) on a composite sets a global size cap (denser than the default)."""
    base = (Shape.rect(0, 0, 4, 1) - Shape.disk(2, 1, 0.4)).extrude(0.6)
    capped = (Shape.rect(0, 0, 4, 1) - Shape.disk(2, 1, 0.4)).sized(0.15).extrude(0.6)
    n0 = base.build()[0].cells[0].data.shape[0]
    mesh1, _dim, ds = capped.build()
    assert ds == 0.15 and mesh1.cells[0].data.shape[0] > 3 * n0


# --------------------------------------------------------------- Shape.domain() one-liner
def test_domain_one_liner():
    """Shape.domain() builds a jno.domain (forwarding kwargs) as a one-liner + composes with batching."""
    import jno

    d = Shape.rect(0, 0, 1, 1, size=0.3).domain()
    assert type(d).__name__ == "domain"

    # a full FEM solve straight off the one-liner
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    assert fem.dofs > 0 and np.asarray(fem.solve()).shape == (fem.dofs,)

    # time= forwards to jno.domain (transient -> has an initial slice); batching binds after .domain()
    dt = Shape.rect(0, 0, 1, 1, size=0.3).domain(time=(0.0, 1.0, 5))
    assert dt.variable("initial", split=True) is not None
    assert type(8 * Shape.rect(0, 0, 1, 1, size=0.3).domain()).__name__ == "domain"


# --------------------------------------------------------------------------- 1-D curve
def test_curve_classify_pure_no_gmsh():
    """Curve.classify names the two overall ends; polyline junctions stay interior (no gmsh)."""
    from jno.geometry.primitives import Curve

    c = Curve((0.0, 0.0, 0.0), (("line", (1.0, 0.0, 0.0), None),))
    assert c.dim == 1
    assert c.classify(0.0, 0.0, 0.0) == "left"
    assert c.classify(1.0, 0.0, 0.0) == "right"
    assert c.classify(0.5, 0.0, 0.0) is None
    poly = Curve((0.0, 0.0, 0.0), (("line", (1.0, 0.0, 0.0), None), ("line", (2.0, 0.0, 0.0), None)))
    assert poly.classify(1.0, 0.0, 0.0) is None  # the junction is interior
    assert poly.classify(0.0, 0.0, 0.0) == "left" and poly.classify(2.0, 0.0, 0.0) == "right"


def test_curve_needs_a_segment():
    with pytest.raises(ValueError):
        Path(0.0, 0.0).curve()


def test_path_curve_1d_mesh_blocks_and_sets():
    """An open line path meshes to a 1-D domain: line volume block + vertex boundary block."""
    sh = Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.25)
    assert sh.dim == 1
    mesh, dim, _ds = sh.build()
    assert dim == 1
    assert [cb.type for cb in mesh.cells] == ["line", "vertex"]
    assert {"interior", "left", "right", "boundary"} <= set(mesh.cell_sets)
    # boundary = exactly the two endpoint vertices
    assert int(sum(np.asarray(a).size for a in mesh.cell_sets["boundary"])) == 2


def test_curve_domain_named_endpoints():
    """.domain() exposes interior + named endpoints at the right coordinates (the 1-D BC fix:
    the endpoint vertex block must survive orphan-node dropping)."""
    d = Path(2.0, 0.0).line_to(5.0, 0.0).curve(size=0.25).domain()  # offset interval, not [0,1]
    assert d.dimension == 1
    assert {"interior", "left", "right", "boundary"} <= set(d._mesh_pool)
    assert np.allclose(np.asarray(d._mesh_pool["left"]).reshape(-1), [2.0])
    assert np.allclose(np.asarray(d._mesh_pool["right"]).reshape(-1), [5.0])
    assert np.allclose(np.sort(np.asarray(d._mesh_pool["boundary"]).reshape(-1)), [2.0, 5.0])
