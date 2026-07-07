"""Tests for the gmsh-OCC ``Shape`` geometry layer (jno/geometry).

Covers the pure naming/selection algebra (no gmsh), the mesh + ``cell_sets`` contract
that ``jno.domain`` consumes, per-shape graded sizing, and -- the assertion that drove
the shared adjacent-element normal fix -- correct outward normals on a concave arc.
"""

import numpy as np
import pytest

from jno.geometry import Shape

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
