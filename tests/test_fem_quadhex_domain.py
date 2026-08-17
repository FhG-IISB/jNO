"""A quadrilateral or hexahedral mesh becomes a domain: measures, normals and tagged facets.

The oracle throughout is the *simplex mesh of the same geometry*. A structured quad mesh and the
right-triangulation of the same grid describe an identical region, so total volume, total surface
area, the boundary node set and every outward normal must agree exactly — not to a tolerance, since
both are computed from the same nodes. That makes these tests independent of any analytic formula
and sensitive to the things that actually break: a reversed cell, a facet dropped on the floor, a
normal oriented inward.

The failure this file exists to prevent is the quiet one. Point-based tags (`tag_indices`) are
derived from node coordinates and looked correct for a hexahedral mesh from the very first run,
while the tagged *facets* underneath them were being discarded — the mesh knew where "left" was but
had no faces to integrate over.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.domain.geometries import Geometries
from jno.domain.mesh_utils import MeshUtils

RECT = dict(x_range=(0.0, 2.0), y_range=(0.0, 1.0), nx=4, ny=3)
BOX = dict(x_range=(0.0, 2.0), y_range=(0.0, 1.0), z_range=(0.0, 1.0), nx=3, ny=2, nz=2)


def _domain(cell=None, three_d=False, mc=True):
    ctor = (
        Geometries.equi_distant_box(**BOX, **({"cell": cell} if cell else {}))
        if three_d
        else Geometries.equi_distant_rect(**RECT, **({"cell": cell} if cell else {}))
    )
    return jno.domain(constructor=ctor, compute_mesh_connectivity=mc)


@pytest.fixture(scope="module")
def pair_2d():
    return _domain(), _domain(cell="quad")


@pytest.fixture(scope="module")
def pair_3d():
    return _domain(three_d=True), _domain(cell="hex", three_d=True)


# ------------------------------------------------------------------ the mesh reaches the domain


def test_a_quad_mesh_builds_a_domain(pair_2d):
    tri, quad = pair_2d
    assert quad.dimension == 2
    assert len(quad.mesh.points) == len(tri.mesh.points), "the two meshes must share their nodes"
    assert quad.mesh_connectivity["cell_type"] == "quad"
    assert tri.mesh_connectivity["cell_type"] == "triangle"


def test_a_hex_mesh_builds_a_domain(pair_3d):
    tet, hexd = pair_3d
    assert hexd.dimension == 3
    assert len(hexd.mesh.points) == len(tet.mesh.points)
    assert hexd.mesh_connectivity["cell_type"] == "hexahedron"


def test_the_constant_p1_gradient_is_absent_not_wrong(pair_2d):
    """`p1_grad_phi` is the CONSTANT per-cell P1 gradient the FD stencils are built on. A bilinear
    quad's gradient is not constant, so the key must be missing rather than filled with a
    plausible-looking value computed from three of the four nodes."""
    tri, quad = pair_2d
    assert "p1_grad_phi" in tri.mesh_connectivity
    assert "p1_grad_phi" not in quad.mesh_connectivity


# --------------------------------------------------------------------------------- the measures


def test_nodal_volumes_sum_to_the_same_region(pair_2d, pair_3d):
    """Total measure is a property of the REGION, so the two discretizations must agree exactly."""
    tri, quad = pair_2d
    tet, hexd = pair_3d
    np.testing.assert_allclose(np.asarray(quad.mesh_connectivity["nodal_volumes"]).sum(), 2.0, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(quad.mesh_connectivity["nodal_volumes"]).sum(),
        np.asarray(tri.mesh_connectivity["nodal_volumes"]).sum(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(hexd.mesh_connectivity["nodal_volumes"]).sum(), 2.0, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(hexd.mesh_connectivity["nodal_volumes"]).sum(),
        np.asarray(tet.mesh_connectivity["nodal_volumes"]).sum(),
        rtol=1e-12,
    )


def test_nodal_ds_sums_to_the_same_surface(pair_2d, pair_3d):
    tri, quad = pair_2d
    tet, hexd = pair_3d
    np.testing.assert_allclose(np.asarray(quad.mesh_connectivity["nodal_ds"]).sum(), 6.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(hexd.mesh_connectivity["nodal_ds"]).sum(), 2 * (2 + 2 + 1), rtol=1e-12)
    for a, b in ((tri, quad), (tet, hexd)):
        np.testing.assert_allclose(
            np.asarray(a.mesh_connectivity["nodal_ds"]).sum(),
            np.asarray(b.mesh_connectivity["nodal_ds"]).sum(),
            rtol=1e-12,
        )


def test_hex_volume_is_not_a_parallelepiped_shortcut():
    """A general hexahedron's volume is not the product of its edge lengths. The divergence-theorem
    formula must reproduce a known non-box volume — here a uniformly sheared cube, whose volume is
    unchanged by the shear (det of a shear is 1)."""
    mesh, _, _ = Geometries.equi_distant_box(nx=2, ny=2, nz=2, cell="hex")(None)
    pts = np.asarray(mesh.points).copy()
    pts[:, 0] += 0.7 * pts[:, 2]  # shear x by z
    cells = {c.type: np.asarray(c.data) for c in mesh.cells}["hexahedron"]
    vols = MeshUtils._hex_signed_volumes(pts, cells)
    assert np.all(vols > 0), "shearing must not invert a cell"
    np.testing.assert_allclose(vols.sum(), 1.0, rtol=1e-12)


# ---------------------------------------------------------------------------------- the normals


def test_2d_boundary_normals_match_the_triangle_mesh(pair_2d):
    """Same nodes, same region ⇒ the same outward normal at every boundary node.

    A quad edge has no opposite vertex, so it is oriented by the owning cell's centroid rather than
    the apex; this asserts that substitution changes nothing. In 2-D the agreement is exact
    everywhere, because a quad mesh and its triangulation have the *same* boundary edges.
    """
    tri, quad = pair_2d
    ns, idx_s = MeshUtils.get_boundary_normals(tri.mesh)
    nt, idx_t = MeshUtils.get_boundary_normals(quad.mesh)
    np.testing.assert_array_equal(idx_s, idx_t)
    np.testing.assert_allclose(np.asarray(ns)[:, :2], np.asarray(nt)[:, :2], atol=1e-12)


def test_3d_boundary_normals_match_the_tet_mesh_away_from_corners(pair_3d):
    """Exact agreement everywhere a vertex normal is well defined — which excludes the corners.

    A per-vertex normal is the area-weighted average of the faces meeting at it. Splitting each
    quadrilateral face into two triangles changes how much of that face's area reaches each of its
    nodes, so at a box CORNER (three faces meeting, no unique normal) the two meshes legitimately
    disagree: measured 3.4e-1 there, against 0.0 on face interiors and 1.6e-16 along edges. Pinning
    corner agreement would be pinning an artefact of the triangulation, so the test pins the
    property that is real.
    """
    tet, hexd = pair_3d
    ns, idx_s = MeshUtils.get_boundary_normals(tet.mesh)
    nt, idx_t = MeshUtils.get_boundary_normals(hexd.mesh)
    np.testing.assert_array_equal(idx_s, idx_t)

    pts = np.asarray(tet.mesh.points)[idx_s]
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])
    n_faces_at_node = ((np.abs(pts - lo) < 1e-12) | (np.abs(pts - hi) < 1e-12)).sum(axis=1)
    not_corner = n_faces_at_node < 3
    assert not_corner.sum() >= 20, "the mesh must have enough non-corner boundary nodes to be a test"
    np.testing.assert_allclose(np.asarray(ns)[not_corner, :3], np.asarray(nt)[not_corner, :3], atol=1e-12)

    # On a face interior the normal is not merely equal to the tet mesh's, it is exactly an axis.
    face_interior = n_faces_at_node == 1
    axes = np.abs(np.asarray(nt)[face_interior, :3])
    np.testing.assert_allclose(axes.max(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sort(axes, axis=1)[:, :2], 0.0, atol=1e-12)


@pytest.mark.parametrize("three_d", [False, True])
def test_boundary_normals_are_outward_and_unit(three_d, pair_2d, pair_3d):
    _, tensor = pair_3d if three_d else pair_2d
    dim = 3 if three_d else 2
    n, idx = MeshUtils.get_boundary_normals(tensor.mesh)
    n = np.asarray(n)[:, :dim]
    pts = np.asarray(tensor.mesh.points)[idx][:, :dim]
    centre = np.array([1.0, 0.5, 0.5][:dim])
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    assert np.all(np.einsum("ij,ij->i", n, pts - centre) > 0), "a normal points inward"


# ------------------------------------------------------------ the tagged facets (the silent drop)


def test_hex_boundary_facets_are_captured_not_dropped(pair_3d):
    """THE regression. A hexahedron's boundary face has four nodes, and the two facet stores held
    only 2- and 3-node facets, so every tagged face of a hex mesh was silently discarded — while
    the point-based tags kept looking correct."""
    tet, hexd = pair_3d
    for tag in ("boundary", "left", "right", "top", "bottom", "front", "back"):
        facets = hexd.tag_facets(tag)
        assert facets is not None and len(facets), f"tag {tag!r} lost its facets"
        assert facets.shape[1] == 4, f"tag {tag!r} facets should be quadrilaterals"
    # each tet-mesh face is one quad split in two
    assert len(hexd.tag_facets("boundary")) * 2 == len(tet.tag_facets("boundary"))


def test_quad_volume_cells_are_captured(pair_2d):
    """In 2-D the 4-node block is the VOLUME cell, not a facet — it must be stored too."""
    tri, quad = pair_2d
    interior = quad.tag_facets("interior")
    assert interior is not None and interior.shape[1] == 4
    assert len(interior) * 2 == len(tri.tag_facets("interior"))


def test_quad_boundary_edges_are_unchanged(pair_2d):
    """A quad's facet is a straight 2-node edge, exactly as a triangle's is."""
    tri, quad = pair_2d
    for tag in ("boundary", "left", "right", "top", "bottom"):
        np.testing.assert_array_equal(np.sort(quad._tag_edges[tag], axis=1), np.sort(tri._tag_edges[tag], axis=1))


@pytest.mark.parametrize("three_d", [False, True])
def test_point_tags_agree_with_the_simplex_mesh(three_d, pair_2d, pair_3d):
    simplex, tensor = pair_3d if three_d else pair_2d
    for tag in ("boundary", "left", "right", "interior"):
        np.testing.assert_array_equal(
            np.sort(np.atleast_1d(simplex.tag_indices[tag])),
            np.sort(np.atleast_1d(tensor.tag_indices[tag])),
        )


def test_tag_facets_returns_none_for_an_unknown_tag(pair_3d):
    _, hexd = pair_3d
    assert hexd.tag_facets("no_such_tag") is None


# --------------------------------------------------------------------------- refusals & extremes


def test_the_facet_table_still_refuses_what_it_cannot_do():
    from jno.utils.solver.fem_facets import _face_table, has_facet_apex

    assert _face_table("quadrilateral")[1] == 2  # a quad's facet is a 2-node edge
    assert _face_table("hexahedron")[1] == 4  # a hex's facet is a 4-node quadrilateral
    assert has_facet_apex("triangle") and not has_facet_apex("hexahedron")
    with pytest.raises(NotImplementedError, match="prism"):
        _face_table("prism")


def test_a_single_cell_mesh():
    """One quad and one hex — the degenerate case where every facet is a boundary facet."""
    d2 = jno.domain(constructor=Geometries.equi_distant_rect(nx=1, ny=1, cell="quad"), compute_mesh_connectivity=True)
    d3 = jno.domain(constructor=Geometries.equi_distant_box(nx=1, ny=1, nz=1, cell="hex"), compute_mesh_connectivity=True)
    np.testing.assert_allclose(np.asarray(d2.mesh_connectivity["nodal_volumes"]).sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(d3.mesh_connectivity["nodal_volumes"]).sum(), 1.0, rtol=1e-12)
    assert len(d3.tag_facets("boundary")) == 6


def test_a_high_aspect_ratio_mesh():
    """1000:1 cells — the boundary-layer shape, where a normal built from a near-degenerate cross
    product is most likely to lose its orientation."""
    d = jno.domain(
        constructor=Geometries.equi_distant_box(x_range=(0.0, 1000.0), nx=2, ny=2, nz=2, cell="hex"),
        compute_mesh_connectivity=True,
    )
    n, idx = MeshUtils.get_boundary_normals(d.mesh)
    n = np.asarray(n)[:, :3]
    pts = np.asarray(d.mesh.points)[idx]
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    assert np.all(np.einsum("ij,ij->i", n, pts - np.array([500.0, 0.5, 0.5])) > 0)
    np.testing.assert_allclose(np.asarray(d.mesh_connectivity["nodal_volumes"]).sum(), 1000.0, rtol=1e-12)
