"""Structured meshes of quadrilaterals and hexahedra — the gmsh-free route to a tensor-product mesh.

``equi_distant_rect`` / ``equi_distant_box`` already build the exact structured node layout and then
throw the structure away: the 2-D constructor splits every rectangle into two triangles, and the 3-D
one splits every voxel into six Kuhn tets. ``cell="quad"`` / ``cell="hex"`` keeps the cell whole.

That matters for three reasons: it is the mesh the topology-optimisation literature is written on, it
is a perfectly graded mesh for layered geometry, and — because it needs no mesher — it is the one
quad/hex source with no gmsh risk attached.

These tests pin the *mesh*, not the solve: block types, cell counts, node ordering, and the geometry
that ordering implies. A wrong vertex order is the failure mode that survives every smoke test and
then silently produces a negative Jacobian, so it is checked directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from jno.domain.geometries import Geometries

# These target the geometry-func factories rather than `jno.domain.equi_distant_rect`, which builds a
# whole domain: mesh INGESTION of a quad/hex block is the next step, and pinning the emitter first
# keeps the two failures distinguishable.


def _build(constructor):
    """Run a geometry constructor the way ``jno.domain`` does, returning its meshio.Mesh."""
    mesh, dim, ds = constructor(None)
    return mesh, dim, ds


def _blocks(mesh):
    return {c.type: np.asarray(c.data) for c in mesh.cells}


# --------------------------------------------------------------------------------------- 2-D quads


def test_rect_emits_one_quad_per_cell():
    """nx*ny quads, not 2*nx*ny triangles — and the boundary stays a 2-node edge block."""
    mesh, dim, _ = _build(Geometries.equi_distant_rect(nx=4, ny=3, cell="quad"))
    blocks = _blocks(mesh)
    assert dim == 2
    assert "triangle" not in blocks, "cell='quad' still emitted triangles"
    assert blocks["quad"].shape == (12, 4)
    # A quad's facet is a straight 2-node edge, exactly as for a triangle: the boundary
    # representation is unchanged, which is why 2-D quads keep the tie/mortar machinery.
    assert blocks["line"].shape[1] == 2


def test_quad_node_order_is_counterclockwise():
    """VTK order (p00, p10, p11, p01). Checked via the shoelace area: counterclockwise is POSITIVE.

    A clockwise cell gives det J < 0 everywhere, which most solves survive with a sign flip and no
    error — this is the assertion that catches it at the source.
    """
    mesh, _, _ = _build(Geometries.equi_distant_rect(x_range=(0.0, 2.0), y_range=(0.0, 1.0), nx=4, ny=2, cell="quad"))
    pts, quads = np.asarray(mesh.points)[:, :2], _blocks(mesh)["quad"]
    v = pts[quads]  # (n_cells, 4, 2)
    x, y = v[..., 0], v[..., 1]
    shoelace = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    assert np.all(shoelace > 0), "quad vertices are not counterclockwise"
    np.testing.assert_allclose(shoelace, (2.0 / 4) * (1.0 / 2), rtol=1e-12)
    np.testing.assert_allclose(shoelace.sum(), 2.0, rtol=1e-12)  # tiles the rectangle exactly


def test_quad_cells_tile_the_rectangle_without_gaps():
    """Every node is shared by its neighbours: a conforming mesh has no duplicated coordinates."""
    mesh, _, _ = _build(Geometries.equi_distant_rect(nx=5, ny=4, cell="quad"))
    pts = np.asarray(mesh.points)[:, :2]
    assert len(np.unique(pts, axis=0)) == len(pts) == 6 * 5
    quads = _blocks(mesh)["quad"]
    assert set(np.unique(quads)) == set(range(len(pts))), "some nodes belong to no cell"


def test_quad_boundary_tags_survive():
    """The cell_sets contract is what `variable('left')` reads — cell='quad' must not disturb it."""
    mesh, _, _ = _build(Geometries.equi_distant_rect(nx=3, ny=3, cell="quad"))
    for name in ("interior", "boundary", "left", "right", "top", "bottom"):
        assert name in mesh.cell_sets, f"missing cell_set {name!r}"
    blocks = [c.type for c in mesh.cells]
    vol = blocks.index("quad")
    n_interior = len(mesh.cell_sets["interior"][vol])
    assert n_interior == 9, "interior must select every volume cell"
    edge = blocks.index("line")
    assert len(mesh.cell_sets["boundary"][edge]) == 4 * 3


def test_triangles_remain_the_default():
    """Backwards compatibility: the existing spelling is untouched."""
    mesh, _, _ = _build(Geometries.equi_distant_rect(nx=3, ny=3))
    blocks = _blocks(mesh)
    assert blocks["triangle"].shape == (18, 3) and "quad" not in blocks


# ---------------------------------------------------------------------------------------- 3-D hexes


def test_box_emits_one_hex_per_voxel():
    """nx*ny*nz hexes instead of 6x that many Kuhn tets, and QUAD boundary faces."""
    mesh, dim, _ = _build(Geometries.equi_distant_box(nx=3, ny=2, nz=2, cell="hex"))
    blocks = _blocks(mesh)
    assert dim == 3
    assert "tetra" not in blocks and "triangle" not in blocks
    assert blocks["hexahedron"].shape == (12, 8)
    assert blocks["quad"].shape[1] == 4
    # 2*(3*2 + 3*2 + 2*2) = 32 boundary quads
    assert blocks["quad"].shape[0] == 32


def test_hex_node_order_gives_positive_volume():
    """VTK hexahedron order: bottom face ccw (0-3) then the top face above it (4-7).

    Volume is computed by the divergence theorem over the 6 faces, which depends on BOTH the
    vertex order and the face table being consistent — a stronger check than a bounding box.
    """
    mesh, _, _ = _build(Geometries.equi_distant_box(x_range=(0.0, 2.0), nx=2, ny=2, nz=2, cell="hex"))
    pts, hexes = np.asarray(mesh.points), _blocks(mesh)["hexahedron"]
    v = pts[hexes]  # (n_cells, 8, 3)
    # VTK faces, outward-oriented
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    vol = np.zeros(len(hexes))
    for a, b, c, d in faces:
        for t in ((a, b, c), (a, c, d)):  # split each quad face into two triangles
            p0, p1, p2 = v[:, t[0]], v[:, t[1]], v[:, t[2]]
            vol += np.einsum("ij,ij->i", p0, np.cross(p1 - p0, p2 - p0)) / 6.0
    assert np.all(vol > 0), "hex vertices are not right-handed (negative volume)"
    np.testing.assert_allclose(vol, (2.0 / 2) * 0.5 * 0.5, rtol=1e-12)
    np.testing.assert_allclose(vol.sum(), 2.0 * 1.0 * 1.0, rtol=1e-12)


def test_hex_bottom_and_top_faces_are_stacked():
    """Node k+4 sits directly above node k — the property the VTK ordering encodes."""
    mesh, _, _ = _build(Geometries.equi_distant_box(nx=2, ny=2, nz=2, cell="hex"))
    v = np.asarray(mesh.points)[_blocks(mesh)["hexahedron"]]
    np.testing.assert_allclose(v[:, 4:, :2], v[:, :4, :2], atol=1e-12)  # same x,y
    assert np.all(v[:, 4:, 2] > v[:, :4, 2]), "top face is not above the bottom face"


def test_box_boundary_tags_survive():
    mesh, _, _ = _build(Geometries.equi_distant_box(nx=2, ny=2, nz=2, cell="hex"))
    for name in ("interior", "boundary", "left", "right", "top", "bottom", "front", "back"):
        assert name in mesh.cell_sets, f"missing cell_set {name!r}"
    blocks = [c.type for c in mesh.cells]
    assert len(mesh.cell_sets["interior"][blocks.index("hexahedron")]) == 8
    assert len(mesh.cell_sets["boundary"][blocks.index("quad")]) == 24


def test_tets_remain_the_default():
    mesh, _, _ = _build(Geometries.equi_distant_box(nx=2, ny=2, nz=2))
    blocks = _blocks(mesh)
    assert blocks["tetra"].shape == (48, 4) and "hexahedron" not in blocks


# --------------------------------------------------------------------------------------- extremes


@pytest.mark.parametrize("nx,ny", [(1, 1), (1, 7), (7, 1)])
def test_degenerate_grid_shapes(nx, ny):
    """A single cell, and single-cell-wide strips — the off-by-one cases in the index arithmetic."""
    mesh, _, _ = _build(Geometries.equi_distant_rect(nx=nx, ny=ny, cell="quad"))
    quads = _blocks(mesh)["quad"]
    assert quads.shape == (nx * ny, 4)
    assert set(np.unique(quads)) == set(range((nx + 1) * (ny + 1)))


def test_negative_and_offset_coordinate_ranges():
    """Ranges that straddle the origin: orientation must not depend on the sign of the coordinates."""
    mesh, _, _ = _build(Geometries.equi_distant_rect(x_range=(-3.0, -1.0), y_range=(-2.0, 5.0), nx=3, ny=3, cell="quad"))
    pts, quads = np.asarray(mesh.points)[:, :2], _blocks(mesh)["quad"]
    v = pts[quads]
    x, y = v[..., 0], v[..., 1]
    shoelace = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    assert np.all(shoelace > 0)
    np.testing.assert_allclose(shoelace.sum(), 2.0 * 7.0, rtol=1e-12)


def test_high_aspect_ratio_cells_stay_oriented():
    """1000:1 cells — the stretched case a boundary layer produces."""
    mesh, _, _ = _build(Geometries.equi_distant_box(x_range=(0.0, 1000.0), nx=2, ny=2, nz=2, cell="hex"))
    v = np.asarray(mesh.points)[_blocks(mesh)["hexahedron"]]
    assert np.all(v[:, 4:, 2] > v[:, :4, 2])
    assert len(v) == 8


def test_an_unknown_cell_kind_is_refused():
    """Loud refusal, naming what was asked for — never a silent fallback to triangles."""
    with pytest.raises(ValueError, match="cell="):
        Geometries.equi_distant_rect(nx=2, ny=2, cell="pentagon")
    with pytest.raises(ValueError, match="cell="):
        Geometries.equi_distant_box(nx=2, ny=2, nz=2, cell="quad")  # quad is 2-D; the box wants "hex"
