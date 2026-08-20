"""Per-cell geometry and the eq. (17)-(19) patch filter on **quadrilateral** meshes.

Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403 write their deformable-mesh method
on triangles, and Sec. 2.3.2 leaves the extension open. A quadrilateral mesh is worth having for a
reason that is measurable rather than aesthetic: eq. (18) is a geometric mean over ``N - 2`` factors,
so its contrast is governed by patch SIZE, and a structured quad mesh has vertex valence **4** where
a triangulation has ~6. ``test_patch_filter_scaling.py`` measures a one-node connection scoring
0.001 of solid at ``N = 4`` against 0.04 at ``N = 6`` -- the sharpest regime of the whole family.

Everything here has a closed form on a lattice of unit squares, so the assertions are against
analysis rather than against a previous run. The reflex case is the one that matters most: a
quadrilateral can go non-convex while keeping a positive area, which a triangle cannot, so it is a
failure mode a deformable quad mesh has and a deformable triangle mesh simply does not.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

PI = np.pi


def _lattice(nx=4, ny=2, lx=4.0, ly=2.0):
    """A lattice of ``nx * ny`` unit squares -- every quantity below is exact on it."""
    return jno.Shape.rect(0, 0, lx, ly, size=1.0).structured(n=(nx, ny)).quad().domain()


class TestQuadCellGeometry:
    """``cell_volume`` / ``cell_angles`` / ``cell_aspect`` on quadrilaterals."""

    def test_the_topology_accessor_reports_the_cell_kind(self):
        """``_cells_topo`` returns the block name, so callers branch on shape instead of guessing.

        ``_cells_p1`` promises a simplex and a great deal of code reads it that way, so it keeps that
        contract and still refuses here -- the two are siblings, not a replacement.
        """
        d = _lattice()
        cells, kind = d._cells_topo()
        assert kind == "quad"
        assert cells.shape == (8, 4)
        with pytest.raises(ValueError, match="no 'triangle' cells"):
            d._cells_p1()

    def test_a_unit_square_lattice_has_exact_area_angles_and_aspect(self):
        """Area 1, four right angles, aspect exactly 1.0 -- the reference cell of the measure."""
        d = _lattice()
        assert np.allclose(np.asarray(d.cell_volume().eval()), 1.0, atol=1e-6)
        assert float(np.asarray(d.cell_volume().eval()).sum()) == pytest.approx(8.0, abs=1e-5)
        assert np.allclose(np.asarray(d.cell_angles().eval()), PI / 2, atol=1e-6)
        assert np.allclose(np.asarray(d.cell_aspect().eval()), 1.0, atol=1e-6)

    def test_the_shoelace_area_is_not_the_simplex_formula(self):
        """A quadrilateral's area is the shoelace sum, exact for any simple polygon.

        ``|det J| / 2`` -- the simplex formula the triangle path uses -- reads a quadrilateral as the
        parallelogram spanned by two of its first edges, which is right only when it happens to be
        one. Dragging a single node turns four unit squares into trapezia whose exact areas are
        1.15 and 0.85; the simplex formula would report 0.5 for the same cell, so the two cannot be
        confused. The total is invariant: moving one node redistributes area, it cannot create any.
        """
        d = _lattice()  # unit squares on [0, 4] x [0, 2]
        pts = np.asarray(d.mesh.points)
        node = int(np.where(np.isclose(pts[:, 0], 2.0) & np.isclose(pts[:, 1], 1.0))[0][0])
        d.mesh.points[node, 0] += 0.3
        area = np.asarray(d.cell_volume().eval(), dtype=float)
        assert float(area.sum()) == pytest.approx(8.0, abs=1e-5)
        assert np.isclose(area, 1.15, atol=1e-5).sum() == 2, f"expected two cells at 1.15, got {np.sort(area)}"
        assert np.isclose(area, 0.85, atol=1e-5).sum() == 2, f"expected two cells at 0.85, got {np.sort(area)}"

    def test_aspect_ratio_sees_elongation(self):
        """A 2x1 rectangle reads 1.5: longest edge 2 over ``2 * inradius``, ``r = 2A/P = 2/3``."""
        d = _lattice(nx=2, ny=2)  # 4.0 x 2.0 domain in 2 x 2 cells -> each cell is 2 x 1
        assert np.allclose(np.asarray(d.cell_aspect().eval()), 1.5, atol=1e-6)

    def test_a_reflex_corner_is_reported_above_pi(self):
        """The failure mode ``arccos`` cannot express, and a triangle cannot have.

        ``arccos`` of two incident edge vectors has range ``[0, π]``, so a dart quadrilateral reports
        its 250° corner as 110° and a minimum-angle bound sees nothing wrong -- while the cell still
        has positive area, so no other constraint objects either. The four angles must still sum to
        ``2π``, reflex corner included, which is what pins the branch.
        """
        d = _lattice()
        pts = np.asarray(d.mesh.points)
        node = int(np.where((pts[:, 0] > 1.5) & (pts[:, 0] < 2.5) & (pts[:, 1] > 0.5) & (pts[:, 1] < 1.5))[0][0])
        d.mesh.points[node, :2] += 0.85
        ang = np.asarray(d.cell_angles().eval())
        assert ang.max() > PI, f"a reflex corner must read above pi, got {np.degrees(ang.max()):.1f} deg"
        assert np.allclose(ang.sum(axis=1), 2 * PI, atol=1e-5), "four interior angles always sum to 2*pi"
        assert float(np.asarray(d.cell_volume().eval()).min()) > 0.0, (
            "the dart still has positive area -- which is exactly why the angle has to be the guard"
        )

    def test_hexahedra_are_refused_by_name(self):
        """Not yet supported, and said so rather than computed wrongly.

        A hexahedron's volume is not ``|det J| / 6`` and its faces need not be planar, so both the
        volume and the dihedral would be silently wrong if the simplex path were reused.
        """
        d = jno.Shape.box(0, 0, 0, 2, 1, 1, size=1.0).structured(n=(2, 1, 1)).quad().domain()
        assert d._cells_topo()[1] == "hexahedron"
        for call in (d.cell_volume, d.cell_angles, d.cell_aspect):
            with pytest.raises(NotImplementedError, match="hexahedral"):
                call()


class TestQuadPatches:
    """``_patch_topology`` and ``patch_filter`` over 4-element vertex patches."""

    def test_an_interior_vertex_patch_has_exactly_four_elements(self):
        """Valence 4, and it stays 4 wherever the nodes go -- the criterion's ``N`` is a constant.

        On an unstructured mesh ``N`` varies per vertex and so does eq. (18)'s sensitivity; a lattice
        fixes it, so the filter behaves identically at every interior vertex of the domain.
        """
        d = _lattice(nx=6, ny=4, lx=6.0, ly=4.0)
        topo = d._patch_topology()
        assert topo["size"].shape[1] == 4, "an element belongs to one patch per corner"
        assert np.array_equal(np.unique(topo["size"][~topo["boundary"]]), np.array([4]))

    def test_the_filter_is_the_exact_identity_on_solid(self):
        d = _lattice(nx=6, ny=4, lx=6.0, ly=4.0)
        n = d._cells_topo()[0].shape[0]
        assert np.allclose(np.asarray(d.patch_filter()(np.ones(n))), 1.0, atol=0.0)

    def test_a_lone_dense_element_is_driven_to_the_n_equals_four_value(self):
        """0.032, the value ``test_patch_filter_scaling.py`` predicts for a patch of four.

        That the real mesh reproduces the synthetic-ring number is what ties the scaling study to
        this element type: it is the same kernel reading a genuine valence-4 patch.
        """
        d = _lattice(nx=6, ny=4, lx=6.0, ly=4.0)
        cells = d._cells_topo()[0]
        pts = np.asarray(d.mesh.points)
        cen = pts[cells].mean(axis=1)
        k = int(np.argmin(np.linalg.norm(cen[:, :2] - np.array([3.5, 2.5]), axis=1)))  # an interior cell
        r = np.full(cells.shape[0], 1e-3)
        r[k] = 1.0
        f = float(np.asarray(d.patch_filter()(r))[k])
        assert f == pytest.approx(0.032, abs=0.006), f"a lone dense element must be suppressed, got {f:.4f}"


class TestQuadFacetsAndTransfer:
    """The two host-side traversals the perimeter and the reanalysis need."""

    def test_interior_facets_tile_the_cell_edge_slots(self):
        """Every one of the ``4 * n_cells`` edge slots is either shared by two cells or on the boundary."""
        d = _lattice(nx=5, ny=3, lx=5.0, ly=3.0)
        cells = d._cells_topo()[0]
        facets = d._interior_facets()
        n_interior = facets["cells"].shape[0]
        n_boundary = 2 * (5 + 3)  # the lattice's outer edges
        assert facets["nodes"].shape[1] == 2, "a quadrilateral's facet is a 2-node edge"
        assert 2 * n_interior + n_boundary == 4 * cells.shape[0]

    def test_a_piecewise_constant_field_survives_transfer_to_a_finer_quad_mesh(self):
        """The reanalysis path: point location on a quadrilateral is a bilinear inverse, not a solve.

        A field that is constant on each half of the domain must come back constant on each half,
        which is the property a density transfer actually relies on.
        """
        src = _lattice(nx=4, ny=2, lx=4.0, ly=2.0)
        tgt = _lattice(nx=12, ny=6, lx=4.0, ly=2.0)
        cells = src._cells_topo()[0]
        cen = np.asarray(src.mesh.points)[cells].mean(axis=1)
        vals = np.where(cen[:, 0] < 2.0, 1.0, 0.25)
        out = np.asarray(src.transfer_cell_field(vals, tgt))
        tcen = np.asarray(tgt.mesh.points)[tgt._cells_topo()[0]].mean(axis=1)
        assert np.allclose(out[tcen[:, 0] < 1.9], 1.0)
        assert np.allclose(out[tcen[:, 0] > 2.1], 0.25)
