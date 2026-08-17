"""A quadrilateral facet is a facet: the boundary region of a hexahedral mesh.

``BoundaryRegion`` carried only ``edges`` and ``triangles``, and every consumer picked between them
with ``dim == 2``. A hexahedral mesh's boundary facet is a *quadrilateral*, so on a hex mesh that
choice selected an empty store and two things degraded silently:

* ``BoundaryRegion.contains`` fell through to a point-distance test against the tag's nodes, so a
  point anywhere between nodes — a facet centre — was reported outside the boundary;
* the assembler's ``_region_faces`` found no named facets and fell back to the all-nodes-in-region
  mask, which re-selects any facet whose corners happen to lie in the tag even when the facet itself
  is not in it.

Both are measured here against a tetrahedral control built on the same box, because the point of the
fix is that the two cell types behave identically.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.domain.boundary_region import BoundaryRegion
from jno.domain.geometries import Geometries

meshio = pytest.importorskip("meshio")


def _box(cell, n=3):
    return jno.domain(constructor=Geometries.equi_distant_box(nx=n, ny=n, nz=n, cell=cell))


def _facet_probe_points(facets):
    """Facet centroids and facet edge midpoints — every one of them is on the boundary."""
    k = facets.shape[1]
    mid = np.concatenate([0.5 * (facets[:, i] + facets[:, (i + 1) % k]) for i in range(k)], axis=0)
    return facets.mean(axis=1), mid


# ------------------------------------------------------------------ the region carries its facets


@pytest.mark.parametrize("cell,field,arity", [("tetra", "triangles", 3), ("hex", "quads", 4)], ids=["tetra", "hex"])
def test_the_boundary_region_stores_facets_by_their_vertex_count(cell, field, arity):
    d = _box(cell)
    region = d._boundary_regions["boundary"]
    assert getattr(region, field) is not None and len(getattr(region, field)) > 0
    assert region.facets.shape[1] == arity
    # 27 hexes give 6*9 = 54 boundary quads; the same box as tets gives twice as many triangles.
    assert len(region.facets) == (54 if cell == "hex" else 108)


def test_a_hex_boundary_is_registered_as_a_boundary_tag():
    """It used to reach the registry only through the per-point normals fallback, as
    ``boundary_points`` — a tag with no entities at all."""
    assert _box("hex")._boundary_registry["boundary"]["entity_kind"] == "quad"


def test_from_facets_dispatches_on_arity_and_refuses_anything_else():
    pts = np.zeros((4, 3))
    quads = np.zeros((2, 4, 3))
    r = BoundaryRegion.from_facets("t", 3, pts, quads)
    assert r.quads is not None and r.triangles is None and r.edges is None
    assert BoundaryRegion.from_facets("t", 3, pts, np.zeros((2, 3, 3))).triangles is not None
    assert BoundaryRegion.from_facets("t", 2, pts, np.zeros((2, 2, 2))).edges is not None
    with pytest.raises(ValueError, match="k in"):
        BoundaryRegion.from_facets("t", 3, pts, np.zeros((2, 5, 3)))


def test_facets_is_none_when_the_region_has_no_entities():
    assert BoundaryRegion(tag="t", dim=3, points=np.zeros((3, 3))).facets is None


# ---------------------------------------------------------------------------- contains, measured


@pytest.mark.parametrize("cell", ["tetra", "hex"])
def test_contains_is_exact_on_every_facet_of_the_boundary(cell):
    """Centroids and edge midpoints of every boundary facet, in the default float32.

    The edge midpoints are the demanding half: they sit on the shared edge of two facets, where the
    membership test evaluates to exactly zero. Splitting each quad into triangles (either diagonal,
    or a centroid fan) puts that zero at the end of a cancellation of two numbers of order 1/2 —
    below the float32 roundoff floor — and missed 36 of these 216 points.
    """
    d = _box(cell)
    region = d._boundary_regions["boundary"]
    for name, probes in zip(("centroids", "edge midpoints"), _facet_probe_points(region.facets)):
        hit = np.array([bool(region.contains(p)) for p in probes])
        assert hit.all(), f"{cell}: {(~hit).sum()} of {len(probes)} facet {name} reported outside"


@pytest.mark.parametrize("cell", ["tetra", "hex"])
def test_contains_rejects_the_interior(cell):
    region = _box(cell)._boundary_regions["boundary"]
    inside = np.random.default_rng(0).uniform(0.2, 0.8, (50, 3))
    assert not any(bool(region.contains(p)) for p in inside)


def test_the_point_distance_fallback_is_what_this_replaces():
    """The old state, reconstructed: the same points with no entities attached. A facet centre is not
    a mesh node, so the nearest-node test puts it outside the boundary it is on."""
    d = _box("hex")
    full = d._boundary_regions["boundary"]
    bare = BoundaryRegion(tag="boundary", dim=3, points=full.points, tol=full.tol)
    centroids, _ = _facet_probe_points(full.facets)
    assert not any(bool(bare.contains(p)) for p in centroids)
    assert all(bool(full.contains(p)) for p in centroids)
    # a mesh node is the one case the fallback did get right, and still does
    node = np.asarray(full.points)[0]
    assert bool(bare.contains(node)) and bool(full.contains(node))


# --------------------------------------------------------------- tagging a subset of a hex boundary


@pytest.mark.parametrize("cell,arity", [("tetra", 3), ("hex", 4)], ids=["tetra", "hex"])
def test_a_coordinate_tagged_face_keeps_its_facets(cell, arity):
    """``d.tag`` builds a sub-region from the parent's facets. It chose the field by dimension, so a
    quad subset was written into ``triangles`` and read back as a triangle."""
    d = _box(cell)
    d.tag("top", lambda x, y, z: z > 1 - 1e-6)
    region = d._boundary_regions["top"]
    assert region.facets is not None and region.facets.shape[1] == arity
    assert len(region.facets) == (9 if cell == "hex" else 18)
    assert bool(region.contains(np.array([0.5, 0.5, 1.0])))
    assert not bool(region.contains(np.array([0.5, 0.5, 0.0])))


@pytest.mark.parametrize("cell,arity", [("tetra", 3), ("hex", 4)], ids=["tetra", "hex"])
def test_a_facet_predicate_tag_works_on_both_cell_types(cell, arity):
    """The richer ``f(x, n, name)`` predicate reads the parent's facet normals, so it needs the same
    facets — and a quad's normal comes from its two diagonals, not from an edge pair."""
    d = _box(cell)
    d.tag("lid", lambda x, n, name: n[:, 2] > 0.5)
    region = d._boundary_regions["lid"]
    assert region.facets.shape == (9 if cell == "hex" else 18, arity, 3)
    np.testing.assert_allclose(np.asarray(d.normals_by_tag["lid"])[:, 2], 1.0, atol=1e-6)


# ------------------------------------------------------- the assembler integrates the right facets


def _patch_mesh(tmp_path, cell, drop_centre):
    """A box whose top face is a named cell set — optionally missing its centre facet.

    The centre facet is the discriminating one: all four of its corners are shared with its
    neighbours, so they stay in the tag's *node* set even when the facet is dropped. A node mask
    therefore re-selects it; matching on facet identity does not.
    """
    vol_name = "hexahedron" if cell == "hex" else "tetra"
    fac_name = "quad" if cell == "hex" else "triangle"
    m, _, _ = Geometries.equi_distant_box(nx=3, ny=3, nz=3, cell=cell)(None)
    blocks = {c.type: np.asarray(c.data) for c in m.cells}
    p, facets = m.points, blocks[fac_name]
    top = np.where(np.all(np.abs(p[facets][:, :, 2] - 1.0) < 1e-9, axis=1))[0]
    if drop_centre:
        centre = [i for i in top if np.linalg.norm(p[facets[i]].mean(axis=0) - [0.5, 0.5, 1.0]) < 1e-9]
        assert centre, "the fixture must actually drop a facet"
        top = np.array([i for i in top if i not in centre], dtype=np.int64)
    path = str(tmp_path / f"patch_{cell}_{int(drop_centre)}.inp")
    meshio.write(
        path,
        meshio.Mesh(
            points=p,
            cells=[(vol_name, blocks[vol_name]), (fac_name, facets)],
            cell_sets={"patch": [np.array([], dtype=np.int64), np.asarray(top, dtype=np.int64)]},
        ),
        file_format="abaqus",
    )
    return path, len(top)


def _applied_load(path):
    """The load a unit flux on ``patch`` actually applies, read off as the Dirichlet reaction.

    ``-Delta u = 0`` with ``u = 0`` on ``z = 0`` and ``du/dn = 1`` on ``patch``: every other face is
    natural, so the whole applied load leaves through the pinned face and the reaction there is
    exactly the integrated flux, i.e. the AREA of the facets the term was assembled on. Exact for a
    constant flux under any element, so the number is an oracle, not a tolerance.
    """
    d = jno.domain(path, compute_mesh_connectivity=False)
    d.tag("bot", lambda x, y, z: z < 1e-9)
    u, v = d.fem_symbols()
    ci, cp, cb = (d.variable(t, split=True) for t in ("interior", "patch", "bot"))
    ui, vi = u.bind(x=ci[0], y=ci[1], z=ci[2]), v.bind(x=ci[0], y=ci[1], z=ci[2])
    lap = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z
    flux = -1.0 * v.bind(x=cp[0], y=cp[1], z=cp[2])
    fem = jno.fem([lap, flux, u(*cb[:3]) - 0.0])
    reaction = np.asarray(fem.eval(lap, np.asarray(fem.solve()).ravel())).ravel()
    return float(-reaction[np.asarray(d.mesh.points)[:, 2] < 1e-9].sum())


@pytest.mark.parametrize("cell", ["tetra", "hex"])
def test_a_surface_term_integrates_over_the_whole_named_face(tmp_path, cell):
    path, n = _patch_mesh(tmp_path, cell, drop_centre=False)
    assert n == (9 if cell == "hex" else 18)
    assert _applied_load(path) == pytest.approx(1.0, rel=2e-3)


def test_a_surface_term_on_a_hex_face_with_a_hole_skips_the_hole(tmp_path):
    """The measurement that names the gap: 8 of the 9 top facets are tagged, and the term used to
    integrate over all 9 (load 1.000 where the truth is 8/9) because the node mask put the untagged
    centre facet back. The 2e-3 tolerance is float32; in float64 this is 0.888889 exactly."""
    path, n = _patch_mesh(tmp_path, "hex", drop_centre=True)
    assert n == 8
    assert _applied_load(path) == pytest.approx(8.0 / 9.0, rel=2e-3)


def test_a_single_hex_facet_is_a_usable_region(tmp_path):
    """The extreme end: one facet, area 1/9 — small enough that picking up one neighbour doubles it."""
    m, _, _ = Geometries.equi_distant_box(nx=3, ny=3, nz=3, cell="hex")(None)
    blocks = {c.type: np.asarray(c.data) for c in m.cells}
    p, quads = m.points, blocks["quad"]
    top = np.where(np.all(np.abs(p[quads][:, :, 2] - 1.0) < 1e-9, axis=1))[0]
    one = np.array([top[int(np.argmin([np.linalg.norm(p[quads[i]].mean(axis=0) - [0.5, 0.5, 1.0]) for i in top]))]])
    path = str(tmp_path / "one.inp")
    meshio.write(
        path,
        meshio.Mesh(
            points=p,
            cells=[("hexahedron", blocks["hexahedron"]), ("quad", quads)],
            cell_sets={"patch": [np.array([], dtype=np.int64), one]},
        ),
        file_format="abaqus",
    )
    assert _applied_load(path) == pytest.approx(1.0 / 9.0, rel=5e-3)


# ---------------------------------------------------------------------------------- 2-D unchanged


@pytest.mark.parametrize("cell", ["triangle", "quad"])
def test_a_2d_boundary_facet_is_an_edge_whatever_the_cell(cell):
    """A quadrilateral *mesh* in 2-D still has line facets — the store follows the facet, not the
    cell, so this is the case the fix must leave exactly as it was."""
    d = jno.domain(constructor=Geometries.equi_distant_rect(nx=4, ny=4, cell=cell))
    region = d._boundary_regions["boundary"]
    assert region.edges is not None and region.quads is None
    assert region.facets.shape[1] == 2
    assert bool(region.contains(np.array([0.5, 0.0])))
    assert not bool(region.contains(np.array([0.5, 0.5])))
