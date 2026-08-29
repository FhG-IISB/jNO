"""A metric-based remesh must PRESERVE named cell regions and the interfaces between them.

``domain.refine()`` hands the mesh to Mmg, which is natively multi-domain: give each tetrahedron
its material reference and Mmg keeps the surface between differing references as a real interface,
refining along it instead of moving nodes through it.

jNO passed ``np.ones(len(elems))`` for those references and rebuilt the remeshed domain with only
``interior`` / ``boundary`` cell-sets. So on any multi-material mesh -- which is every real device --
the material interfaces were invisible to Mmg and the region names were dropped on the way out.
Nothing raised: you got a plausible mesh with the physics geometry quietly dissolved.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D meshing")
pytest.importorskip("mmgpy", reason="mmgpy required for metric-based remeshing")

import jno  # noqa: E402


def _tet_volumes(points, tets):
    p = np.asarray(points)[np.asarray(tets)]
    a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def _two_material_ball_in_box(size=0.34):
    """A sphere embedded in a box: two conforming volume regions with a curved interface."""
    inner = jno.Shape.sphere(0.5, 0.5, 0.5, 0.28)
    outer = jno.Shape.box(0, 0, 0, 1, 1, 1)
    return jno.Shape.regions(ball=inner, block=outer).sized(size).domain()


def _region_volume(d, name):
    m = d.mesh
    blocks = list(m.cells)
    tot = 0.0
    for bi, arr in enumerate(m.cell_sets.get(name, []) or []):
        if arr is None or len(arr) == 0 or bi >= len(blocks):
            continue
        blk = blocks[bi]
        if blk.type != "tetra":
            continue
        tot += float(_tet_volumes(m.points, np.asarray(blk.data)[np.asarray(arr)]).sum())
    return tot


def test_the_fixture_really_is_two_materials():
    """Guard the guard: if the regions were not there to begin with, the tests below prove nothing."""
    d = _two_material_ball_in_box()
    assert {"ball", "block"} <= set(d.mesh.cell_sets), f"got {sorted(d.mesh.cell_sets)}"
    vb, vk = _region_volume(d, "ball"), _region_volume(d, "block")
    assert vb > 0 and vk > 0
    assert np.isclose(vb + vk, 1.0, rtol=2e-2), f"regions must partition the unit box, got {vb + vk}"


def test_remesh_keeps_the_region_names():
    d = _two_material_ball_in_box()
    n = len(d.mesh.points)
    d.refine(np.full(n, 0.22))
    assert {"ball", "block"} <= set(d.mesh.cell_sets), (
        f"named cell regions were dropped by the remesh; got {sorted(d.mesh.cell_sets)}"
    )


def _centroid_radii(d, name):
    """Distance from the sphere centre to the centroid of every cell tagged ``name``."""
    m = d.mesh
    blocks = list(m.cells)
    out = []
    for bi, arr in enumerate(m.cell_sets.get(name, []) or []):
        if arr is None or len(arr) == 0 or bi >= len(blocks) or blocks[bi].type != "tetra":
            continue
        tets = np.asarray(blocks[bi].data)[np.asarray(arr)]
        cen = np.asarray(m.points)[tets].mean(axis=1)
        out.append(np.linalg.norm(cen - np.array([0.5, 0.5, 0.5]), axis=1))
    return np.concatenate(out) if out else np.zeros(0)


def test_remesh_does_not_smear_the_material_interface():
    """The strong invariant. If mmg is not told where the sphere is, nodes cross it and the two
    references interleave -- ball-tagged cells appear well outside the sphere and vice versa. A
    volume tolerance is too weak to catch that; cell POSITIONS are not.

    The tolerance is one element size: a cell straddling the discrete interface legitimately has a
    centroid slightly to either side of the nominal radius.
    """
    d = _two_material_ball_in_box(size=0.18)
    R, h = 0.28, 0.18
    d.refine(np.full(len(d.mesh.points), 0.12))

    r_ball, r_block = _centroid_radii(d, "ball"), _centroid_radii(d, "block")
    assert r_ball.size and r_block.size
    assert r_ball.max() < R + h, f"a ball-tagged cell sits at r={r_ball.max():.3f}, outside the sphere (R={R})"
    assert r_block.min() > R - h, f"a block-tagged cell sits at r={r_block.min():.3f}, inside the sphere (R={R})"


def test_remesh_keeps_the_regions_a_partition():
    """Every cell belongs to exactly one material after the remesh, and they still fill the box."""
    d = _two_material_ball_in_box(size=0.18)
    d.refine(np.full(len(d.mesh.points), 0.12))
    n_tet = len(d.mesh.cells_dict["tetra"])
    idx = [set(np.asarray(a).tolist()) for nm in ("ball", "block") for a in (d.mesh.cell_sets[nm][0],)]
    assert idx[0].isdisjoint(idx[1]), "a cell is tagged as BOTH materials"
    assert len(idx[0]) + len(idx[1]) == n_tet, "the materials do not cover every cell"
    assert np.isclose(_region_volume(d, "ball") + _region_volume(d, "block"), 1.0, rtol=1e-9)


EXACT_SPHERE = 4.0 / 3.0 * np.pi * 0.28**3


def test_remesh_improves_the_curved_interface():
    """mmg does not merely carry the discrete interface across -- it reconstructs it as a SMOOTH
    surface and refines toward that within ``hausd``, so a coarsely facetted sphere gets closer to
    the real one. Measured: -14.5 % of the exact volume before, -4.8 % after at the default hausd
    and -1.3 % at hausd=1e-4.

    Asserting "the volume is unchanged" would therefore be wrong, and would fail for the right
    reason. What must hold is that fidelity improves and never degrades.
    """
    d = _two_material_ball_in_box(size=0.18)
    err0 = abs(_region_volume(d, "ball") / EXACT_SPHERE - 1.0)
    d.refine(np.full(len(d.mesh.points), 0.12))
    err1 = abs(_region_volume(d, "ball") / EXACT_SPHERE - 1.0)
    assert err1 < err0, f"interface fidelity got worse: {err0:.4f} -> {err1:.4f} relative volume error"


def test_a_tighter_hausdorff_tolerance_tracks_the_surface_more_closely():
    """``hausd`` is the knob that controls it, so it must actually bite -- otherwise the improvement
    above is incidental rather than something a caller can ask for."""
    d_loose = _two_material_ball_in_box(size=0.18)
    d_loose.refine(np.full(len(d_loose.mesh.points), 0.12))
    d_tight = _two_material_ball_in_box(size=0.18)
    d_tight.refine(np.full(len(d_tight.mesh.points), 0.12), hausd=1e-4)

    e_loose = abs(_region_volume(d_loose, "ball") / EXACT_SPHERE - 1.0)
    e_tight = abs(_region_volume(d_tight, "ball") / EXACT_SPHERE - 1.0)
    assert e_tight < e_loose, f"hausd had no effect: loose {e_loose:.4f} vs tight {e_tight:.4f}"


def test_a_single_material_mesh_is_unaffected():
    """No named volume region -> the original single-reference path, byte-for-byte behaviour."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain()
    before = set(d.mesh.cell_sets)
    d.refine(np.full(len(d.mesh.points), 0.25))
    assert len(d.mesh.cells_dict["tetra"]) > 0
    assert "interior" in d.mesh.cell_sets and "boundary" in d.mesh.cell_sets
    assert not (set(d.mesh.cell_sets) - before - {"interior", "boundary"}), "invented a region"


# --- the two cases a clean synthetic fixture does not produce, but a real gmsh mesh does ----------


class _FakeMesh:
    def __init__(self, cells, cell_sets):
        self.cells, self.cell_sets = cells, cell_sets


class _Blk:
    def __init__(self, type_, n):
        self.type, self.data = type_, np.zeros((n, 4), dtype=np.int64)


class _FakeDomain:
    def __init__(self, mesh):
        self.mesh = mesh


def test_nested_regions_do_not_overwrite_each_other():
    """A gmsh physical group routinely NESTS another: a mesh generator commonly emits both the
    individual parts and a group spanning them, so a cell belongs to two names at once. mmg carries
    one integer reference per element, so keying it on the region NAME lets whichever name is
    written last win and silently empties the rest. Keying on the membership COMBINATION is what
    keeps every group intact.
    """
    from jno.utils.solver.fem_adapt import _material_refs

    n = 10
    sets = {
        "part_a": [np.arange(0, 3)],
        "part_b": [np.arange(3, 6)],
        "both": [np.arange(0, 6)],  # a group spanning part_a and part_b
        "air": [np.arange(6, n)],
    }
    refs, name_of_ref = _material_refs(_FakeDomain(_FakeMesh([_Blk("tetra", n)], sets)), n, 3)
    assert refs is not None
    per = {}
    for r, nms in name_of_ref.items():
        for nm in nms:
            per[nm] = per.get(nm, 0) + int((refs == r).sum())
    assert per == {"part_a": 3, "part_b": 3, "both": 6, "air": 4}, per


def test_reader_metadata_cell_sets_are_not_treated_as_materials():
    """`gmsh:bounding_entities` is meshio METADATA, not a region: it carries NEGATIVE sentinels, so
    treating it as a material both invents a region and indexes the reference array from the wrong
    end -- corrupting the material of a cell at the other end of the mesh."""
    from jno.utils.solver.fem_adapt import _material_refs

    n = 8
    sets = {
        "steel": [np.arange(0, 4)],
        "copper": [np.arange(4, n)],
        "gmsh:bounding_entities": [np.array([-640, -2, 3])],
    }
    refs, name_of_ref = _material_refs(_FakeDomain(_FakeMesh([_Blk("tetra", n)], sets)), n, 3)
    flat = {nm for nms in name_of_ref.values() for nm in nms}
    assert flat == {"steel", "copper"}, f"metadata leaked in as a material: {flat}"
    assert refs.min() >= 1 and len(refs) == n
