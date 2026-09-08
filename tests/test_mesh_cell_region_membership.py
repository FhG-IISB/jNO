"""``mesh_cell_region_membership``: a mesh file's named volume regions as flat per-cell masks.

The one place that reads ``mesh.cell_sets`` into the cell axis every assembler vmaps over, shared by
the remesh material labels and the per-region coefficient path so the two cannot drift apart.

Each test below pins a way the naive version gets it wrong, and every one of them was observed on a
real mesh rather than imagined.
"""

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")

from jno.domain.mesh_utils import mesh_cell_region_membership, volume_cell_type  # noqa: E402


class _Blk:
    def __init__(self, type_, n, ncol=4):
        self.type, self.data = type_, np.zeros((n, ncol), dtype=np.int64)


class _Mesh:
    def __init__(self, cells, cell_sets):
        self.cells, self.cell_sets = cells, cell_sets
        by_type: dict = {}
        for b in cells:
            by_type.setdefault(b.type, []).append(b.data)
        self.cells_dict = {k: np.concatenate(v, axis=0) for k, v in by_type.items()}


def _counts(mesh, dim=3):
    return {k: int(v.sum()) for k, v in mesh_cell_region_membership(mesh, dim).items()}


def test_indices_are_offset_per_block():
    """`cell_sets[name]` indexes each `mesh.cells` block separately while `cells_dict` is the
    concatenation. A real gmsh mesh has many blocks -- 17 on one device -- so without the cumulative
    offset every block after the first lands on the wrong cells."""
    mesh = _Mesh(
        [_Blk("tetra", 4), _Blk("tetra", 6)],
        {"a": [np.arange(4), None], "b": [None, np.arange(6)]},
    )
    assert _counts(mesh) == {"a": 4, "b": 6}
    m = mesh_cell_region_membership(mesh, 3)
    assert m["a"].tolist() == [True] * 4 + [False] * 6, "block 1's cells were not offset past block 0"


def test_curved_blocks_are_recognised():
    """A curved mesh stores only `tetra10`; matching `blk.type == "tetra"` finds no blocks at all and
    silently returns no regions -- the whole mesh then reads as one material."""
    mesh = _Mesh([_Blk("tetra10", 5, ncol=10)], {"steel": [np.arange(5)]})
    assert volume_cell_type(mesh, 3) == "tetra"
    assert _counts(mesh) == {"steel": 5}


def test_reader_metadata_is_not_a_region():
    """`gmsh:bounding_entities` is meshio metadata carrying NEGATIVE sentinels: treated as a region
    it both invents a material and writes a label from the wrong end of the array."""
    mesh = _Mesh(
        [_Blk("tetra", 6)],
        {"steel": [np.arange(0, 3)], "copper": [np.arange(3, 6)], "gmsh:bounding_entities": [np.array([-640, -2, 3])]},
    )
    assert _counts(mesh) == {"steel": 3, "copper": 3}


def test_jnos_own_whole_domain_sets_are_excluded():
    """`interior` / `boundary` are jNO's own sets, not materials; counting them as regions would make
    every cell a member of a spurious material spanning the mesh."""
    mesh = _Mesh([_Blk("tetra", 4)], {"steel": [np.arange(4)], "interior": [np.arange(4)], "boundary": [np.arange(0)]})
    assert _counts(mesh) == {"steel": 4}


def test_nested_regions_are_both_reported():
    """A generator commonly emits the individual parts AND a group spanning them, so a cell belongs
    to several names at once. Membership is per name, so nesting must survive."""
    mesh = _Mesh([_Blk("tetra", 10)], {"a": [np.arange(0, 3)], "b": [np.arange(3, 6)], "both": [np.arange(0, 6)]})
    assert _counts(mesh) == {"a": 3, "b": 3, "both": 6}


def test_global_indices_are_rebased_like_the_loader():
    """The mesh loader treats a set whose max index exceeds the block length as GLOBAL ids and
    rebases by its min. Reading the same arrays with a different convention would resolve different
    cells for the same region name on the same mesh."""
    mesh = _Mesh([_Blk("tetra", 5)], {"steel": [np.arange(100, 105)]})
    assert _counts(mesh) == {"steel": 5}


def test_a_surface_only_mesh_has_no_volume_regions():
    mesh = _Mesh([_Blk("triangle", 4, ncol=3)], {"skin": [np.arange(4)]})
    assert volume_cell_type(mesh, 3) is None
    assert mesh_cell_region_membership(mesh, 3) == {}
