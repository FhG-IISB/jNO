"""A thin extrusion (thinner than its mesh size) is meshed as one structured layer.

A foil, a film or a PCB copper layer is far thinner than the in-plane size it should be meshed at; free
tetrahedral meshing then either refines everything down to the thickness or makes slivers. jno.shape layers
such an extrusion itself: one layer of prisms split into tets, the in-plane size kept, the thickness exact.
"""

import numpy as np
import pytest

pytest.importorskip("gmsh")

import jno  # noqa: E402
from jno.domain.mesh_utils import mesh_cell_region_membership  # noqa: E402

R, B = jno.shape.rect, jno.shape.box


def _tets(d, region):
    reg = mesh_cell_region_membership(d.mesh, 3)
    P = np.asarray(d.mesh.points)
    return P, np.asarray(d.mesh.cells_dict["tetra"])[np.asarray(reg[region], bool)]


def _vol(p):
    return np.abs(np.einsum("ti,ti->t", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6.0


def _aspect(p):
    """longest edge cubed over volume, scaled to 1 for a regular tet"""
    e = [np.linalg.norm(p[:, i] - p[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)]
    return np.max(e, 0) ** 3 / (6.0 * np.sqrt(2.0) * _vol(p))


@pytest.mark.parametrize("moved", [False, True])
@pytest.mark.parametrize("t", [0.01, 0.002])
def test_a_foil_is_one_structured_layer(t, moved):
    """every foil vertex sits on one of its two faces, the volume is exact, the cell count follows the in-plane
    size, and no tet is worse than the layer itself makes it (aspect ~ h / t). Measured at t = 0.002, h = 0.1:
    worst aspect 91 layered, 357 free-meshed (the slivers this removes); at t = 0.01 both give 18. A translated
    foil (a layer placed in a stack) must keep its layer: moving the extruded solid used to drop it."""
    dz = 0.1 if moved else 0.0
    foil = (R(0, 0, 1, 1) - R(0.3, 0.3, 0.7, 0.7)).extrude(t).translate((0.0, 0.0, dz)).name("foil").size(0.1)
    d = (foil + B(-0.3, -0.3, -0.3, 1.3, 1.3, 0.4).name("air").size(0.2)).sized(0.2).domain()
    P, C = _tets(d, "foil")
    z = np.unique(np.round((P[np.unique(C), 2] - dz) / t, 9))
    assert set(z) == {0.0, 1.0}, f"foil vertices at z/t = {z}: not one layer"
    np.testing.assert_allclose(_vol(P[C]).sum(), t * (1.0 - 0.16), rtol=1e-10)
    # one layer of prisms (3 tets each) over the in-plane triangulation: O(area / h^2), independent of t
    assert len(C) < 3 * 4 * (1.0 - 0.16) / 0.1**2 * 3
    assert _aspect(P[C]).max() < 2.5 * 0.1 / t


def test_a_thick_extrusion_is_meshed_freely():
    """an extrusion thicker than its size keeps the free tetrahedral mesh (several cells through it)"""
    block = R(0, 0, 1, 1).extrude(0.5).name("block").size(0.1)
    d = (block + B(-0.3, -0.3, -0.3, 1.3, 1.3, 0.8).name("air").size(0.2)).sized(0.2).domain()
    P, C = _tets(d, "block")
    assert len(np.unique(np.round(P[np.unique(C), 2], 9))) > 3
