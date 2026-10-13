"""`domain.mask(name)`: which mesh nodes belong to a named region.

The oracle is the geometry itself: two unit squares side by side, so a node is in the left region
exactly when x <= 1 and in the right one exactly when x >= 1. Nodes on the shared edge x = 1 are
vertices of cells on both sides and belong to both.
"""

import numpy as np
import pytest

import jno

TOL = 1e-12


def _two_squares(curved=False):
    left = jno.shape.rect(0.0, 0.0, 1.0, 1.0)
    right = jno.shape.rect(1.0, 0.0, 2.0, 1.0)
    plan = (left.name("left") + right.name("right")).sized(0.25)
    return (plan.curved() if curved else plan).domain()


def test_nodes_follow_the_geometry_and_the_interface_belongs_to_both():
    d = _two_squares()
    x = np.asarray(d.mesh.points)[:, 0]
    a, b = d.mask("left"), d.mask("right")

    assert a.dtype == bool and a.shape == (len(x),)
    np.testing.assert_array_equal(a, x <= 1.0 + TOL)
    np.testing.assert_array_equal(b, x >= 1.0 - TOL)
    on_interface = np.abs(x - 1.0) < TOL
    assert on_interface.sum() >= 2
    assert (a & b).tolist() == on_interface.tolist()
    assert (a | b).all()


def test_curved_mesh_includes_the_mid_side_nodes():
    """A curved mesh stores six-node triangles; the region's mid-side nodes are its nodes too."""
    d = _two_squares(curved=True)
    assert "triangle6" in d.mesh.cells_dict
    x = np.asarray(d.mesh.points)[:, 0]
    np.testing.assert_array_equal(d.mask("left"), x <= 1.0 + TOL)
    np.testing.assert_array_equal(d.mask("right"), x >= 1.0 - TOL)


def test_unknown_region_raises_and_lists_the_regions():
    d = _two_squares()
    with pytest.raises(KeyError, match=r"no region 'middle'.*left.*right"):
        d.mask("middle")
