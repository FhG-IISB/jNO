"""``jno.Shape.contains`` — analytic, shapely-free point-in-region membership (2-D & 3-D).

The point-containment predicate that resolves a geometric ``domain.region(name, shape)`` to a mesh-node
subset, and the membership test the mesh-free PINN sampler rejects against. Covers CSG (leaf +
cut/fuse/inter/regions) and the transforms whose inverse is closed-form (translate/rotate/extrude/
revolve); ``sweep`` and ``fillet`` have none and refuse by name."""

import numpy as np
import pytest

from jno.geometry import Shape


def test_rect_contains():
    s = Shape.rect(0.0, 0.0, 1.0, 1.0)
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.5], [0.0, 0.0], [1.0, 1.0]])
    assert list(s.contains(pts)) == [True, False, False, True, True]  # corners inclusive


def test_disk_contains():
    s = Shape.disk(0.0, 0.0, 1.0)
    pts = np.array([[0.0, 0.0], [0.9, 0.0], [1.1, 0.0], [0.7, 0.7]])
    assert list(s.contains(pts)) == [True, True, False, True]  # (0.7,0.7): r≈0.99 < 1


def test_polygon_contains_triangle():
    s = Shape.polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    pts = np.array([[0.2, 0.2], [0.6, 0.6], [0.9, 0.9], [-0.1, 0.5]])
    assert list(s.contains(pts)) == [True, False, False, False]  # (0.6,0.6) is outside x+y<1


def test_box_contains():
    s = Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    pts = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert list(s.contains(pts)) == [True, False, True, True]


def test_sphere_contains():
    s = Shape.sphere(0.0, 0.0, 0.0, 1.0)
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    assert list(s.contains(pts)) == [True, True, False]  # (.5,.5,.5): r≈0.87<1; (1,1,1): r≈1.73>1


def test_cylinder_contains():
    s = Shape.cylinder(0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0)  # axis +z, height 2, radius 1
    pts = np.array([[0.0, 0.0, 1.0], [0.9, 0.0, 1.0], [1.1, 0.0, 1.0], [0.0, 0.0, 2.5], [0.0, 0.0, -0.1]])
    assert list(s.contains(pts)) == [True, True, False, False, False]


def test_cut_fuse_inter():
    a = Shape.rect(0.0, 0.0, 0.6, 1.0)
    b = Shape.rect(0.4, 0.0, 1.0, 1.0)  # overlap x∈[0.4,0.6]
    pts = np.array([[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]])  # in A only / both / in B only
    assert list((a - b).contains(pts)) == [True, False, False]  # cut: A minus B
    assert list((a | b).contains(pts)) == [True, True, True]  # fuse: union
    assert list((a & b).contains(pts)) == [False, True, False]  # inter: overlap band only


def test_difference_matches_shapely():
    """The cut predicate matches shapely's, node-for-node, on a random cloud — the migration invariant."""
    shp = pytest.importorskip("shapely")
    from shapely.geometry import box

    a_s, b_s = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    diff = a_s.difference(b_s)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.1, 1.1, size=(500, 2))
    ours = (Shape.rect(0.0, 0.0, 0.6, 1.0) - Shape.rect(0.4, 0.0, 1.0, 1.0)).contains(pts)
    theirs = np.asarray(shp.contains_xy(diff.buffer(1e-9), pts[:, 0], pts[:, 1]))
    assert np.array_equal(ours, theirs)


def test_rigid_and_sweep_transforms_are_analytic():
    """translate/rotate/extrude/revolve map the query point into the child's frame, so membership
    stays closed-form; each is checked against the geometry it is supposed to describe."""
    box = Shape.rect(0.0, 0.0, 1.0, 1.0).extrude(2.0)  # the unit square swept to height 2
    pts = np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 2.5], [1.5, 0.5, 1.0], [0.5, 0.5, 0.0]])
    assert list(box.contains(pts)) == [True, False, False, True]  # the z=0 cap is inclusive

    moved = Shape.rect(0.0, 0.0, 1.0, 1.0).translate((5.0, 0.0, 0.0))
    assert list(moved.contains(np.array([[5.5, 0.5], [0.5, 0.5]]))) == [True, False]

    turned = Shape.rect(0.0, 0.0, 2.0, 1.0).rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2)
    assert list(turned.contains(np.array([[-0.5, 1.0], [1.0, 0.5]]))) == [True, False]

    # the profile x in [1, 2] swept a full turn about +y is the ring 1 <= sqrt(x^2+z^2) <= 2
    ring = Shape.rect(1.0, 0.0, 2.0, 1.0).revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2 * np.pi)
    pts = np.array([[1.5, 0.5, 0.0], [0.0, 0.5, 1.5], [0.5, 0.5, 0.0], [1.5, 1.5, 0.0]])
    assert list(ring.contains(pts)) == [True, True, False, False]


def test_fillet_membership_comes_from_the_tessellation():
    """The plans with genuinely no closed form are answered by their boundary mesh instead.

    The oracle is what the fillet actually did: rounding the vertical edges of the unit cube with
    radius 0.1 removes the material at the corner, keeps the centre, and leaves the middle of a face
    untouched. A membership test that merely recursed to the un-filleted solid would say the corner
    is still there.
    """
    solid = Shape.rect(0.0, 0.0, 1.0, 1.0).extrude(1.0).fillet(0.1).size(0.15)
    assert not solid.is_analytic()
    pts = np.array(
        [
            [0.5, 0.5, 0.5],  # centre — kept
            [0.5, 0.5, 0.99],  # middle of a face — kept
            [0.01, 0.01, 0.5],  # inside a rounded vertical edge — REMOVED by the fillet
            [1.5, 0.5, 0.5],  # outside the cube entirely
        ]
    )
    assert list(solid.contains(pts)) == [True, True, False, False]


def test_cut_keeps_its_own_cut_surface():
    """``A - B`` retains the surface it was cut along — that is where a mesh puts nodes, so
    testing the subtrahend inclusively would drop every node on a hole's boundary."""
    holed = Shape.rect(0.0, 0.0, 1.0, 1.0) - Shape.disk(0.5, 0.5, 0.25)
    on_hole = np.array([[0.75, 0.5], [0.5, 0.75], [0.25, 0.5]])  # exactly at radius 0.25
    assert holed.contains(on_hole).all()
    assert not holed.contains(np.array([[0.5, 0.5]])).any()  # the hole's middle is still out
