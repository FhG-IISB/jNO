"""``jno.Shape.contains`` — analytic, shapely-free point-in-region membership (2-D & 3-D).

The point-containment predicate that resolves a geometric ``domain.region(name, shape)`` to a mesh-node
subset. CSG only (leaf + cut/fuse/inter); non-analytic transforms raise."""

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


def test_non_csg_transform_raises():
    solid = Shape.rect(0.0, 0.0, 1.0, 1.0).extrude(1.0)  # a swept solid — no analytic membership
    with pytest.raises(NotImplementedError, match="closed-form"):
        solid.contains(np.zeros((1, 3)))
