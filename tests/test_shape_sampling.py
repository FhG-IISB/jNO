"""``jno.Shape`` mesh-free sampling — the analytic geometry a PINN draws collocation points from.

Three things have to hold for this to replace mesh nodes as the collocation source:

* **bounds** must be a superset of the true extent, or the rejection proposal is invalid;
* **contains** must agree with the solid gmsh actually builds, or the samples are of a different
  body than the FEM path would use;
* **samples** must be continuous — drawn from the geometry, not from any fixed point set — and
  boundary points must land exactly on the analytic surface with exactly outward normals.

The oracles are analytic (known volumes, areas, arclengths) and, for membership, the meshed
geometry itself: every node gmsh puts in the mesh lies in the solid.
"""

import math

import numpy as np
import pytest

from jno.geometry import Shape

pytestmark = pytest.mark.filterwarnings("ignore")


def _mc_volume(shape, n=200_000, seed=0):
    """Volume (area in 2-D) by rejection in the shape's own bounding box."""
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(v, dtype=float) for v in shape.bounds())
    span = hi - lo
    free = span > 0.0
    cand = np.tile(lo, (n, 1))
    cand[:, free] += rng.uniform(0.0, 1.0, size=(n, int(free.sum()))) * span[free]
    return float(np.prod(span[free])) * float(shape.contains(cand, tol=0.0).mean())


# --------------------------------------------------------------------------- bounds


def test_bounds_are_exact_for_primitives():
    assert Shape.rect(0.0, 0.0, 2.0, 1.0).bounds() == ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
    assert Shape.box(0.0, 0.0, 0.0, 2.0, 1.0, 3.0).bounds() == ((0.0, 0.0, 0.0), (2.0, 1.0, 3.0))
    lo, hi = Shape.disk(1.0, 2.0, 0.5).bounds()
    assert np.allclose(lo, (0.5, 1.5, 0.0)) and np.allclose(hi, (1.5, 2.5, 0.0))
    lo, hi = Shape.sphere(1.0, 1.0, 1.0, 2.0).bounds()
    assert np.allclose(lo, (-1.0, -1.0, -1.0)) and np.allclose(hi, (3.0, 3.0, 3.0))
    # a cylinder's box is the caps' centres widened by the disc projected on each axis
    lo, hi = Shape.cylinder(0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0).bounds()
    assert np.allclose(lo, (-1.0, -1.0, 0.0)) and np.allclose(hi, (1.0, 1.0, 4.0))


def test_bounds_bound_the_shape_under_csg_and_transforms():
    """A box that is not a superset silently truncates the sampler, so assert containment
    directly: every point of the shape must lie inside the reported box."""
    shapes = [
        Shape.rect(0.0, 0.0, 1.0, 1.0) - Shape.disk(0.5, 0.5, 0.25),
        Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0) - Shape.sphere(0.5, 0.5, 0.5, 0.3),
        Shape.rect(0.0, 0.0, 2.0, 1.0) | Shape.disk(3.0, 0.0, 1.0),
        Shape.rect(0.0, 0.0, 2.0, 1.0).translate((5.0, 7.0, 0.0)),
        Shape.rect(0.0, 0.0, 2.0, 1.0).rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 6),
        Shape.disk(0.0, 0.0, 1.0).extrude(3.0),
        Shape.rect(1.0, 0.0, 2.0, 1.0).revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2 * math.pi),
    ]
    for shape in shapes:
        lo, hi = (np.asarray(v, dtype=float) for v in shape.bounds())
        pts = shape.sample_interior(4000, np.random.default_rng(0))
        assert (pts >= lo - 1e-9).all() and (pts <= hi + 1e-9).all()


def test_bounds_refuse_by_name_without_a_closed_form():
    solid = Shape.rect(0.0, 0.0, 1.0, 1.0).extrude(1.0).fillet(0.05)
    assert not solid.is_analytic()
    with pytest.raises(NotImplementedError, match="tessellation"):
        solid.bounds()


# --------------------------------------------------------------------------- membership


@pytest.mark.parametrize(
    "shape",
    [
        Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.2),
        (Shape.rect(0.0, 0.0, 1.0, 1.0) - Shape.disk(0.5, 0.5, 0.25)).sized(0.06),
        Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.3),
        (Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0) - Shape.sphere(0.5, 0.5, 0.5, 0.3)).sized(0.15),
        Shape.cylinder(0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, size=0.5),
        Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.2).translate((5.0, 7.0, 0.0)),
        Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.2).rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 6),
        Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).extrude(2.0),
        Shape.disk(0.0, 0.0, 1.0, size=0.3).extrude(3.0),
        Shape.rect(1.0, 0.0, 2.0, 1.0, size=0.25).revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2 * math.pi),
        Shape.rect(1.0, 0.0, 2.0, 1.0, size=0.25).revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), math.pi),
    ],
    ids=[
        "rect",
        "rect-disk",
        "box",
        "box-sphere",
        "cylinder",
        "translate",
        "rotate",
        "extrude",
        "disk-extrude",
        "revolve-full",
        "revolve-half",
    ],
)
def test_contains_accepts_every_node_of_the_mesh_it_describes(shape):
    """The meshed solid is the oracle: gmsh only places nodes inside the body, so a disagreement
    means the analytic membership describes a *different* shape than the FEM path would build."""
    mesh, _dim, _ds = shape.build()
    pts = np.asarray(mesh.points, dtype=float)[:, :3]
    tol = 1e-7 * max(float(np.ptp(pts, axis=0).max()), 1.0)
    assert shape.contains(pts, tol=tol).all()


# --------------------------------------------------------------------------- interior sampling


@pytest.mark.parametrize(
    "shape, volume",
    [
        (Shape.rect(0.0, 0.0, 2.0, 1.0), 2.0),
        (Shape.rect(0.0, 0.0, 1.0, 1.0) - Shape.disk(0.5, 0.5, 0.25), 1.0 - math.pi / 16),
        (Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 1.0),
        (Shape.sphere(0.0, 0.0, 0.0, 1.0), 4.0 / 3.0 * math.pi),
        (Shape.cylinder(0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0), 4.0 * math.pi),
        (Shape.rect(0.0, 0.0, 1.0, 1.0).extrude(2.0), 2.0),
        (Shape.disk(0.0, 0.0, 1.0).extrude(3.0), 3.0 * math.pi),
        # profile x in [1,2] swept about +y: the ring pi*(2^2 - 1^2) * height 1
        (Shape.rect(1.0, 0.0, 2.0, 1.0).revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2 * math.pi), 3.0 * math.pi),
    ],
    ids=["rect", "rect-disk", "box", "sphere", "cylinder", "extrude", "disk-extrude", "revolve"],
)
def test_membership_reproduces_the_analytic_volume(shape, volume):
    assert _mc_volume(shape) == pytest.approx(volume, rel=0.01)


def test_sampled_points_are_inside_and_fill_the_shape():
    shape = Shape.rect(0.0, 0.0, 1.0, 1.0) - Shape.disk(0.5, 0.5, 0.25)
    pts = shape.sample_interior(20_000, np.random.default_rng(0))
    assert pts.shape == (20_000, 3)
    assert shape.contains(pts, tol=0.0).all()
    # and they cover it: on a 10x10 grid the only cells left empty are the ones the hole swallows
    # whole, so the coverage gap IS the hole rather than a blind spot of the sampler.
    hist, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=10, range=[[0, 1], [0, 1]])
    for i, j in zip(*np.where(hist == 0)):
        corners = np.array([(xe[i + a], ye[j + b]) for a in (0, 1) for b in (0, 1)])
        farthest = np.linalg.norm(corners - np.array([0.5, 0.5]), axis=1).max()
        assert farthest <= 0.25 + 1e-12, f"cell ({i},{j}) is empty but not inside the hole"


def test_draws_are_continuous_not_a_fixed_point_set():
    """The whole point of mesh-free sampling: successive draws are different points, and the
    count is not capped by any node set."""
    shape = Shape.disk(0.0, 0.0, 1.0)
    rng = np.random.default_rng(0)
    a = shape.sample_interior(500, rng)
    b = shape.sample_interior(500, rng)
    assert not np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))
    # far more points than a mesh of this shape would ever have nodes, with no cap and no warning
    many = shape.sample_interior(200_000, rng)
    assert len(many) == 200_000 and len(np.unique(many, axis=0)) == 200_000


def test_a_vanishing_shape_raises_instead_of_returning_short():
    empty = Shape.disk(0.0, 0.0, 1.0) & Shape.disk(10.0, 0.0, 1.0)  # disjoint -> empty intersection
    with pytest.raises(RuntimeError, match="vanishing|empty"):
        empty.sample_interior(10, np.random.default_rng(0), max_rounds=3)


# --------------------------------------------------------------------------- boundary sampling


@pytest.mark.parametrize(
    "shape",
    [
        Shape.rect(0.0, 0.0, 2.0, 1.0),
        Shape.disk(1.0, 2.0, 0.5),
        Shape.box(0.0, 0.0, 0.0, 2.0, 1.0, 3.0),
        Shape.sphere(0.0, 0.0, 0.0, 1.0),
        Shape.cylinder(0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0),
        Shape.rect(0.0, 0.0, 1.0, 1.0) - Shape.disk(0.5, 0.5, 0.25),
        Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0) - Shape.sphere(0.5, 0.5, 0.5, 0.3),
        Shape.rect(0.0, 0.0, 2.0, 1.0).translate((5.0, 7.0, 0.0)),
        Shape.rect(0.0, 0.0, 2.0, 1.0).rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 6),
        Shape.disk(0.0, 0.0, 1.0).extrude(3.0),
        # a ring swept about +y, and the half sweep whose flat end caps must also come out outward
        Shape.rect(1.0, 0.0, 2.0, 1.0).revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2 * math.pi),
        Shape.rect(1.0, 0.0, 2.0, 1.0).revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), math.pi),
        Shape.rect(0.0, 0.0, 3.0, 1.0).revolve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2 * math.pi),
    ],
    ids=[
        "rect",
        "disk",
        "box",
        "sphere",
        "cylinder",
        "rect-disk",
        "box-sphere",
        "translate",
        "rotate",
        "disk-extrude",
        "revolve",
        "revolve-half",
        "revolve-x",
    ],
)
def test_boundary_points_are_on_the_surface_with_outward_unit_normals(shape):
    """Stepping a hair along +n must leave the shape and along -n must stay inside — which is a
    statement about the point lying on the boundary *and* the normal pointing the right way."""
    pts, nrm = shape.sample_boundary(4000, np.random.default_rng(0))
    assert pts.shape == (4000, 3) and nrm.shape == (4000, 3)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    lo, hi = (np.asarray(v, dtype=float) for v in shape.bounds())
    eps = 1e-7 * max(float(np.max(hi - lo)), 1.0)
    assert (~shape.contains(pts + eps * nrm, tol=0.0)).all()
    assert shape.contains(pts - eps * nrm, tol=0.0).all()


def test_boundary_points_are_exact_not_faceted():
    """A circle's samples lie on the circle to machine precision and the normal is exactly radial
    — the property a tessellation cannot provide (its points sit on chords)."""
    disk = Shape.disk(1.0, 2.0, 0.5)
    pts, nrm = disk.sample_boundary(5000, np.random.default_rng(0))
    radial = pts[:, :2] - np.array([1.0, 2.0])
    assert np.allclose(np.linalg.norm(radial, axis=1), 0.5, atol=1e-14)
    assert np.allclose(nrm[:, :2], radial / 0.5, atol=1e-14)


def test_boundary_draw_is_uniform_by_measure():
    """A 2x1 rectangle: each edge should receive its own length's share of the draws."""
    pts, _ = Shape.rect(0.0, 0.0, 2.0, 1.0).sample_boundary(80_000, np.random.default_rng(0))
    on = {
        "bottom": np.isclose(pts[:, 1], 0.0),
        "top": np.isclose(pts[:, 1], 1.0),
        "left": np.isclose(pts[:, 0], 0.0),
        "right": np.isclose(pts[:, 0], 2.0),
    }
    assert sum(m.sum() for m in on.values()) == 80_000  # every point is on exactly one edge
    for name, want in (("bottom", 2 / 6), ("top", 2 / 6), ("left", 1 / 6), ("right", 1 / 6)):
        assert on[name].mean() == pytest.approx(want, abs=0.01), name


def test_a_cut_away_surface_contributes_no_boundary_points():
    """Half of the disk's circle lies outside the rectangle after the cut; no sample may land
    there, or the boundary draw would include a surface the solid does not have."""
    shape = Shape.rect(0.0, 0.0, 1.0, 1.0) - Shape.disk(0.0, 0.5, 0.25)  # bite out of the left edge
    pts, _ = shape.sample_boundary(20_000, np.random.default_rng(0))
    assert (pts[:, 0] >= -1e-12).all()  # nothing on the half-circle at x < 0


# --------------------------------------------------------------------------- 1-D


def test_a_1d_curve_is_sampled_along_its_arclength():
    """A 1-D domain has no volume to reject into, so it is parametrised — exactly, and uniformly
    in arclength rather than in any coordinate."""
    from jno.geometry import Path

    line = Path(0.0, 0.0).line_to(1.0, 0.0).curve()
    pts = line.sample_interior(20_000, np.random.default_rng(0))
    assert pts.shape == (20_000, 3)
    assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= 1.0
    assert pts[:, 0].mean() == pytest.approx(0.5, abs=0.01)
    assert np.allclose(pts[:, 1:], 0.0)
    assert line.contains(pts[:, :1], tol=1e-9).all()  # a 1-D domain carries only its x column


def test_a_curved_1d_manifold_refuses_a_projected_query():
    """Asking about a curved 1-D manifold from one coordinate is ambiguous; it says so rather
    than answering from a projection."""
    from jno.geometry.primitives import Curve

    arc = Curve((1.0, 0.0, 0.0), (("arc", (0.0, 1.0, 0.0), (0.7071, 0.7071, 0.0), None),))
    assert arc.measure() == pytest.approx(math.pi / 2, rel=1e-3)
    with pytest.raises(NotImplementedError, match="subspace"):
        arc.contains(np.array([[0.5]]))
