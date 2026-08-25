"""A ``Shape`` whose boundary has no closed form is served by meshing that boundary and nothing else.

``sweep`` follows an arbitrary path and ``fillet`` removes material near edges, so neither has
analytic membership nor an analytic boundary. :meth:`Shape.tessellate` meshes the perimeter (2-D) /
surface (3-D) once -- no volume fill -- and answers all three questions from that one artifact:
where the surface is, which way it faces, and what is inside.

The accuracy trade is the point of these tests. A facet has EXTENT and gmsh conforms it to the CAD
topology, so a facet lies wholly on one face and its normal is **exact at a crease** -- the property
a point cloud cannot have at any density. In exchange a straight facet is a chord of a curved
boundary, so there the normal is O(h).
"""

import numpy as np
import pytest

import jno
from jno.geometry.shape import BoundaryMesh

pytestmark = pytest.mark.filterwarnings("ignore")


def _angles(got, exact):
    return np.degrees(np.arccos(np.clip((got * exact).sum(-1), -1.0, 1.0)))


# --------------------------------------------------------------------------- the mesh itself


def test_tessellate_meshes_the_boundary_and_not_the_volume():
    bm = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).size(0.3).tessellate()
    assert isinstance(bm, BoundaryMesh)
    assert bm.facets.shape[1] == 3, "the boundary of a 3-D shape is triangles"
    assert bm.dim == 3
    # every facet carries a unit normal and a positive measure -- no degenerate slivers survive
    assert np.allclose(np.linalg.norm(bm.normals, axis=1), 1.0)
    assert np.all(bm.measures > 0.0)


def test_a_2d_shape_tessellates_to_its_perimeter():
    bm = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).size(0.25).tessellate()
    assert bm.facets.shape[1] == 2, "the boundary of a 2-D shape is line segments"
    assert bm.measures.sum() == pytest.approx(4.0, abs=1e-12), "the perimeter of a unit square"


def test_the_mesher_runs_once_and_is_cached():
    s = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).fillet(0.15).size(0.3)
    assert s.tessellate() is s.tessellate()


@pytest.mark.parametrize(
    "shape, measure, name",
    [
        (jno.Shape.rect(0.0, 0.0, 2.0, 3.0).size(0.3), 10.0, "rectangle perimeter"),
        (jno.Shape.box(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).size(0.4), 22.0, "box surface area"),
    ],
)
def test_flat_boundaries_are_measured_exactly(shape, measure, name):
    """A polyhedron's own boundary is meshed with no error at all -- nothing is being approximated."""
    assert shape.tessellate().measures.sum() == pytest.approx(measure, rel=1e-12), name


def test_a_curved_boundary_is_a_chord_and_converges_from_below():
    """Facets are straight, so they under-measure a curve -- and the deficit falls with h."""
    got = [jno.Shape.disk(0.0, 0.0, 1.0).size(h).tessellate().measures.sum() for h in (0.4, 0.2, 0.1, 0.05)]
    assert all(g < 2 * np.pi for g in got), f"a chord cannot be longer than its arc: {got}"
    err = [2 * np.pi - g for g in got]
    assert err[0] > err[1] > err[2] > err[3], f"not converging: {err}"


# --------------------------------------------------------------------------- normals


@pytest.mark.parametrize(
    "shape, exact",
    [
        (jno.Shape.rect(0.0, 0.0, 1.0, 1.0).size(0.2), None),
        (jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).size(0.3), None),
    ],
)
def test_a_facet_normal_is_exact_at_a_crease(shape, exact):
    """The headline property. A box is flat everywhere and has 12 edges; a facet lies wholly on one
    face, so *every* sampled normal is one of the six axis directions, exactly.

    This is what a point cloud cannot do: a nearest-neighbour lookup near an edge snaps to the
    adjoining face and is a full 90 degrees wrong, and only the frequency of that falls with
    resolution, never the magnitude.
    """
    _pts, nrm = shape.tessellate().sample_boundary(3000, np.random.default_rng(0))
    axis_aligned = np.sort(np.abs(nrm), axis=1)  # each row must be (0, 0, 1) up to ordering
    assert np.allclose(axis_aligned[:, -1], 1.0, atol=1e-12)
    assert np.allclose(axis_aligned[:, :-1], 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "shape, exact, sizes",
    [
        (jno.Shape.disk(0.0, 0.0, 1.0), lambda p: p / np.linalg.norm(p, axis=1, keepdims=True), (0.4, 0.2, 0.1)),
        (jno.Shape.sphere(0.0, 0.0, 0.0, 1.0), lambda p: p / np.linalg.norm(p, axis=1, keepdims=True), (0.5, 0.25, 0.125)),
    ],
    ids=["disk", "sphere"],
)
def test_a_facet_normal_is_first_order_on_a_curved_boundary(shape, exact, sizes):
    """A chord's normal is off by roughly the half-angle it subtends, so the error halves with h.

    Stated as a scope limit rather than hidden: this is the price of the tessellated path, and
    ``.size(h)`` is the knob that pays it down.
    """
    med = []
    for h in sizes:
        pts, nrm = shape.size(h).tessellate().sample_boundary(4000, np.random.default_rng(0))
        med.append(float(np.median(_angles(nrm, exact(pts)))))
    assert med[0] > med[1] > med[2], f"normal error must fall with h: {med}"
    ratio = med[0] / med[2]
    assert 2.5 < ratio < 6.0, f"halving h twice should shrink the error ~4x, got {ratio:.2f}x ({med})"


def test_normals_point_out_of_the_shape():
    for shape in (
        jno.Shape.rect(0.0, 0.0, 1.0, 1.0).size(0.2),
        jno.Shape.disk(0.0, 0.0, 1.0).size(0.2),
        jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).size(0.4),
        jno.Shape.sphere(0.0, 0.0, 0.0, 1.0).size(0.3),
    ):
        bm = shape.tessellate()
        pts, nrm = bm.sample_boundary(2000, np.random.default_rng(1))
        step = 0.02 * float(np.ptp(np.asarray(bm.points), axis=0).max())
        assert not bm.contains(pts + step * nrm).any(), "a step ALONG the normal must leave the shape"
        assert bm.contains(pts - step * nrm).all(), "a step AGAINST it must stay inside"


# --------------------------------------------------------------------------- membership


def _disagreement(shape, n=20000, seed=0):
    """Volume fraction of the bounding box where the tessellated answer differs from the closed form."""
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(v) for v in shape.bounds())
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    q = lo + rng.uniform(size=(n, 3)) * span
    analytic = np.asarray(shape.contains(q[:, : shape.dim]))
    return float((analytic != shape.tessellate().contains(q)).mean())


@pytest.mark.parametrize(
    "shape",
    [jno.Shape.rect(0.0, 0.0, 1.0, 1.0).size(0.15), jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).size(0.2)],
    ids=["rect", "box"],
)
def test_ray_cast_membership_is_exact_on_a_polyhedron(shape):
    """A polyhedron IS its own tessellation, so the two answers can only differ on the surface itself."""
    assert _disagreement(shape) < 1e-3


@pytest.mark.parametrize(
    "make, sizes",
    [
        (lambda h: jno.Shape.disk(0.5, 0.5, 0.4).size(h), (0.3, 0.15, 0.075)),
        (lambda h: jno.Shape.sphere(0.5, 0.5, 0.5, 0.4).size(h), (0.3, 0.15, 0.075)),
        (lambda h: (jno.Shape.rect(0.0, 0.0, 1.0, 1.0) - jno.Shape.disk(0.5, 0.5, 0.2)).size(h), (0.2, 0.1, 0.05)),
    ],
    ids=["disk", "sphere", "csg-with-a-hole"],
)
def test_ray_cast_membership_converges_to_the_closed_form_on_a_curve(make, sizes):
    """The two can only disagree within one sagitta of a curved boundary, and that band is O(h^2).

    Asserted as convergence rather than against a fixed tolerance, because "how close" is a statement
    about the mesh size, not about the method.
    """
    got = [_disagreement(make(h)) for h in sizes]
    assert got[0] > got[1] > got[2], f"disagreement must fall with h: {got}"
    assert got[-1] < 0.01, f"at the finest size the two should nearly agree, got {got[-1]:.4f}"


def test_interior_draws_land_inside_and_boundary_draws_land_on_the_surface():
    bm = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).fillet(0.2).size(0.2).tessellate()
    q = bm.sample_interior(2000, np.random.default_rng(0))
    assert bm.contains(q).all()
    lo, hi = (np.asarray(v) for v in bm.bounds())
    assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)


def test_two_interior_draws_differ():
    """The tessellated path is still continuous: points come from the facets, not from a node set."""
    bm = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).fillet(0.2).size(0.25).tessellate()
    a = bm.sample_interior(500, np.random.default_rng(0))
    b = bm.sample_interior(500, np.random.default_rng(1))
    assert not np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


# --------------------------------------------------------------------------- the plans it exists for


@pytest.mark.parametrize(
    "make",
    [
        lambda: jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).fillet(0.2),
        lambda: jno.Shape.rect(0.0, 0.0, 1.0, 1.0).extrude(1.0).fillet(0.15),
    ],
    ids=["filleted-box", "filleted-extrusion"],
)
def test_a_plan_with_no_closed_form_is_sampled_through_its_tessellation(make):
    shape = make().size(0.25)
    assert not shape.is_analytic(), "the premise: this plan has no closed form"

    pts, nrm = shape.sample_boundary(500, np.random.default_rng(0))
    assert pts.shape == (500, 3) and nrm.shape == (500, 3)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)

    q = shape.sample_interior(500, np.random.default_rng(1))
    assert q.shape == (500, 3)
    assert np.asarray(shape.contains(q)).all(), "sample_interior and contains must agree"


def test_a_filleted_solid_makes_a_mesh_free_domain():
    """The payoff: no volume mesh, and the faces are still named."""
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).fillet(0.15).size(0.25).domain()
    assert d.__dict__.get("_mesh") is None, "constructing must not run the volume mesher"
    assert {"interior", "boundary"} <= set(d._geometry_tags)
    d.variable("interior", sample=(700, None), split=True)
    d.variable("boundary", sample=(300, None), normals=True, split=True)
    assert d.__dict__.get("_mesh") is None, "sampling must not run it either"
    assert np.asarray(d.context["interior"]).shape == (1, 1, 700, 3)


def test_a_straight_sweep_is_an_extrusion_and_never_needs_the_tessellation():
    """Sweeping along a straight line is an extrusion, and the DSL rewrites it as one -- so it keeps
    the analytic path and no mesher runs at all. Worth pinning, because it means the tessellation is
    not what serves a straight sweep."""
    tube = jno.Shape.disk(0.0, 0.0, 0.3).sweep(jno.Path(0, 0, 0).line_to(0, 0, 3))
    assert tube._node[0] == "extrude"
    assert tube.is_analytic()
    q = tube.size(0.3).sample_interior(400, np.random.default_rng(0))
    assert np.all(np.hypot(q[:, 0], q[:, 1]) <= 0.3 + 1e-9), "every point is within the tube radius"
    assert np.all((q[:, 2] >= -1e-9) & (q[:, 2] <= 3.0 + 1e-9))


def test_a_surface_the_mesher_leaves_open_is_refused_by_name():
    """gmsh does not sew the seam of a solid swept along an ARC: its boundary mesh has edges that
    bound a single facet, and without a closed surface the even-odd rule has no inside to find.

    Refused rather than answered wrongly. This is not a resolution problem -- measured across
    h = 0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.08 the count of open edges runs 7, 7, 7, 10, 13, 16, 24,
    so refining makes it worse. A gentler arc with a thinner profile is open too. The message says
    as much and sends the caller to `.build()`.
    """
    bent = jno.Shape.disk(0.0, 0.0, 0.3).sweep(jno.Path(0, 0, 0).arc_to(2, 0, 2, through=(0.6, 0, 1.4)))
    with pytest.raises(RuntimeError, match="not a closed manifold"):
        bent.size(0.3).tessellate()


def test_the_refusal_names_a_method_that_exists():
    """The error a caller reaches for has to point somewhere real."""
    assert hasattr(jno.Shape, "tessellate")
