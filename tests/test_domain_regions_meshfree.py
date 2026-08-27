"""A multi-region domain does not mesh until something needs the mesh.

Each region of an analytic ``Shape.regions`` plan is a closed-form membership test, so its INTERIOR
tag is servable from the geometry — that is the whole vocabulary a partial-element or collocation
method wants, and meshing to get it is wasted work.

Its INTERFACE tags are the mesher's shared conforming facets. Point sampling does not reconstruct
them (they carry numbered facet groups, `a|b.1`, that only the emitter knows), so naming one builds
the mesh rather than being served halfway or refused.
"""

import numpy as np
import pytest

import jno


def two_lines():
    mk = lambda o: jno.Shape.line([(o, 0, 0), (o, 0, 0.01), (o + 0.02, 0, 0.01), (o + 0.02, 0, 0)], r=2e-4)
    return mk(0.0).name("W1") + mk(0.03).name("W2"), mk(0.0), mk(0.03)


def two_boxes(size=0.5):
    a = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).name("A")
    b = jno.Shape.box(1, 0, 0, 2, 1, 1, size=size).name("B")
    return a + b


def test_building_a_region_domain_does_not_mesh():
    d = two_boxes().domain()
    assert d.__dict__.get("_mesh") is None
    assert d.__dict__.get("_lazy_plan") is not None


def test_a_region_is_a_tag_named_the_way_the_mesher_names_it():
    """Lazy and eager must agree, or the same script means two things."""
    lazy = two_boxes().domain()
    eager = two_boxes().domain()
    eager.mesh  # force the build
    for name in ("A", "B"):
        assert name in lazy._geometry_tags
        assert name in eager.avaiable_mesh_tags


def test_region_points_come_from_that_region_and_no_other():
    shape, w1, w2 = two_lines()
    d = shape.domain()
    rng = np.random.default_rng(0)
    for tag, mine, theirs in [("W1", w1, w2), ("W2", w2, w1)]:
        pts, _ = d._draw_geometry_points("interior", None, d._tag_predicates[tag], 200, rng, False)
        assert np.asarray(mine.contains(pts)).all()
        assert not np.asarray(theirs.contains(pts)).any()
    assert d.__dict__.get("_mesh") is None  # and none of it needed a mesh


def test_naming_an_interface_builds_the_mesh():
    d = two_boxes().domain()
    assert d.__dict__.get("_mesh") is None
    assert d.interface_tags() == ["A|B"]
    assert d.__dict__.get("_mesh") is not None


def test_a_misspelt_tag_is_still_an_error_and_not_a_surprise_mesh():
    """Only a tag whose two halves are BOTH known regions is an interface; the rest still raise."""
    d = two_boxes().domain()
    with pytest.raises(ValueError, match="not in the mesh pool or context"):
        d.variable("A|typo", sample=(4, None))
    assert d.__dict__.get("_mesh") is None


def test_a_singly_named_plan_defers_too():
    """One name is one sub-body: no interface to resolve, so meshing at construction bought nothing.

    Its `_shape_regions` entry is recorded either way -- that is what carries an `.attach`ed property
    -- and its tag vocabulary is the same deferred or not, so nothing downstream can tell.
    """
    named = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).name("solo").domain()
    plain = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    assert named.__dict__.get("_lazy_plan") is not None
    assert named.__dict__.get("_mesh") is None
    assert sorted(named._shape_regions) == ["solo"]
    assert sorted(named._geometry_tags) == sorted(plain._geometry_tags)


def test_a_structured_plan_still_meshes_eagerly():
    """By the time the domain sees it, a structured plan has been swapped for its lattice closure,
    which is not a Shape and carries no closed-form membership."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.3).structured().domain()
    assert d.__dict__.get("_lazy_plan") is None
