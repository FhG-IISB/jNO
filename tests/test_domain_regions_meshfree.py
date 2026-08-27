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


def test_a_singly_named_plan_still_meshes_eagerly():
    """The gate widened for MULTI-region plans only. A lone `.name(...)` is a different thing — it
    asks for one tagged sub-body, and nothing here changed that path."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).name("solo").domain()
    assert d.__dict__.get("_lazy_plan") is None


def test_an_enclosing_region_does_not_claim_the_region_it_encloses():
    """Regions may OVERLAP; the first declared wins. Geometric containment is not belonging.

    `background` encloses `inner` entirely, so `background.contains` is true at (3, 3) — but (3, 3)
    belongs to `inner`. Answering the containment question instead of the belonging one stops the
    masks being a partition, and `by_region` sums `mask * value`, so the background's coefficient
    would be added on top of every other region's: silent, physical, and wrong.
    """
    inner = jno.Shape.rect(2, 2, 4, 4, size=0.5).name("inner")
    background = jno.Shape.rect(0, 0, 6, 6, size=0.5).name("background")  # declared LAST, encloses inner
    d = (inner + background).domain()

    at = lambda x, y: (np.array([float(x)]), np.array([float(y)]))
    assert bool(d._tag_predicates["inner"](*at(3, 3))[0])
    assert not bool(d._tag_predicates["background"](*at(3, 3))[0])
    assert bool(d._tag_predicates["background"](*at(5, 5))[0])  # the leftover void is still its own

    # and over a cloud: every point belongs to exactly one region
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [6, 6], size=(2000, 2))
    owned = sum(d._tag_predicates[n](pts[:, 0], pts[:, 1]).astype(int) for n in ("inner", "background"))
    assert set(np.unique(owned)) == {1}


def test_a_regions_plan_declares_its_resolution_through_its_regions():
    """`(a + b)._size` is None even when a and b both declare one — the sizes live on the REGIONS.

    Reading only the outer value would tell the domain it has no declared resolution when it plainly
    has one, and `variable('interior')` would refuse to hand back a node set it can perfectly well
    build.
    """
    sized = two_boxes(size=0.4).domain()
    assert sized.__dict__.get("_mesh") is None  # still deferred at construction
    xs = sized.variable("interior", split=True)  # no sample= : the declared size settles it
    assert len(xs) >= 3  # x, y, z (and t)
    assert sized.__dict__.get("_mesh") is not None  # asking for the node set built it

    bare = (jno.Shape.box(0, 0, 0, 1, 1, 1).name("A") + jno.Shape.box(1, 0, 0, 2, 1, 1).name("B")).domain()
    with pytest.raises(ValueError, match="no mesh size was declared"):
        bare.variable("interior")
    assert bare.__dict__.get("_mesh") is None  # and it refused without meshing
