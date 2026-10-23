"""Region meshing and enclosure occlusion on a ``Shape``-built domain.

These cover ``jno/geometry/emit.py`` (per-region mesh size under ``conforming=True``, and classifying
OCC entities by an interior probe point rather than their centre of mass) and ``jno/domain/enclosure.py``
(taking the opaque occluder model from a Shape domain, which has no shapely ``_source_regions``).
Both are separate from ``Shape.attach`` and ship with those changes, not with it.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno


def test_enclosure_on_a_shape_domain_has_an_occluder_model():
    """Interface-mode enclosures take their opaque model from `_source_regions`, which a Shape-built
    domain does not have -- leaving NOTHING occluding, so every pair sees every other straight through
    solid material. It fails silently: closure and reciprocity are enforced afterwards, so the F that
    comes out looks perfectly plausible. Here a full-height blocker separates two solids; their
    mutual view factor must be ~0."""
    d = (
        jno.Shape.rect(1, 4, 2, 6, size=0.4).name("left")
        + jno.Shape.rect(4, 0, 5, 10, size=0.4).name("blocker")
        + jno.Shape.rect(7, 4, 8, 6, size=0.4).name("right")
        + jno.Shape.rect(0, 0, 10, 10, size=1.0).name("medium")
    ).domain()

    gap = d.enclosure(["left", "blocker", "right"], medium_tags=["medium"], axisymmetric=True, enforce_closure=False)
    tags = np.array([str(t) for t in np.asarray(gap.element_tags)])
    F = np.asarray(gap.view_factor)
    lo, hi = tags == "left", tags == "right"
    assert lo.any() and hi.any()
    # The blocker spans the full height, so nothing on `left` can see anything on `right`.
    assert F[np.ix_(lo, hi)].max() < 1e-3, "blocked pair is visible -- the occluder model is empty"
    # ...while each still sees the blocker facing it, so this is not just an all-zero matrix.
    assert F[np.ix_(lo, tags == "blocker")].max() > 1e-2


# --------------------------------------------------------------------------------------
# conforming per-region mesh size (the reason `_apply_region_sizes` now covers conforming too)
# --------------------------------------------------------------------------------------
def test_conforming_region_size_reaches_the_region_interior():
    """A Threshold field only refines a BAND around a region's boundary, leaving a wide region coarse
    in the middle. Point sizing carries the request inward, so a 6x finer region must produce
    substantially more than the ~6^2 cells a band-only refinement leaves it with."""
    fine = jno.Shape.rect(0, 0, 1, 1, size=0.02).name("fine")
    coarse = jno.Shape.rect(0, 0, 4, 1, size=0.12).name("coarse")
    mesh, _dim, _ds = (fine + coarse).build()

    tri = mesh.cells[0].data
    p = mesh.points[tri][:, :, :2]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    h = np.sqrt(2.0 * area)  # ~edge length of a right isoceles cell

    inner = mesh.points[tri].mean(axis=1)[:, :2]
    deep = mesh.cell_sets["fine"][0]
    deep = deep[(inner[deep, 0] > 0.3) & (inner[deep, 0] < 0.7) & (inner[deep, 1] > 0.3) & (inner[deep, 1] < 0.7)]
    assert len(deep) > 0, "no cells in the middle of the fine region"
    # Middle-of-region cells must track the region's own size, not the coarse background.
    assert np.median(h[deep]) < 0.5 * 0.12


def test_conforming_interface_point_takes_the_finest_claim():
    """Two ADJACENT regions each own their own entities, so both claim the points on the interface
    between them; min-wins is what lets the coarse side grade into the fine neighbour instead of
    quantising to whichever was written last. Declaration order must not change the outcome."""
    fine = jno.Shape.rect(0, 0, 1, 1, size=0.02).name("fine")
    coarse = jno.Shape.rect(1, 0, 4, 1, size=0.12).name("coarse")
    for first, second in ((fine, coarse), (coarse, fine)):
        _mesh, _dim, ds = (first + second).build()
        assert ds == pytest.approx(0.02), "the finest declared size must survive either ordering"


def test_ring_shaped_region_is_sized_by_an_interior_point_not_its_centre_of_mass():
    """The regression that made a multi-material domain over-refine: after `occ.fragment` a region
    enclosing another is a RING, and a ring's centre of mass lies in its hole -- inside a different
    region. Classifying entities by centre of mass therefore gave the enclosing region the enclosed
    region's mesh size. Here the outer ring must keep its own coarse size far from the inclusion."""
    inner = jno.Shape.rect(4, 4, 6, 6, size=0.06).name("inner")
    outer = jno.Shape.rect(0, 0, 10, 10, size=0.6).name("outer")
    mesh, _dim, _ds = (inner + outer).build()

    tri = mesh.cells[0].data
    p = mesh.points[tri][:, :, :2]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    h = np.sqrt(2.0 * area)

    cen = p.mean(axis=1)
    far = mesh.cell_sets["outer"][0]
    far = far[(np.abs(cen[far, 0] - 5.0) > 3.0) | (np.abs(cen[far, 1] - 5.0) > 3.0)]
    assert len(far) > 0
    # Nowhere near the 0.06 inclusion size: before the fix the whole ring inherited it.
    assert np.median(h[far]) > 5 * 0.06


def test_overlapping_regions_are_resolved_by_priority_not_by_size():
    """Distinct from the test above: when one region CONTAINS another, the earlier declaration owns
    the overlap outright, so the later region's size never applies to it."""
    fine = jno.Shape.rect(0, 0, 1, 1, size=0.02).name("fine")
    coarse = jno.Shape.rect(0, 0, 4, 1, size=0.12).name("coarse")
    _mesh, _dim, ds = (coarse + fine).build()  # coarse first -> it swallows the fine rect
    assert ds == pytest.approx(0.12)
