"""``domain.tag(..., region=...)`` must name ONE body's surface — points and normals both.

Naming a single body's surface is the prerequisite for contact: ``u.gap(secondary, main)`` has to be
handed two surfaces that are actually different, and a normal that points out of the body it belongs
to. Two defects stopped that, and neither was loud.

**P1 — ``region=`` did not restrict which facets the tag got.** The predicate alone cannot separate two
bodies, so a predicate true everywhere handed `sA` and `sB` the SAME facets: measured on two disjoint
squares, 40 boundary points each, of which only 16 and 24 were their own. (On the gear mesh where this
surfaced it was 1120 normals in each of the two tags — the full set, twice.) A contact pair built from
those two tags is a body against itself.

**P2 — the point pool and the normal pool were different sets.** A tag's pool is interior + boundary
(89 points here) while ``normals_by_tag`` holds one normal per BOUNDARY point (16). The sampler draws
indices from the first and indexes the second with them, so ``sample(..., normals=True)`` raised
``IndexError``. Both readings are legitimate — the same tag can be a volume region for one term and a
surface for its normals — so the pool is not narrowed in general, only for a caller that has said it
will index the two together.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver.fem_utils import _cell_region_mask

BOX = {"A": (0.0, 1.0), "B": (1.5, 2.5)}  # two disjoint unit squares, meshed at different sizes


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def two_bodies():
    """Disjoint on purpose: nothing but ``region=`` can tell the two surfaces apart."""
    d = jno.Shape.regions(
        A=jno.Shape.rect(*BOX["A"][:1], 0, BOX["A"][1], 1).sized(0.25),
        B=jno.Shape.rect(BOX["B"][0], 0, BOX["B"][1], 1).sized(0.17),
        conforming=False,
    ).domain()
    _ = d.built_mesh
    everywhere = lambda x, y: x**2 >= -1.0  # noqa: E731 — true on both bodies: `region=` must do the work
    d.tag("sA", everywhere, region="A")
    d.tag("sB", everywhere, region="B")
    return d


def _own_nodes(d, region):
    tri = np.asarray(d.built_mesh.cells_dict["triangle"])
    return np.unique(tri[np.asarray(_cell_region_mask(d, region)).reshape(-1) > 0])


# ----------------------------------------------------------------------------------------------
# P1 — the tag belongs to its region
# ----------------------------------------------------------------------------------------------
def test_a_region_scoped_tag_only_gets_its_own_bodys_facets(two_bodies):
    """Before the fix both tags came back with the whole boundary — 40 points each, 16+24 of them
    misattributed."""
    d = two_bodies
    pts = np.asarray(d.built_mesh.points)[:, :2]
    for tag, region in (("sA", "A"), ("sB", "B")):
        reg = d._boundary_regions[tag]
        rp = np.asarray(reg.points)[:, :2]
        own = {tuple(np.round(q, 9)) for q in pts[_own_nodes(d, region)]}
        stray = [q for q in rp if tuple(np.round(q, 9)) not in own]
        assert not stray, f"{tag} claims {len(stray)} points outside body {region}, e.g. {stray[0]}"


def test_the_two_surfaces_partition_the_boundary(two_bodies):
    """Disjoint, and between them the whole thing: neither over- nor under-claiming."""
    d = two_bodies
    key = lambda p: {tuple(np.round(q, 9)) for q in np.asarray(p)[:, :2]}  # noqa: E731
    a, b = (key(d._boundary_regions[t].points) for t in ("sA", "sB"))
    full = key(d._mesh_pool["boundary"])
    assert not (a & b), f"{len(a & b)} points claimed by both bodies"
    assert a | b == full, f"union misses {len(full - (a | b))} and invents {len((a | b) - full)}"


# ----------------------------------------------------------------------------------------------
# P2 — points and normals describe the same surface
# ----------------------------------------------------------------------------------------------
def test_points_and_normals_pair_and_point_out_of_their_own_body(two_bodies):
    """The sampler indexes both with one index set, so a length mismatch is an ``IndexError`` waiting;
    a normal from the *other* body is worse, because it is silent."""
    d = two_bodies
    for tag in ("sA", "sB"):
        (_, p, n), = d._sampling_groups_for_tag(tag, for_normals=True)
        p, n = np.asarray(p)[:, :2], np.asarray(n)[:, :2]
        assert len(p) == len(n), f"{tag}: {len(p)} points against {len(n)} normals"

        x0, x1 = BOX[tag[-1]]
        assert (p[:, 0] >= x0 - 1e-9).all() and (p[:, 0] <= x1 + 1e-9).all(), f"{tag}: a point off body"
        edge = np.minimum.reduce([abs(p[:, 0] - x0), abs(p[:, 0] - x1), abs(p[:, 1]), abs(p[:, 1] - 1)])
        assert (edge < 1e-9).all(), f"{tag}: an interior point carries a normal"

        assert np.allclose(np.linalg.norm(n, axis=1), 1.0), f"{tag}: normals not unit"
        centre = np.array([0.5 * (x0 + x1), 0.5])
        assert (((p - centre) * n).sum(1) > 0).all(), f"{tag}: a normal points into its own body"


def test_the_volume_reading_of_the_same_tag_keeps_the_whole_pool(two_bodies):
    """The fix must not cost the other reading. A tag can be a volume region for one term and a surface
    for its normals; only the caller that pairs them gets the surface."""
    d = two_bodies
    for tag in ("sA", "sB"):
        (_, whole, _), = d._sampling_groups_for_tag(tag)
        (_, surf, _), = d._sampling_groups_for_tag(tag, for_normals=True)
        assert np.asarray(whole).shape[0] == np.asarray(d._mesh_pool[tag]).shape[0]
        assert np.asarray(whole).shape[0] > np.asarray(surf).shape[0], "the pool is interior + boundary"


def test_sampling_with_normals_no_longer_raises(two_bodies):
    """The user-facing symptom: ``IndexError: index 71 is out of bounds for axis 0 with size 16``."""
    d = two_bodies
    for tag in ("sA", "sB"):
        d.sample({tag: (8, None)}, normals=True)
        assert np.asarray(d.context[tag]).shape[-2:] == (8, 2)


# ----------------------------------------------------------------------------------------------
# The case coordinates cannot decide
# ----------------------------------------------------------------------------------------------
def _own_boundary_facets(d, region):
    """Edges used by exactly one of ``region``'s triangles — the body's own boundary, counted directly."""
    tri = np.asarray(d.built_mesh.cells_dict["triangle"])
    m = np.asarray(_cell_region_mask(d, region)).reshape(-1) > 0
    e = np.sort(tri[m][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uq, c = np.unique(e, axis=0, return_counts=True)
    return uq[c == 1]


@pytest.mark.parametrize("hB,label", [(0.25, "matching"), (0.17, "differing")])
def test_touching_bodies_keep_their_own_interface_side(hB, label):
    """Two bodies that TOUCH, so the interface facets of the two sides are coincident.

    Coordinates cannot separate those, and when the two sides are meshed at the same size their nodes
    land on identical points — measured, `sA` came back with 20 facets where body A has 16, having
    absorbed all four of B's seam facets. The count is the only thing that shows it: the stray facets
    contribute no new POINTS, because those coordinates already belong to both bodies.

    What does separate them is the name. `geometry.emit` writes an interface side as ``"A|B.A"``, so the
    owner is on the tag; the exterior boundary, which carries no owner, falls back to cell topology —
    safe there, because two bodies' exterior faces cannot coincide.
    """
    d = jno.Shape.regions(A=jno.Shape.rect(0, 0, 1, 1).sized(0.25),
                          B=jno.Shape.rect(1, 0, 2, 1).sized(hB), conforming=False).domain()
    _ = d.built_mesh
    everywhere = lambda x, y: x**2 >= -1.0  # noqa: E731
    d.tag("sA", everywhere, region="A")
    d.tag("sB", everywhere, region="B")

    for tag, region in (("sA", "A"), ("sB", "B")):
        got = len(np.asarray(d._boundary_regions[tag].facets))
        want = len(_own_boundary_facets(d, region))
        assert got == want, (
            f"[{label} seam] {tag} has {got} facets where body {region} has {want} — "
            f"{'it absorbed the other side of the interface' if got > want else 'it lost facets'}"
        )
