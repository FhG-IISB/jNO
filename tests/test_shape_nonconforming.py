"""Non-conforming multi-body meshes — ``Shape.regions(..., conforming=False)``.

``Shape.regions`` fragments its pieces so a shared interface meshes conforming (one set of nodes, no
tie needed). ``conforming=False`` skips the fragment: each piece is meshed independently, so two
touching regions end up with two **coincident but non-matching** surfaces and duplicated nodes. Gluing
those with ``u(A) - u(B)`` in ``jno.fem`` is what lets two bodies meshed at different resolutions be
solved as one — the reason the mortar coupling exists.

Two things this pins down:

* **The mesh really separates.** Interface nodes are duplicated, and each side gets its own auto tag
  ``"a|b.a"`` / ``"a|b.b"``, because the two faces are spatially coincident and no ``domain.tag``
  predicate could tell them apart.
* **The tie actually glues.** A tied two-body bar must reproduce the single-conforming-mesh solution.
  This is the regression that matters: with the interface left in the catch-all ``"boundary"`` region
  the Dirichlet pinned it, ``u`` was exactly 0 across the whole interface, and the solve silently
  returned two disconnected bodies — converging to the *wrong* answer rather than failing.
"""

import numpy as np
import pytest

import jno

#: The bar is deliberately **asymmetric** (a 1-tall block under a 1.5-tall one). A symmetric 1x1x2 bar
#: puts the interface exactly on the symmetry plane, where the exact solution already has zero normal
#: flux -- so the natural "do nothing" condition an UNTIED interface gets happens to be the right
#: answer, the tie changes nothing, and every test below would pass without it. Measured: untied gave
#: 0.06995 against a conforming 0.06988. Off the symmetry plane the tie has to do real work.
_LOWER_TOP, _UPPER_TOP = 1.0, 2.5


def _bar(conforming, size):
    """A 1x1x2.5 bar as two stacked blocks, either fragmented or independently meshed."""
    return (
        jno.Shape.regions(
            lower=jno.Shape.box(0, 0, 0, 1, 1, _LOWER_TOP),
            upper=jno.Shape.box(0, 0, _LOWER_TOP, 1, 1, _UPPER_TOP),
            conforming=conforming,
        )
        .sized(size)
        .domain()
    )


def _interface_tags(d):
    return sorted(t for t in d.built_mesh.cell_sets if "|" in t)


def _poisson(d, tie):
    """-lap(u) = 1, u = 0 on the outer boundary; optionally glue the two bodies."""
    u, v = d.fem_symbols()
    c = d.variable("interior", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    terms = [ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi]
    if tie:
        a, b = (d.variable(t, split=True) for t in _interface_tags(d))
        terms.append(u(a[0], a[1], a[2]) - u(b[0], b[1], b[2]))
    zb = d.variable("boundary", split=True)
    terms.append(u(zb[0], zb[1], zb[2]) - 0.0)
    return np.asarray(jno.fem(terms, element_type="TET4").solve()).reshape(-1)


def _interface_interior(pts):
    """Nodes strictly inside the z = 1 interface (not on the bar's outer wall)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return (np.abs(z - _LOWER_TOP) < 1e-9) & (x > 1e-6) & (x < 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6)


def test_conforming_regions_share_one_interface():
    """The default: fragmented, so the interface is a single shared node set and one ``"a|b"`` tag."""
    d = _bar(True, 0.4)
    pts = np.asarray(d.built_mesh.points)
    on_plane = np.abs(pts[:, 2] - _LOWER_TOP) < 1e-9
    assert int(on_plane.sum()) == len(np.unique(np.round(pts[on_plane], 9), axis=0))  # no duplicates
    assert _interface_tags(d) == ["lower|upper"]


def test_nonconforming_regions_duplicate_the_interface_nodes():
    """The fragment is skipped, so each body carries its own copy of the interface surface."""
    d = _bar(False, 0.4)
    pts = np.asarray(d.built_mesh.points)
    on_plane = np.abs(pts[:, 2] - _LOWER_TOP) < 1e-9
    n_nodes, n_distinct = int(on_plane.sum()), len(np.unique(np.round(pts[on_plane], 9), axis=0))
    assert n_nodes > n_distinct, "the two sides must not share nodes"
    assert _interface_tags(d) == ["lower|upper.lower", "lower|upper.upper"]
    for t in _interface_tags(d):
        assert len(np.asarray(d.tag_indices[t]).reshape(-1)) > 0


def test_each_side_is_tagged_and_registered_as_a_boundary_region():
    """The sides are spatially coincident, so a d.tag() predicate cannot separate them — the emitter
    has to name them, and they must bind as regions so ``u(tag)`` works."""
    d = _bar(False, 0.4)
    lo, up = _interface_tags(d)
    regions = getattr(d, "_boundary_regions", {})
    assert lo in regions and up in regions
    pts = np.asarray(d.built_mesh.points)
    for t in (lo, up):
        assert np.allclose(pts[np.asarray(d.tag_indices[t]).reshape(-1), 2], _LOWER_TOP)


def test_interface_is_excluded_from_the_catch_all_boundary():
    """The regression. Each interface face IS a facet of exactly one cell, so it is topologically
    boundary — but it is semantically internal, and a plain ``u(boundary) - g`` must not pin it."""
    d = _bar(False, 0.13)
    pts = np.asarray(d.built_mesh.points)
    inner = np.flatnonzero(_interface_interior(pts))
    assert len(inner) > 0, "the mesh must be fine enough to have interface-interior nodes"
    sol = _poisson(d, tie=True)
    assert np.abs(sol[inner]).max() > 1e-3, "interface nodes were pinned — the bodies solve separately"


def test_the_ring_where_the_interface_meets_the_outer_wall_stays_pinned():
    """The filter is 'on at least one NON-interface facet', not 'not on an interface facet': the
    nodes where the interface meets the bar's side walls belong to both and must stay Dirichlet."""
    d = _bar(False, 0.13)
    pts = np.asarray(d.built_mesh.points)
    z, x = pts[:, 2], pts[:, 0]
    ring = np.flatnonzero((np.abs(z - _LOWER_TOP) < 1e-9) & (np.abs(x) < 1e-9))  # interface edge on the x=0 wall
    assert len(ring) > 0
    sol = _poisson(d, tie=True)
    assert np.abs(sol[ring]).max() < 1e-12, "the outer wall must still be pinned at the interface edge"


@pytest.mark.parametrize("size", [0.25, 0.18])
def test_tied_two_body_bar_reproduces_the_conforming_solution(size):
    """The Phase-B oracle: gluing two independently meshed bodies must give the same physics as one
    conforming mesh. Both discretisations differ, so agreement is at discretisation level, not exact
    — the failure this guards against was a fixed ~20% offset that did not shrink with refinement."""
    ref = _poisson(_bar(True, size), tie=False).max()
    got = _poisson(_bar(False, size), tie=True).max()
    assert abs(got - ref) / ref < 0.05, f"tied {got:.6f} vs conforming {ref:.6f}"


def test_the_tie_is_what_makes_the_field_continuous():
    """The negative control, measuring the tie directly rather than through a scalar peak.

    Gluing means the two coincident node sets carry the *same* value. Without the tie each body is
    solved with a natural (zero-flux) condition on its own interface face, so the two sides drift
    apart. Comparing peaks is a poor control here — untied differs by only ~2.5 %, and on a symmetric
    bar not at all, because zero flux happens to be right on a symmetry plane."""
    d = _bar(False, 0.18)
    pts = np.asarray(d.built_mesh.points)
    lo, up = (np.asarray(d.tag_indices[t]).reshape(-1) for t in _interface_tags(d))
    # pair the two sides by coordinate (they are spatially coincident)
    key = {tuple(np.round(pts[i, :2], 9)): i for i in lo}
    pairs = [(key[k], j) for j in up if (k := tuple(np.round(pts[j, :2], 9))) in key]
    assert len(pairs) > 10, "the two sides must have coincident nodes to compare"

    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    tied, untied = _poisson(d, tie=True), _poisson(d, tie=False)
    scale = float(tied.max())
    assert np.abs(tied[a] - tied[b]).max() < 1e-6 * scale, "the tie must make the field continuous"
    assert np.abs(untied[a] - untied[b]).max() > 1e-2 * scale, "without it the sides must drift apart"


def test_conforming_is_a_reserved_region_name():
    with pytest.raises(TypeError, match="must be a bool"):
        jno.Shape.regions(a=jno.Shape.box(0, 0, 0, 1, 1, 1), b=jno.Shape.box(0, 0, 1, 1, 1, 2), conforming="no")
