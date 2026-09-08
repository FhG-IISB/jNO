"""Non-conforming multi-body meshes — ``shape.regions(..., conforming=False)``.

``shape.regions`` fragments its pieces so a shared interface meshes conforming (one set of nodes, no
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
        jno.shape.regions(
            lower=jno.shape.box(0, 0, 0, 1, 1, _LOWER_TOP),
            upper=jno.shape.box(0, 0, _LOWER_TOP, 1, 1, _UPPER_TOP),
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
    return np.asarray(jno.fem(terms).solve()).reshape(-1)


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


def _stack(base_size, film_size):
    """Two stacked blocks sharing a full face, meshed independently at their own resolutions."""
    return jno.shape.regions(
        base=jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=base_size),
        film=jno.shape.rect(0.0, 1.0, 2.0, 1.4, size=film_size),
        conforming=False,
    ).domain()


def test_each_region_is_meshed_at_its_own_resolution():
    """Mesh size used to be a Distance+Threshold *field*, i.e. a function of POSITION — and the two
    sides of a non-conforming interface sit at the same position, so both bodies were meshed
    identically however different their requested sizes (measured: a 3x ratio still gave 41 nodes on
    each side). Sizing each region's own entities is what makes 'coarse body, fine body' expressible,
    and therefore what makes a genuinely non-matching interface reachable at all."""
    same = _stack(0.25, 0.25)
    lo, hi = (len(np.asarray(same.tag_indices[t]).reshape(-1)) for t in _interface_tags(same))
    assert lo == hi, "equal sizes should still mesh the two sides alike"

    graded = _stack(0.25, 0.08)
    lo, hi = (len(np.asarray(graded.tag_indices[t]).reshape(-1)) for t in _interface_tags(graded))
    assert hi > 2 * lo, f"a 3x size ratio must give genuinely different node counts, got {lo} vs {hi}"


def _poisson_2d(d, order=1):
    u, v = d.fem_symbols(order=order)
    s, m = _interface_tags(d)
    c = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    a1, b1 = d.variable(s, split=True), d.variable(m, split=True)
    ui, vi = u.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])
    terms = [ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(a1[0], a1[1]) - u(b1[0], b1[1]), u(b[0], b[1]) - 0.0]
    return float(np.asarray(jno.fem(terms).solve()).max())


def test_a_graded_interface_uses_the_mortar_coupling():
    """The end of the chain: two bodies meshed at different resolutions, glued, and the tie reaches
    the *integrated* coupling rather than node-to-node matching. Equal sizes must still take the exact
    path — there is nothing to interpolate there, and using mortar would be strictly worse."""
    import jno.utils.solver.fem_utils as fu

    seen = {}
    orig = fu.build_periodic_prolongation
    fu.build_periodic_prolongation = lambda *a, **k: (lambda r: (seen.__setitem__("coupling", r["coupling"]), r)[1])(
        orig(*a, **k)
    )
    try:
        same = _poisson_2d(_stack(0.25, 0.25))
        assert seen["coupling"] == "conforming"
        graded = _poisson_2d(_stack(0.25, 0.08))
        assert seen["coupling"] == "mortar", "a graded interface must reach the integrated coupling"
    finally:
        fu.build_periodic_prolongation = orig
    assert abs(graded - same) / same < 0.05, f"graded {graded:.6f} vs uniform {same:.6f}"


@pytest.mark.parametrize(
    "order",
    [
        2,
        pytest.param(
            3,
            marks=pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason="A cubic edge carries 4 nodes and `_facet_dual_coeffs` derives the biorthogonal "
                "dual basis for 2 (P1) and 3 (P2) only -- a separate gap, and it says so rather than "
                "quietly using the wrong basis. Kept as a parameter so it stays visible.",
            ),
        ),
    ],
)
def test_the_mortar_ties_a_graded_interface_above_order_one(order):
    """A mortar between independently meshed bodies was unreachable above P1.

    A facet belongs to a tag by its **vertices**. The tie resolved one through ``tag_indices``, a P1
    node list, then asked whether ALL of a facet's nodes were in it -- so at P2 every facet was
    rejected on its absent midside node, ``facets[tag]`` came out empty, and the tie died with "no
    main facet connectivity was supplied for interpolation". Taylor-Hood over two independently meshed
    bodies needs exactly this, so it was the standing blocker on two-body flow.

    Order 3 is carried as an xfail: it has TWO nodes per edge, so it would catch a fix that merely
    allowed one extra node per facet -- but it stops earlier, on a dual basis that exists for P1 and
    P2 edges only. Selecting the facet and integrating over it are different gaps; this pins the
    first and names the second."""
    import jno.utils.solver.fem_utils as fu

    seen = {}
    orig = fu.build_periodic_prolongation
    fu.build_periodic_prolongation = lambda *a, **k: (lambda r: (seen.__setitem__("coupling", r["coupling"]), r)[1])(
        orig(*a, **k)
    )
    try:
        got = _poisson_2d(_stack(0.25, 0.08), order=order)
    finally:
        fu.build_periodic_prolongation = orig
    assert seen["coupling"] == "mortar", "a graded interface must reach the integrated coupling"
    ref = _poisson_2d(_stack(0.25, 0.25), order=1)
    assert abs(got - ref) / ref < 0.05, f"order-{order} mortar {got:.6f} vs conforming {ref:.6f}"


def _lap2(u, phi, r):
    a, b = u.bind(x=r[0], y=r[1]), phi.bind(x=r[0], y=r[1])
    return a.x * b.x + a.y * b.y


def _graded_stack():
    """Coarse base under a finer film, meshed independently — a genuinely non-matching interface."""
    return jno.shape.regions(
        base=jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=0.25),
        film=jno.shape.rect(0.0, 1.0, 2.0, 1.4, size=0.08),
        conforming=False,
    ).domain()


def test_tag_region_separates_two_coincident_faces():
    """The two sides of a non-conforming interface share coordinates exactly, so no ``d.tag``
    predicate can separate them — ``region=`` names the owning body, which is the only discriminator.
    Both must register as boundary regions AND resolve to *different* node sets."""
    d = _graded_stack()
    on = lambda x, y: np.abs(y - 1.0) < 1e-9  # noqa: E731
    d.tag("film_face", on, region="film")
    d.tag("base_face", on, region="base")

    assert "film_face" in d._boundary_regions and "base_face" in d._boundary_regions
    assert d._tag_regions == {"film_face": "film", "base_face": "base"}

    from jno._fem import _face_nodes

    pts = np.asarray(d.built_mesh.points)
    bn = np.unique(np.asarray(d.built_mesh.cells_dict["line"]))
    f = _face_nodes(d, pts[:, :2], bn, "film_face")
    b = _face_nodes(d, pts[:, :2], bn, "base_face")
    assert len(set(f.tolist()) & set(b.tolist())) == 0, "the two sides must not share nodes"
    assert len(f) != len(b), "a graded interface should give the two sides different node counts"


def test_a_region_scoped_tag_resolves_in_the_ASSEMBLY_numbering():
    """The same tag, but resolved the way the TIE resolves it -- against the assembly mesh.

    The test above hands `_face_nodes` the P1 mesh, where node ids happen to agree with the ones
    `tag_indices` is keyed on. The tie does not: it passes the ASSEMBLY mesh, which at order 2 has its
    own numbering and roughly twice the nodes. Ownership was applied by intersecting with
    ``tag_indices[region]`` -- an EXCLUSIVE partition of the *P1* mesh -- so P2 ids were intersected
    against P1 ids and the survivors were whichever collided by accident.

    Exclusivity is the second half of it: a node lying on both bodies is handed to one of them, so
    intersecting drops it from the other side's tag and the facet using it fails the subset test. On
    an annular tie split into arcs that punched a one-facet hole and the arc arrived as two chains.

    Owning by cell topology fixes both: the numbering is the assembly's, and a shared node belongs to
    every region whose cells contain it.
    """
    d = _graded_stack()
    on = lambda x, y: np.abs(y - 1.0) < 1e-9  # noqa: E731
    d.tag("film_face", on, region="film")
    d.tag("base_face", on, region="base")

    u, v = d.fem_symbols(order=2)
    c = d.variable("interior", split=True)
    jno.fem([_lap2(u, v, c)])  # force the order-2 assembly mesh into existence

    from jno._fem import _boundary_facets, _face_nodes
    from jno.utils.solver.fem_utils import _cell_region_mask

    pts = np.asarray(d._fem_native_dof_points_all[0])
    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    assert len(pts) > len(np.asarray(d.built_mesh.points)), "order 2 must add nodes to resolve against"
    bn = np.unique(_boundary_facets(pts, cells, 2, 2, "triangle"))

    got = {}
    for tag, region in (("film_face", "film"), ("base_face", "base")):
        sel = np.asarray(_face_nodes(d, pts, bn, tag, cells), dtype=int).reshape(-1)
        own = np.unique(cells[np.asarray(_cell_region_mask(d, region)).reshape(-1) > 0])
        assert len(sel) > 0, f"{tag} resolved to nothing in the assembly numbering"
        assert set(sel.tolist()) <= set(own.tolist()), f"{tag} reached outside {region}'s own cells"
        got[tag] = sel

    assert not (set(got["film_face"].tolist()) & set(got["base_face"].tolist())), "sides must stay apart"
    # and neither side may be truncated. Compare each side against ITS OWN P1 count -- the seam holds
    # two coincident node sets, so the combined count is not the bar. Order 2 adds a midpoint per edge,
    # so a side with `n` P1 nodes must come back with about `2n - 1`.
    p1pts = np.asarray(d.built_mesh.points)[:, :2]
    bn1 = np.unique(np.asarray(d.built_mesh.cells_dict["line"]))
    for tag, sel in got.items():
        n1 = len(np.asarray(_face_nodes(d, p1pts, bn1, tag), dtype=int).reshape(-1))
        assert len(sel) >= 2 * n1 - 1, f"{tag} kept {len(sel)} of the ~{2 * n1 - 1} nodes its P1 side implies"


def test_tag_region_reaches_the_interface_at_all():
    """A non-conforming interface is deliberately kept OUT of the catch-all ``"boundary"`` region, so
    a predicate over the interface plane finds nothing there. ``region=`` has to widen the facet search
    to interface facets, or the tag silently never registers."""
    d = _graded_stack()
    with pytest.raises(ValueError, match="unknown region"):
        d.tag("bad", lambda x, y: np.abs(y - 1.0) < 1e-9, region="not_a_body")
    d.tag("plain", lambda x, y: np.abs(y - 1.0) < 1e-9)  # no region= -> never reaches the interface
    assert "plain" not in d._boundary_regions
    d.tag("owned", lambda x, y: np.abs(y - 1.0) < 1e-9, region="film")
    assert "owned" in d._boundary_regions


def test_a_coarse_secondary_is_reordered_rather_than_left_wrong():
    """In ``u(A) - u(B)`` the secondary A is eliminated in favour of an interpolation from B, so the secondary
    must be the FINER side or the fine mesh's interface resolution is discarded. Measured on a
    coating/substrate tie (81 nodes against 10), the wrong order was off by 10.62% with no error at
    all — so the tie reorders itself and says so, rather than trusting the caller to know the rule."""
    ref = None
    for first, second in (("film", "base"), ("base", "film")):
        d = _graded_stack()
        on = lambda x, y: np.abs(y - 1.0) < 1e-9  # noqa: E731
        a = d.variable("A", where=on, region=first, split=True)
        b = d.variable("B", where=on, region=second, split=True)
        u, v = d.fem_symbols()
        c = d.variable("interior", split=True)
        bb = d.variable("boundary", split=True)
        ui, vi = u.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])
        got = float(
            np.asarray(
                jno.fem(
                    [ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(a[0], a[1]) - u(b[0], b[1]), u(bb[0], bb[1]) - 0.0]
                ).solve()
            ).max()
        )
        if ref is None:
            ref = got
        else:
            assert abs(got - ref) / ref < 1e-3, f"both orderings must agree: {got:.6f} vs {ref:.6f}"


def test_p2_on_a_nonconforming_domain_keeps_the_bodies_apart():
    """``_promote_to_degree`` used to deduplicate synthesised nodes by physical COORDINATE -- the right
    conformity test for one body and the wrong one for two. A ``conforming=False`` interface is
    coincident *on purpose*, so every P2 node added there was merged across the bodies and welded them:
    measured on this two-body bar, **37 nodes were referenced by cells of BOTH bodies**, all at the
    interface. Benign for a tie, wrong for contact (those DOFs could then never separate), and silent
    either way -- so it was refused outright, which also put Taylor-Hood (P2 velocity / P1 pressure)
    out of reach on independently meshed bodies.

    The promotion now keys on the topological ENTITY (the P1 vertices its reference point's non-zero
    weights span), which separates entities that merely coincide in space while still collapsing a
    genuinely shared one from either neighbouring cell."""

    def build(conforming, order):
        d = _bar(conforming, 0.6)
        u, v = d.fem_symbols(order=order)
        c = d.variable("interior", split=True)
        b = d.variable("boundary", split=True)
        ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(b[0], b[1], b[2]) - 0.0])
        return d, fem

    d, fem = build(False, 2)  # previously raised
    from jno.utils.solver.fem_utils import _cell_region_mask

    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    na, nb = (np.unique(cells[np.asarray(_cell_region_mask(d, n)).reshape(-1) > 0]) for n in ("lower", "upper"))
    assert len(np.intersect1d(na, nb)) == 0, "the two bodies must not share a node -- that is the weld"

    assert build(True, 2)[1] is not None  # a conforming interface shares its surface anyway
    assert build(False, 1)[1] is not None  # P1 duplicated the interface nodes correctly all along


def test_conforming_is_a_reserved_region_name():
    with pytest.raises(TypeError, match="must be a bool"):
        jno.shape.regions(a=jno.shape.box(0, 0, 0, 1, 1, 1), b=jno.shape.box(0, 0, 1, 1, 1, 2), conforming="no")
