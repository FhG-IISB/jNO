"""``fem.solve(contact=...)`` — the pairing follows the solution instead of being frozen.

``u.gap(secondary, main)`` precomputes, once, which main nodes every secondary quadrature point reads.
That is correct only while displacements stay far below the element size; past it a point is still tied
to the facet it faced before anything moved. **Nothing reports this.** The solve converges perfectly
well, just for a contact configuration that is not the one being solved — measured on an involute gear
pair the torque ratio drifted up to 43% off the kinematic oracle and got *worse* under refinement.

The slot solves, re-runs the search at ``x + u``, and repeats until the pairing and the solution both
stop moving. What has to be true for that to be sound is pinned below: re-pairing at zero displacement
must reproduce the frozen tables exactly, the re-paired ``g0`` must not double-count the displacement
the residual re-applies, and a loop that does not settle must raise rather than return.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stacked_bars(size=0.4, c=1.0e3):
    """Two boxes meeting at z = 1, the upper one loaded into the lower through a penalised gap."""
    d = jno.Shape.regions(lower=jno.Shape.box(0, 0, 0, 1, 1, 1),
                          upper=jno.Shape.box(0, 0, 1, 1, 1, 2.5), conforming=False).domain(size=size)
    _ = d.built_mesh
    sec, main = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    u, v = d.fem_symbols(value_shape=(3,))
    ci, bb, sb = (d.variable(t, split=True) for t in ("interior", "boundary", sec))
    nrm = d.variable(sec, normals=True)
    gu, gv = jno.np.grad(u, [ci[0], ci[1], ci[2]]), jno.np.grad(v, [ci[0], ci[1], ci[2]])
    g = u.gap(sec, main, domain=d)
    terms = [jno.np.inner(gu, gv, n_contract=2) - 1.0 * v.bind(x=ci[0], y=ci[1], z=ci[2])[2],
             (-c * g) * jno.np.inner(nrm, v.bind(x=sb[0], y=sb[1], z=sb[2]), n_contract=1)]
    terms += [u(bb[0], bb[1], bb[2])[i] - 0.0 for i in range(3)]
    return d, jno.fem(terms)


# ----------------------------------------------------------------------------------------------
# The invariant the whole thing rests on
# ----------------------------------------------------------------------------------------------
def test_repairing_at_zero_displacement_reproduces_the_frozen_pairing():
    """Bit-identical, not merely close. The search runs the same projection the build-time tables did,
    so at ``u = 0`` it must land on the same facets with the same weights and the same ``g0``; asserted
    through the residual, at two states, because that is what every consumer of the tables sees."""
    _, fem = _stacked_bars()
    op = fem._op
    n = int(op.size)
    tb = op.repair_contact(np.zeros(n))
    rng = np.random.default_rng(0)
    for state in (np.zeros(n), 0.02 * rng.standard_normal(n)):
        frozen = np.asarray(op.residual(state, None)).reshape(-1)
        repaired = np.asarray(op.residual(state, {"__gap_tables__": tb})).reshape(-1)
        assert np.array_equal(frozen, repaired), (
            f"re-pairing at u=0 moved the residual by {np.abs(frozen - repaired).max():.3e}"
        )


def test_the_repaired_gap_does_not_double_count_the_displacement():
    """``g0`` is what the residual STARTS from, and it then subtracts ``n . D(u)`` with ``u`` measured
    from the REFERENCE configuration. Taking ``g0`` from the deformed frame would subtract the same
    displacement twice and the gap would read roughly double, so contact would never close.

    The check: re-pair at the solved state, then evaluate the residual THERE. If ``g0`` were measured
    in the deformed frame the contact term would jump by the full penalty times the displacement; it
    must instead stay near the converged residual, because the equation barely changed."""
    _, fem = _stacked_bars()
    op = fem._op
    sol = np.asarray(fem.solve()).reshape(-1)
    r_frozen = float(np.abs(op.residual(sol, None)).max())
    r_repaired = float(np.abs(op.residual(sol, {"__gap_tables__": op.repair_contact(sol)})).max())
    scale = float(np.abs(sol).max())
    assert scale > 1e-3, "the test needs a solution that actually moved"
    assert r_repaired < 1e-3 * scale, (
        f"re-pairing at the solution moved the residual to {r_repaired:.3e} (frozen {r_frozen:.3e}); "
        "a doubled gap would show up here"
    )


# ----------------------------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------------------------
def test_the_loop_settles_and_reports_how_many_rounds():
    _, fem = _stacked_bars()
    u = np.asarray(fem.solve(contact=jno.solve.contact())).reshape(-1)
    assert 2 <= int(fem.contact_rounds) <= 8
    assert np.isfinite(u).all() and np.abs(u).max() > 1e-3


def test_a_loop_that_cannot_settle_raises_rather_than_returning():
    """One round leaves nothing to compare the solution against, however good its answer looks. A
    contact solve that quietly stops iterating is the classic plausible-wrong answer, so this raises
    rather than returning."""
    _, fem = _stacked_bars()
    with pytest.raises(RuntimeError, match="cannot show that the search has settled"):
        fem.solve(contact=jno.solve.contact(rounds=1))


def test_rounds_must_be_at_least_one():
    _, fem = _stacked_bars()
    with pytest.raises(ValueError, match="rounds must be at least 1"):
        fem.solve(contact=jno.solve.contact(rounds=0))


# ----------------------------------------------------------------------------------------------
# What it refuses, and why — each by name
# ----------------------------------------------------------------------------------------------
def test_contact_refuses_a_form_that_declares_no_gap():
    d = jno.Shape.box(0, 0, 0, 1, 1, 1).domain(size=0.5)
    u, v = d.fem_symbols()
    ci, bb = d.variable("interior", split=True), d.variable("boundary", split=True)
    gu, gv = jno.np.grad(u, [ci[0], ci[1], ci[2]]), jno.np.grad(v, [ci[0], ci[1], ci[2]])
    fem = jno.fem([jno.np.inner(gu, gv, n_contract=1) - 1.0 * v.bind(x=ci[0], y=ci[1], z=ci[2]),
                   u(bb[0], bb[1], bb[2]) - 0.0])
    with pytest.raises(ValueError, match="declares no contact pair"):
        fem.solve(contact=jno.solve.contact())


def test_contact_refuses_the_assembled_tangent_by_name():
    """The assembled tangent hoists the contact block's sparsity pattern ONCE, from the pairing's
    concrete node ids. Re-pairing changes which main nodes each point reads, so that pattern is stale —
    and a stale pattern does not error, it silently drops the new couplings."""
    _, fem = _stacked_bars()
    with pytest.raises(NotImplementedError, match="newton\\(direct=True\\)"):
        fem.solve(contact=jno.solve.contact(), nonlinear=jno.solve.newton(direct=True))


def test_contact_refuses_a_load_path_march_by_name():
    """The march compiles one step and replays it under ``lax.scan`` — which is what keeps a load path
    reverse-mode differentiable — and a host-side search cannot run inside a scan."""
    _, fem = _stacked_bars()
    with pytest.raises(NotImplementedError, match="lax.scan"):
        fem.solve(contact=jno.solve.contact(), tau=jno.solve.adaptive())


# ----------------------------------------------------------------------------------------------
# Large sliding, against a closed form
# ----------------------------------------------------------------------------------------------
R_DISK, Y0, SLIDE = 1.0, 1.05, 0.9


def _block_on_disk(size=0.09):
    """A flat-bottomed block standing over the top of a disk, with a gap term on the block's underside.

    Chosen so the search has a closed form. The block's bottom is horizontal, so every secondary
    quadrature point sits at ``y = Y0`` with outward normal ``(0, -1)``; the main surface is a circle,
    so the closest point to ``(x, Y0)`` is ``R (x, Y0) / |(x, Y0)|`` and

        g0 = n . (Phi(x) - x) = Y0 - R Y0 / sqrt(x^2 + Y0^2)

    -- the closest-point separation resolved on the normal, which is what
    :func:`~.contact_search.project_points` computes. (NOT a ray cast along ``n``: for a curved main
    surface those differ, by 0.25 against 0.18 at the offsets used here.) And because the slide below
    is a purely horizontal RIGID translation, ``n . D`` vanishes, so the re-paired ``g0`` *is* the
    deformed gap with nothing left to model.
    """
    blk = jno.Shape.rect(-0.3, Y0, 0.3, Y0 + 0.4)
    d = jno.Shape.regions(disk=jno.Shape.disk(0, 0, R_DISK).sized(size),
                          blk=blk.sized(size), conforming=False).domain()
    _ = d.built_mesh
    eps = 1e-6
    d.tag("s_blk", lambda x, y: y < Y0 + eps, region="blk")
    d.tag("s_disk", lambda x, y: y > -R_DISK + eps, region="disk")

    u, v = d.fem_symbols(value_shape=(2,))
    Rg = {k: d.variable(k, split=True) for k in ("disk", "blk")}
    terms = []
    for k in Rg:
        ui, vi = u.bind(x=Rg[k][0], y=Rg[k][1]), v.bind(x=Rg[k][0], y=Rg[k][1])
        eu, ev = jno.np.symgrad(ui, [Rg[k][0], Rg[k][1]]), jno.np.symgrad(vi, [Rg[k][0], Rg[k][1]])
        terms.append(jno.np.inner(eu, ev, n_contract=2))
    sb = d.variable("s_blk", split=True)
    nb = d.variable("s_blk", normals=True)
    g = u.gap("s_blk", "s_disk", domain=d)
    terms.append(jno.np.maximum(0.0, -1.0e3 * g) * jno.np.inner(nb, v.bind(x=sb[0], y=sb[1]), n_contract=1))
    terms += [u(Rg["disk"][0], Rg["disk"][1]) - 0.0]
    return d, jno.fem(terms)


def _rigid_slide(d, fem, s):
    """The DOF vector for 'the block translates by (s, 0), the disk stays put'. No solve involved —
    this isolates the search from every solver question."""
    from jno.utils.solver.fem_utils import _cell_region_mask

    tri = np.asarray(d.built_mesh.cells_dict["triangle"])
    blk = np.unique(tri[np.asarray(_cell_region_mask(d, "blk")).reshape(-1) > 0])
    uu = np.zeros((int(np.asarray(d.built_mesh.points).shape[0]), 2))
    uu[blk, 0] = s
    return uu.reshape(-1)


def _min_gap(tables):
    """The smallest live separation in a re-paired payload.

    Read ``g0``, not ``g0_full``: the latter is scattered up to every boundary face and is ZERO off the
    secondary ones, so its minimum is the padding, not a gap."""
    from jno.utils.solver.contact_search import OPEN_GAP

    g = np.concatenate([np.asarray(t["g0"]).reshape(-1) for t in tables.values()])
    live = g[np.abs(g) < 0.5 * OPEN_GAP]
    assert live.size, "every slot inactive"
    return float(live.min())


def _closest_point_gap(x):
    """``Y0 - R Y0 / sqrt(x^2 + Y0^2)`` — the exact separation for this geometry. Increasing in |x|."""
    return Y0 - R_DISK * Y0 / np.sqrt(x**2 + Y0**2)


def test_the_search_follows_a_slide_of_many_facet_widths():
    """The whole point of the slot, against a closed form.

    The block starts centred on the disk's crown, where the separation is ``Y0 - R = 0.05``. Slid right
    by 0.9 -- ten element widths -- its nearest point is over ``x = 0.6``, where the disk has fallen
    away and the true separation is ``0.182``: nearly four times larger.

    A frozen pairing sees none of it. Its ``g0`` was measured once, at the crown, and the slide is
    tangential so nothing in ``g0 - n . D`` moves it -- it still reports 0.05 for a block that is
    nowhere near touching. Not a small error, and completely silent.

    The bracket: the leftmost secondary quadrature point lies somewhere in the leftmost facet, i.e. in
    ``x in [0.6, 0.6 + h]``, and the gap is increasing in x -- so the measured minimum must fall
    between the two, up to the faceting of the circle.
    """
    h = 0.09
    d, fem = _block_on_disk(size=h)
    op = fem._op
    frozen = _min_gap(op.repair_contact(np.zeros(int(op.size))))
    slid = _min_gap(op.repair_contact(_rigid_slide(d, fem, SLIDE)))

    assert abs(frozen - (Y0 - R_DISK)) < h**2, f"the reference gap should be {Y0 - R_DISK}, got {frozen:.4f}"

    lo, hi = _closest_point_gap(SLIDE - 0.3), _closest_point_gap(SLIDE - 0.3 + h)
    sag = h**2 / (8 * R_DISK)  # the polygon's chord sits inside the circle by this much
    assert lo - sag <= slid <= hi + sag, (
        f"after sliding {SLIDE} the gap should be in [{lo:.4f}, {hi:.4f}] (faceting {sag:.1e}), got {slid:.4f}"
    )
    assert slid > 3.0 * frozen, (
        f"the frozen pairing reports {frozen:.4f} where the truth is ~{lo:.4f}; if the search were "
        "doing nothing these would agree"
    )


# ----------------------------------------------------------------------------------------------
# Several candidate surfaces, and a surface against itself
# ----------------------------------------------------------------------------------------------
def _plates_domain(size=0.14):
    """Geometry and tags only — no ``u.gap``, so a caller can exercise the registration itself."""
    d = jno.Shape.regions(
        left=jno.Shape.rect(-1.5, 0.0, -0.1, 1.0).sized(size),
        right=jno.Shape.rect(0.1, 0.0, 1.5, 0.8).sized(size),
        blk=jno.Shape.rect(-0.6, 1.1, 0.6, 1.5).sized(size),
        conforming=False,
    ).domain()
    _ = d.built_mesh
    e = 1e-6
    d.tag("s_blk", lambda x, y: y < 1.1 + e, region="blk")
    d.tag("s_left", lambda x, y: y > 1.0 - e, region="left")
    d.tag("s_right", lambda x, y: y > 0.8 - e, region="right")
    return d


def _plates_and_block(main, size=0.14):
    """A block over two plates at DIFFERENT heights, so which one a point pairs with is visible in g0.

    Left plate top at y = 1.0, right at y = 0.8, block bottom at y = 1.1 — so a point over the left
    reads 0.1 and one over the right reads 0.3, and a pairing that quietly picked a single surface
    would report one of those everywhere.
    """
    d = _plates_domain(size)
    u, v = d.fem_symbols(value_shape=(2,))
    Rg = {k: d.variable(k, split=True) for k in ("left", "right", "blk")}
    terms = []
    for k in Rg:
        ui, vi = u.bind(x=Rg[k][0], y=Rg[k][1]), v.bind(x=Rg[k][0], y=Rg[k][1])
        eu, ev = jno.np.symgrad(ui, [Rg[k][0], Rg[k][1]]), jno.np.symgrad(vi, [Rg[k][0], Rg[k][1]])
        terms.append(jno.np.inner(eu, ev, n_contract=2))
    sb, nb = d.variable("s_blk", split=True), d.variable("s_blk", normals=True)
    g = u.gap("s_blk", main, domain=d)
    terms.append(jno.np.maximum(0.0, -1.0e3 * g) * jno.np.inner(nb, v.bind(x=sb[0], y=sb[1]), n_contract=1))
    terms += [u(Rg["left"][0], Rg["left"][1]) - 0.0, u(Rg["right"][0], Rg["right"][1]) - 0.0]
    return d, jno.fem(terms)


def _g0(fem):
    op = fem._op
    tb = op.repair_contact(np.zeros(int(op.size)))
    return np.concatenate([np.asarray(t["g0"]).reshape(-1) for t in tb.values()])


def test_a_candidate_list_pairs_each_point_with_the_nearest_of_them():
    """Every point takes one candidate or the other — and BOTH are used. A pairing that silently
    collapsed to one main surface would return that surface's answer everywhere."""
    gl, gr = _g0(_plates_and_block("s_left")[1]), _g0(_plates_and_block("s_right")[1])
    gm = _g0(_plates_and_block(["s_left", "s_right"])[1])
    assert gl.shape == gr.shape == gm.shape

    picked_l, picked_r = np.isclose(gm, gl, atol=1e-9), np.isclose(gm, gr, atol=1e-9)
    assert (picked_l | picked_r).all(), (
        f"{int((~(picked_l | picked_r)).sum())} point(s) took neither candidate's value"
    )
    assert picked_l.any() and picked_r.any(), (
        f"the list collapsed to one surface: {int(picked_l.sum())} left, {int(picked_r.sum())} right"
    )
    assert abs(gm.min() - 0.1) < 0.02, f"the closest approach is the left plate at 0.1, got {gm.min():.4f}"
    assert abs(gm.max() - 0.3) < 0.05, f"the furthest is the right plate at 0.3, got {gm.max():.4f}"


def test_a_candidate_list_is_refused_without_the_search():
    """A frozen pairing is built once, against ONE surface. With candidates there is no single answer
    to freeze, and choosing one silently is the failure this whole mechanism exists to remove."""
    _, fem = _plates_and_block(["s_left", "s_right"])
    with pytest.raises(ValueError, match="names several candidate surfaces"):
        fem.solve()


def test_self_contact_must_be_asked_for_as_a_list():
    d = _plates_domain()
    u, _v = d.fem_symbols(value_shape=(2,))
    with pytest.raises(ValueError, match="must be different regions"):
        u.gap("s_blk", "s_blk", domain=d)
    assert u.gap("s_blk", ["s_blk"], domain=d) is not None


def _slotted_block(size=0.075):
    """A C: a 2x1 bar with a 0.2-wide slot cut in from the right. The two slot faces look at each other
    across 0.2 and are the only part of the boundary that does."""
    body = jno.Shape.rect(0, 0, 2.0, 1.0) - jno.Shape.rect(0.6, 0.4, 2.1, 0.6)
    d = body.sized(size).domain()
    _ = d.built_mesh
    d.tag("surf", lambda x, y: x**2 >= -1.0)  # one body: its boundary IS the whole boundary
    d.tag("clamp", lambda x, y: x < 1e-6)

    u, v = d.fem_symbols(value_shape=(2,))
    r = d.variable("interior", split=True)
    ui, vi = u.bind(x=r[0], y=r[1]), v.bind(x=r[0], y=r[1])
    eu, ev = jno.np.symgrad(ui, [r[0], r[1]]), jno.np.symgrad(vi, [r[0], r[1]])
    sb, nb = d.variable("surf", split=True), d.variable("surf", normals=True)
    cl = d.variable("clamp", split=True)
    g = u.gap("surf", ["surf"], domain=d)
    return d, jno.fem([
        jno.np.inner(eu, ev, n_contract=2),
        jno.np.maximum(0.0, -1.0e3 * g) * jno.np.inner(nb, v.bind(x=sb[0], y=sb[1]), n_contract=1),
        u(cl[0], cl[1]) - 0.0,
    ])


def test_a_surface_searching_against_itself_finds_the_slot_and_not_its_own_neighbours():
    """Self-contact's whole difficulty is not pairing a facet with the surface it is part of.

    Excluding the adjacency ring is not sufficient: two facets a few rings apart on a FLAT stretch are
    collinear, so ``Phi(x) - x`` lies along the surface, ``g0`` comes out ~0, and the bar reads as
    touching itself everywhere. What separates them from a genuine fold is the normal — parallel, not
    opposed — so the pairing keeps only facets that face the query. The measurable consequence is that
    the smallest separation found is the slot's 0.2, not 0.
    """
    _, fem = _slotted_block()
    g = _g0(fem)
    live = g[np.abs(g) < 1e29]
    assert live.size > 10, f"only {live.size} live slots — the filter cannot be rejecting everything"
    assert live.min() > 0.15, (
        f"the closest the bar comes to itself is the 0.2 slot, but the search found {live.min():.4f} — "
        "that is a facet pairing with its own flat continuation"
    )
    assert abs(live.min() - 0.2) < 0.03, f"expected the slot width 0.2, got {live.min():.4f}"


def test_a_plain_solve_after_a_searched_one_is_not_the_searched_answer():
    """``contact=`` swaps the operator's residual for the duration of the loop, and the operator caches
    a ``jax.jit`` of a closure that reads that residual at TRACE time. Without dropping that cache an
    ordinary ``fem.solve()`` afterwards would silently return the searched answer — a wrong result with
    no symptom, which is the worst kind."""
    _, fem = _stacked_bars()
    plain_first = np.asarray(fem.solve()).reshape(-1)
    fem.solve(contact=jno.solve.contact())
    plain_after = np.asarray(fem.solve()).reshape(-1)
    assert np.allclose(plain_first, plain_after, rtol=1e-10, atol=1e-12), (
        f"a plain solve changed by {np.abs(plain_first - plain_after).max():.3e} after a searched one"
    )


def test_a_candidate_list_actually_solves_with_the_search():
    """The refusal above must not also fire on the driver's OWN rounds — it re-enters the dispatch with
    ``contact=None`` by design, so an unguarded check would refuse the very solve it was asked to run."""
    _, fem = _plates_and_block(["s_left", "s_right"])
    u = np.asarray(fem.solve(contact=jno.solve.contact(capture=0.5))).reshape(-1)
    assert np.isfinite(u).all()
    assert 2 <= int(fem.contact_rounds) <= 8


# ----------------------------------------------------------------------------------------------
# Hertz — the physics oracle, and what its resolution costs
# ----------------------------------------------------------------------------------------------
E_H, NU_H, R_H, H_H, CN_H = 1.0e3, 0.30, 1.0, 0.030, 2.0e5
ESTAR = E_H / (2.0 * (1 - NU_H**2))  # 1/E* = 2(1-v^2)/E for two identical bodies


def _hertz(press):
    """A cylinder pressed onto a flat, plane strain, frictionless. Returns ``(P, a)``.

    Both are measured, neither is read out of the form: ``P`` is the reaction on the driven top from
    the free internal force ``A u``, and ``a`` is the half-width of the contact patch taken from the
    DEFORMED surface geometry.
    """
    from jno.utils.solver.contact_search import project_points
    from jno.utils.solver.fem_utils import _cell_region_mask
    import jax.numpy as jnp

    lam, mu = E_H * NU_H / ((1 + NU_H) * (1 - 2 * NU_H)), E_H / (2 * (1 + NU_H))
    gapy = 0.004
    d = jno.Shape.regions(cyl=jno.Shape.disk(0.0, R_H + gapy, R_H).sized(H_H),
                          blk=jno.Shape.rect(-1.6, -1.2, 1.6, 0.0).sized(H_H), conforming=False).domain()
    _ = d.built_mesh
    d.tag("s_cyl", lambda x, y: y < R_H + gapy, region="cyl")
    d.tag("s_blk", lambda x, y: y > -1e-9, region="blk")
    d.tag("top", lambda x, y: y > 2 * R_H + gapy - 0.05, region="cyl")
    d.tag("bot", lambda x, y: y < -1.2 + 1e-9, region="blk")

    u, v = d.fem_symbols(value_shape=(2,))
    Rg = {k: d.variable(k, split=True) for k in ("cyl", "blk")}
    terms = []
    for k in Rg:
        ui, vi = u.bind(x=Rg[k][0], y=Rg[k][1]), v.bind(x=Rg[k][0], y=Rg[k][1])
        eu, ev = jno.np.symgrad(ui, [Rg[k][0], Rg[k][1]]), jno.np.symgrad(vi, [Rg[k][0], Rg[k][1]])
        terms.append(2 * mu * jno.np.inner(eu, ev, 2) + lam * jno.np.trace(eu) * jno.np.trace(ev))
    sc, nc = d.variable("s_cyl", split=True), d.variable("s_cyl", normals=True)
    g = u.gap("s_cyl", "s_blk", domain=d)
    terms.append(jno.np.maximum(0.0, -CN_H * g) * jno.np.inner(nc, v.bind(x=sc[0], y=sc[1]), 1))
    tp, bt = d.variable("top", split=True), d.variable("bot", split=True)
    terms += [u(tp[0], tp[1])[0] - 0.0, u(tp[0], tp[1])[1] - (-press), u(bt[0], bt[1]) - 0.0]

    fem = jno.fem(terms)
    uu = np.asarray(fem.solve(contact=jno.solve.contact(capture=4 * H_H))).reshape(-1, 2)

    saved = d.__dict__.pop("_contact_pairs", None)   # the free form drops the gap; the check is right
    try:
        A, b = jno.fem(terms[:2])._op
    finally:
        if saved is not None:
            d.__dict__["_contact_pairs"] = saved
    f = np.asarray(A @ jnp.asarray(uu.reshape(-1)) - b).reshape(-1, 2)
    pts = np.asarray(d.built_mesh.points)[:, :2]
    tri = np.asarray(d.built_mesh.cells_dict["triangle"])
    P = abs(float(f[np.asarray(d.tag_indices["top"]).reshape(-1), 1].sum()))

    m = np.asarray(_cell_region_mask(d, "blk")).reshape(-1) > 0
    e = np.sort(tri[m][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uq, cnt = np.unique(e, axis=0, return_counts=True)
    fB = uq[cnt == 1]
    mc = np.asarray(_cell_region_mask(d, "cyl")).reshape(-1) > 0
    ec = np.sort(tri[mc][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uqc, cntc = np.unique(ec, axis=0, return_counts=True)
    nC = np.unique(uqc[cntc == 1])
    Pd = pts + uu
    low = nC[Pd[nC, 1] < R_H + gapy - press]
    nrm = np.tile(np.array([0.0, -1.0]), (len(low), 1))
    _i, _w, gg, act = project_points(Pd[low], fB, Pd, nrm, capture=4 * H_H)
    touch = low[act & (gg < 0)]
    return P, (float(np.abs(pts[touch, 0]).max()) if len(touch) else 0.0)


@pytest.mark.slow  # two full re-pairing solves on a few-thousand-node mesh: ~120 s
def test_hertz_half_width_scales_as_the_square_root_of_the_load():
    """``a = sqrt(4 P R / (pi E*))`` for a cylinder on a flat (Johnson, *Contact Mechanics* §4.2).

    Two things are worth separating. The ABSOLUTE half-width is resolution-limited here — the patch is
    only 2.5 to 4.5 elements across, so ``a`` is quantised to the surface node spacing, and it comes
    out 15.4% and 11.6% under Hertz, the error shrinking as the patch resolves. Refining that is a
    meshing problem (`.sized()` cannot yet grade a contact patch), not a contact one.

    The SCALING is far better conditioned, because the same +-h/2 quantisation sits in both
    measurements and largely cancels: ``a2/a1`` must be ``sqrt(P2/P1)``. Measured 1.7965 against
    1.7184. That is the statement that the pressure really is distributed as Hertz says, and it is the
    assertion worth making.
    """
    P1, a1 = _hertz(0.020)
    P2, a2 = _hertz(0.045)
    assert a1 > 2 * H_H and a2 > a1, f"the patch must resolve at all: a1={a1:.4f}, a2={a2:.4f}"
    for P, a in ((P1, a1), (P2, a2)):
        a_h = np.sqrt(4 * P * R_H / (np.pi * ESTAR))
        assert 0.75 * a_h < a < a_h, (
            f"a={a:.4f} against Hertz {a_h:.4f}; a discrete patch UNDER-reports, but not by this much"
        )
    assert abs((a2 / a1) / np.sqrt(P2 / P1) - 1.0) < 0.12, (
        f"a should scale as sqrt(P): measured ratio {a2/a1:.4f}, Hertz {np.sqrt(P2/P1):.4f}"
    )


# ----------------------------------------------------------------------------------------------
# The same statement in 3-D
# ----------------------------------------------------------------------------------------------
R_BALL, Z0_3D, SLIDE_3D, H_3D = 1.0, 1.06, 0.75, 0.20


def test_the_search_follows_a_large_slide_in_three_dimensions():
    """The 2-D sliding check above pins edges against a circle; this pins TRIANGLES against a sphere.

    The closed form is the same with ``x^2 -> x^2 + y^2``: the block's bottom is a horizontal plane at
    ``z = Z0`` with outward normal ``(0, 0, -1)``, the main surface is a sphere of radius ``R``, so the
    closest point is ``R (x, y, Z0) / |(x, y, Z0)|`` and

        g0 = Z0 - R Z0 / sqrt(x^2 + y^2 + Z0^2)

    A purely horizontal rigid slide keeps ``n . D`` at zero, so the re-paired ``g0`` is the deformed
    gap outright. Worth having separately from the 2-D case because the 3-D path is different code --
    :func:`~.contact_search.closest_point_on_triangle` and ``_tri_shape`` rather than the segment and
    edge routines -- and "the implementation looks dimension-generic" is not evidence.
    """
    from jno.utils.solver.contact_search import OPEN_GAP
    from jno.utils.solver.fem_utils import _cell_region_mask

    blk = jno.Shape.box(-0.35, -0.35, Z0_3D, 0.35, 0.35, Z0_3D + 0.4)
    d = jno.Shape.regions(ball=jno.Shape.sphere(0, 0, 0, R_BALL).sized(H_3D),
                          blk=blk.sized(H_3D), conforming=False).domain()
    _ = d.built_mesh
    e = 1e-6
    d.tag("s_blk", lambda x, y, z: z < Z0_3D + e, region="blk")
    d.tag("s_ball", lambda x, y, z: z > -R_BALL + e, region="ball")

    u, v = d.fem_symbols(value_shape=(3,))
    Rg = {k: d.variable(k, split=True) for k in ("ball", "blk")}
    terms = []
    for k in Rg:
        ui, vi = u.bind(x=Rg[k][0], y=Rg[k][1], z=Rg[k][2]), v.bind(x=Rg[k][0], y=Rg[k][1], z=Rg[k][2])
        eu = jno.np.symgrad(ui, [Rg[k][0], Rg[k][1], Rg[k][2]])
        ev = jno.np.symgrad(vi, [Rg[k][0], Rg[k][1], Rg[k][2]])
        terms.append(jno.np.inner(eu, ev, n_contract=2))
    sb, nb = d.variable("s_blk", split=True), d.variable("s_blk", normals=True)
    g = u.gap("s_blk", "s_ball", domain=d)
    terms.append(jno.np.maximum(0.0, -1.0e3 * g) * jno.np.inner(nb, v.bind(x=sb[0], y=sb[1], z=sb[2]), 1))
    terms += [u(Rg["ball"][0], Rg["ball"][1], Rg["ball"][2]) - 0.0]
    op = jno.fem(terms)._op

    tet = np.asarray(d.built_mesh.cells_dict["tetra"])
    blk_nodes = np.unique(tet[np.asarray(_cell_region_mask(d, "blk")).reshape(-1) > 0])
    uu = np.zeros((int(np.asarray(d.built_mesh.points).shape[0]), 3))
    uu[blk_nodes, 0] = SLIDE_3D

    def live(tb):
        gg = np.concatenate([np.asarray(t["g0"]).reshape(-1) for t in tb.values()])
        return gg[np.abs(gg) < 0.5 * OPEN_GAP]

    exact = lambda x, y: Z0_3D - R_BALL * Z0_3D / np.sqrt(x**2 + y**2 + Z0_3D**2)  # noqa: E731
    frozen = live(op.repair_contact(np.zeros(int(op.size)))).min()
    slid = live(op.repair_contact(uu.reshape(-1))).min()

    sag = H_3D**2 / (8 * R_BALL)
    assert abs(frozen - exact(0.0, 0.0)) < sag + 1e-3, (
        f"the reference gap at the crown should be {exact(0.0, 0.0):.4f}, got {frozen:.4f}"
    )
    lo, hi = exact(SLIDE_3D - 0.35, 0.0), exact(SLIDE_3D - 0.35 + H_3D, 0.0)
    assert lo - sag - 1e-3 <= slid <= hi + sag + 1e-3, (
        f"after sliding {SLIDE_3D} the gap should be in [{lo:.4f}, {hi:.4f}], got {slid:.4f}"
    )
    assert slid > 1.8 * frozen, (
        f"the frozen pairing reports {frozen:.4f} where the truth is ~{lo:.4f}; if the 3-D search were "
        "doing nothing these would agree"
    )
