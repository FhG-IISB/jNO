"""A node that is both TIE-ELIMINATED and DIRICHLET-PRESCRIBED loses its prescribed value, silently.

Diagnosed from a linear-field patch test across a non-matching interface, which should be exact and is
not. The chain of evidence, each step a test below:

1. the prolongation **is** linearly complete — its weights sum to 1 and it reproduces the nodal
   coordinates, ``P @ X_kept == X_full``, to ~1e-15. So the mortar rows are not the problem;
2. with a **conforming** tie the patch test is exact to machine precision, at prescribed and free nodes
   alike — so neither is the elimination itself;
3. with a **mortar** tie, prescribed nodes *off* the interface are exact (error 0.0) while prescribed
   nodes *on* it are wrong by ~7e-05 — and several of those are eliminated by the tie.

So where a tied face meets a constrained outer boundary, the Dirichlet condition is not imposed. Nothing
raises. Two distinct mechanisms are at work, and the second is the one that matters (see below): a
prescribed node may be *eliminated* by the tie, and a prescribed node that survives may still have its
row destroyed by ``P^T``.

jNO already refuses exactly this conflict for a slip condition — "a node on both a tied face and a slip
wall would be eliminated twice" (`_fem.py`, the `slip_bcs and periodic_ties` branch). The Dirichlet case
is the same geometry and is unguarded.

**Why the obvious fix is not enough** (tried, measured, reverted). Excluding prescribed nodes from the
elimination — the standard treatment — removes the overlap but makes the error *worse*, 7e-05 -> 1.3e-04,
because the damage is on the other side of the reduction. Measured: after excluding, the prescribed nodes
that are wrong are **exactly** the ones some eliminated node's weights point at (8 of 8; every prescribed
target wrong, no non-target wrong). The reduction is Galerkin, ``P^T A P``, and Dirichlet is applied
**before** it — so ``P^T`` sums an eliminated node's equation into its target's row and destroys the unit
row that was holding the prescribed value. Keeping the node in the system does not protect it.

The complete fix therefore has to restore Dirichlet rows *after* the reduction (or apply Dirichlet after
it), in each representation the reduction has: linear and nonlinear reduce lazily at solve time in
``FEM.solve``, transient eagerly in ``_reduce_transient_block_periodic``, complex per leg. That is a
change to the solve path for every tied/periodic problem, not a local one.

The cheap alternative remains **refusing the overlap loudly**, as the slip path already does.

Both defects are now fixed and every assertion here is live. The file is kept as the record of what was
wrong and how it was measured, because the measurements are the only thing that distinguished two bugs
that presented identically.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

A_LIN, C_LIN = (0.011, -0.004, 0.007), 0.002


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _exact(pts):
    return A_LIN[0] * pts[:, 0] + A_LIN[1] * pts[:, 1] + A_LIN[2] * pts[:, 2] + C_LIN


def _outer(x, y, z):
    return (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9) | (z < 1e-9) | (z > 2 - 1e-9)


def _geom(conforming, size=0.5, **dkw):
    """Two stacked unit blocks, meshed together (``conforming``) or independently (a tied interface)."""
    return (
        jno.Shape.regions(
            lower=jno.Shape.box(0, 0, 0, 1, 1, 1),
            upper=jno.Shape.box(0, 0, 1, 1, 1, 2),
            conforming=conforming,
        )
        .sized(size)
        .domain(**dkw)
    )


def _tie_term(d, u):
    """``u(secondary) - u(main)`` across the shared face — the tie the reduction is built from."""
    sec, main = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    sv, mv = d.variable(sec, split=True), d.variable(main, split=True)
    return u(sv[0], sv[1], sv[2]) - u(mv[0], mv[1], mv[2])


def _patch(conforming, size=0.5):
    """Laplace with a linear field prescribed on the whole outer boundary — exact by construction."""
    d = _geom(conforming, size)
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    ob = d.variable("ob", where=_outer, split=True)
    lin = A_LIN[0] * ob[0] + A_LIN[1] * ob[1] + A_LIN[2] * ob[2] + C_LIN
    terms = [ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(ob[0], ob[1], ob[2]) - lin]
    if not conforming:
        terms.insert(1, _tie_term(d, u))
    fem = jno.fem(terms)
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    pts = np.asarray(d.built_mesh.points)
    return fem, pts, sol, np.abs(sol - _exact(pts))


def _patch_complex(conforming, size=0.5):
    """The same patch test through the FUSED COMPLEX path: a complex coefficient makes the operator
    complex, so it solves as the real 2n block ``x = [x_r; x_i]``. ``(1+2j) grad u . grad phi = 0`` with a
    real linear field prescribed on the boundary still has that field as its exact solution, in both legs.
    """
    d = _geom(conforming, size)
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    ob = d.variable("ob", where=_outer, split=True)
    lin = A_LIN[0] * ob[0] + A_LIN[1] * ob[1] + A_LIN[2] * ob[2] + C_LIN
    terms = [(1.0 + 2.0j) * (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z), u(ob[0], ob[1], ob[2]) - lin]
    if not conforming:
        terms.insert(1, _tie_term(d, u))
    sol = np.asarray(jno.fem(terms).solve(linear=jno.solve.lu(backend="host")))
    return np.abs(sol - _exact(np.asarray(d.built_mesh.points))).max()


def _patch_transient(conforming, size=0.5):
    """The same patch test through the TRANSIENT path. ``u_t = lap u`` started AT the linear field holds
    it for all time (the Laplacian vanishes and the boundary never moves), so every saved step must equal
    it exactly -- an oracle that needs no time-discretisation error budget."""
    d = _geom(conforming, size, time=(0.0, 0.2, 5))
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    ui = u.bind(x=co[0], y=co[1], z=co[2], t=co[3])
    vi = phi.bind(x=co[0], y=co[1], z=co[2], t=co[3])
    ob = d.variable("ob", where=_outer, split=True)
    ci = d.variable("initial", split=True)
    lin = A_LIN[0] * ob[0] + A_LIN[1] * ob[1] + A_LIN[2] * ob[2] + C_LIN
    ic = A_LIN[0] * ci[0] + A_LIN[1] * ci[1] + A_LIN[2] * ci[2] + C_LIN
    terms = [
        ui.t * vi + ui.x * vi.x + ui.y * vi.y + ui.z * vi.z,
        u(ob[0], ob[1], ob[2]) - lin,
        u(ci[0], ci[1], ci[2]) - ic,
    ]
    if not conforming:
        terms.insert(1, _tie_term(d, u))
    w = jno.fem(terms).solve()
    w = np.asarray(w.fn() if hasattr(w, "fn") and not w.args else w)
    return np.abs(w - _exact(np.asarray(d.built_mesh.points))[None, :]).max()


# ----------------------------------------------------------------------------------------------
# 1. the prolongation is NOT at fault
# ----------------------------------------------------------------------------------------------


def test_the_mortar_prolongation_is_linearly_complete():
    """Weights sum to one (constants reproduced) and the coordinates are reproduced exactly (linears).
    This is what clears the mortar rows of blame for the patch-test failure below."""
    fem, pts, _sol, _err = _patch(conforming=False)
    per = fem._periodic
    P = per["P_node"]
    P = np.asarray(P.todense()) if hasattr(P, "todense") else np.asarray(P)
    kept = np.asarray(per["kept_nodes"])

    assert np.abs(P.sum(axis=1) - 1.0).max() < 1e-12, "a partition of unity reproduces constants"
    assert np.abs(P @ pts[kept] - pts).max() < 1e-12, "a linearly complete map reproduces coordinates"


def test_a_conforming_tie_passes_the_patch_test_exactly():
    """The control: same geometry, matching meshes. Exact at prescribed and free nodes alike, so the
    elimination machinery is sound when no node is doubly constrained."""
    _fem, _pts, _sol, err = _patch(conforming=True)
    assert err.max() < 1e-12


# ----------------------------------------------------------------------------------------------
# 2. the defect
# ----------------------------------------------------------------------------------------------


def test_no_prescribed_dof_is_eliminated_by_the_tie():
    """The invariant the fix rests on. A prescribed DOF must survive as a KEPT DOF, because that is the
    only way it has a row of its own for the Dirichlet condition to be imposed into after the reduction
    — an eliminated one would need the multipoint constraint `sum w_j x_j = g`, which is not a unit row.

    Before the fix this geometry had 8 prescribed-and-eliminated DOFs, and they silently lost their
    values. The exclusion is decided per DOF rather than per node, so a roller (one component prescribed
    on a tied node) keeps the tie on its free components.
    """
    fem, pts, _sol, _err = _patch(conforming=False)
    kept = np.zeros(len(pts) * max(1, int(fem._periodic.get("vec", 1))), bool)
    kept[np.asarray(fem._periodic["kept_nodes"])] = True
    prescribed = [int(d) for d, _g in (getattr(fem.domain, "_fem_native_dirichlet_pairs", None) or [])]
    assert prescribed, "sanity: this problem has essential conditions"
    assert all(kept[d] for d in prescribed), "a prescribed DOF was eliminated by the tie"


def test_prescribed_nodes_away_from_the_interface_are_still_exact():
    """Scoping the damage: ordinary Dirichlet is untouched. Only nodes caught by the elimination lose
    their value, which is why the symptom looks like a mild accuracy problem rather than a broken BC."""
    _fem, pts, _sol, err = _patch(conforming=False)
    on_iface = np.abs(pts[:, 2] - 1.0) < 1e-9
    prescribed = _outer(pts[:, 0], pts[:, 1], pts[:, 2])
    assert err[prescribed & ~on_iface].max() < 1e-12


def test_every_prescribed_node_holds_its_prescribed_value():
    """A Dirichlet condition is not an approximation — a prescribed node must hold its value exactly,
    whatever else the system does. This FAILED before the fix (~7e-05, entirely on the interface
    perimeter) and is now exact, because Dirichlet is imposed after the tie reduction rather than
    destroyed by it."""
    _fem, pts, _sol, err = _patch(conforming=False)
    mask = _outer(pts[:, 0], pts[:, 1], pts[:, 2])
    assert err[mask].max() < 1e-12, f"prescribed nodes deviate by {err[mask].max():.3e}"


def test_the_mortar_tie_reproduces_a_linear_field():
    """The classical mortar patch test. Exact now: 6.2e-05 -> 1.0e-17.

    It took both halves. One formula per interface fixed the normal mode; the boundary-modified
    multiplier space fixed the tangential ones."""
    _fem, _pts, _sol, err = _patch(conforming=False)
    assert err.max() < 1e-12, f"linear field not reproduced: {err.max():.3e}"


# ----------------------------------------------------------------------------------------------
# 3. the SECOND defect, isolated: the mortar reduction is not variationally consistent
# ----------------------------------------------------------------------------------------------


def _free_residual_after_restriction():
    """``max |P^T (A u* - b)|`` over the FREE reduced rows, at the exact solution.

    A Galerkin reduction is consistent when the exact solution satisfies the reduced equations, i.e.
    when the restriction annihilates the free system's residual on every free row. Prescribed rows are
    excluded because they legitimately carry the constraint's reaction.
    """
    fem, pts, _sol, _err = _patch(conforming=False)
    per = fem._periodic
    A = np.asarray(fem.A.todense() if hasattr(fem.A, "todense") else fem.A)
    b = np.asarray(fem.b)
    P = per["P"]
    P = np.asarray(P.todense() if hasattr(P, "todense") else P)
    pr = P.T @ (A @ _exact(pts) - b)
    kept = np.asarray(per["kept_nodes"])
    prescribed = _outer(pts[:, 0], pts[:, 1], pts[:, 2])
    free = [i for i, k in enumerate(kept) if not prescribed[k]]
    return float(np.abs(pr[free]).max())


def test_the_tie_restriction_does_not_annihilate_the_free_residual():
    """The second defect, stated at its source rather than through the patch test it breaks.

    `P` is linearly complete — it reproduces the exact field to ~1e-17 (asserted above) — so the exact
    solution IS in the constrained space. For the Galerkin solve to return it, the exact solution must
    also satisfy the reduced equations, which means `P^T (A u* - b)` must vanish on every free row. It
    does not: measured ~8.9e-05, entirely on interface rows, with **no Dirichlet involved**.

    So a linearly complete prolongation is necessary but not sufficient: eliminating with dual-mortar
    weights makes the prolonged basis continuous only in the weak (integral) sense, and the resulting
    method carries a consistency error. That is a property of the mortar weights against the assembled
    operator, and it is what remains of the patch-test error now that Dirichlet is imposed correctly.
    The control is `test_a_conforming_tie_passes_the_patch_test_exactly` above: with matching meshes the
    same elimination machinery is exact, so the inconsistency is in the mortar weights, not the machinery.
    """
    assert _free_residual_after_restriction() < 1e-10


def test_one_interface_uses_exactly_one_formula():
    """The interface is classified ONCE and every secondary is tied the same way.

    Choosing per node was the bug: on a stacked-block interface the two bodies share their outer edges,
    so most secondaries coincided with a main node and took the weight-1 shortcut while the rest were
    integrated — measured 8 exact against 4 mortar. A weight-1 row inside a dual-mortar operator
    destroys the biorthogonality the method's consistency rests on.

    The exact path is not removed, and must not be: it is the only tie mechanism for a 1-D interface,
    for Morley's value block, and for quad/hex and 3-D P2 facets, none of which have a dual basis of
    this form. What is removed is the possibility of two formulas meeting inside one interface.
    """
    fem, _pts, _sol, _err = _patch(conforming=False)
    counts = fem._periodic["tie_counts"]
    assert len([n for n in counts if n]) == 1, f"one interface, one formula — got {counts}"
    assert counts[1] > 0, "a non-matching interface with facets on both sides should integrate"


def _mode_defect():
    """``max |P^T A u|`` at INTERIOR interface rows for each linear mode, with no Dirichlet anywhere.

    The sharpest instrument available on this defect, and the one that localised it. For a consistent
    reduction the transferred traction of any linear field must vanish on rows that are interior to the
    interface — there is no boundary term there to balance it. Returns ``{mode: value}``.

    KNOWN BLIND SPOT, stated so nobody trusts this alone. It measures INTERIOR interface rows only, and
    on this geometry the interface rim lies on the outer boundary, where the transferred traction is
    legitimately non-zero (there is a real surface flux there) and cannot be asserted to vanish. A fix
    that merely DROPS the rim multipliers without redistributing their weight is O(1) wrong on the
    rim-adjacent layer and would pass this probe cleanly. The gate that catches it is the full patch
    test, `test_the_mortar_tie_reproduces_a_linear_field`, which sums over every free row. Treat the two
    as a pair: this one localises, that one falsifies.
    """
    d = (
        jno.Shape.regions(lower=jno.Shape.box(0, 0, 0, 1, 1, 1), upper=jno.Shape.box(0, 0, 1, 1, 1, 2), conforming=False)
        .sized(0.5)
        .domain()
    )
    sec, main = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    sv, mv = d.variable(sec, split=True), d.variable(main, split=True)
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(sv[0], sv[1], sv[2]) - u(mv[0], mv[1], mv[2])])
    per = fem._periodic
    A = np.asarray(fem.A.todense() if hasattr(fem.A, "todense") else fem.A)
    P = per["P"]
    P = np.asarray(P.todense() if hasattr(P, "todense") else P)
    pts = np.asarray(d.built_mesh.points)
    kept = np.asarray(per["kept_nodes"])
    onout = (
        (pts[:, 0] < 1e-9)
        | (pts[:, 0] > 1 - 1e-9)
        | (pts[:, 1] < 1e-9)
        | (pts[:, 1] > 1 - 1e-9)
        | (pts[:, 2] < 1e-9)
        | (pts[:, 2] > 2 - 1e-9)
    )
    iface = np.abs(pts[:, 2] - 1.0) < 1e-9
    rows = [i for i, k in enumerate(kept) if iface[k] and not onout[k]]
    modes = {"1": np.ones(len(pts)), "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
    return {n: float(np.abs((P.T @ (A @ m))[rows]).max()) for n, m in modes.items()}


def test_the_constant_and_normal_modes_transfer_exactly():
    """Two of the four linear modes are already exact, and must stay that way.

    The normal mode `z` was 1.27e-02 while the interface mixed formulas and is 3.6e-16 now that it does
    not — this is the assertion that pins what one-formula-per-interface bought.
    """
    dfct = _mode_defect()
    assert dfct["1"] < 1e-12, f"constants must transfer exactly: {dfct}"
    assert dfct["z"] < 1e-12, f"the normal mode must transfer exactly: {dfct}"


def test_the_tangential_modes_transfer_exactly():
    """The remaining half of the defect, stated at its source.

    Multipliers belonging to nodes on the RIM of the interface — where the tied face runs into the rest
    of the boundary — are enforcing an averaged continuity condition at a place where their support
    leaves the interface. The consequence is that outer-boundary flux is dragged into interior interface
    equations, which the tangential modes see and the normal one does not. Measured: `x` at 1.07e-02,
    `y` at 8.4e-03, against `1` and `z` at round-off.

    Fixed by Wohlmuth's boundary-modified multiplier space (SIAM J. Numer. Anal. 38(3):989-1012, 2000,
    §3; general order in Lamichhane's thesis §2.3 Thm 2.8): the rim multipliers are redistributed onto
    their interior neighbours and the rim nodes are left untied, so the space still reproduces constants
    and linears. Measured after: `x` 1.07e-02 -> 2.0e-16, `y` 8.4e-03 -> 1.1e-16.
    """
    dfct = _mode_defect()
    assert dfct["x"] < 1e-12 and dfct["y"] < 1e-12, f"tangential modes must transfer exactly: {dfct}"


# ----------------------------------------------------------------------------------------------
# 5. the same repair on every reduced-space path
#
# The steady real path was fixed first, and the others stayed wrong for as long as they each had their
# own copy of the reduction — or, for the transient, their own second construction site that never
# annotated the dict at all. Both are measured here against the conforming control rather than against
# a threshold, so a regression shows up as a mode that no longer matches its own control.
# ----------------------------------------------------------------------------------------------


def test_the_fused_complex_path_imposes_dirichlet_after_the_tie_reduction():
    """A complex operator solves as the real 2n block, whose reduction is blkdiag(P, P). That transform
    used to DROP `dirichlet_reduced`, so the restoration silently became a no-op: measured 5.8e-04 here,
    against 7e-18 for the same problem with a conforming interface."""
    assert _patch_complex(conforming=True) < 1e-12
    assert _patch_complex(conforming=False) < 1e-12


def test_the_transient_path_imposes_dirichlet_after_the_tie_reduction():
    """The transient march reduces its block eagerly, through a construction site that built the
    prolongation itself and skipped both the elimination exclusion and the annotation. Measured
    4.2e-04, against exactly 0.0 conforming — the field is held for all time, so there is no
    time-discretisation error to hide behind."""
    assert _patch_transient(conforming=True) < 1e-12
    assert _patch_transient(conforming=False) < 1e-12
