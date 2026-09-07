"""An involute gear pair, held to its closed-form torque oracle at every drive angle.

For involute teeth the contact force lies along the line of action, which is common to both gears, so
the moment arm about each centre is that gear's *base* radius and

    |T_B / T_A|  =  r_bB / r_bA  =  z_B / z_A        exactly, at every drive angle

That is a rare thing in contact mechanics: an oracle that does not care about the mesh, the material,
or the load. It is also unforgiving -- it measures the *direction* of the transmitted force, so any
pairing that puts load somewhere the involute geometry does not predict shows up immediately, while
the animation still looks perfect.

**What this caught, and it was not the solver.** The ratio came back up to 61% wrong at particular
drive angles and got *worse* under refinement (7.4 -> 5.7 -> 16.4 -> 7.4% as h went 0.030 -> 0.011,
penetration growing). Both causes were in the gear itself:

* at the textbook 20 deg pressure angle the undercut limit ``z_min = 2/sin^2(phi)`` is 17.1 teeth, so
  a 12-tooth pinion **has no involute** near its base circle -- there is nothing there for the oracle
  to describe. At 25 deg the limit falls to 11.2 and 12 teeth clears it;
* the contact surface included the root arc and the flat tip arc. Their normals are ~70 deg and 90 deg
  off the line of action, so a facet of either carrying load transmits force in a direction
  ``|T_B/T_A| = z_B/z_A`` simply does not describe. It bites hardest at the hand-off angles, where the
  trailing pair disengages *at the tip*.

The first test below is the cheap one that would have found all of this in milliseconds, and it is the
reason it is here: the classical admissibility limits are closed-form, and checking a benchmark
against them costs nothing next to debugging a contact solver that was never at fault.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

# ---- the gear pair -------------------------------------------------------------------------------
# 25 deg, not the textbook 20 -- see the module docstring. Tooth counts stay 12:20, so the oracle
# z_B/z_A = 5/3 is unchanged: this fixes the geometry the oracle assumes, not the oracle.
M, PHI = 0.10, np.radians(25.0)  # module, pressure angle
ZA, ZB = 12, 20  # tooth counts -> ratio 5/3
BACKLASH, ADDENDUM = 0.004, 1.00
EXACT = ZB / ZA

E_Y, NU = 2.0e5, 0.30
LAM, MU = E_Y * NU / ((1 + NU) * (1 - 2 * NU)), E_Y / (2 * (1 + NU))
RHA, RHB, CN = 0.36, 0.60, 4.0e6  # hub radii; contact penalty
R_LO, R_HI = 1.00, 0.97  # the tagged flank, as fractions of r_base and r_tip


def radii(z):
    """pitch, base, tip, root."""
    rp = M * z / 2.0
    return rp, rp * np.cos(PHI), rp + ADDENDUM * M, rp - 1.25 * M


RPA, RBA, RAA, RFA = radii(ZA)
RPB, RBB, RAB, RFB = radii(ZB)
CENTRE = RPA + RPB + BACKLASH


def _inv(a):
    return np.tan(a) - a


def _profile(z, centre=(0.0, 0.0), spin=0.0, nf=12, nt=5, nr=5):
    """One gear's closed outline: root arc, involute flank, tip arc, involute flank, root arc."""
    _rp, rb, ra, rf = radii(z)
    psi, half = np.pi / (2 * z), np.pi / z
    rr = np.linspace(max(rf, rb), ra, nf)
    fl = _inv(np.arccos(np.clip(rb / np.maximum(rr, rb + 1e-12), -1.0, 1.0))) - _inv(PHI)
    aL, aR = -psi + fl, psi - fl[::-1]
    seg = [
        np.stack([np.linspace(-half, aL[0], nr, endpoint=False), np.full(nr, rf)], 1),
        np.stack([aL, rr], 1),
        np.stack([np.linspace(aL[-1], aR[0], nt + 2)[1:-1], np.full(nt, ra)], 1),
        np.stack([aR, rr[::-1]], 1),
        np.stack([np.linspace(aR[-1], half, nr, endpoint=False)[1:], np.full(nr - 1, rf)], 1),
    ]
    t = np.vstack(seg)
    pts = [
        np.stack(
            [t[:, 1] * np.cos(t[:, 0] + 2 * np.pi * k / z + spin), t[:, 1] * np.sin(t[:, 0] + 2 * np.pi * k / z + spin)], 1
        )
        for k in range(z)
    ]
    return np.vstack(pts) + np.asarray(centre)


def _pair(theta=0.0):
    """A driven to `theta`; B turns the other way, slower by z_A/z_B, phased so a tooth GAP faces A
    at theta = 0 -- otherwise the two outlines overlap and the booleans cannot produce a mesh."""
    return (_profile(ZA, (0.0, 0.0), spin=theta), _profile(ZB, (CENTRE, 0.0), spin=np.pi - np.pi / ZB - theta * ZA / ZB))


# ---- the fast test: is this gear pair even cuttable? ---------------------------------------------
def test_the_demo_gear_satisfies_the_classical_admissibility_limits():
    """The oracle assumes an involute flank carrying the load. These are the two closed-form limits
    that decide whether such a flank exists -- both were *violated* by the 20 deg version, and that,
    not the contact solver, is what made the ratio wrong.

    * **Undercut.** Generating a gear with fewer than ``z_min = 2/sin^2(phi)`` teeth cuts away the
      profile near the base circle, leaving no involute there (Buckingham, *Analytical Mechanics of
      Gears*, 1949, ch. 4).
    * **Interference.** A tip reaching past the other gear's base tangent point digs into a region
      where its mate has no involute at all; the limit is ``r_a,max = sqrt(r_b^2 + (C sin phi)^2)``.
    """
    z_min = 2.0 / np.sin(PHI) ** 2
    assert ZA >= z_min, (
        f"a {ZA}-tooth pinion undercuts at phi = {np.degrees(PHI):.0f} deg "
        f"(needs >= {z_min:.1f} teeth): there is no involute near its base circle"
    )

    for name, r_b, r_a in (("A", RBA, RAA), ("B", RBB, RAB)):
        r_max = np.sqrt(r_b**2 + (CENTRE * np.sin(PHI)) ** 2)
        assert r_a <= r_max, (
            f"gear {name}'s tip r_a = {r_a:.4f} passes the interference limit "
            f"{r_max:.4f} -- it reaches into its mate's non-involute root"
        )

    # and the tagged contact surface must lie strictly inside the flank, r in (r_base, r_tip): both
    # ends of that OPEN interval are other surfaces (root arc below, tip arc above) whose normals do
    # not lie on the line of action.
    for name, r_b, r_a in (("A", RBA, RAA), ("B", RBB, RAB)):
        assert r_b <= R_LO * r_b < R_HI * r_a <= r_a, f"gear {name}: the tagged band is not inside the flank"


def test_the_pair_meshes_without_the_outlines_overlapping():
    """A cheap guard on the phasing: if B is mis-phased the two outlines intersect and every solve
    below fails in the mesher instead of in the physics, which is a confusing way to learn it."""
    from scipy.spatial import cKDTree

    for theta in (0.0, 0.1745, 0.3491):
        a, b = _pair(theta)
        gap = float(cKDTree(a).query(b)[0].min())
        assert gap > 0.0, f"theta = {theta}: the outlines touch or overlap (min flank gap {gap:.2e})"


# ---- the slow test: the solver, against the oracle ------------------------------------------------
@pytest.fixture
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _solve(theta, drive=9.0e-3, h_rim=0.022, h_hub=0.060, rounds=6, capture=0.06):
    """Two gears coupled ONLY through contact, then the two hub reaction torques read back out."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_utils import _cell_region_mask

    n = jno.np
    sym, tr, inner = n.symgrad, n.trace, n.inner

    a, b = _pair(theta)
    hub_a, hub_b = jno.Shape.disk(0, 0, RHA), jno.Shape.disk(CENTRE, 0, RHB)
    d = jno.Shape.regions(
        hubA=hub_a.sized(h_hub),
        rimA=(jno.Shape.polygon(a) - hub_a).sized(h_rim),
        hubB=hub_b.sized(h_hub),
        rimB=(jno.Shape.polygon(b) - hub_b).sized(h_rim),
        conforming=True,
    ).domain()
    _ = d.built_mesh

    # `region=` is what isolates ONE gear's surface: the predicate alone cannot, because both rims
    # span the same radii about their own centres.
    d.tag("sA", lambda x, y: (x**2 + y**2 > (R_LO * RBA) ** 2) & (x**2 + y**2 < (R_HI * RAA) ** 2), region="rimA")
    d.tag(
        "sB",
        lambda x, y: ((x - CENTRE) ** 2 + y**2 > (R_LO * RBB) ** 2) & ((x - CENTRE) ** 2 + y**2 < (R_HI * RAB) ** 2),
        region="rimB",
    )

    u, phi = d.fem_symbols(value_shape=(2,), names=("u", "phi"), order=1)
    R = {k: d.variable(k, split=True) for k in ("hubA", "rimA", "hubB", "rimB")}

    elastic = []
    for k in R:
        x, y = R[k][0], R[k][1]
        eu, ev = sym(u.bind(x=x, y=y), [x, y]), sym(phi.bind(x=x, y=y), [x, y])
        elastic.append(2 * MU * inner(eu, ev, 2) + LAM * tr(eu) * tr(ev))

    sa, nA = d.variable("sA", split=True), d.variable("sA", normals=True)
    g = u.gap("sA", "sB", domain=d)
    p = n.maximum(0.0, -CN * g)  # g < 0 penetrating -> p > 0 pressure
    terms = elastic + [
        p * inner(nA, phi.bind(x=sa[0], y=sa[1]), 1),
        u(R["hubA"][0], R["hubA"][1])[0] - (-drive * R["hubA"][1]),
        u(R["hubA"][0], R["hubA"][1])[1] - (+drive * R["hubA"][0]),
        u(R["hubB"][0], R["hubB"][1]) - 0.0,
    ]

    fem = jno.fem(terms)
    uu = np.asarray(fem.solve(contact=jno.solve.contact(capture=capture, rounds=rounds))).reshape(-1, 2)

    # ---- the oracle. At a driven node the equilibrium equation reads `internal force = the reaction
    # applied there`, and the contact traction acts on the rim, not the hub -- so the two hub
    # reactions ARE the two torques. Read them from the FREE (no-BC, no-contact) elastic operator.
    # Popping the registration is a measurement scaffold: the library rightly refuses to assemble a
    # form that declares a `u.gap` and never reads it.
    saved = d.__dict__.pop("_contact_pairs", None)
    try:
        free = jno.fem(elastic)
    finally:
        if saved is not None:
            d.__dict__["_contact_pairs"] = saved
    A, rhs = free._op
    f = np.asarray(A @ jnp.asarray(uu.reshape(-1)) - rhs).reshape(-1, 2)

    pts = np.asarray(d.built_mesh.points)[:, :2]
    tri = np.asarray(d.built_mesh.cells_dict["triangle"])

    def torque(region, cx):
        nodes = np.unique(tri[np.asarray(_cell_region_mask(d, region)).reshape(-1) > 0])
        r = pts[nodes] - np.array([cx, 0.0])
        return float(np.sum(r[:, 0] * f[nodes, 1] - r[:, 1] * f[nodes, 0]))

    return torque("hubA", 0.0), torque("hubB", CENTRE), getattr(fem, "contact_rounds", None)


@pytest.mark.slow
@pytest.mark.parametrize("theta", [0.0, 0.1745, 0.2327, 0.3491])
def test_the_torque_ratio_matches_the_kinematic_oracle(theta, _x64):
    """The acceptance criterion this whole investigation was held to: a few percent at EVERY drive
    angle, not on average. The four angles here include the two that were hardest -- 0.1745 rad
    (10 deg) was the worst in the cycle and 0.2327 (13.3 deg) is a hand-off, where the trailing pair
    disengages at the tip.

    Measured after the fix: 0.64%, 1.74%, 1.02%, 0.43%, each settling in 3-4 search rounds. The gate
    sits at 5% -- far below the 61% the defect produced, with headroom for mesh and platform drift.
    """
    t_a, t_b, rounds = _solve(theta)
    assert t_a != 0.0, "gear A carries no torque: nothing was transmitted through the contact"

    ratio = abs(t_b / t_a)
    err = abs(ratio - EXACT) / EXACT
    assert err < 0.05, (
        f"theta = {theta}: |T_B/T_A| = {ratio:.4f} is {err:.2%} off the oracle "
        f"{EXACT:.4f} (settled in {rounds} search rounds)"
    )
