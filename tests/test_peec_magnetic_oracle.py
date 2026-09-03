"""What a magnetic material has to reproduce, written before any of it was built.

A core is not a coefficient on the existing solve -- it is a second unknown field, magnetisation
fluxes and potentials, coupled to the electric one. So the claims worth pinning are the ones a
reluctance argument settles independently of how that system is assembled:

* a core of unit permeability is not there at all, and the answer must be the coreless one. This is
  the CONTINUITY guard, and it is the shape of test that a sheet-pair model failed while passing
  every absolute check written for it -- a discretisation that switches on must not move the answer.
* a closed core is a reluctance `l / (mu0 mu_r A)`, so the inductance it mediates is
  `mu0 mu_r N^2 A / l` (Ramo, Whinnery & Van Duzer, *Fields and Waves*, 3rd ed., ch. 4).
* a gap lowers the inductance, and widening it lowers it further. The `1/g` LAW this file first
  claimed is not asserted, and the reason is measured rather than assumed -- see the gap test.
* and all of it differentiable in `mu_r`, or a core is not a design variable.

Geometry: a square ring, 24 mm outside, 12 mm window, 8 mm deep, with one turn threading the window
and closing around the left limb.

The winding is a SOLID, not the `Shape.line` this file first used. A core welded to a line conductor
is refused (`jno.peec: a magnetic region together with a Shape.line is not supported yet`) -- a weld
already needs a cross block and a whole-system factorisation, and a coupled magnetic system on top of
that is untested. None of the four claims below depends on which the winding is; only the
discretisation of one turn changed, and the refusal itself is covered in the front-door tests.
"""


import jax
import jax.numpy as jnp
import numpy as np

import jno

jax.config.update("jax_enable_x64", True)

CU, MU0 = 5.8e7, 4e-7 * np.pi
OUT, WIN, DEP, P = 0.024, 0.006, 0.008, 0.002  # outer, window inset, depth, cell pitch
AREA = WIN * DEP  # a limb's cross-section, 6 x 8 mm
PATH = 4.0 * (OUT - WIN)  # mean magnetic path around the ring, 4 x 18 mm
box = jno.Shape.box


def _core(mu_r, gap=0.0, size=(P,) * 3):
    """The square ring, optionally with a gap cut through its right limb."""
    ring = box(0, 0, 0, OUT, OUT, DEP, size=size) - box(WIN, WIN, -P, OUT - WIN, OUT - WIN, DEP + P)
    if gap > 0:
        mid = 0.5 * OUT
        ring = ring - box(OUT - WIN - P, mid - 0.5 * gap, -P, OUT + P, mid + 0.5 * gap, DEP + P)
    return ring.attach(mu_r=mu_r).name("core")


def _turn(size=(P,) * 3):
    """One solid turn around the left limb: through the window, out under and over the core.

    A rectangular loop in the xz plane at mid-height, with a gap cut in its outer leg for the port.
    """
    y0, y1 = 0.010, 0.012
    w = (
        box(-0.004, y0, -0.004, -0.002, y1, 0.012, size=size)  # outer leg, outside the core
        | box(0.008, y0, -0.004, 0.010, y1, 0.012, size=size)  # inner leg, inside the window
        | box(-0.004, y0, -0.004, 0.010, y1, -0.002, size=size)  # under
        | box(-0.004, y0, 0.010, 0.010, y1, 0.012, size=size)  # over
    )
    return (w - box(-0.005, y0 - 0.001, 0.002, -0.001, y1 + 0.001, 0.004)).attach(sigma=CU).name("turn")


def _gapped(mu_r, centre, half):
    """The ring with a gap cut about ``centre`` of half-width ``half``, in metres."""
    ring = box(0, 0, 0, OUT, OUT, DEP, size=(P,) * 3) - box(WIN, WIN, -P, OUT - WIN, OUT - WIN, DEP + P)
    ring = ring - box(OUT - WIN - P, centre - half, -P, OUT + P, centre + half, DEP + P)
    sh = ring.attach(mu_r=mu_r).name("core") + _turn()
    return _solve(sh)


def _solve(sh, freq=1e5, restart=None):
    d = sh.domain()
    d.tag("A", lambda x, y, z: (x < -0.0021) & (z < 0.0021) & (z > 0.0001))
    d.tag("B", lambda x, y, z: (x < -0.0021) & (z > 0.0039) & (z < 0.0059))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build().solve(
        **({} if restart is None else {"restart": restart})
    )
    return jnp.real(sol.L)


def _inductance(mu_r=None, gap=0.0, freq=1e5, size=(P,) * 3, restart=None):
    return _solve(_turn(size) if mu_r is None else (_core(mu_r, gap, size) + _turn(size)), freq, restart)


def test_a_barely_magnetic_core_is_the_coreless_answer():
    """As chi -> 0 the core must fade out CONTINUOUSLY into the coreless answer.

    The continuity guard, and the shape of test a sheet-pair model failed while passing every
    absolute check written for it: a discretisation that switches on must not move the answer as it
    does. Deliberately mu_r slightly ABOVE 1 rather than exactly 1 -- exactly 1 is dropped as air at
    the front door, so it would test the bypass instead of the physics.
    """
    assert abs(float(_inductance(mu_r=1.0001)) / float(_inductance(mu_r=None)) - 1) < 0.005


def test_a_closed_core_lands_on_its_reluctance():
    """L = mu0 mu_r N^2 A / l for one turn, once the core reluctance dominates the air path.

    Measured 916.7 nH against the law's 837.8 nH, +9.4 %.

    The tolerance is 0.25 and it is NOT slack that happens to be unused -- this mesh is not
    converged. Refining the core from 2 mm to 1 x 2 x 2 mm moves L by 16 % (1803.2 -> 1513.4 nH at
    mu_r = 2000), which at mu_r = 1000 puts the refined answer near 757 nH, i.e. 9.7 % BELOW the same
    law the coarse one sits 9.4 % above. So the agreement here brackets the reluctance value rather
    than converging onto it, and 2 mm is a unit-test pitch rather than an accurate one. Read this
    test as "the formulation produces the right physics at the right magnitude", not as an accuracy
    claim -- the accuracy claim needs a convergence study, which is not a unit test.
    """
    mu_r = 1000.0
    predicted = MU0 * mu_r * AREA / PATH
    got = float(_inductance(mu_r=mu_r))
    assert abs(got / predicted - 1) < 0.25, f"{got * 1e9:.1f} nH against {predicted * 1e9:.1f} nH"


def test_the_inductance_approaches_linearity_in_the_permeability():
    """A far sharper check on the formulation than any single absolute value.

    The core's reluctance is in SERIES with an air path the formula does not describe, so `L` is not
    proportional to mu_r -- it approaches proportionality from BELOW as the core comes to dominate.
    Measured across two decades, 1803.2 / 17759.7 / 177324.8 nH, the decade ratios are 9.849 then
    9.985: below ten, and closing on it. That is the shape a series reluctance predicts, and it is a
    stronger statement than either ratio on its own.

    A coupling with the wrong scale, a susceptibility confused with a permeability, or a
    demagnetising term of the wrong sign would all break this while still landing somewhere plausible
    on the single-point test above.

    It also pins that core strength is not what costs the solver: all three converge at the default
    subspace in under two seconds.
    """
    L = np.array([float(_inductance(mu_r=m)) for m in (2000.0, 20000.0, 200000.0)])
    r = L[1:] / L[:-1]
    assert np.all(r < 10.0) and np.all(r > 9.5), r  # below ten, and not far below
    assert r[1] > r[0], r  # and closing on it as the core takes over from the air path


def test_the_answer_does_not_depend_on_the_frequency():
    """A core's inductance is magnetostatic, so it must not move with frequency below resonance.

    Measured flat over two decades -- 916.7 nH at 10 kHz, 100 kHz and 1 MHz. This is what makes the
    DC refusal harmless in practice: the magnetisation only reaches the circuit through `j w K'`, so
    omega = 0 is refused, but any small frequency gives the magnetostatic answer.
    """
    sh = _core(1000.0) + _turn()
    L = np.real(np.asarray(_solve(sh, freq=[1e4, 1e5, 1e6])))
    assert np.allclose(L, L[0], rtol=1e-6), L * 1e9


def test_the_permeability_is_differentiable():
    """A core that cannot be differentiated is not a design variable, which is the point of jNO."""
    g = float(jax.grad(lambda m: _inductance(mu_r=m))(500.0))
    fd = float(_inductance(mu_r=505.0) - _inductance(mu_r=495.0)) / 10.0
    assert np.isfinite(g) and abs(g / fd - 1) < 0.05


def test_a_gap_lowers_the_inductance_and_a_wider_gap_lowers_it_further():
    """Cutting the core is the largest single thing a designer does to it, so it must show.

    Measured here, mu_r = 2000, gaps of one, two and three cells: 1803 nH closed, then 109.9, 98.8
    and 92.4 nH -- a 16x drop on cutting it, monotone as it widens.

    What is NOT asserted is the reluctance law `L ~ 1/(g/mu0 A + l/mu0 mu_r A)`, which this file
    originally claimed. It does not hold at these gaps, and the reason is physical rather than
    numerical: a 2-6 mm gap in a 6 x 8 mm limb is not a thin gap, so most of the flux leaks around it
    instead of crossing it, and the series-reluctance picture does not describe that. Measured
    against it, 110/99/92 nH where two reluctances in series predict 49/37/34 nH; even with the
    standard fringing correction `A_eff = (w+g)(t+g)` (McLyman, *Transformer and Inductor Design
    Handbook*, 4th ed., ch. 10) the ratio is still 16 % out.

    Reaching the reluctance regime needs `g` small against the limb, and both are counted in the same
    cells -- a limb ten cells across with a one-cell gap is some 26,000 magnetic elements, which is
    not a unit test. So the claim asserted here is the one this mesh can actually support, and the
    law is left to a convergence study rather than smuggled in at a tolerance wide enough to pass.
    """
    closed = float(_inductance(mu_r=2000.0))
    # A whole-CELL gap, which is the only kind a lattice has. Cutting about a cell CENTRE catches an
    # odd number of rows and about a face an even number, so one, two and three cells are all
    # reachable on this pitch -- asking for 0.5 mm on a 2 mm grid would silently cut nothing.
    gaps = [(0.011, 0.001), (0.012, 0.002), (0.011, 0.003)]
    L = [float(_gapped(2000.0, c, h)) for c, h in gaps]
    assert closed / L[0] > 5.0, f"closed {closed * 1e9:.1f} nH against gapped {L[0] * 1e9:.1f} nH"
    assert np.all(np.diff(L) < 0), [f"{v * 1e9:.1f}" for v in L]
