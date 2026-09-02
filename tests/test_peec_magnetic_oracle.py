"""What a magnetic material has to reproduce, stated before any of it is built.

A core is not a coefficient on the existing solve -- it is a second unknown field, magnetic fluxes
and potentials, coupled to the electric one. So the claims worth pinning are the ones a reluctance
argument settles independently of how that system is assembled:

* a core of unit permeability is not there at all, and the answer must be the coreless one. This is
  the CONTINUITY guard, and it is the shape of test that a sheet-pair model failed while passing
  every absolute check written for it -- a discretisation that switches on must not move the answer.
* a closed core is a reluctance `l / (mu0 mu_r A)`, so the inductance it mediates is
  `mu0 mu_r N^2 A / l` (Ramo, Whinnery & Van Duzer, *Fields and Waves*, 3rd ed., ch. 4).
* a gapped core is two reluctances in series and the gap dominates, so `L` tracks `1/g`. This one is
  a SHAPE, insensitive to the air leakage that the reluctance formula does not describe.

Geometry: a square ring, 24 mm outside, 12 mm window, 8 mm deep, with one turn threading the window
and closing around the left limb.
"""

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

CU, MU0 = 5.8e7, 4e-7 * np.pi
OUT, WIN, DEP, P = 0.024, 0.006, 0.008, 0.002  # outer, window inset, depth, cell pitch
AREA = WIN * DEP  # a limb's cross-section, 6 x 8 mm
PATH = 4.0 * (OUT - WIN)  # mean magnetic path around the ring, 4 x 18 mm


def _core(mu_r, gap=0.0):
    """The square ring, optionally with a gap cut through its right limb."""
    ring = jno.Shape.box(0, 0, 0, OUT, OUT, DEP, size=(P, P, DEP)) - jno.Shape.box(
        WIN, WIN, -DEP, OUT - WIN, OUT - WIN, 2 * DEP
    )
    if gap > 0:
        mid = 0.5 * OUT
        ring = ring - jno.Shape.box(OUT - WIN - P, mid - 0.5 * gap, -DEP, OUT + P, mid + 0.5 * gap, 2 * DEP)
    return ring.attach(mu_r=mu_r).name("core")


def _turn():
    """One turn around the left limb: through the window, out under and over the core."""
    y = 0.5 * OUT
    z0, z1 = -0.003, DEP + 0.003
    x0, x1 = -0.003, WIN + 0.003
    pts = [(x0, y, z0 + 0.0025), (x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), (x0, y, z0 + 0.005)]
    return jno.Shape.line(pts, r=6e-4, size=0.002).attach(sigma=CU).name("turn")


def _inductance(mu_r=None, gap=0.0, freq=1e5):
    sh = _turn() if mu_r is None else (_core(mu_r, gap) + _turn())
    d = sh.domain()
    y, z0 = 0.5 * OUT, -0.003
    d.tag("A", lambda x, yy, z: (z < z0 + 0.0013) & (x < -0.002))
    d.tag("B", lambda x, yy, z: (z > z0 + 0.004) & (x < -0.002))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build().solve()
    return float(np.real(sol.L))


@pytest.mark.xfail(
    reason="the coupled magnetic solve is not wired yet -- Phase 3. These are its acceptance "
    "criteria, kept red on purpose so they cannot be quietly forgotten.",
    strict=True,
)
def test_a_barely_magnetic_core_is_the_coreless_answer():
    """As chi -> 0 the core must fade out CONTINUOUSLY into the coreless answer.

    The continuity guard, and the shape of test a sheet-pair model failed this week while passing
    every absolute check written for it: a discretisation that switches on must not move the answer
    as it does. Deliberately mu_r slightly ABOVE 1 rather than exactly 1 -- exactly 1 is dropped as
    air at the front door, so it would test the bypass instead of the physics.
    """
    assert abs(_inductance(mu_r=1.0001) / _inductance(mu_r=None) - 1) < 0.005


@pytest.mark.xfail(
    reason="the coupled magnetic solve is not wired yet -- Phase 3. These are its acceptance "
    "criteria, kept red on purpose so they cannot be quietly forgotten.",
    strict=True,
)
def test_a_closed_core_lands_on_its_reluctance():
    """L = mu0 mu_r N^2 A / l for one turn, once the core reluctance dominates the air path."""
    mu_r = 1000.0
    predicted = MU0 * mu_r * AREA / PATH
    got = _inductance(mu_r=mu_r)
    assert abs(got / predicted - 1) < 0.25, f"{got * 1e9:.1f} nH against {predicted * 1e9:.1f} nH"


@pytest.mark.xfail(
    reason="the coupled magnetic solve is not wired yet -- Phase 3. These are its acceptance "
    "criteria, kept red on purpose so they cannot be quietly forgotten.",
    strict=True,
)
def test_a_gapped_core_tracks_one_over_the_gap():
    """Two reluctances in series with the gap dominating, so L falls as 1/g -- a SHAPE, not a value.

    Insensitive to the air leakage the reluctance formula omits, which is why it is the robust one.
    """
    mu_r = 2000.0
    gaps = np.array([0.5e-3, 1.0e-3, 2.0e-3])
    L = np.array([_inductance(mu_r=mu_r, gap=float(g)) for g in gaps])
    assert np.all(np.diff(L) < 0)  # a wider gap can only lower it
    # gap reluctance g/(mu0 A) against core l/(mu0 mu_r A): the ratio of the two L values should
    # follow the ratio of total reluctances, not merely decrease
    R = gaps / (MU0 * AREA) + PATH / (MU0 * mu_r * AREA)
    assert abs((L[0] / L[-1]) / (R[-1] / R[0]) - 1) < 0.3


@pytest.mark.xfail(
    reason="the coupled magnetic solve is not wired yet -- Phase 3. These are its acceptance "
    "criteria, kept red on purpose so they cannot be quietly forgotten.",
    strict=True,
)
def test_the_permeability_is_differentiable():
    """A core that cannot be differentiated is not a design variable, which is the point of jNO."""
    g = float(jax.grad(lambda m: _inductance(mu_r=m))(500.0))
    fd = (_inductance(mu_r=505.0) - _inductance(mu_r=495.0)) / 10.0
    assert np.isfinite(g) and abs(g / fd - 1) < 0.05
