"""Core loss: the magnetisation working against a COMPLEX permeability.

`magnetic_reluctance` already documents a complex `mu_r` as the way core loss enters -- the
imaginary part of chi is the lossy component, exactly as a complex permittivity carries dielectric
loss. What was missing is that the magnetisation current it acts on was solved for and then thrown
away: the state vector is `[I_c, phi, x_m]` and only the electric slices were unpacked.

The oracle here is POWER BALANCE, which needs no reference value and no convention of its own: a
passive network driven by 1 V takes in `Re(V conj(I)) = Re(1/Z)`, and that must come back out as
ohmic loss plus core loss and nothing else. It is checked first with no core at all, so the
convention is pinned before the new channel is asked to fit inside it.
"""

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

CU, FREQ = 5.8e7, 1e5
OUT, WIN, DEP, P = 0.024, 0.006, 0.008, 0.002  # outer, window inset, depth, cell pitch
box = jno.Shape.box


def _core(mu_r):
    """The square ring, as `test_peec_magnetic_oracle` builds it -- inlined rather than imported so
    this file stands alone and the two cannot drift into each other."""
    ring = box(0, 0, 0, OUT, OUT, DEP, size=(P,) * 3) - box(WIN, WIN, -P, OUT - WIN, OUT - WIN, DEP + P)
    return ring.attach(mu_r=mu_r).name("core")


def _turn():
    """One solid turn around the left limb: through the window, out under and over the core."""
    y0, y1 = 0.010, 0.012
    sz = (P,) * 3
    w = (
        box(-0.004, y0, -0.004, -0.002, y1, 0.012, size=sz)
        | box(0.008, y0, -0.004, 0.010, y1, 0.012, size=sz)
        | box(-0.004, y0, -0.004, 0.010, y1, -0.002, size=sz)
        | box(-0.004, y0, 0.010, 0.010, y1, 0.012, size=sz)
    )
    return (w - box(-0.005, y0 - 0.001, 0.002, -0.001, y1 + 0.001, 0.004)).attach(sigma=CU).name("turn")


def _net(core=None):
    """The oracle file's transformer: a square ring core with one solid turn through its window.

    Deliberately NOT a slab beside a straight bar. That geometry was tried first and its
    magnetisation came out at 1e-26 A -- the port inductance did not move between mu_r = 2000 and
    mu_r = 200000 -- so it could not tell a right core-loss formula from a wrong one. A closed
    magnetic path around a driven turn is the case where the core carries real flux.
    """
    sh = _turn() if core is None else (_core(core) + _turn())
    d = sh.domain()
    d.tag("A", lambda x, y, z: (x < -0.0021) & (z < 0.0021) & (z > 0.0001))
    d.tag("B", lambda x, y, z: (x < -0.0021) & (z > 0.0039) & (z < 0.0059))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=FREQ).build().solve()


def _total(sol, channel):
    """A channel's W/m^3 back to watts, on whichever mesh owns it."""
    if channel == "sigma":
        own, vol, names = np.asarray(sol._owner), np.asarray(sol._vol), sol._names
    else:
        own, vol, names = np.asarray(sol._mag_owner), np.asarray(sol._mag_vol), sol._mag_names
    d = getattr(sol.dissipation(), channel)
    return sum(float(np.real(d[n])) * vol[own == k].sum() for k, n in enumerate(names) if n in d)


def test_the_power_balance_closes_with_no_core():
    """The convention, pinned before anything new is fitted into it: driven by 1 V, the network
    takes in `Re(1/Z)` and it all leaves as ohmic loss."""
    sol = _net()
    p_in = float(np.real(1.0 / complex(sol.Z)))
    assert abs(_total(sol, "sigma") / p_in - 1) < 1e-9, (p_in, _total(sol, "sigma"))
    assert abs(float(sol.joule) / p_in - 1) < 1e-9


def test_a_real_permeability_is_a_LOSSLESS_core():
    """A real `mu_r` stores energy and dissipates none, so the channel exists and is zero -- not
    absent, and not some small numerical residue standing in for a physical effect."""
    sol = _net(core=2000.0)
    assert _total(sol, "mu_r") == 0.0, "a real permeability has a real reluctance and no loss at all"


def test_a_complex_permeability_dissipates_and_the_balance_still_closes():
    """The feature. With a lossy core the input power no longer equals the ohmic loss, and the
    difference is exactly what the magnetisation dissipates -- which is the whole claim."""
    sol = _net(core=2000.0 - 200.0j)
    p_in = float(np.real(1.0 / complex(sol.Z)))
    ohmic, core = _total(sol, "sigma"), _total(sol, "mu_r")
    assert core > 0.0, "a lossy core must dissipate"
    assert core / ohmic > 1e-6, (core, ohmic)  # and measurably, not at round-off
    assert abs((ohmic + core) / p_in - 1) < 1e-6, (p_in, ohmic, core)


def test_the_loss_scales_with_the_imaginary_part():
    """Doubling the lossy component of chi must move the core loss and leave its sign alone. A
    channel that was actually reading the REAL part would not track it."""
    a = _total(_net(core=2000.0 - 100.0j), "mu_r")
    b = _total(_net(core=2000.0 - 200.0j), "mu_r")
    assert a > 0 and b > 0 and b > a, (a, b)


def test_the_mapping_is_the_total_and_the_channels_are_named_by_what_was_attached():
    """`dissipation()` is still the `{region: W/m^3}` a heat source consumes, so a region that
    dissipates two ways contributes both -- and the split is read back under the same spelling it
    was declared with."""
    sol = _net(core=2000.0 - 200.0j)
    d = sol.dissipation()
    assert set(d) == {"turn", "core"}
    assert set(d.sigma) == {"turn"} and set(d.mu_r) == {"core"}
    for region in d:
        parts = [ch[region] for ch in (d.sigma, d.mu_r) if region in ch]
        assert abs(float(np.real(d[region])) - float(np.real(sum(parts)))) < 1e-12 * abs(float(np.real(d[region])))


def test_a_model_with_no_core_says_so_rather_than_returning_zero():
    """An empty dict would read as "no loss"; the honest answer is that nothing here dissipates
    that way at all, and which property to attach if it should."""
    d = _net().dissipation()
    assert set(d.sigma) == {"turn"}
    with pytest.raises(ValueError, match="dissipates through 'mu_r'"):
        d.mu_r
