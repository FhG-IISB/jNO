# --8<-- [start:code]
"""A transformer: a ferrite ring, two windings, and an open secondary.

Everything so far was conductors in air. A region that declares ``mu_r`` instead of (or as well as)
``sigma`` is a MAGNETIC material, and it is solved for: the core becomes a second family of elements
on the shared cell grid, carrying magnetisation currents that couple back into the circuit
(Torchio et al., *IEEE Trans. MTT* **66**(5), 2018). Nothing is switched on -- what a region carries
decides what it is.

That gives the two readouts this tutorial is about.

**An open terminal has a voltage.** The secondary carries no current by construction, so ``current``
and the port impedance say nothing about it -- the induced voltage lives in the nodal potentials,
and ``sol.voltage`` is how they are read. This is the ideal-transformer law appearing on its own:
with one turn on each limb the secondary must approach one volt per volt in, and it gets closer as
the permeability rises, because a better core leaks less flux. Nothing here imposes a turns ratio.

**A complex permeability is a lossy core.** The imaginary part of chi = mu_r - 1 is the lossy
component of the magnetisation, exactly as a complex permittivity carries dielectric loss. It comes
back from ``dissipation()`` under ``.mu_r`` -- named for the property that caused it, the same
spelling in and out -- while the ohmic loss stays under ``.sigma``.

The check on the loss is POWER BALANCE, which needs no reference value: driven by one volt, a
passive network takes in ``Re(1/Z)``, and that must leave as copper loss plus core loss and nothing
else.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

box, CU = jno.Shape.box, 5.8e7
OUT, WIN, DEP, P = 0.024, 0.006, 0.008, 0.002  # outer size, window inset, depth, cell pitch


def ring(mu_r):
    """A square core: a solid block with its window cut out."""
    r = box(0, 0, 0, OUT, OUT, DEP, size=(P,) * 3) - box(WIN, WIN, -P, OUT - WIN, OUT - WIN, DEP + P)
    return r.attach(mu_r=mu_r).name("core")


def turn(x_outer, x_inner, name):
    """One rectangular turn: down the outside, under, up through the window, over the top."""
    y0, y1, s = 0.010, 0.012, (P,) * 3
    lo, hi = min(x_outer, x_inner), max(x_outer, x_inner)
    w = (
        box(x_outer, y0, -0.004, x_outer + P, y1, 0.012, size=s)
        | box(x_inner, y0, -0.004, x_inner + P, y1, 0.012, size=s)
        | box(lo, y0, -0.004, hi + P, y1, -0.002, size=s)
        | box(lo, y0, 0.010, hi + P, y1, 0.012, size=s)
    )
    return (w - box(x_outer - 0.001, y0 - 0.001, 0.002, x_outer + 0.003, y1 + 0.001, 0.004)).attach(sigma=CU).name(name)


PRI, SEC = turn(-0.004, 0.008, "pri"), turn(0.026, 0.014, "sec")  # around the left and right limbs
CORE_VOLUME = (OUT**2 - (OUT - 2 * WIN) ** 2) * DEP  # outer block less its window


def solve(mu_r, freq=1e5):
    d = (ring(mu_r) + PRI + SEC).domain()
    d.tag("P0", lambda x, y, z: (x < -0.0021) & (z < 0.0021) & (z > 0.0001))
    d.tag("P1", lambda x, y, z: (x < -0.0021) & (z > 0.0039) & (z < 0.0059))
    d.tag("S0", lambda x, y, z: (x > 0.0269) & (z < 0.0021) & (z > 0.0001))
    d.tag("S1", lambda x, y, z: (x > 0.0269) & (z > 0.0039) & (z < 0.0059))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec(
        [
            v(*at("P0")) - v(*at("P1")) - 1.0,  # drive the primary with one volt
            i(*at("S0")) - 0.0,  # the secondary is OPEN -- no current
            v(*at("S1")) - 0.0,  # and a reference to measure the induced volts against
        ],
        freq=freq,
    ).solve()


# --- 1. the turns ratio emerges from the geometry -------------------------------------------------
print("  a better core leaks less flux, so V_sec -> the 1:1 turns ratio\n")
print("      mu_r      L_pri        V_sec / V_pri")
ratios = []
for mu in (200.0, 2000.0, 20000.0):
    s = solve(mu)
    r = abs(complex(s.voltage("S0", "S1")))
    ratios.append(r)
    print(f"  {mu:9.0f}   {float(np.real(s.L)) * 1e9:9.1f} nH      {r:.4f}")

assert ratios[0] < ratios[1] < ratios[2] < 1.0, "the ratio must rise toward 1 and never exceed it"
assert ratios[-1] > 0.99, "at mu_r = 20000 the core should couple almost all of the flux"

# --- 2. a complex mu_r is a lossy core, and the energy balances ------------------------------------
# The balance needs no volume and no reference value. `joule` is the total copper watts and
# `Re(1/Z)` is what the port delivers, so whatever separates them is the core -- zero for a real
# permeability, and dominant for a lossy one.
#
# (`dissipation()` is deliberately VOLUMETRIC, per unit of the discretisation's own summed element
# volume, because a heat source is a density -- it is shaped for `d.by_region`, not for totalling.
# `joule` is the total that pairs with it on the copper side; the core channel has no such total.)
print("\n  a complex mu_r dissipates, and the power balance says how much\n")
for mu in (2000.0, 2000.0 - 200.0j):
    s = solve(mu)
    q = s.dissipation()  # {region: W/m^3} total, with .sigma / .mu_r channels
    copper, p_in = float(s.joule), float(np.real(1.0 / complex(s.Z)))
    core = p_in - copper  # delivered but not dissipated in the copper
    density = float(np.real(q.mu_r["core"]))
    print(f"  mu_r = {str(mu):>16}   in {p_in:.4e} W   copper {copper:.4e} W   core {core:.4e} W")
    print(f"  {'':16}     dissipation().mu_r['core'] = {density:.4e} W/m^3")
    if np.imag(mu) == 0:
        assert abs(copper / p_in - 1) < 1e-4, "a lossless core: it all comes back out of the copper"
        assert density == 0.0, "and the core channel is exactly zero, not merely small"
    else:
        assert core > 10 * copper, "at this permeability the core is the dominant loss"
        assert density > 0.0, "and the readout attributes it to the core"
