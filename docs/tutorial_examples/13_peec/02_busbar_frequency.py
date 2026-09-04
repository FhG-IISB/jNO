# --8<-- [start:code]
"""Where the current goes at frequency: a busbar pair from 1 kHz to 10 MHz.

A DC-link busbar is two flat conductors carrying equal and opposite current. At low frequency the
current spreads evenly and the pair behaves like its DC resistance. Raise the frequency and two
things happen at once, both of them redistribution:

* **skin effect** -- current retreats to the surface of each bar;
* **proximity effect** -- and, because the return is right there, it crowds onto the FACING edges,
  where the two currents are closest and the loop encloses least flux.

Both are the same statement: current arranges itself to minimise stored magnetic energy. So the
resistance climbs (less copper is doing the work) while the inductance falls (the effective loop
shrinks) -- and the inductance flattens out, because once the current has reached the facing edges
there is nowhere further for it to go.

This is what a partial-element method is FOR. Nothing here is imposed: there is no skin-depth
formula in the input and no current profile assumed. The bars are cut into filaments across their
width, each filament is free to carry what it likes, and the redistribution is what the circuit
solves for. That is also the limit -- see the note at the end.

An array ``freq=`` solves at every frequency and turns each port readout into an array over it.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

CU, mm = 5.8e7, 1e-3
LB, WB, TB, GAP = 0.060, 0.006, 1e-3, 0.002  # length, bar width, thickness, gap between bars

# --- a coplanar go/return pair, shorted at the far end --------------------------------------------
# One cell through the thickness on purpose: an element may take the surface impedance only when it
# IS the whole thickness, and `jno.peec` says so loudly if a conductor is thick against the skin
# depth. Resolution goes ACROSS the width (1 mm cells on a 6 mm bar), which is the direction the
# current actually redistributes in here.
cells = (2 * mm, 1 * mm, TB)
go = jno.Shape.box(0, 0, 0, LB, WB, TB, size=cells).attach(sigma=CU).name("go")
ret = jno.Shape.box(0, WB + GAP, 0, LB, 2 * WB + GAP, TB, size=cells).attach(sigma=CU).name("return")
link = jno.Shape.box(LB - 2 * mm, 0, 0, LB, 2 * WB + GAP, TB, size=cells).attach(sigma=CU).name("link")

d = (go + ret + link).domain()
d.tag("A", lambda x, y, z: (x < 1.1 * mm) & (y < WB))
d.tag("B", lambda x, y, z: (x < 1.1 * mm) & (y > WB + GAP))

i, v = d.peec_symbols()
at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]

freqs = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freqs).solve()

R, L = np.real(np.asarray(sol.R)), np.real(np.asarray(sol.L))
print(f"  {int(np.asarray(sol.i).shape[-1])} elements, {len(freqs)} frequencies, one solve\n")
print("     f          R            L        skin depth")
for f, r, ind in zip(freqs, R, L):
    delta = 1.0 / np.sqrt(np.pi * f * CU * 4e-7 * np.pi)
    print(f"  {f:8.0e} Hz  {r * 1e6:9.1f} uOhm  {ind * 1e9:7.3f} nH   {delta * 1e3:6.3f} mm")

print(f"\n  R rises {R[-1] / R[0]:.0f}x while L falls {100 * (1 - L[-1] / L[0]):.1f} %")

# --- what must be true ----------------------------------------------------------------------------
assert np.all(np.diff(R) > 0), "resistance rises monotonically as the current crowds"
assert np.all(np.diff(L) < 0), "and inductance falls monotonically as the loop tightens"
assert L[-1] / L[0] > 0.5, "the inductance flattens; it cannot collapse to nothing"
assert abs(L[-2] / L[-1] - 1) < 0.01, "and it has flattened by the top of the sweep"

# NOTE, up front: each filament carries ONE current, so the skin effect WITHIN a filament is not
# represented -- what is captured is redistribution BETWEEN them. That is the right model here,
# because the bars are one cell thick and the crowding is across the width. A conductor thick
# against its skin depth needs either one cell through the thickness (exact, a current sheet per
# face) or cells fine enough to resolve the profile; `jno.peec` refuses to guess and warns by name.
# --8<-- [end:code]
