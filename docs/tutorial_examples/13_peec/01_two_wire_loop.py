# --8<-- [start:code]
"""Loop resistance and inductance of a two-wire line, with nothing meshed.

The first thing to know about ``jno.peec`` is what it does NOT do: it never meshes the air. A
partial-element method (Ruehli, *IEEE Trans. MTT* **22**(3), 1974) discretises only the metal, and
the field between conductors is carried by the partial inductances coupling one filament to another.
So the input is a conductor and a port, and the output is a circuit.

The conductor here is a hairpin: out along one wire, across the far end, back along the other. The
port is the pair of open ends. Two closed forms say what the answer must be.

**Resistance** is exact and unforgiving -- at DC the current fills the section, so

    R = rho * l / A

with `l` the whole routed length, the far-end link included. There is no modelling freedom in it,
which makes it the right thing to check first.

**Inductance** needs more care, and the care is the lesson. The textbook two-wire line carries

    L_ext / l = (mu0 / pi) * acosh(D / 2a)

(e.g. Grover, *Inductance Calculations*, 1946) -- but that is the **external** inductance, the flux
in the air between the wires. ``sol.L`` is the total magnetic energy, so it also holds the flux
INSIDE the copper, which for a round wire carrying a uniform DC current adds `mu0 / 8pi` per unit
length per wire. Compare against the external term alone and jNO looks 9 % high; add the internal
term and it lands 2 % LOW, which is the finite length -- the closed form is for a line with no ends,
and a 100 mm hairpin has two.

That 9 % is not an error bar. It is a different quantity, and knowing which one a formula reports is
most of the work in validating an inductance.
"""

import jax

jax.config.update("jax_enable_x64", True)  # partial inductance is a difference of large numbers

import numpy as np  # noqa: E402

import jno  # noqa: E402

CU, MU0 = 5.8e7, 4e-7 * np.pi  # copper S/m, vacuum permeability
ELL, SEP, RAD = 0.100, 0.005, 5e-4  # leg length, wire spacing, wire radius (m)

# --- the conductor: a hairpin of round wire, and a pad at each open end ---------------------------
route = [(0, 0, 0), (ELL, 0, 0), (ELL, SEP, 0), (0, SEP, 0)]
loop = jno.Shape.line(route, r=RAD, size=2e-3).attach(sigma=CU).name("loop")
pads = jno.Shape.sphere(0, 0, 0, 2 * RAD).name("A") + jno.Shape.sphere(0, SEP, 0, 2 * RAD).name("B")

d = (loop + pads).domain()
i, v = d.peec_symbols()  # terminal current, nodal potential
at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]

# --- the port: one volt across the open ends. That is the whole problem statement -----------------
sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).solve()

# --- what it must equal --------------------------------------------------------------------------
length = 2 * ELL + SEP  # the far-end link is conductor too
R_closed = length / (CU * np.pi * RAD**2)
L_external = ELL * (MU0 / np.pi) * np.arccosh(SEP / (2 * RAD))
L_internal = 2 * ELL * MU0 / (8 * np.pi)  # both wires, uniform DC current

R, L = float(sol.R), float(sol.L)
print(f"  elements            {int(np.asarray(sol.i).size)}   (nothing else was meshed)")
print(f"  R  {R * 1e3:8.4f} mOhm   vs rho*l/A {R_closed * 1e3:8.4f}   {100 * (R / R_closed - 1):+.4f} %")
print(f"  L  {L * 1e9:8.2f} nH     vs external {L_external * 1e9:8.2f}   {100 * (L / L_external - 1):+.1f} %")
print(
    f"     {'':8}          vs ext+int  {(L_external + L_internal) * 1e9:8.2f}   "
    f"{100 * (L / (L_external + L_internal) - 1):+.1f} %"
)

assert abs(R / R_closed - 1) < 1e-6, "the DC resistance is rho*l/A exactly; nothing else is allowed"
assert abs(L / (L_external + L_internal) - 1) < 0.05, "within 5 % of the closed form, internal included"
assert L > L_external, "the total energy must exceed the external flux alone"

# --8<-- [end:code]
