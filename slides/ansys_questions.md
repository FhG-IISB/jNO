# Questions on the Ansys Q3D reference

Context for whoever answers: two independent PEEC codes (jNO and pypeec 5.8.0) now agree on this
layout at ~22.5–23.0 nH once each is corrected for its own known discretisation bias. Q3D reports
**20.641 nH**, about 10 % below both. jNO has separately been validated against Grover's analytic
rectangular-bar formula (+1.9 %) and the exact DC resistance (−0.2 %), so the disagreement is not a
solver being obviously wrong — it is most likely a difference in **what is being reported** or **what
is in the model**.

Ordered by how much the answer would change our conclusion.

## 1. What quantity does `ACLoop` report?

Specifically, is the inductance

- `Im(Z)/ω` at the port,
- an energy form `2W/|I|²`,
- a DC or partial inductance, or
- a loop inductance defined against a particular return path?

*Why it matters:* these differ once currents are out of phase. In jNO the energy form and `Im(Z)/ω`
differ by ~3 %, so this alone will not explain 10 %, but every other comparison depends on fixing the
definition first.

## 2. What is the return path in the extracted loop?

Are **Baseplate** and **Bottom_Metal**

- galvanically part of the circuit (connected to a terminal),
- floating conductors carrying induced eddy currents, or
- excluded from the RL solve entirely?

*Why it matters:* this is the single largest lever on loop area. We model both as **floating** — the
ceramic isolates them — and the return therefore flows through the traces and bond wires. If Q3D has
them in the circuit, the loop is far smaller and a lower inductance follows immediately.

## 3. Did the adaptive solution converge, and at what frequency?

- Solution frequency for the reported number.
- Convergence criterion (Δ % per adaptive pass), number of passes, and the final delta.
- Was the reported value taken from the last pass or from a converged sweep?

*Why it matters:* the mesh statistics show **3,416 tetrahedra** in `PowerLoop` with a **3.4 mm RMS
edge**, against a **66 µm** skin depth at 1 MHz. Both PEEC codes were still moving under refinement
at far finer discretisations, so it is worth knowing what Q3D's own error estimate said.

## 4. Surface impedance, or resolved skin depth?

Does the model use an impedance/surface boundary condition on the conductors, or resolve the current
distribution volumetrically? If a surface condition — on **which** faces (all of them, or only the
large ones)?

*Why it matters:* at 3.4 mm elements it cannot be resolving 66 µm, so it must be an SIBC. Which faces
it applies to sets the conducting perimeter and therefore both R and the current's position.

## 5. Is `PowerLoop` a single fused object?

The mesh statistics list one `PowerLoop` object rather than separate traces plus wires.

- Are the bond wires geometry inside it, or merged into the trace body?
- If modelled: the ribbon cross-section (we assumed **0.25 × 0.08 mm**, loop height 0.35 mm), and
  aluminium or copper?

*Why it matters:* we tested this — even at 56× the real cross-section the bond-wire model moves the
answer by only −5 %, so it is unlikely to be the explanation, but it should be ruled out on the
record rather than by our assumption.

## 6. Where are the two terminals, geometrically?

We drive the **DC+** and **DC−** connector pads (3 × 3 mm, on the top surface of the traces). Are
Q3D's source and sink the same rectangles, on the same faces?

## 7. What is `peec_coarse`?

Which code and version, and is its formulation available to read? It agrees with Q3D to ~1 % on all
three layouts but appears in the JSON as a bare label with no provenance.

*Note:* we have already ruled out one candidate explanation — if it carries a charge/capacitance
matrix that jNO lacks, that cannot matter here. The trace-to-plane capacitance is 1.45 nF, whose
impedance at 1 MHz is 663× the inductive path, with self-resonance at 25.8 MHz.

## 8. Material properties

Conductivities used for the traces, Bottom Metal and Baseplate (we use σ = 5.8 × 10⁷ S/m), and for
the bond wires (we use aluminium, 3.8 × 10⁷ S/m).

## 9. Is there a measurement?

Anything measured on hardware — an impedance-analyser loop inductance, or a switching-transient
estimate — would anchor all three codes at once and is worth more than any of the above.
