# Three questions on the Q3D reference

**Context.** Two independent PEEC codes (jNO and pypeec 5.8.0) agree on this layout at **22.5–23.0 nH**
once each is corrected for its own known discretisation bias. Q3D reports **20.641 nH**, ~10 % below
both. jNO is separately validated against Grover's analytic rectangular-bar formula (+1.9 %) and the
exact DC resistance (−0.2 %), so this is most likely a difference in **what is reported** or **what is
in the model** — not a solver being wrong.

### 1. Are Baseplate and Bottom_Metal in the circuit, or floating?

Connected to a terminal, floating with induced eddy currents, or excluded from the RL solve?

*This is the big one.* We model both as **floating** — the ceramic isolates them — so the return
current flows through the traces and bond wires. If Q3D has them galvanically in the circuit, the loop
is far smaller and a lower inductance follows immediately.

### 2. What does `ACLoop` report — `Im(Z)/ω`, or an energy form?

And against which return path? These differ once the currents are out of phase, and every other
comparison depends on fixing the definition first.

### 3. Did the adaptive solution converge, and at what frequency?

Solution frequency, the Δ % convergence criterion, and the final delta on the last pass.

The mesh statistics show **3,416 tetrahedra with a 3.4 mm RMS edge**, against a **66 µm** skin depth at
1 MHz. Both PEEC codes were still moving under refinement at far finer discretisations, so Q3D's own
error estimate is worth knowing.

---

**And if it exists: is there a measurement?** An impedance-analyser loop inductance, or a switching
transient — anything on hardware anchors all three codes at once and is worth more than the three
questions above put together.

<details>
<summary>Already ruled out on our side — no need to ask</summary>

- **Bond wires.** Even at 56× the real cross-section they move the answer by only −5 %.
- **Capacitance.** Trace-to-plane C is 1.45 nF, 663× the inductive impedance at 1 MHz, self-resonance
  at 25.8 MHz. A charge matrix jNO lacks cannot explain this.
- **Frequency.** jNO is flat 1–10 MHz (26.448 / 26.435 / 26.430 nH).
</details>
