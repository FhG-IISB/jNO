# Phase-Field Fracture, SENT — the whole study as a term list

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/phase_field_fracture_sent.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A single-edge-notched specimen pulled in tension, with the crack regularized into a damage field
$d\in[0,1]$. What makes this worth its own tutorial is not the physics — that is the standard
second-order AT2 model — but that **none of it is driven from outside `jno.fem`**. The coupled system,
the irreversible history, the bound on the damage, the alternating solver and the non-uniform load path
are all either a term in the list or a slot on `fem.solve`.

| what | how |
|---|---|
| the stress **is** the energy derivative | `sigma = jno.np.diff(psi, eps(u))` |
| irreversible history $H=\max_\tau \psi^+$ | `Hs.evolves(maximum(Hs.i(-1), psi_p))` — on a *coupled* system |
| damage is a fraction | `dm.bounds(0.0, 1.0)` — an inequality, in the term list |
| the energy is non-convex in $(u,d)$ jointly | `nonlinear=jno.solve.staggered([u, dm])` |
| the load path is not uniform | `tau=jno.solve.adaptive(limit=[(dm, 0.5)])` |
| the reaction force | `fem.eval(momentum, u_k)[grip]` |

## The model

The elastic energy is split volumetric/deviatoric so only the **tensile** part drives damage and a
closed crack still carries compression (Amor, Marigo & Maurini, *JMPS* **57** (2009) 1209–1229):

$$
\psi^+ = \tfrac12 K \langle \operatorname{tr}\varepsilon\rangle_+^2 + \mu\, \varepsilon_{\text{dev}}\!:\!\varepsilon_{\text{dev}},
\qquad
\sigma = \frac{\partial}{\partial\varepsilon}\Big[ g(d)\,\psi^+ + \psi^- \Big].
$$

That last line is written literally — `jno.np.diff` differentiates the energy, so there is no
hand-derived stress anywhere. Irreversibility comes from the history field $H=\max_\tau\psi^+$, so a
crack cannot heal (Miehe, Welschinger & Hofacker, *IJNME* **83** (2010) 1273–1311), and the damage
equation is the standard AT2 form $(G_c/\ell)\,d - G_c\ell\,\Delta d = 2(1-d)H$.

The notch is **cut**, not painted on: a real slit removed from the geometry with `jno.Shape`, so the
stress concentration at its tip is the mesh's own and the damage field needs no seeding.

```python
plate = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=h)
slit  = jno.Shape.rect(-0.01, 0.5 - w_slit, 0.5, 0.5 + w_slit, size=h)
dom   = (plate - slit).domain(tau=(0.0, 1.0, NOUT))     # tau: the pseudo-time LOAD path
```

Loading is **displacement-controlled** — `u(*ct)[1] - DELTA * ct[-1]`, the grip displacement ramped in
$\tau$. That is not a stylistic choice: under load control a brittle specimen snaps at the peak and
there is no branch left to follow, so a softening response can only be traced by prescribing the grip.

## What it shows, and what it refuses to

Damage initiates at the notch tip and grows smoothly while the load rises. That part is **stable**, and
the adaptive controller resolves it — the schedule comes back as `[0, 0.143, 0.357, 0.679, 1.0]`, taking
large steps while nothing happens and cutting as the tip loads up. The reaction rises to a peak and then
softens, read straight off `fem.eval`.

Propagation across the ligament is **not** stable for this geometry. Ask for a finer resolution of the
same path and the controller cannot deliver one:

```
tighter limit (0.25): refused, and named the unstable branch rather than returning a path
```

That is the correct answer, not a solver failure. Past the peak there is no nearby equilibrium, so no
amount of load-step refinement finds one — verified separately: a **5× finer uniform grid gives the
same jump**, the crack tip going from $x=0.05$ to $x=1.0$ in a single increment either way. Following
an unstable branch needs arc-length (dissipation-controlled) continuation, which jNO does not have.
The alternative to reporting this is grinding to the step floor and returning a plausible-looking path
that never happened.

![Damage field at three load levels, initiating ahead of the notch tip and localizing into a band on the crack plane, beside the force–displacement curve with its peak marked and the adaptive load steps shown as vertical rules.](/jNO/assets/phase_field_fracture_sent.png)

## Notes on the numbers

* The bound is **solved, not clipped** — damage stays in $[0,1]$ through the KKT conditions of the
  variational inequality, and is monotone because the history field is a running max.
* `fem.block_index(dm)` resolves the damage block: field order is *first appearance in the term walk*,
  and the degradation factor puts `dm` ahead of `u`. Never hardcode the index.
* The degradation floor $\eta$ is still needed even with the bound: at $d=1$ exactly, $(1-d)^2$ makes
  the displacement block singular. The bound keeps $d$ in range; it does not make the operator
  well-posed.
* AT2 has **no elastic threshold**, so $d>0$ everywhere from the first increment. Statements about
  localization have to be about *substantial* damage, not about where the field is nonzero.
* The two stragglers outside the crack band are the **grip corners**, where a fully clamped edge meets
  a free lateral edge and the elastic field is singular — a property of the boundary conditions rather
  than of the model.

For the **fourth-order** regularization on a biharmonic (Morley/Argyris) element, see
[Brittle Fracture — 4th-order phase-field](phase-field-fracture-2d.md); that model needs a $C^1$-ish
space, which the load-path march does not yet carry.
