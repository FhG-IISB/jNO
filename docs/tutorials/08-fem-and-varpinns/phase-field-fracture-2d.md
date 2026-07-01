# Brittle Fracture — 4th-order Phase-Field on the C¹ Element (coupled multiphysics)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/phase_field_fracture_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A cracking solid minimizes elastic energy plus fracture (crack-surface) energy. The **variational
phase-field** model regularizes the sharp crack by a smooth damage field $d\in[0,1]$ (0 = intact, 1 = broken)
over a length $\ell$. Borden, Hughes, Landis & Verhoosel (2014) introduced a **fourth-order** regularization
whose crack-surface density carries a $(\Delta d)^2$ term, giving the damage operator the biharmonic
$(1-\ell^2\Delta)^2 d$. Its 1D optimal crack profile is

$$
d(x) = \left(1 + \tfrac{|x|}{\ell}\right) e^{-|x|/\ell},
$$

which is **$C^1$-continuous at the crack** ($d'(0)=0$) — unlike the 2nd-order model's kinked $e^{-|x|/\ell}$.
Representing that smooth profile *requires* a $C^1$ element: **this is a problem the Argyris element solves
that $C^0$ Lagrange cannot.** (By $\Gamma$-convergence all three terms balance at $\sim d^2/\ell$, so
$\Delta^2 d$ is leading-order, not a small correction.)

## Alternate minimization — two linear solves, coupled by two scalar fields

A monolithic Argyris + Lagrange system is not assembled here, so we use the *canonical* **alternate
minimization** (Bourdin–Francfort–Marigo). With the AT2 degradation $g(d)=(1-d)^2+\eta$ each sub-problem is
**linear** given the other field:

```python
# elasticity (P1 vector, native): ∫ g(d) σ(u):ε(v) = 0,  displacement-controlled tension
fem_e = jno.fem([gd * (lam*trace(eu)*trace(ep) + 2*mu*inner(eu, ep, n_contract=2)),
                 u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0, u(xt, yt)[0] - 0.0, u(xt, yt)[1] - 1.0])

# damage (Argyris C¹): Borden 4th-order, coercive ⇒ no essential BC; crack seeded through H
fem_d = jno.fem([(2*Hpar + Gc/ell)*(di*vi) + 2*Gc*ell*(di.x*vi.x + di.y*vi.y)
                 + Gc*ell**3*(laplacian(di,[xi,yi])*laplacian(vi,[xi,yi])) - 2*Hpar*vi])
```

The fields couple through two scalars: $g(d)$ degrades the stiffness, and the tensile strain-energy history
$H=\max_t \psi^+(\varepsilon(u))$ (irreversible — no crack healing) drives damage. Both flow through the
existing `jno.np.parameter` field-coefficient path. Each Argyris assemble+solve is wrapped in `jax.jit` so it
compiles **once** (eager re-assembly in a Python loop is ~100× slower).

## Result

A single-edge-notched specimen is pulled in tension; the crack initiates at the notch and propagates across.

```
4th-order phase-field fracture (Argyris C¹):
  mesh nv=303 nc=544  damage-dofs=6363   ℓ=0.15  h/ℓ=0.47
  Part 1  crack profile: RMS vs 4th-order=0.014  vs 2nd-order=0.183  (12.9× better);  d'(0)=0.000 (4th→0)
  Part 2  crack front x: 0.50 → 1.00  (228/303 damaged)
          peak reaction 1.622e-02 at δ=0.061;  final 2.869e-03 (softening)
```

![Left: the computed Argyris damage profile lies on the 4th-order (1+r/ℓ)e^(−r/ℓ) curve (C¹-smooth peak) and off the kinked 2nd-order one. Middle: the regularized crack spanning the notched specimen. Right: brittle rise-to-peak-then-soften force–displacement response.](/jNO/assets/phase_field_fracture_2d.png)

The controlled profile check (left) confirms the $C^1$ requirement quantitatively: the Argyris solution matches
the 4th-order profile to $\text{RMS}\approx10^{-2}$ — **13× closer** than the 2nd-order shape — with $d'(0)=0$,
the smooth peak a $C^0$ element cannot represent. The coupled solve (middle) drives a real crack across the
specimen, and the reaction force (right) shows the textbook brittle response: a peak as the crack advances,
then softening. The damage band is regularized over $\ell$ on a coarse tutorial-budget mesh — a finer mesh
sharpens it — and the reaction is an energy-based proxy ($\propto \sum g(d)\,\psi^+\,\mathrm{area}$), so its
*shape* is meaningful rather than its calibrated magnitude.

!!! note "References"
    M.J. Borden, T.J.R. Hughes, C.M. Landis, C.V. Verhoosel, *A higher-order phase-field model for brittle
    fracture: Formulation and analysis within the isogeometric analysis framework*, Comput. Methods Appl.
    Mech. Engrg. **273** (2014) 100–118 — the fourth-order regularization. B. Bourdin, G.A. Francfort,
    J.-J. Marigo, *The variational approach to fracture*, J. Elasticity **91** (2008) 5–148 — alternate
    minimization. C. Miehe, M. Hofacker, F. Welschinger, CMAME **199** (2010) 2765–2778 — the
    tension/compression (spectral) split. J.H. Argyris, I. Fried, D.W. Scharpf (1968); R.C. Kirby, SMAI J.
    Comput. Math. **4** (2018) — the $C^1$ element.
