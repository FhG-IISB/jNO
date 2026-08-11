# Brittle Fracture — 4th-order Phase-Field on the cheap Morley element (coupled multiphysics)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/phase_field_fracture_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A cracking solid minimizes elastic energy plus fracture (crack-surface) energy. The **variational
phase-field** model regularizes the sharp crack by a smooth damage field $d\in[0,1]$ (0 = intact, 1 = broken)
over a length $\ell$. Borden, Hughes, Landis & Verhoosel (2014) use a **fourth-order** regularization whose
crack-surface density carries a second-derivative term, giving the damage a biharmonic operator with the 1D
optimal profile

$$
d(x) = \left(1 + \tfrac{|x|}{\ell}\right) e^{-|x|/\ell}.
$$

A 4th-order weak form needs a **special biharmonic element** — plain $C^0$ Lagrange is non-convergent. Two
work: the conforming $C^1$ **Argyris** (21 DOF, accurate) and the **non-conforming Morley** triangle (6 DOF:
value at the 3 vertices + normal derivative at the 3 edge midpoints). This tutorial uses **Morley**: at ~3.5×
fewer DOF it clears the Argyris construction memory ceiling and scales to the **fine mesh a sharp crack
needs**. Because Morley is non-conforming, the biharmonic form is the **full-Hessian inner product**
`inner(hessian(d), hessian(φ))` — the Laplacian form `∫Δd·Δφ` is *singular* for Morley.

## Alternate minimization — two linear solves, coupled by two scalar fields

We use the *canonical* **alternate minimization** (Bourdin–Francfort–Marigo). With the AT2 degradation
$g(d)=(1-d)^2+\eta$ each sub-problem is **linear** given the other field:

```python
# elasticity (P1 vector, native): ∫ g(d) σ(u):ε(v) = 0,  displacement-controlled tension
fem_e = jno.fem([gd * (lam*trace(eu)*trace(ep) + 2*mu*inner(eu, ep, n_contract=2)),
                 u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0, u(xt, yt)[0] - 0.0, u(xt, yt)[1] - 1.0])

# damage (Morley): full-Hessian biharmonic term (∫D²d:D²φ), coercive ⇒ crack seeded through H
dd, dphi = d.fem_symbols(space="Morley")
fem_d = jno.fem([(2*Hpar + Gc/ell)*(di*vi) + 2*Gc*ell*(di.x*vi.x + di.y*vi.y)
                 + Gc*ell**3*inner(hessian(di,[xi,yi]), hessian(vi,[xi,yi]), n_contract=2) - 2*Hpar*vi])
```

The fields couple through two scalars: $g(d)$ degrades the stiffness, and the tensile strain-energy history
$H=\max_t \psi^+(\varepsilon(u))$ (irreversible — no crack healing) drives damage. Both flow through the
existing `jno.np.parameter` field-coefficient path. Each Morley assemble+solve is wrapped in `jax.jit` so it
compiles **once** (eager re-assembly in a Python loop is ~100× slower).

## Result

A single-edge-notched specimen is pulled in tension; the crack initiates at the notch and propagates across.

```
4th-order phase-field fracture (Morley, non-conforming):
  mesh nv=790 nc=1478  damage-dofs=3057  ℓ=0.08  h/ℓ=0.50   (Argyris OOMs at this mesh)
  Part 1  crack profile: RMS vs 4th-order=0.010  vs 2nd-order=0.130  (13.6× better)
  Part 2  crack front x: 0.40 → 1.00  (307/790 damaged)
          peak reaction 2.805e-02 at δ=0.071;  final 2.605e-03 (softening)
```

![Left: the computed Morley damage profile lies on the 4th-order (1+r/ℓ)e^(−r/ℓ) curve and off the kinked 2nd-order one. Middle: a sharp crack spanning the notched specimen on a fine mesh (nv≈790). Right: brittle rise-to-peak-then-soften force–displacement response.](/jNO/assets/phase_field_fracture_2d.png)

The controlled profile check (left) confirms that the non-conforming Morley element **captures the smooth
4th-order profile** to $\text{RMS}\approx10^{-2}$ — an order of magnitude closer than the 2nd-order shape.
Because Morley is cheap (6 DOF vs Argyris's 21), the coupled solve (middle) runs on a **fine mesh
$nv\approx790$ that the conforming $C^1$ element cannot reach** (its construction OOMs there), giving a
markedly **sharper crack** than a coarse conforming solve — the crack band is set by $\ell$, and Morley lets
you afford the mesh that resolves a small $\ell$. The reaction force (right) shows the textbook brittle
response: a peak as the crack advances, then softening (the reaction is an energy-based proxy
$\propto \sum g(d)\,\psi^+\,\mathrm{area}$, so its *shape* is meaningful rather than its calibrated magnitude).

!!! note "References"
    L.S.D. Morley, *The triangular equilibrium element in the solution of plate bending problems*, Aeronautical
    Quarterly **19** (1968) 149–169 — the non-conforming element. M.J. Borden, T.J.R. Hughes, C.M. Landis,
    C.V. Verhoosel, CMAME **273** (2014) 100–118 — the fourth-order phase-field model. B. Bourdin, G.A.
    Francfort, J.-J. Marigo, *The variational approach to fracture*, J. Elasticity **91** (2008) 5–148 —
    alternate minimization. C. Miehe, M. Hofacker, F. Welschinger, CMAME **199** (2010) 2765–2778 — the
    tension/compression (spectral) split.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/phase_field_fracture_2d.py:code"
```
