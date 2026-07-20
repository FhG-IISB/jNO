# NN-enriched finite elements: correct a network prior with a coarse solve

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/nn_enriched_fem_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A neural-network prior $u_{NN}$ is fast but only approximate. **FE-basis enrichment** certifies and
sharpens it with a finite-element correction $u_h$ on a **coarse** mesh: seek $u \approx u_{NN} + u_h$
where

$$ a(u_h, v) = (f, v) - a(u_{NN}, v) \qquad \forall v \in V_h, $$

so $u_h$ is the Galerkin projection of the prior's error $e = u_{exact} - u_{NN}$. Enrichment beats
standard FEM on the *same* mesh **iff the prior carries sub-grid content the coarse space cannot** —
which means the prior's gradient must enter the weak form **continuously at the quadrature points**.

Method: **NN-enriched finite elements** — Barucq, Faucher, Pham & Tonnoir, "Enriching continuous
Lagrange finite element approximation spaces using neural networks", 2025 (arXiv:2502.04947).

## The capability that makes it work

`jnn.grad(frozen_net, x)` assembles the network's continuous spatial gradient at the quadrature
points, landing it in the RHS (the frozen net has no live trial, so it is constant in the unknown).
You write the ordinary weak form with an **enriched trial gradient** $\nabla(u_{NN}+u_h)$:

```python
fnet = net.freeze()                          # trained prior → a known field
gx = ui.x + jnn.grad(fnet(xi, yi), xi)       # ∇(u_NN + u_h)·x̂  — the new assembler capability
gy = ui.y + jnn.grad(fnet(xi, yi), yi)
u_h = jno.fem([gx * vi.x + gy * vi.y - f * vi, u(xb, yb) - 0.0]).solve()
```

A P1-nodal `ui.freeze(values)` would **not** work here: it projects the prior onto $V_h$, carries no
sub-grid content, and gives standard FEM back exactly. The continuous `jnn.grad(net)` is essential.

## Two things the prior needs

- **Beat the spectral bias.** A plain tanh MLP cannot represent the high-frequency content the coarse
  mesh is missing (it fits it to ~100 % error). The prior uses **random Fourier features**
  ($x \mapsto [\sin 2\pi Bx, \cos 2\pi Bx]$; Tancik et al., NeurIPS 2020) to represent it.
- **Fit the gradient, not just the value.** Enrichment integrates $\nabla u_{NN}$, and differentiating
  a frequency-$K$ field amplifies error by $\sim K\pi$. So the prior is trained with a **Sobolev
  loss** — matching value *and* gradient — not value alone.

## The result

![Pointwise error: standard coarse FEM vs NN-enriched on the same mesh](/jNO/assets/nn_enriched_fem_2d.png)

On a coarse mesh (75 DOFs) that under-resolves the $\sin(5\pi x)\sin(5\pi y)$ term, standard FEM gives
rel-$L^2 \approx 2.9\times10^{-2}$; enrichment with the trained prior reaches
$\approx 3.7\times10^{-3}$ — **~8× more accurate at the same cost**. The gain scales with the prior's
gradient fidelity: an *exact* prior recovers the smooth part to near machine precision, a trained one
is bounded by how well it matches $\nabla u_{NN}$.

## What to notice

- **Coarser mesh, same accuracy.** The point of enrichment: reach a target accuracy on a mesh too
  coarse for standard FEM, by letting the network carry the sub-grid structure.
- **The correction owns the boundary.** The prior need not satisfy the BC — $u_h$ does, so a bare
  network prior is fine (here it vanishes on $\partial\Omega$, so homogeneous $u_h$ is exact).
- **`jnn.grad(frozen_net)` is general.** A known network field and its gradient are now usable
  operators in any weak form — a network source $\mathrm{net}\cdot v$, flux $\mathrm{net}\cdot\nabla v$,
  or advection — not just enrichment.
