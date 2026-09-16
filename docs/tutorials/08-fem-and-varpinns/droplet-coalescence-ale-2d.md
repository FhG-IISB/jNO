# Two droplets coalesce: a sharp interface that changes topology

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/droplet_coalescence_ale_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Two drops sit in a void, their rims 0.02 apart. Surface tension pulls them together; they touch, fuse, and
the merged drop begins to round itself. The [Cahn–Hilliard tutorial](droplet-merge-chns-2d.md) does the same
physics the *capturing* way — the interface is a field, so topology costs nothing. This is the other route:
the liquid **is** the mesh, its surface is a real boundary, and the topology changes because the mesh is
re-triangulated on its own moving nodes.

$$\rho\Big(\left.\frac{\partial\mathbf u}{\partial t}\right|_X+((\mathbf u-\mathbf w)\!\cdot\!\nabla)\mathbf u\Big)=-\nabla p+\nabla\!\cdot\!\big(2\eta\,\mathbf D(\mathbf u)\big),\qquad \nabla\!\cdot\!\mathbf u=0$$

$$\mathbf T\mathbf n=-\sigma H\mathbf n \quad\text{on the whole surface},\qquad \frac{d\mathbf X}{dt}=\mathbf u \quad\text{on the whole surface}$$

with $\mathbf w$ the **mesh** velocity and $\mathbf D$ the symmetric gradient. There is no Dirichlet condition
anywhere and no pressure pin: the traction sets the pressure level by itself.

## The surface is a term, and so is its motion

The capillary traction is a boundary term; the kinematic condition is a geometry term. Both are ordinary
entries in the `jno.fem([...])` list:

```python
capillary = SIGMA * div_G(vs)                 # T n = -sigma H n, integrated by parts onto the test function
uf = u.bind(x=xs, y=ys).freeze(np.zeros((n_node, 2)))   # the solved velocity, delivered each step
fem = jno.fem([momentum, continuity, capillary, u(x0, y0)[0] - 0.0, u(x0, y0)[1] - 0.0,
               xs.d(ts) - uf[0], ys.d(ts) - uf[1]])     # the surface is Lagrangian: dX/dt = u
```

`xs.d(ts) - u` moves the mesh, and inside the weak form `xi.d(ti)` **is** $\mathbf w$, so the convective
velocity is $\mathbf u-\mathbf w$ — the ALE form. The nodal values ride with their vertices; nothing is
projected between meshes.

## How the topology changes

```python
traj = fem.solve(nonlinear=jno.solve.newton(direct=True),
                 adapt=jno.solve.remesh(alpha=1.2, every=1))
```

`alpha=` re-triangulates the **nodes** — a Delaunay triangulation filtered to the triangles whose
circumradius is below $\alpha h$, the alpha shape of the Particle Finite Element Method. When the gap
between two bodies closes below about $2\alpha h$, the bridging triangles survive the filter and two meshes
become one. Every node stays where it is, so the P1 state carries across **by identity**.

## What it produces

246 nodes, 738 dofs, 300 steps — 1.06 capillary times $t_\sigma=\sqrt{\rho R^3/\sigma}$, in about 9 minutes
on this machine (CPU), with **124 reconnections**.

| $t/t_\sigma$ | bodies | area | width | height | neck |
|---|---|---|---|---|---|
| 0.00 | **2** | 0.24972 | 0.820 | 0.400 | 0.078 |
| 0.27 | **1** | 0.25911 | 0.817 | 0.392 | 0.304 |
| 0.53 | 1 | 0.25892 | 0.806 | 0.388 | 0.388 |
| 1.06 | 1 | 0.25798 | 0.783 | 0.499 | 0.499 |

The neck opens from 0.078 to 0.499 — as tall as the drop itself — and the free surface shrinks from 2.509
to 1.950, which is what surface tension is for. Peak speed is 9.5, against the capillary velocity
$\sqrt{\sigma/\rho R}=7.07$.

Area is **two** statements, and conflating them would flatter the scheme. Merging *adds* liquid: the bridge
fills the 0.02 gap the drops left between them, +3.3 % here and nearly all of it in the first few steps.
That is the contact model, not an error in the solver. What the solver owes is conservation once the
topology has settled, and there it holds to **−0.44 %** from just after the merge to the end, across 124
re-triangulations that each rebuild the problem.

![Two drops merging, speed on the mesh each step ran on](/jNO/assets/droplet_coalescence_ale_2d.png)

*Speed $|\mathbf u|$ at $t=0$, $0.53\,t_\sigma$ and $1.06\,t_\sigma$, each drawn on the mesh that step ran
on. The drops start at rest; the flow is fastest near the neck as it opens.*

!!! danger "Three things that are silently wrong if you get them wrong"
    **The viscous term must be $2\eta\,\mathbf D(\mathbf u)\!:\!\mathbf D(\mathbf v)$.** The
    $\eta\,\nabla\mathbf u\!:\!\nabla\mathbf v$ form — correct behind Dirichlet walls, and what the
    stabilised-flow and Cahn–Hilliard tutorials use — makes the natural condition a *pseudo-traction*. A
    rigidly rotating drop then loses its spin: $\omega$ decays from 2.0 to 1.17 in 0.1 time units.

    **The SUPG/PSPG $\tau$ must be scaled for this regime.** A capillary drop is nearly inviscid and nearly
    stagnant, the opposite of advection-dominated flow. With the unscaled recipe a drop does not oscillate at
    all — the fitted frequency collapses from Lamb's 63.9 to 0.23 — while a spurious $n=4$ mode grows
    sevenfold. Scaling $\tau$ by $10^{-4}$ recovers the physics.

    **Reconnection must run every step.** At `every=2` the mesh tangles (after 352 s at $h=0.05$, 224 s at
    $h=0.04$): the neck opens faster than the triangulation is refreshed.

## Scope — what this is not

- **One capillary time, not full relaxation.** The merged drop is still a rounded dumbbell at
  $1.06\,t_\sigma$ (width 0.783 against the final circle's 0.566). Relaxing all the way takes about
  $5\,t_\sigma$ — roughly an hour here, because each reconnection re-assembles the problem
  (~2 s/step at 246 nodes). Cells are compile-time constants in the assembler; making this cheap needs
  connectivity as a runtime, padded argument.
- **The time step is tighter than the usual estimate.** $\sqrt{\rho h^3/2\pi\sigma}$ suggests $8\times10^{-4}$
  at $h=0.03$; $5\times10^{-4}$ already tangles, and $2\times10^{-4}$ tangles at $h=0.05$ even reconnecting
  every step. This run uses $10^{-4}$.
- **Merging is a mesh-length contact model.** Bodies join when their gap is comparable to the element size
  ($\approx 2\alpha h$), not by resolving film drainage. A larger $\alpha$ merges earlier and adds area
  through wider bridges: at $1.06\,t_\sigma$ the liquid has grown +3.3 % at $\alpha=1.2$, +3.6 % at 1.6 and
  +6.4 % at 2.2 — most of it the bridge that fills the initial 0.02 gap.
- **The filter can still flicker.** Re-deciding every cell each step makes a triangle sitting near
  $\alpha h$ drop out and return, so the surface loses and regains wedges. `alpha_reconnect` holds an
  existing cell to a wider threshold (hysteresis), which removes most of it: over 300 steps the worst
  one-step change in surface length falls from +25 % to +13 %, and the events from many to **2**. Those
  two remain because a Delaunay edge flip creates triangles that are *new*, and a new triangle faces
  $\alpha$ alone. Node insertion — which real PFEM does, and this does not — is what would close it.
- **2-D and P1 only.** Reconnection refuses P2 — a new edge bridging two bodies has its midpoint in the void
  — and 3-D, where Delaunay plus an alpha filter leaves sliver tetrahedra.

The same machinery is pinned by tests: `tests/test_fem_free_surface_tension.py` (the Laplace jump, per-drop
curvature, a rigid rotation, and the pseudo-traction trap), `tests/test_fem_drop_oscillation.py` (Lamb's
frequency to −0.4 % and the viscous decay to +4 %), `tests/test_fem_droplets_merge.py` and
`tests/test_reconnect_alpha.py` (the merge threshold and the alpha filter itself).

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/droplet_coalescence_ale_2d.py:code"
```

**References.** S. R. Idelsohn, E. Oñate & F. Del Pin, *Int. J. Numer. Methods Eng.* **61** (2004) 964–989
(PFEM). H. Edelsbrunner & E. P. Mücke, *ACM Trans. Graph.* **13** (1994) 43–72 (alpha shapes). H. Lamb,
*Hydrodynamics*, 6th ed. (1932), §275 (drop oscillation). E. Bänsch, *Numer. Math.* **88** (2001) 203–235
(free-surface finite elements).
