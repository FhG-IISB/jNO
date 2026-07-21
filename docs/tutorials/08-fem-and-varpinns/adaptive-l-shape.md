# Adaptive mesh refinement (L-shape re-entrant corner)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The Laplace solution on the L-shape carries the classic $r^{2/3}$ re-entrant-corner singularity. With
Dirichlet data equal to the exact singular mode $u = r^{2/3}\sin(2\varphi/3)$ about the corner
$(0.5,0.5)$, $u$ is harmonic — so **all** the discretization error comes from resolving that one corner.
This tutorial resolves it two ways on the *same* problem, measured by the energy-norm error
$E-E_\text{ref}$ with $E=\tfrac12\int|\nabla u_h|^2$ and $E_\text{ref}$ from a fine mesh:

- **h-adaptivity** — *add* elements at the corner (the DOF count grows).
- **differentiable r-adaptivity** — *relocate* a fixed set of nodes down the energy gradient, computed
  **through the differentiable solve** (the DOF count is fixed).

![Two meshes refining side by side: on the left, h-adaptivity adds triangles at the re-entrant corner in
discrete jumps; on the right, r-adaptivity slides a fixed set of interior vertices toward the corner
continuously.](/jNO/assets/adaptive_l_shape.gif)

The left panel jumps at each discrete remesh — new elements, more DOFs. The right panel flows
continuously: the same nodes slide toward the corner, connectivity and DOF count unchanged.

## Both live in the `adapt=` slot

**h-adaptivity is one call.** `FEM.solve(adapt=AdaptSpec(...))` runs the whole classical loop internally
— solve → [Zienkiewicz–Zhu](https://doi.org/10.1002/nme.1620240206) estimate → Dörfler mark → local mmg
remesh — then rebinds the FEM and mutates the domain to the final adapted mesh, recording each round on
`fem.adapt_history`:

```python
sol = fem.solve(adapt=AdaptSpec(theta=0.6, max_iters=4, refine_factor=1.7))   # ADD elements
```

**r-adaptivity is the same slot with `relocate=True`.** First make the interior vertices trainable:
`.trainable()` on a spatial coordinate turns that region's mesh vertices into a design variable, so the
assembler routes them into the element geometry and `fem.solve()` becomes differentiable in the node
positions — the keystone $\partial(\texttt{fem.solve})/\partial X$. The driver then moves them down the
FE-energy gradient with a backtracking mesh-validity line search:

```python
xm, ym, _ = d.variable("mov", where=interior, split=True)
xm.trainable(name="ix")            # literal, per component — x and y are separate coordinates
ym.trainable(name="iy")
...
sol = fem.solve(adapt=AdaptSpec(relocate=True, max_iters=60, lr=3e-3))         # RELOCATE nodes
```

No new DOFs, fixed connectivity, one JAX graph — no remeshing. The boundary is left fixed, so the L-shape
itself never changes; only its interior nodes move. If `relocate=True` and no coordinate is tagged
`.trainable()`, it fails loud.

## The result

![The h-refined mesh (elements added at the corner) beside the r-relocated mesh (a fixed node set pulled
toward the corner), and both mechanisms on one energy-norm-error-versus-DOF axis.](/jNO/assets/adaptive_l_shape.png)

From a 92-DOF coarse start (error $4.5\times10^{-3}$): h-adaptivity reaches $8.4\times10^{-4}$ at 161 DOFs
(+69, **81 % lower**); r-adaptivity reaches $2.0\times10^{-3}$ at the **same 92 DOFs** (**55 % lower**).
h-adaptivity buys accuracy with DOFs; r-adaptivity buys it by moving the DOFs you already have.

## What to notice

- **The h-estimator is [Zienkiewicz–Zhu](https://doi.org/10.1002/nme.1620240206)** — an inexpensive
  recovered-gradient indicator; **Dörfler** bulk-marking then selects the smallest set of elements
  carrying a fixed fraction of the total error. The remesh is a discrete, non-differentiable outer loop.
- **The r-objective is the FE energy, minimized through the solve.** Because the assembly geometry is pure
  JAX in the node coordinates, $\partial(\texttt{fem.solve})/\partial X$ flows through the existing
  differentiable solve — the mechanism behind differentiable r-adaptivity (cf.
  [G-Adaptivity](https://arxiv.org/abs/2407.04516), ICML 2025).
- **Mesh validity is step control, not a loss term.** The relocation checks $\det J>0$ on the *joint* step
  and backtracks; a mesh-tangling barrier folded into the loss overshoots the huge near-corner gradients
  before it can react. Validity belongs in the line search, so the driver hand-rolls backtracking.
- **The re-entrant corner is pinned** for both mechanisms — h-adaptivity via mmg `set_corners`,
  r-adaptivity by leaving the boundary fixed — so the singularity stays put and the benchmark is honest.
- **`relocate=True` composes across modes** — linear, nonlinear, transient, periodic, vector and complex
  problems all relocate (the energy objective sums over every field block); only complex-transient is not
  yet supported, and it fails loud.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py:code"
```
