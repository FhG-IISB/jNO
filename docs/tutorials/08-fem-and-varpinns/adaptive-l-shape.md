# Adaptive mesh refinement (L-shape re-entrant corner)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Starting from a *coarse* uniform mesh, `jno` repeatedly

$$\text{solve}\ \to\ \text{estimate (Zienkiewicz–Zhu)}\ \to\ \text{mark (Dörfler)}\ \to\ \texttt{domain.refine}$$

so the mesh automatically concentrates elements at the re-entrant corner, where the Laplace solution
carries the classic $r^{2/3}$ singularity. The model problem is $-\nabla^2 u = 0$ on the L-shape with
Dirichlet data equal to the exact singular mode $u=r^{2/3}\sin(2\varphi/3)$ about the corner
$(0.5,0.5)$; $u$ is harmonic, so the only error comes from resolving the corner.

## The loop, in one call

`FEM.solve(adapt=...)` drives the whole loop and rebinds the FEM (and mutates the domain) to the final
adapted mesh; the per-round trace is on `fem.adapt_history`. Or compose the public building blocks
directly:

```python
from jno.utils.solver.fem_adapt import dorfler_mark, size_field_from_marks, zz_error_indicators

d = jno.domain(jno.domain.l_shape(size=1.0, mesh_size=0.3))
for _ in range(n_rounds):
    fem = jno.fem(build_constraints(d))          # constraints reference the domain -> reassemble
    u   = np.asarray(fem.solve()).reshape(-1)     # scalar P1: one value per vertex
    eta, est = zz_error_indicators(d, u)          # recovered-gradient error per element
    marked   = dorfler_mark(eta, theta=0.6)       # bulk-mark the worst elements
    if marked.size == 0:
        break
    d.refine(size_field_from_marks(d, marked, refine_factor=1.7))   # in-place metric remesh (mmg)
```

Refinement is an **outer** Python loop — the mesh is a static argument to the assembler, so
`domain.refine` mutates the domain in place and re-calling `jno.fem(constraints)` reassembles on the
new mesh automatically. Differentiability is exact on the *frozen* adapted mesh, not through the discrete refinement.

## The result

![Filmstrip of the L-shape mesh refining round by round with the singular solution beside it; elements
concentrate sharply at the re-entrant corner while the error estimate falls.](/jNO/assets/adaptive_l_shape.gif)

The Zienkiewicz–Zhu indicator concentrates ~13× at the corner, and the adaptive run reaches a lower
error estimate at ~455 DOFs than a uniform mesh at ~828 DOFs.

## What to notice

- **The estimator is [Zienkiewicz–Zhu](https://doi.org/10.1002/nme.1620240206)** — an inexpensive
  recovered-gradient indicator; **Dörfler** bulk-marking then selects the smallest set of elements
  carrying a fixed fraction of the total.
- **The re-entrant corner is pinned** during remeshing (mmg `set_corners`), so the singularity stays
  put and the benchmark is honest.
- **Works in 2D and 3D** — the same loop drives isotropic refinement on triangle *and*
  tetrahedron meshes (`domain.refine` / `FEM.solve(adapt=...)` on a `jno.domain.cube(...)`
  remeshes via mmg3d, preserving the polyhedral geometry). Scalar P1 for now (one DOF/vertex);
  anisotropic metrics are 2D-only so far.
- **Knowing when to stop:** `AdaptSpec` takes `tol` (stop once the error estimate falls
  below it), `max_dofs` / `max_iters` (budget caps), and `eps` (stop once the estimate stops
  improving between rounds) — so the loop refines only until the solution is *good enough*.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py:code"
```
