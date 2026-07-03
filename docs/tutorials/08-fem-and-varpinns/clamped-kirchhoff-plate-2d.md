# Clamped Kirchhoff Plate under Uniform Pressure (Timoshenko benchmark)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/clamped_kirchhoff_plate_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A thin plate under uniform transverse pressure $q$, clamped on all four edges, bends according to
$D\,\Delta^2 w = q$ ($D$ = flexural rigidity). A clamped edge imposes $w=0$ and $\partial w/\partial n=0$ — but
it does **not** flatten the plate: the boundary curvature $\partial^2 w/\partial n^2$ is nonzero and equals the
reaction bending moment $M_n = -D\,\partial^2 w/\partial n^2$ the clamp carries. That curvature is a *natural*
boundary condition the solve determines, and it is exactly what the Argyris **proper clamped BC** leaves free:

```python
u, phi = d.fem_symbols(space="Argyris")
q = 1.0 + 0.0 * xi                                    # uniform pressure, D = 1
fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - q * vi,
               u(xb, yb) - 0.0, u.dn(xb, yb) - 0.0])  # clamped = deflection w=0 AND rotation ∂w/∂n=0
sol = fem.solve(linear=jno.solve.lu())                # sparse-direct: O(nnz) memory on the refined mesh
```

Clamped composes the two essential plate traces — `u(reg)-g` (deflection) and `u.dn(reg)-h` (rotation) — while
$\partial^2 w/\partial n^2$ is always left free. Dropping the rotation term gives a **simply-supported** plate
instead (`w_max = 0.00406`); dropping both gives a free edge. See the *Plate boundary conditions* table in the
[FEM guide](../../fem.md).

## A real benchmark, not a manufactured solution

This is the textbook **clamped square plate under uniform load** (Timoshenko & Woinowsky-Krieger, *Theory of
Plates and Shells*, 2nd ed. 1959, Table 35). With $D=1$, $q=1$ on the unit square the tabulated coefficients
are a center deflection $w_\max = 0.00126\,qa^4/D$ and an edge-midpoint moment $M = 0.0513\,qa^2$ (which *is*
the freed boundary curvature $\partial^2 w/\partial n^2$). We solve on a refined mesh and recover both:

```
Clamped square plate under uniform pressure (Argyris C¹, proper clamped BC):
  mesh nv=379  dofs=7959
  center deflection   w_max = 1.2569e-03   (Timoshenko 1.26e-03,  ratio 0.998)
  edge-mid curvature ∂²w/∂n² = 0.0511     (Timoshenko 0.0513,     ratio 0.995)
  peak bending moment |M| = 0.0454 at the clamped edge (stress concentration)
```

![Left: the 3D deflected plate. Middle: the bending-moment field, read directly off the C¹ Hessian DOFs, concentrating at the clamped-edge midpoints. Right: the freed boundary curvature ∂²w/∂n² along a clamped edge, peaking at Timoshenko's edge-moment coefficient 0.0513.](/jNO/assets/clamped_kirchhoff_plate_2d.png)

Both classical coefficients come back to **within 0.5 %**. The middle panel is the real payoff of a $C^1$
element: it carries the **full Hessian as degrees of freedom**, so the entire bending-moment field
$M(x,y) = -D(\nabla^2 w + \nu\,\nabla^2 w^{\mathsf T})$ is read straight off the solution — no post-hoc
differentiation — and it correctly concentrates at the clamped-edge midpoints, exactly where the plate is most
stressed and would first crack. The right panel confirms the freed boundary curvature carries the clamp's
reaction moment, landing on Timoshenko's tabulated value. (For a *manufactured* companion where the clamped
data is exact, see the [order-of-accuracy study](biharmonic-argyris-convergence-2d.md).)

!!! note "References"
    S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed. (1959), §30 & Table 35 —
    clamped-plate coefficients. J.H. Argyris, I. Fried, D.W. Scharpf, *The TUBA family of plate elements*,
    Aeronautical Journal **72** (1968) 701–709. R.C. Kirby, *A general approach to transforming finite
    elements*, SMAI J. Comput. Math. **4** (2018) 197–224.
