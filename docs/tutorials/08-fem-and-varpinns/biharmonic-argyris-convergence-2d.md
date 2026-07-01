# Conforming Biharmonic — Argyris C¹ Element (order-of-accuracy study)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/biharmonic_argyris_convergence_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The biharmonic operator $\Delta^2 u = f$ (Kirchhoff plate bending, Cahn–Hilliard, Stokes stream-function)
needs an **$H^2$-conforming** finite element: the weak form $\int \Delta u\,\Delta v = \int f v$ is only
convergent if the discrete normal derivative $\partial u/\partial n$ is *continuous* across element edges.
Standard Lagrange is $C^0$, **not** $C^1$, so $\int \Delta u\,\Delta v$ over $P_k$ is non-conforming and does
not converge. The **Argyris** quintic triangle — 21 degrees of freedom: the value, gradient and Hessian at
each vertex, and the normal derivative at each edge midpoint — is the classical $C^1$-conforming element
that does. Select it with `space="Argyris"`.

## The conforming weak form

The whole problem is two terms in `jno.fem` — the biharmonic bilinear form and the clamped boundary data:

```python
u, phi = d.fem_symbols(space="Argyris")
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
f = 4.0 * PI**4 * sin(PI * xi) * sin(PI * yi)         # f = Δ²u*  for u* = sin(πx)sin(πy)
g = sin(PI * xb) * sin(PI * yb)                        # clamped to the exact trace
fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi, u(xb, yb) - g])
sol = fem.solve()
```

For an Argyris field the Dirichlet term `u(boundary) - g` imposes the **proper clamped** condition by
*autodiff* of the known field `g`: it pins the value and gradient (so $u=g$ and $\partial u/\partial n =
\partial g/\partial n$) and the edge normal-derivative, while leaving the boundary curvature $\partial^2
u/\partial n^2$ **free** — the physical clamped-plate BC, where the normal–normal second derivative is a
*natural* condition recovered by the solve. For a **manufactured** solution the pinned value and gradient are
exact and the freed curvature converges to $\partial^2 u^\ast/\partial n^2$, so the discrete solution still
converges to $u^\ast$ at the optimal rate. (The companion [clamped Kirchhoff plate](clamped-kirchhoff-plate-2d.md)
tutorial reads that freed curvature off the solution as the clamp reaction moment.)

## Order of accuracy

We verify the *a-priori* theory. For a degree $k=5$, $C^1$ element on a 4th-order ($2m=4$, $m=2$) problem,

$$
\bigl\lVert \Delta(u-u_h)\bigr\rVert_{L^2} = O(h^{k-1}) = O(h^4),
\qquad
\lVert u-u_h\rVert_{L^2} = O(h^{k+1}) = O(h^6)\ \text{(Aubin–Nitsche)}.
$$

Error norms are computed by per-cell Gauss quadrature, reconstructing $u_h$ and $\Delta u_h$ from the
solution DOFs with the **same** push-forward the assembler uses — i.e. we audit the discrete solution, not
a hand-built field. Solving on a sequence of unstructured meshes recovers exactly these rates:

```
         h    dofs       L2 err   rate   energy err   rate
    0.5303      97    1.425e-04     --    2.926e-02     --
    0.4226     165    1.451e-05  10.07    8.879e-03   5.25
    0.3112     251    2.931e-06   5.23    3.590e-03   2.96
    0.2021     495    1.967e-07   6.26    6.265e-04   4.04

  least-squares order:  L2 ≈ 6.59  (theory 6)   energy ≈ 3.87  (theory 4)
```

![Argyris C¹ biharmonic convergence: measured L² and energy errors against the O(h⁶) and O(h⁴) reference slopes.](/jNO/assets/biharmonic_argyris_convergence_2d.png)

The measured orders (energy $\approx 3.87$, $L^2 \approx 6.6$) sit right on the optimal $O(h^4)$ / $O(h^6)$
theory — the high-order convergence a $C^0$ Hessian assembly **cannot** deliver. The same element composes
with nonlinear ($\Delta^2 u + u^3 = f$) and transient (the dissipative biharmonic heat flow) solves; see
`tests/test_fem_argyris.py`.

!!! note "References"
    J.H. Argyris, I. Fried, D.W. Scharpf, *The TUBA family of plate elements for the matrix displacement
    method*, Aeronautical Journal **72** (1968) 701–709 — the quintic $C^1$ triangle.
    R.C. Kirby, *A general approach to transforming finite elements*, SMAI J. Comput. Math. **4** (2018)
    197–224 — the affine DOF-transform $M(\text{cell})$.
    P.G. Ciarlet, *The Finite Element Method for Elliptic Problems* (2002), §6.
