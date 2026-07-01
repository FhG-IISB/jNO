# Clamped Kirchhoff Plate — the *proper* clamped BC (free boundary curvature)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/clamped_kirchhoff_plate_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A thin plate bends under a transverse load $q$ according to the biharmonic equation $D\,\Delta^2 w = q$,
with $D$ the flexural rigidity. A **clamped** edge holds the plate flat: $w = 0$ and $\partial w/\partial n = 0$
on $\partial\Omega$. What it does **not** do is flatten the curvature — a clamped plate still bends right up
to the wall, so the boundary curvature $\partial^2 w/\partial n^2$ is **nonzero**. It is proportional to the
reaction bending moment $M_n = -D\,\partial^2 w/\partial n^2$ that the clamp exerts on the plate, and it is a
*natural* boundary condition: the solve determines it, you do not prescribe it.

This is exactly the distinction the Argyris **proper clamped BC** captures. Writing `u(region) - g` pins the
value and gradient (and edge normal-derivative) but **leaves $\partial^2 u/\partial n^2$ free**:

```python
u, phi = d.fem_symbols(space="Argyris")
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
# q = Δ²(sin²πx · sin²πy): a clamped field (w = ∂w/∂n = 0 on ∂Ω) with nonzero boundary curvature
fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - q * vi, u(xb, yb) - 0.0])
```

Pinning the *full* $C^1$ trace instead (forcing $\partial^2 w/\partial n^2 = 0$) would annihilate the clamp
moment and artificially stiffen the plate — the wrong physics. On an axis-aligned edge the $(n,t)$ frame *is*
the $(x,y)$ frame, so $\partial^2 w/\partial n^2$ is a single Argyris Hessian DOF ($\partial_{xx}$ on an
$x=\text{const}$ edge), and proper-clamped simply skips it; a corner, where two edges meet, is fully clamped
automatically. (A non-axis-aligned boundary edge would need the $(n,t)$ rotation and is rejected loudly rather
than silently mistreated.)

## Reading the clamp reaction moment off the solution

We manufacture $w^\ast = \sin^2(\pi x)\sin^2(\pi y)$ — clamped on the unit square ($w^\ast = \partial w^\ast/\partial n = 0$
on $\partial\Omega$) yet with nonzero edge curvature — and take $q = \Delta^2 w^\ast$ (with $D=1$). After the
solve, the boundary curvature is read **directly off the very DOF proper-clamped left free**: on the $x=0$ edge,
$\partial^2 w/\partial n^2 = \partial_{xx} w$ is the Argyris $\partial_{xx}$ DOF at each edge vertex.

```
Clamped Kirchhoff plate (Argyris C¹, proper clamped BC):
  deflection rel-L² error:  h=0.28: 4.047e-05   h=0.18: 2.856e-06
  clamped-edge curvature ∂²w/∂n² (x=0): max |computed| = 19.723  (a real clamp: nonzero)
  vs exact 2π²sin²(πy):     rel error 7.288e-04
```

![Clamped plate deflection and the clamp-edge reaction curvature ∂²w/∂n²: the freed Argyris DOF recovers the exact 2π²sin²(πy), while the full-trace over-pin would wrongly give zero.](/jNO/assets/clamped_kirchhoff_plate_2d.png)

The deflection converges to $w^\ast$, and the freed boundary curvature recovers the exact
$\partial^2 w^\ast/\partial x^2\big|_{x=0} = 2\pi^2\sin^2(\pi y)$ to $\sim 7\times10^{-4}$ — the reaction moment a
real clamp carries. Had we pinned the full trace, that curve would sit flat at zero (the dashed line): a
plausible-looking but physically wrong answer. This is why the essential BC for a genuine clamped BVP must free
the boundary curvature, in contrast to a **manufactured** convergence study where pinning the exact full trace
is also consistent (see the [order-of-accuracy study](biharmonic-argyris-convergence-2d.md)).

!!! note "References"
    S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed. (1959) — clamped-plate
    bending and edge reaction moments. J.H. Argyris, I. Fried, D.W. Scharpf, *The TUBA family of plate
    elements*, Aeronautical Journal **72** (1968) 701–709. R.C. Kirby, *A general approach to transforming
    finite elements*, SMAI J. Comput. Math. **4** (2018) 197–224.
