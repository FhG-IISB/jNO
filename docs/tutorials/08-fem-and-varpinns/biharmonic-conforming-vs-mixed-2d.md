# Why C¹? Conforming Argyris vs. the Mixed Method (biharmonic)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/biharmonic_conforming_vs_mixed_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A 4th-order operator $\Delta^2 u = f$ has two routes through `jno.fem`, and this tutorial puts them
head-to-head on the **same** manufactured problem to answer a practical question: *is the $C^1$ element
worth the trouble?*

1. **Conforming** — the $C^1$ **Argyris** element ($H^2$-conforming): write $\int \Delta u\,\Delta v$
   directly.
2. **Mixed (Ciarlet–Raviart)** — introduce $w=\Delta u$ and solve two coupled 2nd-order problems with
   ordinary $C^0$ Lagrange (here $P_2$). No $C^1$ element needed, but the auxiliary variable costs accuracy.

```python
# conforming: one bilinear form, clamped to the known u*
fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi, u(xb, yb) - g])

# mixed: w = Δu and Δw = f, simply supported u = w = 0 on ∂Ω
fem = jno.fem([(ui.x*pi.x + ui.y*pi.y) + wi*pi,  wi.x*qi.x + wi.y*qi.y + f*qi,  u(xb,yb)-0.0, w(xb,yb)-0.0])
```

With $u^\ast=\sin(\pi x)\sin(\pi y)$ on the unit square (so $u^\ast=\Delta u^\ast=0$ on $\partial\Omega$ —
simply supported), the displacement nodal-$L^2$ error at *comparable cost* is:

```
       h | Argyris dofs      L2 err | mixed dofs      L2 err | Argyris/mixed
   0.420 |          165   9.140e-06 |        130   1.574e-03 |         172×
   0.300 |          251   1.992e-06 |        202   7.646e-04 |         384×
   0.210 |          373   3.731e-07 |        306   3.255e-04 |         872×
```

For essentially the same number of degrees of freedom, the conforming $C^1$ element is **two to three
orders of magnitude more accurate**, and the gap *widens* under refinement (it converges faster too — the
dedicated [convergence study](biharmonic-argyris-convergence-2d.md) measures Argyris's asymptotic $L^2$
order $\approx 6$ on finer meshes). That accuracy-per-DOF is exactly why the $C^1$ element earns its extra
machinery — the price of the mixed method's "no $C^1$ element needed" convenience is paid in accuracy.

!!! note "References"
    P.G. Ciarlet, P.-A. Raviart, *A mixed finite element method for the biharmonic equation*, in Mathematical
    Aspects of Finite Elements in PDE (1974) 125–145. J.H. Argyris, I. Fried, D.W. Scharpf (1968);
    R.C. Kirby, SMAI J. Comput. Math. **4** (2018).
