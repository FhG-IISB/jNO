# Inverse Plate Stiffness — recover a spatially-varying $k(x)$ from a deflection

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/inverse_plate_stiffness_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A plate with a hidden defect — a locally stiffer or thinner region — has a spatially varying flexural
rigidity $k(x)$. Its deflection under a known load obeys $k(x)\,\Delta^2 w = q$. The **inverse** problem: given
a *measured* deflection $w$, recover the field $k(x)$. This is a genuinely 4th-order field-inversion, and it is
well posed here: under a uniform load, $\Delta^2 w = q/k > 0$ everywhere, so the deflection is sensitive to $k$
at every point in the domain.

It exercises the deepest capabilities of the C¹ stack *together*:

- the **Argyris** $C^1$ element for the biharmonic forward solve,
- a spatially varying **P1 field parameter** `k = jno.np.parameter(kf)` — its unknowns live at the mesh
  vertices, interpolated with $P_1$ shape functions independently of the Argyris trial,
- a **differentiable** `fem.solve()` — the whole system is re-assembled at each $k$ and reverse-mode
  differentiated, so gradients of the deflection with respect to $k$ flow through the entire solve.

```python
kf, _ = d.fem_symbols()                 # P1 stiffness field (nodes = mesh vertices)
k = jno.np.parameter(kf, name="k")
u, phi = d.fem_symbols(space="Argyris")
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
fem = jno.fem([k * (laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi])) - q * vi, u(xb, yb) - 0.0])

# recover k(x) end-to-end through crux, from a flat wrong guess k = 1
crux = jno.core([(fem.solve(solver) - w_obs).mse], domain=_DUMMY)
crux.solve(350)
```

## Recovery

We plant a stiffer central patch $k^\ast(x) = 1 + 0.6\sin(\pi x)\sin(\pi y)$, generate the clamped-plate
deflection under a uniform load $q=1$, then hand only that deflection to `crux`, starting from a flat guess
$k = 1$:

```
Inverse plate stiffness (Argyris C¹ + P1 field parameter):
  mesh nodes = 30   recovered k(x) rel-L² error: 2.265e-02
  k* range [1.000, 1.586]   recovered [0.925, 1.585]
```

![True k*(x) vs recovered k(x): the stiffer central patch is reconstructed from the deflection to ~2% relative L² error.](/jNO/assets/inverse_plate_stiffness_2d.png)

The hidden stiffness field is recovered to $\sim 2\%$ relative $L^2$ error — the central stiff patch and its
peak magnitude both come back (recovered max $1.585$ vs. true $1.586$) — in a few seconds. The forward solve
being a *conforming* $C^1$ biharmonic is what makes the sensitivity of $w$ to $k$ clean enough to invert; the
same differentiable-solve machinery underlies the [inverse diffusivity](inverse-diffusivity-field.md) and
[transient inverse](transient-inverse-heat.md) tutorials, here lifted to a 4th-order operator and a field
parameter that lives on a different element than the trial.

!!! note "References"
    The differentiable-solve / inverse pattern is documented in
    [Inverse problems](/jNO/inverse-problems/). J.H. Argyris, I. Fried, D.W. Scharpf (1968);
    R.C. Kirby, SMAI J. Comput. Math. **4** (2018) — the $C^1$ element.
