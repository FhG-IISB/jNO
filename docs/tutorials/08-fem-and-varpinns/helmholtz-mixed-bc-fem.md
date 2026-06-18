# Helmholtz with Mixed BCs (FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A 2D Helmholtz solve $-\Delta u - k^2 u = f$ ($k = 4$) with mixed boundary conditions:
Dirichlet on the left/bottom (one non-homogeneous, $u=\sin\pi x$) and Neumann on the right/top.

## Indefinite reaction + Neumann fluxes

The $-k^2 u$ term is an *indefinite* reaction (a genuine Helmholtz term); with $k=4$ it stays
below the first Dirichlet eigenvalue $\sim 2\pi^2$, so the system is still solvable. The Neumann
fluxes enter as natural terms $-g\,\phi$ on their edges:

```python
volume = ui.x * vi.x + ui.y * vi.y - k**2 * u * vi - f * vi
neumann_right = -flux_right(xr, yr) * phi.bind(x=xr, y=yr)
neumann_top = -flux_top(xt, yt) * phi.bind(x=xt, y=yt)
fem = jno.fem([volume, neumann_right, neumann_top, u(xl, yl) - 0.0, u(xbo, ybo) - sin(pi * xbo)], quad_degree=3)
```

## What to notice

- A negative reaction coefficient (`- k**2 * u * vi`) assembles like any other term.
- Neumann is `-g * phi.bind(<edge>)`; non-homogeneous Dirichlet is `u(<edge>) - g(x)`.
- Recovers $u^\*=\sin(\pi x)(\cos(\pi y)+y)$ to rel-$L^2 \approx 4\times10^{-2}$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem.py"
```
