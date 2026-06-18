# Reaction-Diffusion with Robin BCs (FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A reaction-diffusion solve $-\Delta u + \sigma u = f$ with **mixed boundary conditions**:
Dirichlet on the left/bottom, Robin on the right/top.

## BCs are terms in the same list

A Robin condition $\partial u/\partial n + a\,u = g$ enters the weak form as the natural
surface term $(a\,u - g)\,\phi$, bound to its edge; Dirichlet conditions are `u(region) - g`
(one of them non-homogeneous, `u = y`). Every condition is just another entry passed to
`jno.fem`, classified by the region it is bound to:

```python
volume = ui.x * vi.x + ui.y * vi.y + sigma * u * vi - f * vi
robin_right = (a_r * u.bind(x=xr, y=yr) - (sin(pi * yr) + a_r * (sin(pi * yr) + yr))) * phi.bind(x=xr, y=yr)
robin_top = (a_t * u.bind(x=xt, y=yt) - (1.0 - pi * xt + a_t)) * phi.bind(x=xt, y=yt)
fem = jno.fem([volume, robin_right, robin_top, u(xl, yl) - yl, u(xbo, ybo) - 0.0], quad_degree=3)
```

## What to notice

- Robin / Neumann terms ride on `phi.bind(<edge>)`; Dirichlet on `u(<edge>) - g`.
- A reaction term is `sigma * u * vi` (value × test); diffusion is `ui.x * vi.x + ui.y * vi.y`.
- Recovers the manufactured field $u^\*=x\sin(\pi y)+y$ to rel-$L^2 \approx 7\times10^{-3}$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py"
```
