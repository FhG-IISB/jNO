# Linear-Elastic Cantilever Beam (vector FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/linear_elasticity_cantilever.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A real solid-mechanics solve: a slender cantilever (length $L=10$, height $H=1$) clamped at the
root and loaded by a downward shear traction on the tip. The unknown is the **vector**
displacement $u=(u_x,u_y)$.

## Vector field + traction + isotropic elasticity

`fem_symbols(value_shape=(2,), order=2)` gives a P2 vector field (constant-strain TRI3 is too
stiff in bending). The weak form is the isotropic elasticity bilinear form
$\lambda(\nabla\!\cdot u)(\nabla\!\cdot\phi)+2\mu\,\varepsilon(u){:}\varepsilon(\phi)$; the tip
load is a traction term, and the root is clamped with an all-component condition:

```python
u, phi = d.fem_symbols(value_shape=(2,), order=2)
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
traction = -1.0 * inner(jnp.array([0.0, -q]), phi.bind(x=xr, y=yr), n_contract=1)
fem = jno.fem([weak, u(xl, yl) - (0.0, 0.0), traction])
sol = jnp.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2)
```

The slender-beam P2 stiffness matrix is too ill-conditioned for a Jacobi-preconditioned Krylov
solve in float32, so the solve picks the sparse-direct `lu` slot; the interleaved solution is
read with `.reshape(-1, 2)` (`[:, 0]` = $u_x$, `[:, 1]` = $u_y$), aligned with `fem.points`.

## What to notice

- `value_shape=(2,)` makes `u` a vector; `vi.component(i)` / `[i]` index components.
- A traction is `inner(t, phi.bind(<edge>), n_contract=1)`; an all-component clamp is `u(root) - (0, 0)`.
- The tip deflection matches Euler-Bernoulli $\delta=PL^3/3EI$ to ~1% (a slender-beam
  approximation; Timoshenko & Goodier, *Theory of Elasticity*).

## Result

![Top: the cantilever mesh deformed under the tip load (displacement exaggerated x2) coloured by displacement magnitude, with the undeformed outline behind it. Bottom: bar chart comparing the FEM tip deflection to the Euler-Bernoulli value.](/jNO/assets/linear_elasticity_cantilever.png)

The tip bends downward under the end shear (top, displacement shown $\times2$ for visibility, coloured by $|u|$). The computed tip deflection is $0.4031$ against the Euler-Bernoulli prediction $0.4000$ — a ratio of $1.008$, i.e. the P2 solve matches slender-beam theory to $\sim1\%$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/linear_elasticity_cantilever.py:code"
```
