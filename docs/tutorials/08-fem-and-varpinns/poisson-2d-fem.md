# Poisson 2D (FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A pure finite-element solve of $-\Delta u = f$ on the unit square with $u = 0$ on the boundary,
assembled through `jno.fem`. It is the primer for the API: write the weak form as a list of
residual terms — the volume physics plus the essential boundary condition as `u(region) - g` —
and solve the assembled `A u = b`.

## The weak form

The trial/test functions come from `d.fem_symbols()`; binding them to the quadrature
coordinates gives the `.x` / `.y` derivatives, and the Dirichlet condition is simply the term
`u(xb, yb) - 0.0`:

```python
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)
u_fem = jnp.asarray(fem.solve(linear=jno.solve.cg(), precond=jno.precond.jacobi()))  # SPD -> CG
```

## What to notice

- One call, `jno.fem([...])`, assembles the stiffness `fem.A` and load `fem.b`.
- Boundary conditions are terms in the same list — no separate BC objects.
- `fem.solve()` alone would also work (matrix-free Jacobi-BiCGStab on the sparse operator);
  the **solver slots** pick a structure-appropriate method — CG for this SPD system — without
  ever densifying. See [Choosing the solver](../../fem.md) for the full `jno.solve` /
  `jno.precond` namespaces.
- The solution is audited against the manufactured field $u^\*=x(1-x)y(1-y)$ (rel-$L^2 \approx 8\times10^{-3}$).

## Result

![Left: computed Poisson solution on the unit square. Middle: signed nodal error versus the manufactured exact field. Right: relative L2 error against DOFs on a log-log axis from five real re-solves.](/jNO/assets/poisson_2d_fem.png)

The computed field (left) matches the manufactured $u^\*=x(1-x)y(1-y)$ to rel-$L^2\approx8.6\times10^{-3}$ (middle, signed error). Re-solving on five successively finer meshes (right) shows the error falling as the mesh is refined.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py:code"
```
