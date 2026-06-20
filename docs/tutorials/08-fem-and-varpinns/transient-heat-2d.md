# Transient Heat Diffusion (time-dependent FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/transient_heat_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A time-dependent solve: the heat equation $u_t=\nu\Delta u$ with a single Laplacian-eigenmode
initial condition, which decays as $e^{-2\nu\pi^2 t}$.

## `time=(...)` makes the form semidiscrete

Adding `time=(t0, t1, n)` to the domain and a `u.t` term to the weak form produces a
semidiscrete system $M\dot u + A u = 0$. `jno.fem` then exposes the mass `fem.M`, operator
`fem.operator.A`, and initial state `fem.state0` for any time integrator — here backward Euler:

```python
d = jno.domain(box(0, 0, 1, 1), mesh_size=0.08, time=(0.0, 0.05, 26))
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
ic = u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [xi0, yi0])
fem = jno.fem([ui.t * vi + nu * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, ic])

M, A, dt = fem.M, dense(fem.operator.A), float(fem.dt)  # fem.M is dense; operator.A is raw sparse
w = fem.state0
for _ in range(round((fem.t1 - fem.t0) / dt)):
    w = jnp.linalg.solve(M + dt * A, M @ w)            # backward Euler
```

## What to notice

- A `u.t` term + `time=(...)` switch `jno.fem` to the transient (mass-matrix) route.
- The initial condition is the residual `u(initial) - u0`.
- Backward Euler over `(M, A)` reproduces $e^{-2\nu\pi^2 t}$ to rel-$L^2 \approx 10^{-2}$.
- For a *parametric* transient inverse, train through `fem.solve()` —
  see [transient inverse](transient-inverse-heat.md).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/transient_heat_2d.py"
```
