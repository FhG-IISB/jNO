# Stationary Allen-Cahn (nonlinear FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A nonlinear FEM solve: the stationary Allen-Cahn equation $-\varepsilon^2 \Delta u + (u^3 - u) = 0$,
whose stable equilibrium is a sharp `tanh` phase interface (Allen & Cahn, *Acta Metall.* 1979).

## Nonlinear weak form → residual operator

The cubic term `(u**3 - u) * vi` makes the form nonlinear in `u`, so `jno.fem` returns a
**residual operator** (`fem.residual`, `fem.jacobian`) rather than a linear `A, b`:

```python
fem = jno.fem([eps**2 * (ui.x * vi.x + ui.y * vi.y) + (u**3 - u) * vi,
               u(xl, yl) - exact(0.0), u(xr, yr) - exact(1.0)], quad_degree=3)
assert not fem.is_linear
```

Newton (here SciPy's `root`) is driven by `fem.residual` and `fem.jacobian`, started from an
over-wide interface and sharpened to the analytic equilibrium:

```python
sol = spo.root(lambda v: np.asarray(fem.residual(jnp.asarray(v))), u0,
               jac=lambda v: np.asarray(fem.jacobian(jnp.asarray(v))), method="hybr")
```

## What to notice

- A nonlinear weak form switches `jno.fem` to the residual route automatically.
- `fem.residual(u)` and `fem.jacobian(u)` plug into any Newton / root-finder.
- Converges to the `tanh` interface at rel-$L^2 \approx 10^{-3}$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py"
```
