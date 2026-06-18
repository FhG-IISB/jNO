# Inverse: Recover a Diffusivity Field (differentiable FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/inverse_diffusivity_field.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

An inverse problem: recover the entire spatially-varying diffusivity field $k(x)$ in
$-\nabla\!\cdot(k\nabla u)=f$ — a hidden high-conductivity inclusion — from the measured
response $u$, by **differentiating the FEM solve end to end**. This is the FEM flavour of
parameter-field identification / diffuse tomography.

## A nodal field parameter + smoothness prior

`jno.np.parameter(phi)` is a trainable P1 field (one DOF per node) on the trial space;
`fem.solve()` is the differentiable forward; field inversion is ill-posed, so an H1-seminorm
prior `k.regularize("h1seminorm")` keeps it stable. `crux` fits the data plus the prior:

```python
k = jno.np.parameter(phi, name="k")
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
crux = jno.core([(fem.solve() - u_obs).mse, 1e-3 * k.regularize("h1seminorm").mean], domain=obs)
crux.solve(500)
rec = np.asarray(crux.eval([k])).reshape(-1)     # the recovered field (do not index [0])
```

## What to notice

- `jno.np.parameter(phi)` makes the unknown a full nodal field, not a scalar.
- `fem.solve()` is differentiable, so the gradient reaches every node of `k`.
- `k.regularize(...)` (`h1seminorm`/`l2`/`tv`/`nonneg`/`bounded`) is the FE-exact prior; the
  inclusion is recovered to rel-$L^2 \approx 6\times10^{-2}$ (the prior smooths its peak slightly).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/inverse_diffusivity_field.py"
```
