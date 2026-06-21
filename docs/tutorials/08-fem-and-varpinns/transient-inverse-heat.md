# Inverse: Recover a Rate from a Time Series (transient FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/transient_inverse_heat.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A time-dependent inverse problem: recover the unknown diffusion rate $\alpha$ in
$u_t=\alpha\Delta u$ from an observed trajectory $u(t)$, by differentiating the **time
integration** itself.

## `fem.solve()` returns a differentiable trajectory

For a transient weak form, `fem.solve()` integrates the semidiscrete system and returns the
trajectory `u(save_ts)` (default: backward Euler over the assembled `dt`), differentiable in
the parameters — so the gradient flows through the integrator back to $\alpha$ and `crux` fits it:

```python
alpha = jno.np.parameter((1,), name="alpha")
fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs)
crux.solve(200)                                   # recovers alpha from the trajectory
```

`fem.solve(my_integrator, save_ts=...)` swaps the integrator — build your own from the block's
`M` / `A` / `state0`; the implicit backward-Euler default suits Dirichlet problems.

## What to notice

- `fem.solve()` on a transient form yields the trajectory `(len(save_ts), n_dofs)`.
- The gradient flows through the time stepping to the parameter — no manual adjoint.
- $\alpha$ is recovered to within a fraction of a percent.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/transient_inverse_heat.py"
```
