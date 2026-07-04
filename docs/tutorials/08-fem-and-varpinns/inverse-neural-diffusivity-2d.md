# Inverse: Recover a Diffusivity Field with a Neural Coefficient

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/inverse_neural_diffusivity_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The same identification problem as the [nodal-field tutorial](inverse-diffusivity-field.md) —
recover the hidden diffusivity $k(x)$ in $-\nabla\!\cdot(k\nabla u)=f$ from the measured
response — but the unknown is a **network**, written straight into the weak form:
$k(x,y) = 1 + \mathrm{net}(x,y)$. The system assembles as usual; the kernel re-evaluates the
network at the quadrature points on every re-assembly, and `crux` trains the weights through the
differentiable `fem.solve()`.

## A neural coefficient instead of a nodal field

```python
net = jno.nn.wrap(foundax.mlp(2, hidden_dims=32, num_layers=2,
                              activation=jax.nn.tanh, key=key))
net.dtype(jnp.float64)                        # match the f64 assembly
net.optimizer(optax.adam(2e-2))

fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi,
               u(xb, yb) - 0.0], quad_degree=3)
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs)   # no explicit prior needed
crux.solve(2500)
trained = crux.eval([jno.trace.ModelWeights(net)])          # the trained module
```

## What to notice

- **No new API**: calling `jno.nn.wrap(net)` inside the weak form *is* the feature. The weak
  form's real FE trial makes the network a coefficient (a network written *in place of* the
  trial would be a VPINN).
- **Mesh-independent**: the unknown lives in the weights, not on the mesh — remeshing (or the
  adaptive loop) never touches the parametrisation, unlike `jno.np.parameter(phi)`.
- **Architecture as prior**: the nodal-field version needs an explicit
  `k.regularize("h1seminorm")`; here the MLP's smoothness bias regularises by itself — the
  Gaussian inclusion is recovered to rel-$L^2 \approx 5\times 10^{-2}$.
- The `1 +` offset keeps the operator nonsingular at the (near-zero) network init — the same
  practice as starting a nodal field at $k=1$.
- The same mechanism trains **constitutive laws** `net(u)` / `net(∇u)` (the form then routes to
  the Newton path automatically) — see the neural-coefficients section of the
  [FEM guide](../../fem.md).

This is the unsupervised coefficient/constitutive-recovery setting of NN-EUCLID
(M. Flaschel, S. Kumar, L. De Lorenzis, *NN-EUCLID: Deep-learning hyperelasticity without
stress data*, J. Mech. Phys. Solids 165 (2022) 105076, §2.2–2.3) and A. Tartakovsky et al.,
*Learning Parameters and Constitutive Relationships with Physics-Informed Deep Neural
Networks* (Water Resour. Res. 56, 2020, §2).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/inverse_neural_diffusivity_2d.py"
```
