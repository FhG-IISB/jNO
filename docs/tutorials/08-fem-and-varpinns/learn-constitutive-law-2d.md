# Learn a Nonlinear Constitutive Law $k(T)$ (NN-EUCLID style)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/learn_constitutive_law_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The [neural-diffusivity tutorial](inverse-neural-diffusivity-2d.md) recovered a spatial map
$k(x)$. Here the unknown is a **material law**: the conductivity depends on the *solution itself*,
$k=k(T)$ (temperature-dependent conduction in $-\nabla\!\cdot(k(T)\nabla T)=f$). A network takes
the trial value as its input, so the weak form is **nonlinear in the unknown** — `jno.fem` routes
it to the matrix-free Newton path automatically, and the network's $T$-dependence enters the
element Jacobians through per-element forward AD. Training recovers the law from a single measured
field, with no flux/stress labels: the unsupervised constitutive-learning setting of NN-EUCLID.

## Positivity by construction

A conductivity must be positive, or the operator goes indefinite and Newton diverges. Rather than
police that with a penalty, enforce it **in the architecture** — a softplus output,
$k(T)=\mathrm{softplus}(\mathrm{net}(T))>0$ for any weights the optimiser visits (the same idea as
input-convex networks for hyperelastic potentials):

```python
net = jno.nn.wrap(foundax.mlp(1, hidden_dims=32, num_layers=2,
                              activation=jax.nn.tanh, key=key)).dtype(jnp.float64)

def k_model(T):
    return jno.np.log1p(jno.np.exp(net(T)))          # softplus(net(T)) > 0

fem = jno.fem([k_model(ui) * (ui.x*vi.x + ui.y*vi.y) - f*vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - T_obs).mse], domain=obs).solve(1500)
```

Because `k_model(ui)` feeds the trial `ui` into the network, `jno.fem` classifies the form
nonlinear and solves it with per-step Newton; the gradient to the weights flows through the
`custom_root` implicit differentiation without unrolling the Newton iterations.

## What to notice

- The unknown is a **function of the state**, not of position — impossible to express as a nodal
  field `jno.np.parameter(phi)`. A learned law $k(T)$ is also *transferable*: reuse it on any
  geometry or load.
- **Score on the observed range**: the network only sees $k(T)$ where $T$ was actually sampled, so
  the recovered law is verified on $[0, T_{\max}]$ (rel-$L^2\approx 2\times 10^{-2}$).
- Heavy per-step Newton inverse ⇒ the script pins `JAX_PLATFORMS=cpu` and enables `x64` at the top
  (the differentiable nonlinear solve is float64 and CPU avoids GPU-memory contention).

Reference: M. Flaschel, S. Kumar, L. De Lorenzis, *NN-EUCLID: Deep-learning hyperelasticity without
stress data*, J. Mech. Phys. Solids 165 (2022) 105076, §2.2–2.3; A. Tartakovsky et al., *Learning
Parameters and Constitutive Relationships with Physics-Informed Deep Neural Networks*, Water
Resour. Res. 56 (2020), §2.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/learn_constitutive_law_2d.py"
```
