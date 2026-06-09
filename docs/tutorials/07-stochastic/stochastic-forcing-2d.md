# 2-D Poisson with Stochastic Forcing

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/07_stochastic/stochastic_forcing_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/07-stochastic/">Back to chapter</a>
</div>

This example solves a **2-D Poisson equation whose source term is random** — a model for PDEs driven by turbulent forces, uncertain body loads, or random media.  Unlike the Fokker-Planck tutorial (where noise appeared on boundary observations), here `jno.noise` is placed **directly inside the PDE residual**.  The network still recovers the correct deterministic solution; the tutorial explains exactly why.

---

## Problem

=== "Stochastically Forced PDE"

    $$-\Delta u(x,y) = f(x,y) + \sigma \, \xi(x,y), \qquad (x,y) \in [0,1]^2, \quad u = 0 \text{ on } \partial\Omega$$

    - $f(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$ — deterministic part of the forcing.
    - $\xi(x,y) \sim \mathcal{N}(0, 1)$ pointwise — random fluctuation, freshly drawn each epoch.
    - $\sigma = 0.5$ — noise amplitude.

    The exact **mean-field solution** (the expectation over all noise realisations) is:

    $$u^*(x, y) = \sin(\pi x)\sin(\pi y)$$

=== "Why noise doesn't change the solution"

    The MSE loss at training step $k$ is

    $$\mathcal{L}_k(\theta) = \frac{1}{N}\sum_{i=1}^{N}\bigl(-\Delta u_\theta(x_i, y_i) - f_i - \sigma\,\xi_i^{(k)}\bigr)^2$$

    Taking the expectation over the i.i.d. noise $\xi^{(k)} \sim \mathcal{N}(0,1)$ and expanding the square:

    $$\mathbb{E}_\xi[\mathcal{L}_k(\theta)] = \underbrace{\frac{1}{N}\sum_i(-\Delta u_\theta - f_i)^2}_{\text{deterministic MSE}} + \underbrace{\sigma^2}_{\text{constant}}$$

    The cross term vanishes because $\mathbb{E}[\xi_i] = 0$.  The noise adds a **constant** $\sigma^2$ that does not depend on $\theta$, so:

    $$\arg\min_\theta \, \mathbb{E}[\mathcal{L}_k(\theta)] = \arg\min_\theta \, \frac{1}{N}\sum_i(-\Delta u_\theta - f_i)^2$$

    **The minimiser is identical to the deterministic case.** Stochastic PINN training is a Monte Carlo estimator of the expected loss, and the estimator is unbiased.

=== "Boundary conditions"

    Hard Dirichlet BCs are enforced via the ansatz:

    $$u_\theta(x,y) = \hat{u}_\theta(x,y) \cdot x(1-x) \cdot y(1-y)$$

    The factor $x(1-x)y(1-y)$ is zero on all four edges, so $u_\theta = 0$ on $\partial\Omega$ for any network output $\hat{u}_\theta$.  No boundary loss term is needed.

---

## Code Walkthrough

### Step 1 — Domain and forcing

```python
domain = jno.domain.rect(mesh_size=0.05)
x, y, _ = domain.variable("interior")

f       = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
```

`mesh_size=0.05` gives roughly 500 interior collocation points on $[0,1]^2$.

### Step 2 — Network with hard BCs

```python
net = jno.nn.wrap(
    foundax.mlp(in_features=2, hidden_dims=64, num_layers=5,
                activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

u = net(x, y) * x * (1 - x) * y * (1 - y)
```

The hard-BC ansatz means the only constraint needed is the PDE residual.

### Step 3 — Stochastic PDE residual

```python
σ     = 0.5
noise = jno.noise.gaussian(std=σ)
pde   = -jno.np.laplacian(u, [x, y]) - f - noise
```

`jno.noise.gaussian(std=σ)` is a **symbolic Placeholder** — it participates in the expression tree like any other traced node.  At each training step the solver splits its PRNG key and draws a fresh $(N, 1)$ sample, so every gradient update sees a different noisy residual.

!!! tip "Noise amplitude and convergence"
    With $\sigma = 0.5$ and a Laplacian residual that is typically $O(1)$, the signal-to-noise ratio is moderate.  Training takes slightly more epochs than the deterministic problem, but the final accuracy is the same.  Larger $\sigma$ adds more variance to the stochastic gradient without changing the bias.

### Step 4 — Single-loss solve

```python
crux    = jno.core([pde.mse])
history = crux.solve(40_000)
```

Because hard BCs remove the boundary term, there is only one loss.  The optimizer drives the stochastic PDE residual to zero by learning the mean-field solution $u^* = \sin(\pi x)\sin(\pi y)$.

---

## Comparison: Deterministic vs Stochastic

| | Deterministic | Stochastic (this tutorial) |
|---|---|---|
| Loss | $\lVert{-\Delta u - f}\rVert^2$ | $\lVert{-\Delta u - f - \sigma\xi}\rVert^2$ |
| Gradient | exact | noisy estimate (unbiased) |
| Minimiser $u^*$ | $\sin(\pi x)\sin(\pi y)$ | **same** |
| Extra cost | — | one `fold_in` per step |
| Noise source | — | solver PRNG, seeded via `jno.setup` |

---

## What to Notice

- **One loss, no boundary term.** Hard BCs eliminate the need for a separate boundary loss, so the entire problem reduces to a single stochastic PDE constraint.
- **Noise on the physics, not the data.** This is the complement of the Fokker-Planck tutorial: there, noise modelled uncertain measurements; here, it models an uncertain forcing within the governing equation itself.
- **`ndim` for vector noise.** If your field has multiple components (e.g., a 2-D velocity vector), use `jno.noise.gaussian(std=σ, ndim=2)` to draw a correlated $(N, 2)$ sample in one call rather than two separate scalar noise nodes.
- **Reproducibility via seed.** Fix `jno.setup(seed=42)` (or set `seed = 42` in `.jno.toml`) to get identical noise sequences across runs.

---

## Full Script

```python
--8<-- "tutorial_examples/07_stochastic/stochastic_forcing_2d.py"
```

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/07_stochastic/stochastic_forcing_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/07-stochastic/">Back to 07 Stochastic</a>
</div>
