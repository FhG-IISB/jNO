# Fokker-Planck 2D (Ornstein-Uhlenbeck)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/07_stochastic/fokker_planck_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/07-stochastic/">Back to chapter</a>
</div>

This example solves the **stationary Fokker-Planck equation** for a 2-D Ornstein-Uhlenbeck process — a PDE that describes how a probability density evolves under drift and diffusion.  It introduces `jno.noise` to add stochastic measurement uncertainty to the boundary observations, so the network sees freshly-sampled noisy data every training step.

---

## The Stochastic Process

=== "SDE"

    The underlying stochastic differential equations (Itô form) are:

    $$dX_t = -X_t \, dt + dW_t^{(1)}, \qquad dY_t = -Y_t \, dt + dW_t^{(2)}$$

    - The **drift** $-X_t$, $-Y_t$ pulls the process back toward the origin (restoring force with rate 1).
    - The **diffusion** coefficient $\sigma = 1$ sets the strength of the Brownian noise.

    This is the simplest mean-reverting process in 2-D and is analytically tractable — an ideal test case for a PINN.

=== "Fokker-Planck PDE"

    The **forward Kolmogorov (Fokker-Planck) equation** for the stationary density $p(x, y)$ is:

    $$\frac{\partial (x \, p)}{\partial x} + \frac{\partial (y \, p)}{\partial y} + \frac{\sigma^2}{2} \left(\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2}\right) = 0, \qquad (x, y) \in \Omega = [-3, 3]^2$$

    The first two terms are the **drift divergence** $\nabla \cdot (\mathbf{b} \, p)$ with $\mathbf{b}(x,y) = (-x, -y)$.  The last term is the **diffusion Laplacian** $\tfrac{\sigma^2}{2} \Delta p$.

    Boundary condition: $p \approx 0$ on $\partial\Omega$ (the Gaussian decays to $e^{-9}/\pi \approx 4 \times 10^{-5}$ at the domain edges).

=== "Analytical Solution"

    For unit restoring rate and $\sigma = 1$ the stationary density is a bivariate Gaussian:

    $$p^\infty(x, y) = \frac{1}{\pi} \exp\!\left(-(x^2 + y^2)\right)$$

    **Verification.** Substituting $p = C e^{-(x^2+y^2)}$:

    | Term | Value |
    |------|-------|
    | $\partial(xp)/\partial x$ | $p - 2x^2 p$ |
    | $\partial(yp)/\partial y$ | $p - 2y^2 p$ |
    | $\tfrac{1}{2}\Delta p$ | $(−2 + 2x^2 − 2 + 2y^2) p / 2 = (-2 + x^2 + y^2 - 1)p$... |

    Working through all terms cancels exactly to zero. ✓

---

## Code Walkthrough

### Step 1 — Domain centred at the origin

```python
domain = jno.domain.rect(x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), mesh_size=0.15)
x,  y,  _ = domain.variable("interior")
xb, yb, _ = domain.variable("boundary")
```

The domain is centred at $(0, 0)$ so the Gaussian peak sits at the domain centre.  `mesh_size=0.15` gives roughly 1 900 interior points and 160 boundary points.

### Step 2 — Network

```python
net = jno.nn.wrap(
    foundax.mlp(in_features=2, hidden_dims=64, num_layers=5,
                activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

p = net(x, y)
```

A tanh MLP is a natural fit because the target $e^{-(x^2+y^2)}$ is smooth and bell-shaped.

### Step 3 — Fokker-Planck residual

```python
drift = jno.np.grad(x * p, x) + jno.np.grad(y * p, y)
diff  = 0.5 * jno.np.laplacian(p, [x, y])
fp    = drift + diff
```

`jno.np.grad(x * p, x)` computes $\partial(x\,p)/\partial x$ via automatic differentiation.  The Laplacian $\Delta p$ is computed by `jno.np.laplacian`.

### Step 4 — Normalization constraint

```python
norm = p.integrate() - 1.0
```

`.integrate()` reduces the field $p(x, y)$ to the scalar $\iint_\Omega p \, dx \, dy$ using mesh-based quadrature weights precomputed at domain creation.  Subtracting 1 creates a loss that drives the total probability mass to 1 — an essential physical constraint for any density.

### Step 5 — Noisy boundary condition

```python
p_bc = net(xb, yb) - (p_exact_bc + jno.noise.gaussian(std=1e-4))
```

`jno.noise.gaussian(std=1e-4)` is a **lazy Placeholder** — no random numbers are generated at graph-build time.  Each training step the solver splits its PRNG key, calls `jax.random.fold_in` with the node's unique ID, and samples a fresh $(N_b, 1)$ array.

This simulates the scenario where boundary observations come from noisy physical measurements (e.g., estimated from Monte Carlo paths of the SDE that happen to cross the domain boundary).

!!! note "Reproducibility"
    The noise sequence is fully determined by the global seed.  Set it with `jno.setup(seed=42)` or in `.jno.toml` to reproduce the exact same training run.

### Step 6 — Solve

```python
crux = jno.core([fp.mse, norm.mse, p_bc.mse], domain)
history = crux.solve(50_000)
```

Three losses compete: PDE residual, normalization, and noisy boundary data.  The solver balances them using its built-in loss weighting.

---

## What to Notice

- **Noise on observations, not the physics.** The Fokker-Planck residual is deterministic; only the boundary data carries noise.  This mirrors the real-world setting where the governing equation is known but measurements are uncertain.
- **The network still converges.** Because $\operatorname{std} = 10^{-4}$ is of the same order as the true boundary values ($\sim 4 \times 10^{-5}$), the signal-to-noise ratio at the boundary is low.  The PDE and normalization constraints compensate and anchor the interior solution.
- **`.integrate()` is differentiable.** Gradients flow through the normalization term so the optimizer simultaneously adjusts the global scale of $p$ and the shape of the Fokker-Planck residual.
- **No manual key management.** The user never calls `jax.random.split` or threads keys through the loss — `jno.noise` handles all of that inside the solver.

---

## Full Script

```python
--8<-- "tutorial_examples/07_stochastic/fokker_planck_2d.py"
```

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/07_stochastic/fokker_planck_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/07-stochastic/">Back to 07 Stochastic</a>
</div>
