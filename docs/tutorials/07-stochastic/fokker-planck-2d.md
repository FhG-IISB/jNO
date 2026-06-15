# Fokker-Planck 2D (Ornstein-Uhlenbeck)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/07_stochastic/fokker_planck_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/07-stochastic/">Back to chapter</a>
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

### Step 1 — Domain centred at the origin and analytical boundary

The domain is centred at $(0, 0)$ so the Gaussian peak sits at the domain centre. The analytical boundary value $p_\text{exact}$ is reused later as the (near-zero) target for the noisy boundary observations.

```python
--8<-- "tutorial_examples/07_stochastic/fokker_planck_2d.py:setup"
```

### Step 2 — Fokker-Planck residual

`jno.np.vector(...)` builds a typed `VectorView` from scalar components without manual `concat`, and `.div(x, y)` reads exactly like the math $\nabla \cdot (\mathbf{b} p)$. The Laplacian $\Delta p$ is computed by `jno.np.laplacian`.

```python
--8<-- "tutorial_examples/07_stochastic/fokker_planck_2d.py:residual"
```

### Step 3 — Normalization and noisy boundary

`.integrate()` reduces $p(x, y)$ to the scalar $\iint_\Omega p \, dx \, dy$ using mesh-based quadrature weights. Subtracting 1 creates a loss that drives total probability mass to 1.

`jno.noise.gaussian(std=1e-4)` is a **lazy Placeholder** — each training step the solver splits its PRNG key and samples a fresh $(N_b, 1)$ array, simulating noisy physical measurements at the boundary.

```python
--8<-- "tutorial_examples/07_stochastic/fokker_planck_2d.py:constraints"
```

!!! note "Reproducibility"
    The noise sequence is fully determined by the global seed.  Set it with `jno.setup(seed=42)` or in `.jno.toml` to reproduce the exact same training run.

### Step 4 — Solve

Three losses compete: PDE residual, normalization, and noisy boundary data. The solver balances them using its built-in loss weighting.

```python
--8<-- "tutorial_examples/07_stochastic/fokker_planck_2d.py:solve"
```

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
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/07_stochastic/fokker_planck_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/07-stochastic/">Back to 07 Stochastic</a>
</div>
