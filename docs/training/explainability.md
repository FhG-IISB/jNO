# Explainability

Four callbacks give insight into what is happening inside the training loop. They work by differentiating through the constraint functions after each outer step, independently of the gradient updates that drive training. Results are stored as numpy arrays and, when a W&B run is active, pushed automatically to your dashboard.

---

## How they work

Each callback registers itself in `on_solve_begin` — called once after the initial JIT compilation — and pre-compiles its JAX function against the current parameter shapes. The first call at `epoch % interval == 0` therefore runs a pre-warmed XLA kernel with no recompilation overhead.

Internally, callbacks 1–3 share a single `jacrev`-based function that computes the full gradient matrix $G \in \mathbb{R}^{N \times P}$ where $N$ is the number of constraints and $P$ the number of (selected) parameters. The loss landscape callback uses a separate `jax.lax.map` + `vmap` sweep.

---

## Gradient norms

```python
cb = jno.callbacks.gradient_norms(interval=100)
crux.solve(5000, callbacks=[cb])

norms = cb.result["norms"]    # (n_samples, n_constraints)
epochs = cb.result["epochs"]  # (n_samples,)
```

Tracks $\|\nabla L_i\|_2$ for each constraint $i$. A constraint whose norm is orders of magnitude larger than the others will dominate the parameter update regardless of its loss value.

W&B keys: `explainability/gradient_norm/constraint_0`, …, `explainability/gradient_norm/constraint_N`

---

## Cosine similarity

```python
cb = jno.callbacks.cos_similarity(interval=100)
crux.solve(5000, callbacks=[cb])

cos_mat = cb.result["cos_sim_matrix"]   # (n_samples, N, N)
```

Computes the full $(N \times N)$ pairwise cosine similarity matrix between constraint gradients. The diagonal is always 1; the upper triangle carries the meaningful values.

$$\text{sim}_{ij} = \frac{\nabla L_i \cdot \nabla L_j}{\|\nabla L_i\| \|\nabla L_j\|}$$

| Value | Meaning |
|-------|---------|
| $\approx +1$ | Gradients reinforce each other — constraints are compatible |
| $\approx 0$ | Independent directions |
| $\approx -1$ | Gradient conflict — one constraint actively hurts the other |

When W&B is active the matrix is uploaded as a heatmap image at each sampled epoch.

W&B keys: `explainability/cos_sim/0_1`, …, `explainability/cos_sim_matrix` (image)

---

## Gradient alignment

```python
cb = jno.callbacks.gradient_alignment(interval=100)
crux.solve(5000, callbacks=[cb])

alignment = cb.result["alignment"]   # (n_samples,)  — values in [0, 1]
```

A single scalar measuring global agreement across all constraints (Eq. 3.1 of [[2502.00604](https://arxiv.org/abs/2502.00604)]):

$$\text{alignment} = \frac{\left\|\sum_i \nabla L_i\right\|}{\sum_i \|\nabla L_i\|}$$

- **1.0** — all gradients point in exactly the same direction
- **0.0** — gradients cancel completely (destructive interference)

A value that drops steadily during training is a reliable early warning of constraint conflict.

W&B key: `explainability/gradient_alignment`

---

## Loss landscape

```python
cb = jno.callbacks.loss_landscape(
    interval=500,     # expensive — n_grid² forward passes each call
    n_grid=15,
    alpha_range=1.0,  # perturbation range in units of ‖θ‖
)
crux.solve(5000, callbacks=[cb])

landscapes = cb.result["landscapes"]   # (n_samples, n_grid, n_grid)
```

At every `interval` steps, two random filter-normalised directions are sampled and the total loss is evaluated on an $(n\_\text{grid} \times n\_\text{grid})$ perturbation grid centred on the current parameters (based on [Li et al., 2018](https://arxiv.org/abs/1712.09913)).

A smooth bowl indicates a well-conditioned landscape. Sharp ridges or flat saddle regions may explain slow convergence or oscillating loss.

!!! warning "Cost"
    Each call requires $n\_\text{grid}^2$ full forward passes. Keep `interval` large (500–1000) for real training runs, or use `mask` to restrict perturbations to a small subset of parameters.

W&B key: `explainability/loss_landscape` (heatmap image)

---

## Restricting to a parameter subset

All four callbacks accept an optional `mask` — a pytree of booleans matching the `trainable` structure. Only the selected parameters are differentiated or perturbed. This is essential for large networks and strongly recommended even for medium-sized ones.

```python
import equinox as eqx, jax

all_false   = jax.tree_util.tree_map(lambda _: False, u_net.params)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

cb_norms   = jno.callbacks.gradient_norms(interval=50,  mask=output_mask)
cb_cos     = jno.callbacks.cos_similarity(interval=50,  mask=output_mask)
cb_align   = jno.callbacks.gradient_alignment(interval=50,  mask=output_mask)
cb_land    = jno.callbacks.loss_landscape(interval=500, mask=output_mask, n_grid=11)
```

The output-layer weight matrix typically gives the dominant gradient directions at a fraction of the cost of the full parameter set.

---

## W&B logging

All four callbacks push their results to W&B automatically when a run is active (see [Weights & Biases](../misc/wandb.md)). No extra code is needed — enabling `jno.setup(..., wandb=True)` is sufficient.

---

## Combining with other callbacks

```python
crux.solve(
    10_000,
    callbacks=[
        jno.callbacks.gradient_norms(interval=50),
        jno.callbacks.gradient_alignment(interval=50),
        jno.callbacks.loss_landscape(interval=500, n_grid=11),
        jno.callbacks.checkpoint(save_interval_epochs=1000),
        jno.callbacks.early_stopping(patience=2000),
    ],
)
```
