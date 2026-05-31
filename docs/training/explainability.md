# Explainability

Several callbacks give insight into what is happening inside the training loop. They work by differentiating through the constraint functions after each outer step, or by directly inspecting the residuals, independently of the gradient updates that drive training. Results are stored as numpy arrays and, when a W&B run is active, pushed automatically to your dashboard.

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

## Residual statistics

```python
cb = jno.callbacks.residual_stats(interval=100)
crux.solve(5000, callbacks=[cb])

means = cb.result["means"]    # (n_samples, n_constraints)
maxes = cb.result["maxes"]    # (n_samples, n_constraints)
p99   = cb.result["p99"]      # (n_samples, n_constraints)
```

For each constraint $i$, evaluates the un-reduced residual array $r_i$ produced by the compiled constraint function (i.e. the values *before* the training loss applies its mean) and records four scalar statistics — mean, std, max, and 99th percentile — plus a histogram of the raw residuals when W&B is active. A constraint whose ``max`` or ``p99`` stays orders of magnitude above the others points to a region of the domain where the PDE is poorly satisfied, complementing [gradient norms](#gradient-norms) which only reflect each constraint's *aggregated* contribution to the parameter update.

W&B keys: `explainability/residual/constraint_{i}/{mean,std,max,p99}`, `.../histogram` (image)

### Scoping to a subset of constraints

Pass the constraint expressions you care about via `constraints=`. The callback validates them by Python identity against what was given to `jno.core(...)`, so you must assign your constraints to variables — `.mse` is a property that returns a fresh placeholder every access:

```python
pde_loss = pde.mse                       # assign once
bc_loss  = bc.mse
solver = jno.core([pde_loss, bc_loss], domain)

cb = jno.callbacks.residual_stats(interval=100, constraints=[pde_loss])
crux.solve(5000, callbacks=[cb])

print(cb.result["means"].shape)   # (n_samples, 1)  — just pde_loss
print(cb.result["indices"])       # [0]  — solver-side index
```

W&B keys use the **solver-side index** so the dashboard remains stable when you add or remove unrelated constraints later. `cb.result["indices"]` records that mapping.

Reference: per-point residual magnitudes as a sampling / diagnostic signal — Sec. 3 of [[2207.10289](https://arxiv.org/abs/2207.10289)] (Wu et al., 2023).

---

## Input sensitivity / saliency

```python
cb = jno.callbacks.input_sensitivity(u.d(x), interval=100)
crux.solve(5000, callbacks=[cb])

values = cb.result["values"]    # (n_samples, *expr_shape)
```

Evaluates an arbitrary jno placeholder expression at the training collocation points and records its value every `interval` outer steps. The intended use is *input-gradient saliency* — for a scalar network output $u$ and a coordinate variable $x$, $\partial u/\partial x$ measures how strongly the network response at a given point depends on that input dimension. High-magnitude regions are where small input perturbations produce large output changes (the PINN analogue of the class-saliency map of [Simonyan, Vedaldi & Zisserman, 2014](https://arxiv.org/abs/1312.6034), Sec. 3).

Common expressions to pass:

| Expression                       | Meaning                                              |
|----------------------------------|------------------------------------------------------|
| `u.d(x)`                         | $\partial u/\partial x$ — scalar per point           |
| `jno.Jacobian(u, [x, y])`        | full input Jacobian — shape `(N, 2)` for 2-D inputs  |
| `u.d(x)**2 + u.d(y)**2`          | squared $\lvert\nabla u\rvert^2$ as a scalar field   |

Any composite expression compiles, because the callback uses the same `TraceCompiler.compile_multi_expression` pathway that the solver uses for constraints and trackers.

W&B keys: `explainability/saliency/{mean_abs,max_abs,std_abs}`, `.../histogram` (image)

Reference: input-gradient saliency — Sec. 3 of [[1312.6034](https://arxiv.org/abs/1312.6034)] (Simonyan et al., 2014).

---

## Empirical NTK spectrum

```python
cb = jno.callbacks.ntk_spectrum(
    u.grad(u_net),
    n_points=256,
    top_k=10,
    interval=500,
)
crux.solve(10_000, callbacks=[cb])

eigvals = cb.result["eigvals_topk"]        # (n_samples, top_k)
cond    = cb.result["condition_number"]    # (n_samples,)
```

Compiles a `NetworkGradient` placeholder to obtain the per-point parameter Jacobian $J \in \mathbb{R}^{N \times P}$, subsamples ``n_points`` rows (with a fixed seed so the same points are used at every call), and reports the eigenvalue spectrum of the empirical NTK $K = J J^\top$. A wide spread between the largest and smallest eigenvalues is the canonical diagnostic for PINN spectral bias.

$$K_{ij} = \langle \nabla_\theta u(x_i), \, \nabla_\theta u(x_j) \rangle$$

Restrict to a parameter subset via `net.mask(...)` chained into the placeholder:

```python
cb = jno.callbacks.ntk_spectrum(u.grad(u_net.mask(out_mask)), n_points=128, top_k=10)
```

!!! warning "Cost"
    Cost is $O(n\_\text{points}^2 \times P)$.  Use both subsampling (`n_points`) **and** placeholder masking on large networks.  Scalar output only — for vector-valued $u$, project first (e.g. `u[..., 0].grad(net)`).

W&B keys: `explainability/ntk/eigval_{0..k-1}`, `.../lambda_max`, `.../lambda_min`, `.../condition_number`, `.../spectrum_hist`

Reference: NTK spectrum for PINN spectral-bias diagnosis — Sec. 3-4 of [[2007.14527](https://arxiv.org/abs/2007.14527)] (Wang, Wang & Perdikaris, 2022).

---

## Hessian eigenspectrum (sharpness)

```python
cb = jno.callbacks.hessian_spectrum(
    k=10,
    n_iter=30,
    interval=500,
    # mask=output_mask  # strongly recommended for large models
)
crux.solve(10_000, callbacks=[cb])

eigvals   = cb.result["eigvals"]      # (n_samples, k)  — descending
sharpness = cb.result["sharpness"]    # (n_samples,)    — largest eigenvalue
```

Computes the top-$k$ eigenvalues of the total training loss Hessian $\nabla^2_\theta L$ via Lanczos with Hessian-vector products. The largest eigenvalue is the **sharpness** of the loss surface at the current iterate (Sec. 2.2 of [[1609.04836](https://arxiv.org/abs/1609.04836)] — Keskar et al., 2017).

### Per-constraint Hessian (`constraints=`)

The total-loss Hessian conflates conditioning across all constraints. Pass `constraints=[...]` to scope the Hessian to a subset of the constraint losses — the spectrum then reflects the Hessian of `mean(L_i for i in constraints)`:

```python
pde_loss = pde.mse
bc_loss  = bc.mse
solver = jno.core([pde_loss, bc_loss], domain)

cb_pde = jno.callbacks.hessian_spectrum(k=5, n_iter=20, interval=500, constraints=[pde_loss])
cb_bc  = jno.callbacks.hessian_spectrum(k=5, n_iter=20, interval=500, constraints=[bc_loss])
crux.solve(5000, callbacks=[cb_pde, cb_bc])
```

!!! warning "Cost"
    Each call performs ``n_iter`` HVPs, each roughly the cost of one full forward+backward pass.  Keep ``interval`` large (500–1000) for real runs and use ``mask`` to restrict to a parameter subset.

W&B keys: `explainability/hessian/eigval_{0..k-1}`, `.../sharpness`, `.../n_iter`

Reference: HVP-based Lanczos for neural-network Hessian spectra — Sec. 3.1-3.2 of [[1912.07145](https://arxiv.org/abs/1912.07145)] (Yao et al., 2020).  Sharpness concept — Sec. 2.2 of [[1609.04836](https://arxiv.org/abs/1609.04836)] (Keskar et al., 2017).

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

The gradient-analysis callbacks (gradient norms, cosine similarity, gradient alignment, loss landscape) accept an optional `mask` — a pytree of booleans matching the `trainable` structure. Only the selected parameters are differentiated or perturbed. This is essential for large networks and strongly recommended even for medium-sized ones. (`residual_stats` does not need a mask — it inspects pre-existing constraint residuals rather than computing parameter gradients.)

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

All callbacks push their results to W&B automatically when a run is active (see [Weights & Biases](../misc/wandb.md)). No extra code is needed — enabling `jno.setup(..., wandb=True)` is sufficient.

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
