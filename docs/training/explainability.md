# Explainability

A family of trackers that give insight into what is happening inside the training loop. They work by differentiating through the constraint functions after each outer step, or by directly inspecting the residuals, independently of the gradient updates that drive training. Results are stored as numpy arrays on the tracker object and, when a W&B run is active, pushed automatically to your dashboard.

Each tracker registers itself in `on_solve_begin` — called once after the initial JIT compilation — and pre-compiles its JAX function against the current parameter shapes. The first call at `epoch % interval == 0` therefore runs a pre-warmed XLA kernel with no recompilation overhead. Internally, the three gradient trackers share a single `jacrev`-based function that computes the full gradient matrix $G \in \mathbb{R}^{N \times P}$ where $N$ is the number of constraints and $P$ the number of (selected) parameters.

Trackers are surfaced under two equivalent namespaces:

- **`jno.trackers.*`** — preferred; matches the tracker mental model and ships factories that mirror `jno.callbacks.*` 1-for-1.
- **`jno.callbacks.*`** — the historical entry point; remains supported. Both return the same classes.

---

## Live access: `tracker.value` vs `tracker.result`

Each tracker exposes two complementary views of its data:

| Attribute             | Updated when                                  | Type                | Used for                                                                                  |
|-----------------------|-----------------------------------------------|---------------------|-------------------------------------------------------------------------------------------|
| `tracker.value`       | Every time `epoch % interval == 0` fires      | `dict | None`       | **Live**: read by adaptive components (loss balancing) at the next step. `None` until the first interval. |
| `tracker.latest_epoch`| Same as `tracker.value`                       | `int | None`        | Tells consumers how stale `value` is.                                                     |
| `tracker.result`      | After `crux.solve()` returns                  | `dict[str, ndarray]`| **Post-training**: full history of every fire — `epochs` plus the per-metric stacked array. |

The live channel is what enables tracker-driven loss balancing (next section). Until the first measurement, consumers see `tracker.value is None` and must fall back to a default (typically uniform weights).

---

## Gradient norms

Tracks $\|\nabla L_i\|_2$ for each constraint $i$ every `interval` outer steps. A constraint whose norm is orders of magnitude larger than the others will dominate the parameter update regardless of its loss value — a useful early signal of constraint imbalance.

```python
jno.callbacks.gradient_norms(
    interval = 100,
    mask     = None,
)
```

| Argument   | Type             | Default | Description                                                                                          |
|------------|------------------|---------|------------------------------------------------------------------------------------------------------|
| `interval` | `int`            | `100`   | Compute every *n* outer training steps.                                                              |
| `mask`     | pytree of `bool` | `None`  | Restrict the differentiated parameter subset (see [Restricting to a parameter subset](#restricting-to-a-parameter-subset)). Recommended for large models. |

**`cb.result` keys**

| Key      | Shape              | Description                              |
|----------|--------------------|------------------------------------------|
| `epochs` | `(S,)` int         | Sampled outer-step indices.              |
| `norms`  | `(S, N)` float32   | Per-constraint gradient $L_2$ norms.     |

**W&B keys:** `explainability/gradient_norm/constraint_0`, …, `explainability/gradient_norm/constraint_{N-1}`

---

## Cosine similarity

Computes the full $(N \times N)$ pairwise cosine similarity matrix between constraint gradients every `interval` outer steps. The diagonal is always 1; the upper triangle carries the meaningful values.

$$\text{sim}_{ij} = \frac{\nabla L_i \cdot \nabla L_j}{\|\nabla L_i\| \|\nabla L_j\|}$$

| Value | Meaning |
|-------|---------|
| $\approx +1$ | Gradients reinforce each other — constraints are compatible |
| $\approx 0$ | Independent directions |
| $\approx -1$ | Gradient conflict — one constraint actively hurts the other |

```python
jno.callbacks.cos_similarity(
    interval = 100,
    mask     = None,
)
```

| Argument   | Type             | Default | Description                                       |
|------------|------------------|---------|---------------------------------------------------|
| `interval` | `int`            | `100`   | Compute every *n* outer training steps.           |
| `mask`     | pytree of `bool` | `None`  | Restrict the differentiated parameter subset.     |

**`cb.result` keys**

| Key              | Shape               | Description                                     |
|------------------|---------------------|-------------------------------------------------|
| `epochs`         | `(S,)` int          | Sampled outer-step indices.                     |
| `cos_sim_matrix` | `(S, N, N)` float32 | Pairwise cosine similarity matrix at each step. |

**W&B keys:** `explainability/cos_sim/{i}_{j}` (per upper-triangle pair), `explainability/cos_sim_matrix` (heatmap image).

---

## Gradient alignment

A single scalar measuring global agreement across *all* constraints (Eq. 3.1 of [[2502.00604](https://arxiv.org/abs/2502.00604)]). Each gradient is unit-normalised *first*, so the metric reflects pure direction agreement and is invariant to per-loss scale; for $N = 2$ it reduces to the ordinary cosine similarity.

$$\text{alignment} \;=\; 2\left\|\frac{1}{N}\sum_{i=1}^{N} \frac{\nabla L_i}{\|\nabla L_i\|}\right\|^2 - 1$$

- **+1** — all gradients point in exactly the same direction
- **0** — gradients are mutually orthogonal on average
- **−1** — gradients perfectly cancel (anti-aligned)

A value that drops steadily during training — especially one that crosses zero into negative territory — is a reliable early warning of constraint conflict.

```python
jno.callbacks.gradient_alignment(
    interval = 100,
    mask     = None,
)
```

| Argument   | Type             | Default | Description                                   |
|------------|------------------|---------|-----------------------------------------------|
| `interval` | `int`            | `100`   | Compute every *n* outer training steps.       |
| `mask`     | pytree of `bool` | `None`  | Restrict the differentiated parameter subset. |

**`cb.result` keys**

| Key         | Shape         | Description                                                  |
|-------------|---------------|--------------------------------------------------------------|
| `epochs`    | `(S,)` int    | Sampled outer-step indices.                                  |
| `alignment` | `(S,)` float32| Total gradient alignment scalar, $\in [-1, 1]$.              |

**W&B key:** `explainability/gradient_alignment`

---

## Residual statistics

For each constraint $i$, evaluates the un-reduced residual array $r_i$ produced by the compiled constraint function (i.e. the values *before* the training loss applies its mean) and records four scalar statistics — mean, std, max, and 99th percentile — plus a histogram of the raw residuals when W&B is active. A constraint whose `max` or `p99` stays orders of magnitude above the others points to a region of the domain where the PDE is poorly satisfied, complementing [gradient norms](#gradient-norms) which only reflect each constraint's *aggregated* contribution to the parameter update (per-point residual magnitudes as a sampling / diagnostic signal — Sec. 3 of [[2207.10289](https://arxiv.org/abs/2207.10289)]).

```python
jno.callbacks.residual_stats(
    interval    = 100,
    constraints = None,
)
```

| Argument      | Type                       | Default | Description                                                                                                                                                                                |
|---------------|----------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `interval`    | `int`                      | `100`   | Compute every *n* outer training steps.                                                                                                                                                    |
| `constraints` | `list[Placeholder] \| None`| `None`  | Scope to a subset of the solver's constraints. Must be the *same* Python objects passed to `jno.core([...])` — assign them to variables first (`.mse` returns a fresh placeholder each access). `None` tracks all. |

**`cb.result` keys** (`K = len(constraints)` when scoped, else `N`)

| Key       | Shape              | Description                                                  |
|-----------|--------------------|--------------------------------------------------------------|
| `epochs`  | `(S,)` int         | Sampled outer-step indices.                                  |
| `means`   | `(S, K)` float32   | Per-constraint residual mean.                                |
| `stds`    | `(S, K)` float32   | Per-constraint residual std.                                 |
| `maxes`   | `(S, K)` float32   | Per-constraint residual max.                                 |
| `p99`     | `(S, K)` float32   | Per-constraint 99th-percentile residual.                     |
| `indices` | `(K,)` int         | Solver-side constraint indices for the columns above.        |

**W&B keys** use the **solver-side index** so the dashboard remains stable when you add or remove unrelated constraints later: `explainability/residual/constraint_{i}/{mean,std,max,p99}` and `.../histogram` (image).

---

## Input sensitivity / saliency

Evaluates an arbitrary jno placeholder expression at the training collocation points and records its value every `interval` outer steps. The intended use is *input-gradient saliency* — for a scalar network output $u$ and a coordinate variable $x$, $\partial u/\partial x$ measures how strongly the network response at a given point depends on that input dimension; high-magnitude regions are where small input perturbations produce large output changes (the PINN analogue of the class-saliency map of [[1312.6034](https://arxiv.org/abs/1312.6034)], Sec. 3 — Simonyan, Vedaldi & Zisserman, 2014).

```python
jno.callbacks.input_sensitivity(
    expr,
    interval = 100,
)
```

| Argument   | Type          | Default | Description                                                                                                                                                                  |
|------------|---------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `expr`     | `Placeholder` | —       | Any jno placeholder expression — see the table below for common choices. Any composite expression compiles via the same `TraceCompiler` pathway used by constraints/trackers. |
| `interval` | `int`         | `100`   | Compute every *n* outer training steps.                                                                                                                                      |

Common expressions to pass:

| Expression                       | Meaning                                              |
|----------------------------------|------------------------------------------------------|
| `u.d(x)`                         | $\partial u/\partial x$ — scalar per point           |
| `jno.Jacobian(u, [x, y])`        | full input Jacobian — shape `(N, 2)` for 2-D inputs  |
| `u.d(x)**2 + u.d(y)**2`          | squared $\lvert\nabla u\rvert^2$ as a scalar field   |

**`cb.result` keys**

| Key      | Shape                  | Description                              |
|----------|------------------------|------------------------------------------|
| `epochs` | `(S,)` int             | Sampled outer-step indices.              |
| `values` | `(S, *expr_shape)`     | Expression evaluated at the collocation points. |

**W&B keys:** `explainability/saliency/{mean_abs,max_abs,std_abs}`, `.../histogram` (image).

---

## Empirical NTK spectrum

Compiles a `NetworkGradient` placeholder to obtain the per-point parameter Jacobian $J \in \mathbb{R}^{N \times P}$, subsamples `n_points` rows (with a fixed seed so the same points are used at every call), and reports the eigenvalue spectrum of the empirical NTK $K = J J^\top$. A wide spread between the largest and smallest eigenvalues is the canonical diagnostic for PINN spectral bias — directions in parameter space train orders of magnitude faster than others, the classic mechanism behind PINNs lagging on high-frequency features (Sec. 3-4 of [[2007.14527](https://arxiv.org/abs/2007.14527)] — Wang, Wang & Perdikaris, 2022).

$$K_{ij} = \langle \nabla_\theta u(x_i), \, \nabla_\theta u(x_j) \rangle$$

```python
jno.callbacks.ntk_spectrum(
    grad_expr,
    n_points = 256,
    top_k    = 10,
    interval = 500,
)
```

| Argument    | Type                | Default | Description                                                                                                                                  |
|-------------|---------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `grad_expr` | `NetworkGradient`   | —       | Built from `expr.grad(model)`, e.g. `u.grad(u_net)`. Restrict to a parameter subset by chaining `net.mask(mask)` *inside* the placeholder.   |
| `n_points`  | `int`               | `256`   | Subsample cap for kernel rows. Cost is $O(n\_\text{points}^2 \times P)$.                                                                     |
| `top_k`     | `int`               | `10`    | Number of largest eigenvalues to retain.                                                                                                     |
| `interval`  | `int`               | `500`   | Compute every *n* outer training steps. Keep large for real runs.                                                                            |

!!! warning "Cost"
    Use **both** subsampling (`n_points`) **and** placeholder masking on large networks. Scalar output only — for vector-valued $u$, project first (e.g. `u[..., 0].grad(net)`).

**`cb.result` keys**

| Key                | Shape                | Description                                       |
|--------------------|----------------------|---------------------------------------------------|
| `epochs`           | `(S,)` int           | Sampled outer-step indices.                       |
| `eigvals_topk`     | `(S, top_k)` float32 | Top-$k$ eigenvalues (descending).                 |
| `lambda_min`       | `(S,)` float32       | Smallest eigenvalue of the subsampled kernel.     |
| `lambda_max`       | `(S,)` float32       | Largest eigenvalue.                               |
| `condition_number` | `(S,)` float32       | $\lambda_{\max} / \lambda_{\min}$.                |
| `all_eigvals`      | `(S, n_points)`      | Full eigenvalue spectrum (descending).            |

**W&B keys:** `explainability/ntk/eigval_{0..k-1}`, `.../lambda_max`, `.../lambda_min`, `.../condition_number`, `.../spectrum_hist`.

---

## Hessian eigenspectrum (sharpness)

Computes the top-$k$ eigenvalues of the total training loss Hessian $\nabla^2_\theta L$ via Lanczos with Hessian-vector products and full reorthogonalisation (Sec. 3.1-3.2 of [[1912.07145](https://arxiv.org/abs/1912.07145)] — Yao et al., 2020). The largest eigenvalue is the **sharpness** of the loss surface at the current iterate (Sec. 2.2 of [[1609.04836](https://arxiv.org/abs/1609.04836)] — Keskar et al., 2017), with high values predicting a sharp minimum typically associated with worse generalisation.

```python
jno.callbacks.hessian_spectrum(
    k           = 10,
    n_iter      = 30,
    interval    = 500,
    mask        = None,
    constraints = None,
)
```

| Argument      | Type                       | Default | Description                                                                                                                                                                                  |
|---------------|----------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `k`           | `int`                      | `10`    | Number of largest eigenvalues to report.                                                                                                                                                     |
| `n_iter`      | `int`                      | `30`    | Number of Lanczos iterations. Each is roughly one full forward+backward pass.                                                                                                                |
| `interval`    | `int`                      | `500`   | Compute every *n* outer training steps.                                                                                                                                                      |
| `mask`        | pytree of `bool`           | `None`  | Restrict the Hessian to a parameter subset. Essential for large models.                                                                                                                      |
| `constraints` | `list[Placeholder] \| None`| `None`  | Scope the Hessian to the *mean* of the selected constraint losses instead of the full training loss — useful for diagnosing which constraint drives the sharpness. Same identity rules as [residual statistics](#residual-statistics). |

!!! warning "Cost"
    Keep `interval` large (500–1000) for real runs and use `mask` to restrict to a parameter subset.

**`cb.result` keys**

| Key         | Shape           | Description                                |
|-------------|-----------------|--------------------------------------------|
| `epochs`    | `(S,)` int      | Sampled outer-step indices.                |
| `eigvals`   | `(S, k)` float32| Top-$k$ eigenvalues (descending).          |
| `sharpness` | `(S,)` float32  | Largest eigenvalue.                        |

**W&B keys:** `explainability/hessian/eigval_{0..k-1}`, `.../sharpness`, `.../n_iter`.

---

## Loss landscape

At every `interval` steps, two random filter-normalised directions are sampled and the total loss is evaluated on an $(n\_\text{grid} \times n\_\text{grid})$ perturbation grid centred on the current parameters (based on [[1712.09913](https://arxiv.org/abs/1712.09913)] — Li et al., 2018). A smooth bowl indicates a well-conditioned landscape; sharp ridges or flat saddle regions may explain slow convergence or oscillating loss.

```python
jno.callbacks.loss_landscape(
    interval    = 500,
    mask        = None,
    n_grid      = 15,
    alpha_range = 1.0,
)
```

| Argument      | Type             | Default | Description                                                                                          |
|---------------|------------------|---------|------------------------------------------------------------------------------------------------------|
| `interval`    | `int`            | `500`   | Compute every *n* outer training steps. Each call costs $n\_\text{grid}^2$ forward passes.           |
| `mask`        | pytree of `bool` | `None`  | Only the selected parameters are perturbed; the rest stay at their current values. Strongly recommended for large models. |
| `n_grid`      | `int`            | `15`    | Grid points per axis. Total evaluations = $n\_\text{grid}^2$.                                        |
| `alpha_range` | `float`          | `1.0`   | Perturbation range in units of $\|\theta_\text{selected}\|$.                                         |

!!! warning "Cost"
    Each call requires $n\_\text{grid}^2$ full forward passes. Keep `interval` large (500–1000) for real training runs, or use `mask` to restrict perturbations to a small subset of parameters.

**`cb.result` keys**

| Key          | Shape                      | Description                                            |
|--------------|----------------------------|--------------------------------------------------------|
| `epochs`     | `(S,)` int                 | Sampled outer-step indices.                            |
| `landscapes` | `(S, n_grid, n_grid)` f32  | Loss evaluated on the 2-D perturbation grid each call. |

**W&B key:** `explainability/loss_landscape` (heatmap image).

---

## Restricting to a parameter subset

The gradient-analysis callbacks (`gradient_norms`, `cos_similarity`, `gradient_alignment`, `hessian_spectrum`, `loss_landscape`) accept an optional `mask` — a pytree of booleans matching the `trainable` structure. Only the selected parameters are differentiated or perturbed. This is essential for large networks and strongly recommended even for medium-sized ones. (`residual_stats` does not need a mask — it inspects pre-existing constraint residuals rather than computing parameter gradients. `ntk_spectrum` masks via `net.mask(...)` *inside* its `grad_expr`.)

```python
import equinox as eqx, jax

all_false   = jax.tree_util.tree_map(lambda _: False, u_net.params)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

cb_norms = jno.callbacks.gradient_norms(interval=50, mask=output_mask)
cb_land  = jno.callbacks.loss_landscape(interval=500, mask=output_mask, n_grid=11)
```

The output-layer weight matrix typically gives the dominant gradient directions at a fraction of the cost of the full parameter set.

---

## Driving adaptive loss balancing from a tracker

Two weight schemes consume tracker objects directly, reading their live `.value` inside the loss evaluation. Both follow the same staleness pattern as `ReLoBRaLo`: the tracker updates on its `interval`, the weight scheme reads the most recent cached value every step.

```python
from jno.utils.adaptive.weights import gradient_norm_balanced, ntk_balanced
```

### Gradient-norm balancing

`gradient_norm_balanced(tracker)` reads `tracker.value["norms"]` and emits $w_i \propto 1 / \lVert\nabla L_i\rVert$, normalised so the weights sum to $N$. A constraint with a large gradient norm is down-weighted so it stops dominating the parameter update.

```python
gn = jno.trackers.gradient_norms(interval=50)
w  = gradient_norm_balanced(gn)

# Use the balancer just like any other host-side weight scheme — pass the
# scalar per-loss values and multiply by the returned weights inside the
# constraint expression.
w_pde, w_bc = w(pde_loss_scalar, bc_loss_scalar)
```

### NTK-trace balancing (Wang, Yu & Perdikaris, 2022)

`ntk_balanced([ntk_a, ntk_b, ...])` takes one NTK tracker per loss term — each measuring the kernel of *that* loss's network output — and emits $w_i = \mathrm{tr}(K_\text{total}) / \mathrm{tr}(K_i)$, normalised to sum to $N$. Constraints whose NTK trace is small converge slowly and receive more weight.

```python
ntk_pde = jno.trackers.ntk_spectrum(pde.grad(net), n_points=128, interval=200)
ntk_bc  = jno.trackers.ntk_spectrum(bc.grad(net),  n_points=128, interval=200)
w       = ntk_balanced([ntk_pde, ntk_bc], ema=0.9)

# Register both trackers AND the balancer in the solver's callback list so
# the trackers actually fire during training.
crux.solve(10_000, callbacks=[ntk_pde, ntk_bc])
w_pde, w_bc = w(pde_loss_scalar, bc_loss_scalar)
```

Until *every* listed tracker has fired at least once, the scheme returns uniform weights — the cold-start fallback. Reference: Sec. 3 of [[2007.14527](https://arxiv.org/abs/2007.14527)]. EMA convention: `ema=0.0` means "use the latest measurement immediately"; higher values smooth toward history (same direction as `ReLoBRaLo`'s `alpha`).

---

## W&B logging

All callbacks push their results to W&B automatically when a run is active (see [Weights & Biases](../misc/wandb.md)). No extra code is needed — enabling `jno.setup(..., wandb=True)` is sufficient.

---

## Combined example

Every callback plugs into `crux.solve(...)` the same way — pass them all in a single `callbacks=[...]` list. Cheap diagnostics (`interval=50–100`) and expensive ones (`interval=500–1000`) coexist without conflict; each maintains its own pre-compiled JAX function and writes its own `cb.result`.

```python
import jno
# `output_mask` constructed as shown in "Restricting to a parameter subset" above.

# --- Cheap, every 50 steps ---
cb_norms = jno.callbacks.gradient_norms(interval=50)
cb_cos   = jno.callbacks.cos_similarity(interval=50)
cb_align = jno.callbacks.gradient_alignment(interval=50)
cb_res   = jno.callbacks.residual_stats(interval=50)
cb_sal   = jno.callbacks.input_sensitivity(u.d(x), interval=100)

# --- Expensive, every 500 steps; mask to the output-layer weights ---
cb_ntk   = jno.callbacks.ntk_spectrum(u.grad(u_net.mask(output_mask)), n_points=128)
cb_hess  = jno.callbacks.hessian_spectrum(k=5, n_iter=20, interval=500, mask=output_mask)
cb_land  = jno.callbacks.loss_landscape(interval=500, mask=output_mask, n_grid=11)

crux.solve(
    10_000,
    callbacks=[
        cb_norms, cb_cos, cb_align, cb_res, cb_sal,
        cb_ntk, cb_hess, cb_land,
        jno.callbacks.checkpoint(save_interval_epochs=1000),
        jno.callbacks.early_stopping(patience=2000),
    ],
)

# After training — every cb.result is a plain numpy dict
print(cb_norms.result["norms"].shape)         # (S, N)
print(cb_align.result["alignment"])           # (S,) in [-1, 1]
print(cb_ntk.result["condition_number"])      # (S,)
print(cb_hess.result["sharpness"])            # (S,)
```
