# Training & Model Controls

This page covers the complete jNO training pipeline: constructing the core solver, running training, parallelism, evaluation and profiling, learning rate schedules, adaptive loss weights, adaptive resampling, callbacks, and per-model controls (freeze, LoRA, dtype, diagnostics, IREE deployment).

## Contents

- [Training](#training)
- [Parallelism](#parallelism)
- [Evaluation & Explainability](#evaluation--explainability)
- [Schedules](#schedules)
- [Adaptive Resampling](#adaptive-resampling)
- [Callbacks](#callbacks)
- [Model Controls](#model-controls)
- [Mask & Freeze](#mask--freeze)
- [LoRA & Parameter-Efficient Fine-Tuning](#lora--parameter-efficient-fine-tuning)
- [Optimizer & Learning Rate](#optimizer--learning-rate)
- [Initialize, Dtype & Tune](#initialize-dtype--tune)
- [Diagnostics](#diagnostics)
- [IREE Deployment](#iree-deployment)

---

## Training

`jno.core` is the central training object. It:

1. Builds the symbolic computation graph from your constraints.
2. Performs common sub-expression elimination (CSE).
3. Initialises all neural-network parameters.
4. Compiles a JIT-optimised step function.
5. Runs the training loop and returns training statistics.

```python
crux = jno.core(
    constraints=[pde.mse, boc.mse], 
    rng_seed=42,                       # optional; also set in .jno.toml → [jno] seed
    mesh=(1, 1),                       # (batch_devices, model_devices)
)
```

### Attaching Optimisers

**Every non-frozen model must have an optimiser before calling `solve()`.**

```python
u_net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.9, 2000, 1e-5))
v_net.optimizer(optax.adamw).scale(lrs.warmup_cosine(5000, 500, 1e-3, 1e-4))
```

`model.optimizer()` returns `self` for chaining:

```python
u_net = jno.nn.mlp(2, key=key).optimizer(optax.adam).scale(lrs(1e-3))
```

#### After `core.load()`

When loading a saved solver the `Model` references in the expression tree are disconnected from Python variables. Use `set_optimizer` to reassign:

```python
crux = jno.core.load("runs/crux.pkl")
crux.set_optimizer(optax.adam, scale=lrs(1e-4))
crux.solve(1000)
```

### Running Training

```python
stats = crux.solve(
    epochs=5000,
    batchsize=128,              # None = full batch (all collocation points)
    checkpoint_gradients=False, # True → gradient checkpointing (saves memory, ~30% slower)
    offload_data=False,         # True → keep dataset on CPU, stream mini-batches
)
stats.plot("history.png")
```

Returns a `statistics` object with `.plot()` and loss arrays.

#### Memory Optimisations

| Option | Effect | Use When |
|--------|--------|----------|
| `batchsize=N` | Mini-batch gradient estimation | Dataset doesn't fit in GPU memory |
| `checkpoint_gradients=True` | Rematerialise activations during backward pass | Very deep networks or long time sequences |
| `offload_data=True` | Keep dataset on CPU; stream each mini-batch | Very large datasets |

`offload_data` requires `batchsize < total_samples`.

### Multi-Phase Training

Call `solve()` multiple times with different optimisers or schedules. The solver resumes from where it left off:

```python
# Phase 1: Adam warm-up
u_net.optimizer(optax.adam).scale(lrs.warmup_cosine(3000, 300, 1e-3, 1e-5))
crux.solve(3000).plot("phase1.png")

# Phase 2: L-BFGS quasi-Newton refinement
u_net.optimizer(optax.lbfgs).scale(lrs(5e-5))
crux.solve(500).plot("phase2.png")

# Phase 3: SOAP second-order method
from soap_jax import soap
u_net.optimizer(soap(1)).scale(lrs(1e-5))
crux.solve(500).plot("phase3.png")
```

---

## Parallelism

jNO supports data parallelism, model parallelism, and hybrid parallelism via JAX's device mesh.

### Device Mesh

```python
# No parallelism (single device, default)
crux = jno.core(constraints,  mesh=(1, 1))

# Pure data parallelism: split batches across 4 GPUs
crux = jno.core(constraints,  mesh=(4, 1))

# Pure model parallelism: shard model weights across 2 GPUs
crux = jno.core(constraints,  mesh=(1, 2))

# Hybrid (2 data × 2 model = 4 GPUs total)
crux = jno.core(constraints,  mesh=(2, 2))

# Auto-scale to all available devices
n = len(jax.devices())
crux = jno.core(constraints,  mesh=(n, 1))
```

### Mesh Shape Rules

- `batch × model` must equal the total number of available devices.
- Data parallelism (`(n, 1)`) maximises throughput when the model fits on a single device.
- Model parallelism (`(1, n)`) allows training models too large for a single device.

---

## Evaluation & Explainability

### Evaluation

After training, use `crux.eval()` to evaluate any symbolic expression:

```python
# On the training domain
pred = crux.eval(u)    # shape: (B, T, N, out_dim)

# On a different domain (e.g., fine test grid)
test_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
pred_fine = crux.eval(u, domain=test_domain)

# Prediction on arbitrary point arrays
import numpy as np
points = np.tile(test_domain.points[None, ...], (B, 1, 1))  # (B, N, 2)
pred = crux.predict(points=points, operation=u, context=test_domain.context)
```

### Debugging

#### Print Computation Tree

```python
crux.print_tree()              # to stdout
crux.print_tree("tree.txt")    # to file
```

#### Print Tensor Shapes

```python
crux.print_shapes()   # per-node shape trace for all constraints and trackers
```

#### Debug Print Inside Expressions

Use JAX's debug print inside expressions for verbose intermediate inspection (expensive — use sparingly):

```python
pde = jno.np.laplacian(u, [x, y]) + 1.0
pde.debug._shape = True   # print shape at this node each step
pde.debug._mean = True    # print mean value
```

### Profiler

Pass `profile=True` to `solve()` to capture a JAX performance trace:

```python
history = crux.solve(5000, profile=True)
```

jNO skips the first outer step (startup JIT compilation) and then records the next 50 steady-state steps. The traces are written to `<logger.path>/traces/` in **Perfetto format** — open them at [ui.perfetto.dev](https://ui.perfetto.dev) to inspect the timeline.

#### What the trace shows

| Symptom | Likely cause |
|---------|-------------|
| A single long "unflatten" span every few steps | Python GC pause — jNO disables cyclic GC during training to suppress these |
| Many short gaps between ops | Host–device sync points; consider fusing ops or using `jax.block_until_ready` less aggressively |
| One constraint takes 10× longer than others | That expression has much higher compute cost — consider finite-difference vs AD trade-off |
| Compile time dominates the first step | Normal for JIT; the skip-first-step logic means this is excluded from the captured trace |

#### Controlling the output directory

```python
import jno
jno.setup("runs/my_experiment")   # traces go to runs/my_experiment/traces/

crux = jno.core([pde.mse])
history = crux.solve(5000, profile=True)
```

- `profile=True` is a **side effect only** — it does not change the return value of `solve()`.
- Profiling adds negligible overhead to the captured steps themselves; the trace serialisation happens asynchronously.
- For very short runs (fewer than 52 epochs) the capture window is clamped: `min(50, epochs - 1)` steps are recorded.

### Training Statistics

`solve()` returns a `statistics` object:

```python
stats = crux.solve(5000)

stats.plot("history.png")           # save loss curves

# Access raw data
stats.epoch                          # epoch indices
stats.total_loss                     # total weighted loss per logged epoch
stats.losses                         # per-constraint losses, shape (log_steps, n_constraints)
stats.weights                        # constraint weights, shape (log_steps, n_constraints)
stats.training_time                  # wall-clock time in seconds
stats.trainable_params               # number of trainable parameters
stats.total_params                   # total parameters
```

### Checkpoints

`solve()` automatically saves a checkpoint (model weights, optimiser state, RNG key) after every call. All checkpoints are accessible via `crux.checkpoints`:

```python
# Inspect available checkpoints
for i, ckpt in enumerate(crux.checkpoints):
    print(i, ckpt["step"], ckpt["time"])

# Restore a specific checkpoint
crux.models = crux.checkpoints[-1]["models"]
```

### Explainability Trackers

A family of trackers gives insight into what is happening inside the training loop. They differentiate through the constraint functions after each outer step, or directly inspect residuals, independently of the gradient updates that drive training. Results are stored as numpy arrays on the tracker object and, when a W&B run is active, pushed automatically to your dashboard.

Trackers are surfaced under two equivalent namespaces:

- **`jno.trackers.*`** — preferred; matches the tracker mental model.
- **`jno.callbacks.*`** — historical entry point; remains supported.

#### Live access: `tracker.value` vs `tracker.result`

| Attribute             | Updated when                                  | Type                | Used for                                                                                  |
|-----------------------|-----------------------------------------------|---------------------|-------------------------------------------------------------------------------------------|
| `tracker.value`       | Every time `epoch % interval == 0` fires      | `dict | None`       | **Live**: read by adaptive components (loss balancing) at the next step. `None` until the first interval. |
| `tracker.latest_epoch`| Same as `tracker.value`                       | `int | None`        | Tells consumers how stale `value` is.                                                     |
| `tracker.result`      | After `crux.solve()` returns                  | `dict[str, ndarray]`| **Post-training**: full history of every fire — `epochs` plus the per-metric stacked array. |

#### Gradient norms

Tracks $\|\nabla L_i\|_2$ for each constraint $i$ every `interval` outer steps. A constraint whose norm is orders of magnitude larger than the others will dominate the parameter update regardless of its loss value.

```python
jno.callbacks.gradient_norms(
    interval = 100,
    mask     = None,
)
```

| Argument   | Type             | Default | Description                                                                                          |
|------------|------------------|---------|------------------------------------------------------------------------------------------------------|
| `interval` | `int`            | `100`   | Compute every *n* outer training steps.                                                              |
| `mask`     | pytree of `bool` | `None`  | Restrict the differentiated parameter subset. Recommended for large models. |

**`cb.result` keys**

| Key      | Shape              | Description                              |
|----------|--------------------|------------------------------------------|
| `epochs` | `(S,)` int         | Sampled outer-step indices.              |
| `norms`  | `(S, N)` float32   | Per-constraint gradient $L_2$ norms.     |

#### Cosine similarity

Computes the full $(N \times N)$ pairwise cosine similarity matrix between constraint gradients every `interval` outer steps.

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

#### Gradient alignment

A single scalar measuring global agreement across *all* constraints (Eq. 3.1 of [[2502.00604](https://arxiv.org/abs/2502.00604)]). Each gradient is unit-normalised first, so the metric reflects pure direction agreement and is invariant to per-loss scale.

$$\text{alignment} \;=\; 2\left\|\frac{1}{N}\sum_{i=1}^{N} \frac{\nabla L_i}{\|\nabla L_i\|}\right\|^2 - 1$$

```python
jno.callbacks.gradient_alignment(
    interval = 100,
    mask     = None,
)
```

#### Residual statistics

For each constraint $i$, evaluates the un-reduced residual array $r_i$ and records mean, std, max, and 99th percentile, plus a histogram when W&B is active (Sec. 3 of [[2207.10289](https://arxiv.org/abs/2207.10289)]).

```python
jno.callbacks.residual_stats(
    interval    = 100,
    constraints = None,
)
```

#### Input sensitivity / saliency

Evaluates an arbitrary jno placeholder expression at the training collocation points every `interval` steps — use for input-gradient saliency (PINN analogue of [[1312.6034](https://arxiv.org/abs/1312.6034)]).

```python
jno.callbacks.input_sensitivity(
    expr,
    interval = 100,
)
```

Common expressions:

| Expression                       | Meaning                                              |
|----------------------------------|------------------------------------------------------|
| `u.d(x)`                         | $\partial u/\partial x$ — scalar per point           |
| `jno.Jacobian(u, [x, y])`        | full input Jacobian — shape `(N, 2)` for 2-D inputs  |
| `u.d(x)**2 + u.d(y)**2`          | squared $\lvert\nabla u\rvert^2$ as a scalar field   |

#### Empirical NTK spectrum

Reports the eigenvalue spectrum of the empirical NTK $K = J J^\top$ — a wide spread diagnoses spectral bias (Sec. 3-4 of [[2007.14527](https://arxiv.org/abs/2007.14527)]).

```python
jno.callbacks.ntk_spectrum(
    grad_expr,
    n_points = 256,
    top_k    = 10,
    interval = 500,
)
```

!!! warning "Cost"
    Use **both** subsampling (`n_points`) **and** placeholder masking on large networks. Scalar output only — for vector-valued $u$, project first (e.g. `u[..., 0].grad(net)`).

#### Hessian eigenspectrum (sharpness)

Top-$k$ eigenvalues of the total training loss Hessian via Lanczos (Sec. 3.1-3.2 of [[1912.07145](https://arxiv.org/abs/1912.07145)]).

```python
jno.callbacks.hessian_spectrum(
    k           = 10,
    n_iter      = 30,
    interval    = 500,
    mask        = None,
    constraints = None,
)
```

!!! warning "Cost"
    Keep `interval` large (500–1000) and use `mask` to restrict to a parameter subset.

#### Loss landscape

Two random filter-normalised directions are sampled and the total loss is evaluated on an $n\_\text{grid} \times n\_\text{grid}$ perturbation grid (based on [[1712.09913](https://arxiv.org/abs/1712.09913)]).

```python
jno.callbacks.loss_landscape(
    interval    = 500,
    mask        = None,
    n_grid      = 15,
    alpha_range = 1.0,
)
```

!!! warning "Cost"
    Each call requires $n\_\text{grid}^2$ full forward passes. Keep `interval` large (500–1000).

#### Restricting to a parameter subset

The gradient-analysis callbacks accept an optional `mask` — a pytree of booleans. Only the selected parameters are differentiated or perturbed.

```python
import equinox as eqx, jax

all_false   = jax.tree_util.tree_map(lambda _: False, u_net.params)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

cb_norms = jno.callbacks.gradient_norms(interval=50, mask=output_mask)
cb_land  = jno.callbacks.loss_landscape(interval=500, mask=output_mask, n_grid=11)
```

#### Driving adaptive loss balancing from a tracker

```python
from jno.utils.adaptive.weights import gradient_norm_balanced, ntk_balanced
```

**Gradient-norm balancing** — emits $w_i \propto 1 / \lVert\nabla L_i\rVert$, normalised so weights sum to $N$:

```python
gn = jno.trackers.gradient_norms(interval=50)
w  = gradient_norm_balanced(gn)
w_pde, w_bc = w(pde_loss_scalar, bc_loss_scalar)
```

**NTK-trace balancing** (Wang, Yu & Perdikaris, 2022) — emits $w_i = \mathrm{tr}(K_\text{total}) / \mathrm{tr}(K_i)$, normalised to sum to $N$ (Sec. 3 of [[2007.14527](https://arxiv.org/abs/2007.14527)]):

```python
ntk_pde = jno.trackers.ntk_spectrum(pde.grad(net), n_points=128, interval=200)
ntk_bc  = jno.trackers.ntk_spectrum(bc.grad(net),  n_points=128, interval=200)
w       = ntk_balanced([ntk_pde, ntk_bc], ema=0.9)
crux.solve(10_000, callbacks=[ntk_pde, ntk_bc])
w_pde, w_bc = w(pde_loss_scalar, bc_loss_scalar)
```

#### Combined example

```python
import jno
# `output_mask` constructed as shown above.

cb_norms = jno.callbacks.gradient_norms(interval=50)
cb_cos   = jno.callbacks.cos_similarity(interval=50)
cb_align = jno.callbacks.gradient_alignment(interval=50)
cb_res   = jno.callbacks.residual_stats(interval=50)
cb_sal   = jno.callbacks.input_sensitivity(u.d(x), interval=100)

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

print(cb_norms.result["norms"].shape)         # (S, N)
print(cb_align.result["alignment"])           # (S,) in [-1, 1]
print(cb_ntk.result["condition_number"])      # (S,)
print(cb_hess.result["sharpness"])            # (S,)
```

---

## Schedules

### Learning Rate

The simplest way to set a learning rate is to bake it into the optax constructor:

```python
net.optimizer(optax.adam(1e-3))
net.optimizer(optax.adamw(5e-4, weight_decay=1e-2))
```

optax chains work the same way:

```python
net.optimizer(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-3),
    )
)
```

For **dynamic schedules**, attach the schedule with `.scale(...)`. Construct the optimizer with a placeholder rate of `1` so `.scale` sets the effective learning rate:

```python
from jno import LearningRateSchedule as lrs

net.optimizer(optax.adam(1)).scale(lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

A loss-adaptive `dlrs(...)` plugs in the same way:

```python
net.optimizer(optax.adam(1)).scale(jno.fn.adaptive.dlrs(lr0=1e-3, window=10))
```

### Learning Rate Schedules

`LearningRateSchedule` wraps **any** callable `(epoch, individual_losses) → scalar`:

```python
from jno import LearningRateSchedule as lrs

# Any (epoch, losses) -> scalar callable is a schedule
lrs(lambda epoch, losses: 1e-4 * (0.9 ** (epoch / 500)))

# Adapt to a runtime signal
lrs(lambda epoch, losses: 1e-3 if losses[0] > 1e-2 else 1e-5)
```

#### Built-in schedule factories

```python
# Constant
lrs.constant(1e-3)
lrs(1e-3)          # shorthand

# Exponential decay: lr(t) = max(lr_end, lr0 * decay_rate^(t/decay_steps))
lrs.exponential(lr0=1e-3, decay_rate=0.9, decay_steps=1000, lr_end=1e-5)

# Cosine decay
lrs.cosine(total_steps=5000, lr0=1e-3, lr_end=1e-6)

# Linear warm-up then cosine decay
lrs.warmup_cosine(total_steps=5000, warmup_steps=500, lr0=1e-3, lr_end=1e-6)

# Piecewise constant
lrs.piecewise_constant(
    boundaries=[1000, 3000],
    values=[1e-3, 5e-4, 1e-4],   # len(boundaries) + 1 values
)
```

All factories accept `min_lr` and `max_lr` keyword arguments to clamp the output.

### Adaptive Loss Weights

Loss weights are **traced placeholders** — call an adaptive balancer with your losses before passing them to `jno.core`:

```python
w_pde, w_bc = jno.fn.adaptive.relobralo([pde, bc])

crux = jno.core([w_pde * pde, w_bc * bc])
```

The weights are recomputed inside the compiled JAX function every step — no Python callback overhead.

#### Logging weights

```python
crux = jno.core([w_pde * pde, w_bc * bc, w_pde.tracker(), w_bc.tracker()])
```

#### Available balancers

**`relobralo`** — Relative Loss Balancing via Residual Algorithms. Balances losses relative to their initial values.

```python
w0, w1 = jno.fn.adaptive.relobralo(
    [pde, bc],
    alpha=0.99,          # exponential moving average factor
    tau=0.1,             # temperature for softmax normalisation
    expected_rho=0.999,  # target ratio for balancing
    seed=42,
)
```

**`softadapt`** — Weights losses by the softmax of their recent rate of change.

```python
w0, w1 = jno.fn.adaptive.softadapt([pde, bc], beta=0.1)
```

**`dwa`** — Dynamic Weight Average. Weights by ratio of current to previous-step value.

```python
w0, w1 = jno.fn.adaptive.dwa([pde, bc], temperature=2.0)
```

**`lbpinns_loss_balancing`** — Learnable log-variance weights updated via internal Adam step.

```python
w0, w1 = jno.fn.adaptive.lbpinns_loss_balancing([pde, bc], init_s=0.0, lr_s=1e-2)
```

**`rlw`** — Random Loss Weighting. Draws weights from a Dirichlet distribution each step.

```python
w0, w1 = jno.fn.adaptive.rlw([pde, bc], alpha=1.0, seed=42)
```

#### Loss preprocessing (`mode`)

All balancers accept a `mode` keyword:

| `mode` | Effect |
|--------|--------|
| `"raw"` (default) | Use loss values as-is |
| `"minmax"` | Scale each loss to [0, 1] over its observed range |
| `"l2"` | Normalise by the L2 norm of the loss vector |

```python
w0, w1 = jno.fn.adaptive.relobralo([pde, bc], mode="minmax")
```

---

## Adaptive Resampling

Adaptive resampling strategies dynamically replace collocation points during training to focus on regions of high PDE residuals or high predicted error. All strategies are available via the `jno.sampler` factory class.

### Why Adaptive Resampling?

Standard PINNs fix collocation points at the start of training. In problems with sharp gradients, boundary layers, or discontinuities, a uniform distribution is inefficient. Adaptive strategies periodically replace a fraction of points based on some criterion (residual, gradient, influence), concentrating them where the network struggles.

### Build your own strategy

Any subclass of `ResamplingStrategy` is a valid strategy — override one abstract method:

```python
def resample(self, points, residuals, domain, tag, epoch, rng_key, candidates=None) -> jnp.ndarray
```

```python
from jno.utils.adaptive.resampling import ResamplingStrategy
import jax, jax.numpy as jnp

class TopResidual(ResamplingStrategy):
    """Replace the worst-residual points with fresh draws from the candidate pool."""

    def resample(self, points, residuals, domain, tag, epoch, rng_key, candidates=None):
        if residuals.ndim > 1:
            residuals = jnp.mean(residuals, axis=0)
        n_replace = int(len(points) * self.resample_fraction)
        n_keep    = len(points) - n_replace
        order     = jnp.argsort(residuals)
        kept      = points[order[:n_keep]]
        idx       = jax.random.choice(rng_key, len(candidates), shape=(n_replace,))
        return jnp.concatenate([kept, candidates[idx]], axis=0)

x, y = domain.variable(
    "interior",
    sample=(None, None),
    resampling_strategy=TopResidual(resample_every=100, resample_fraction=0.2, start_epoch=1000),
)
```

### Built-in strategies

#### `sampler.random` — Baseline

Replaces a random subset of points from a new uniform sample.

```python
from jno import sampler

s = sampler.random(resample_every=100, resample_fraction=0.1, start_epoch=1000)
```

#### `sampler.rad` — Residual-Adaptive Distribution

Selects new points by clustering around the top-`k` points with the highest PDE residuals.

```python
s = sampler.rad(resample_every=100, resample_fraction=0.1, start_epoch=1000, k=10)
```

**Reference:** Lu et al., "Residual-based adaptivity for two-phase flow simulation in porous media using Physics-informed Neural Networks"

#### `sampler.rard` — Residual-Adaptive Refinement with Distribution

Uses importance sampling based on `residual^power`.

```python
s = sampler.rard(resample_every=100, resample_fraction=0.1, start_epoch=1000, power=2.0)
```

#### `sampler.ha` — Hybrid Adaptive

Alternates between random-refresh phases and adaptive phases.

```python
s = sampler.ha(
    resample_every=100,
    resample_fraction=0.5,
    start_epoch=1000,
    alternate=True,
    random_first=True,
)
```

#### `sampler.cr3` — Causal Retain-Resample

Designed for **time-dependent PDEs** where causality matters. Uses a learnable causal gate that progressively exposes later time steps.

```python
s = sampler.cr3(
    resample_every=100,
    resample_fraction=0.5,
    start_epoch=1000,
    t_index=-1,
    alpha=5.0,
    gamma0=-0.5,
    eta_g=1e-3,
    epsilon=20.0,
    delta_max=0.1,
    min_keep_frac=0.1,
    max_keep_frac=0.9,
)
```

**Reference:** Adapted from "Respecting Causality for Training Physics-Informed Neural Networks"

#### `sampler.pinnfluence` — PINNFluence

Uses gradient-based influence scores to identify high-impact collocation points.

```python
s = sampler.pinnfluence(
    resample_every=500,
    resample_fraction=0.2,
    start_epoch=2000,
    alpha=1.0,
    c=1.0,
    candidate_factor=3.0,
)
```

### Usage

Attach the strategy when creating variables:

```python
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y = domain.variable(
    "interior",
    sample=(None, None),
    resampling_strategy=sampler.rad(resample_every=100, resample_fraction=0.1, start_epoch=1000),
)
```

### Strategy Comparison

| Strategy | Computational Cost | Sensitivity | Best For |
|----------|--------------------|-------------|----------|
| `random` | Very low | None (baseline) | Preventing grid overfitting |
| `rad` | Low | High-residual clustering | Localized high-error regions |
| `rard` | Low | Importance sampling | Smooth residual fields |
| `ha` | Low | Adaptive with regularization | General use, avoids over-concentration |
| `cr3` | Medium | Time-causal | Time-dependent PDEs |
| `pinnfluence` | High | Gradient-based influence | Small-data or high-accuracy regimes |

### Tips

- **Start epoch**: Always delay resampling (`start_epoch > 0`) to let the network form a rough global solution first.
- **Fraction**: Values between 0.1 and 0.3 are typical.
- **Resample interval**: Shorter intervals react faster but add overhead.
- **Combining with mini-batching**: Adaptive resampling works with `batchsize` in `solve()`.

---

## Callbacks

Callbacks hook into the training loop without modifying the solver itself. Pass a list of callbacks to `solve()`:

```python
crux.solve(10000, callbacks=[cb1, cb2])
```

Each callback's `on_epoch_end` is called after every outer training step and can optionally signal early termination by returning `True`.

### Build your own callback

Any subclass of `Callback` is a valid callback — override one or more of three hooks:

| Hook | Signature | When it fires |
|------|-----------|---------------|
| `on_solve_begin(**kw)` | returns `None` | Once, after `solve()` finishes JIT setup, before the loop |
| `on_epoch_end(**kw)` | returns `bool` (`True` to stop training) | After every outer training step |
| `on_training_end(**kw)` | returns `None` | Once, after the loop finishes |

The most useful keys inside `on_epoch_end` are `epoch`, `total_loss`, `individual_losses`, `trainable`, `rng`, and `log`.

```python
from jno.utils.adaptive.callbacks import Callback

class LossPrinter(Callback):
    def __init__(self, every: int = 100):
        self.every = every

    def on_epoch_end(self, **kw) -> bool:
        if kw["epoch"] % self.every == 0:
            print(f"epoch {kw['epoch']}: loss = {float(kw['total_loss']):.4e}")
        return False

crux.solve(10_000, callbacks=[LossPrinter(every=500)])
```

### Built-in callbacks

#### Early Stopping

```python
cb = jno.callbacks.early_stopping(
    patience=1000,
    min_delta=1e-6,
    mode="min",        # "min", "max", or "rel"
)

crux.solve(100_000, callbacks=[cb])

print(cb.stopped_epoch)
print(cb.best_metric)
```

| `mode` | Stops when |
|--------|-----------|
| `"min"` | metric hasn't dropped by more than `min_delta` for `patience` epochs |
| `"max"` | metric hasn't risen by more than `min_delta` for `patience` epochs |
| `"rel"` | metric hasn't improved by a fraction of `min_delta` relative to best value |

Monitor a custom metric:

```python
cb = jno.callbacks.early_stopping(
    patience=500,
    metric_fn=lambda **kw: float(kw["individual_losses"][1]),
)
```

Starting from a baseline:

```python
cb = jno.callbacks.early_stopping(patience=500, baseline=1e-3)
```

#### Checkpointing

```python
cb = jno.callbacks.checkpoint(
    directory="runs/my_experiment/checkpoints",
    save_interval_epochs=500,
    max_to_keep=3,
)

crux.solve(10000, callbacks=[cb])
```

Keep the best checkpoint:

```python
cb = jno.callbacks.checkpoint(
    save_interval_epochs=200,
    max_to_keep=2,
    best_fn=lambda m: m["total_loss"],
)
```

Restore a checkpoint:

```python
state = cb.restore()          # latest
state = cb.restore(step=2000) # specific step
print(state["metadata"])      # {"epoch": 2000, "total_loss": ..., "timestamp": ...}
```

Async checkpointing is on by default (`async_checkpointing=True`). Set to `False` for synchronous writes:

```python
cb = jno.callbacks.checkpoint(async_checkpointing=False)
```

### Combining callbacks

```python
crux.solve(
    50_000,
    callbacks=[
        jno.callbacks.checkpoint(save_interval_epochs=1000, max_to_keep=3),
        jno.callbacks.early_stopping(patience=2000, mode="rel", min_delta=1e-3),
    ],
)
```

---

## Model Controls

The recommended style is:

- build a model with `foundax` (`fx.mlp`, `fx.fno2d`, `fx.poseidon.T`, ...) **or write your own `equinox.Module`**
- wrap it with `jno.nn.wrap(...)`
- apply model controls on the wrapped `Model`

`jno.nn.wrap` accepts any [Equinox](https://docs.kidger.site/equinox/) module.

```python
import optax
import foundax as fx
import jno
from jno import LearningRateSchedule as lrs

net = jno.nn.wrap(
    fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3),
    name="u_net",
)
net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.8, 5000, 1e-5))

# custom equinox model — works identically
import equinox as eqx

class MyNet(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [eqx.nn.Linear(2, 64, key=k1), eqx.nn.Linear(64, 1, key=k2)]

    def __call__(self, x, y):
        h = jax.nn.tanh(self.layers[0](jnp.stack([x, y])))
        return self.layers[1](h)

custom_net = jno.nn.wrap(MyNet(jax.random.PRNGKey(0)))
custom_net.optimizer(optax.adam(1e-3))
```

### Available Methods

`Model` (returned by `jno.nn.wrap(...)`) supports:

- `dont_show()`
- `summary()`
- `freeze()` / `unfreeze()`
- `mask(param_mask=None)`
- `lora(rank=4, alpha=1.0, *, target=None, wrapper=None, specs=None)`
- `optimizer(opt_fn)`
- `scale(schedule_or_scalar)`
- `initialize(weights_or_path_or_initializer, *, key=None)`
- `dtype(dtype)`
- `tune(...)`
- `reset()`

All methods return `self` and are chainable.

---

## Mask & Freeze

### Mask

`mask(...)` takes a boolean pytree mask:

```python
import equinox as eqx
import jax

all_true = jax.tree_util.tree_map(lambda _: True, eqx.filter(net.module, eqx.is_array))
net.mask(all_true).optimizer(optax.adam)
net.mask(all_true).scale(lrs(1e-4))
```

#### Regex-style targeting

```python
import re
import equinox as eqx
import jax

def regex_mask(module, pattern: str):
    arrays = eqx.filter(module, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(arrays)

    def part(k):
        if hasattr(k, "name"): return str(k.name)
        if hasattr(k, "idx"):  return str(k.idx)
        if hasattr(k, "key"):  return str(k.key)
        return str(k)

    leaves = []
    for path, _ in flat:
        path_str = "/".join(part(p) for p in path)
        leaves.append(bool(re.search(pattern, path_str)))

    return jax.tree_util.tree_unflatten(treedef, leaves)

decoder_mask = regex_mask(net.module, r"decoder")
net.mask(decoder_mask).optimizer(optax.adam)
net.mask(decoder_mask).scale(lrs(3e-4))
```

### Freeze / Unfreeze

```python
net.freeze()                      # freeze entire model
net.mask(decoder_mask).freeze()   # freeze only selected leaves
```

With `mask(...).freeze()`, non-selected leaves remain trainable.

---

## LoRA & Parameter-Efficient Fine-Tuning

LoRA inserts trainable low-rank adapter matrices into matching layers while keeping base weights frozen. By default both linear and conv layers are wrapped automatically.

### Selecting Layers

```python
net.lora(rank=8, target="encoder")                         # path-regex
net.mask(encoder_mask).lora(rank=8)                        # boolean mask
net.mask(encoder_mask).lora(rank=8, target="encoder")      # both combined
```

### Uniform and Per-Target Specs

```python
net.lora(rank=8, alpha=16)                    # all layers
net.lora(rank=8, alpha=16, target="encoder")  # restricted to a subset

net.lora(
    specs=[
        {"target": "encoder", "rank": 4,  "alpha": 1.0},
        {"target": "decoder", "rank": 16, "alpha": 4.0},
    ]
)
```

`target` is regex-matched against the slash-joined pytree path.

### Combining with Mask and Freeze

```python
net.mask(encoder_mask).lora(rank=8, alpha=16)
net.freeze().lora(rank=8, alpha=16)                     # freeze all; only adapters train
net.freeze().mask(encoder_mask).lora(rank=8, alpha=16)
```

### Default Layer Types

```python
net.lora(rank=4, alpha=1.0)           # Linear + Conv1d/2d/3d
net.lora(rank=4, wrapper=LoRALinear)  # linear only
net.lora(rank=4, wrapper=LoRAConv)    # conv only
```

### Build your own adapter

Any subclass of `LoRAWrapper` is a valid adapter — implement `applies_to`, `__init__`, `__call__`, and `merge()`:

```python
from jno.lora import LoRAWrapper
import equinox as eqx
import jax, jax.numpy as jnp

class MyEmbeddingAdapter(LoRAWrapper):
    adapter_fields = ("delta",)
    base: eqx.Module
    delta: jax.Array
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf):
        return isinstance(leaf, eqx.nn.Embedding) and not isinstance(leaf, LoRAWrapper)

    def __init__(self, base, rank, alpha, *, key):
        self.base, self.rank, self.alpha = base, rank, alpha
        self.delta = jnp.zeros_like(base.weight)

    def __call__(self, x):
        return self.base(x) + self.delta[x] * (self.alpha / self.rank)

    def merge(self):
        w = self.base.weight + self.delta * (self.alpha / self.rank)
        return eqx.tree_at(lambda m: m.weight, self.base, w)

net.lora(rank=4, wrapper=MyEmbeddingAdapter)
```

### LoRA Zoo — Linear

```python
from jno.lora import (
    LoRALinear,    # standard LoRA (default)
    rsLoRALinear,  # rank-stabilized
    LoRAFALinear,  # frozen A — fewer trainable params
    DoRALinear,    # weight-decomposed
    PiSSALinear,   # SVD init — fastest convergence on pretrained models
    LoRAXSLinear,  # extra-small r×r core
    VeRALinear,    # frozen random A,B; only b,d vectors trained
    MiLoRALinear,  # minor SVD components — preserves pretrained knowledge
    IA3Linear,     # output scaling vector — no low-rank matrices
    LoKrLinear,    # Kronecker product adapter
    OFTLinear,     # block-diagonal orthogonal fine-tuning
)
```

| Class | Trainable params | Key idea |
|-------|-----------------|----------|
| `LoRALinear` | `r·(in + out)` | Standard LoRA; scale = `α/r` |
| `rsLoRALinear` | `r·(in + out)` | Scale = `α/√r` — stable across ranks ([rsLoRA](https://arxiv.org/abs/2312.03732)) |
| `LoRAFALinear` | `r·out` | Frozen A; halves adapter params ([LoRAFA](https://arxiv.org/abs/2308.03303)) |
| `DoRALinear` | `r·(in + out) + out` | Magnitude + direction decomposition ([DoRA](https://arxiv.org/abs/2402.09353)) |
| `PiSSALinear` | `r·(in + out)` | SVD principal-component init ([PiSSA](https://arxiv.org/abs/2404.02948)) |
| `LoRAXSLinear` | `r²` | Frozen A,B; trainable r×r core only ([LoRA-XS](https://arxiv.org/abs/2405.17604)) |
| `VeRALinear` | `out + r` | Seed-based frozen A,B; only b,d vectors trained ([VeRA](https://arxiv.org/abs/2310.11454)) |
| `MiLoRALinear` | `r·(in + out)` | Adapts minor SVD components ([MiLoRA](https://arxiv.org/abs/2405.09913)) |
| `IA3Linear` | `out` | Per-output scale vector; no rank hyperparameter ([IA³](https://arxiv.org/abs/2205.05638)) |
| `LoKrLinear` | `r² + ⌈out/r⌉·⌈in/r⌉` | Kronecker product adapter ([LoKr](https://arxiv.org/abs/2212.10650)) |
| `OFTLinear` | `n_blocks·r²` | Orthogonal fine-tuning via Cayley map ([OFT](https://arxiv.org/abs/2306.07280)) |

**When to use which:**

- **rsLoRALinear** — default upgrade; use higher ranks without numerical issues.
- **LoRAFALinear** — memory-constrained; halves adapter parameter count.
- **DoRALinear** — pretrained models where preserving weight norms matters.
- **PiSSALinear** — pretrained models; adapters start at the most informative directions.
- **VeRALinear** — fewest trainable params; A, B not stored in checkpoints.

### LoRA Zoo — Conv

```python
from jno.lora import (
    LoRAConv, rsLoRAConv, LoRAFAConv, DoRAConv, PiSSAConv,
    LoRAXSConv, VeRAConv, MiLoRAConv, IA3Conv, LoKrConv, OFTConv,
)
```

Mix linear and conv adapters per layer group:

```python
net.lora(
    specs=[
        {"target": "encoder", "rank": 8,  "alpha": 16,  "wrapper": PiSSALinear},
        {"target": "decoder", "rank": 4,  "alpha": 1.0, "wrapper": rsLoRALinear},
        {"target": "conv",    "rank": 4,  "alpha": 1.0, "wrapper": rsLoRAConv},
    ]
)
```

---

## Optimizer & Learning Rate

### Global

```python
net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

### Parameter Groups

Assign different optimizers or learning rates to different parameter groups via masks:

```python
net.optimizer(optax.adamw).scale(lrs(1e-3))                      # global fallback
net.mask(decoder_mask).optimizer(optax.adam)
net.mask(decoder_mask).scale(lrs(5e-4))
net.mask(encoder_mask).optimizer(optax.sgd)
net.mask(encoder_mask).scale(lrs(1e-4))
```

`mask(...)` is consumed by the next mutator call. A bare global `optimizer(...)` clears all previously configured groups.

### LR-Only Updates

```python
net.mask(decoder_mask).scale(lrs(1e-5))   # group-specific LR
net.scale(lrs(1e-5))                      # global LR
```

During `solve()`, jNO logs group coverage, overlap, and uncovered-parameter diagnostics.

---

## Initialize, Dtype & Tune

### Initialize

`initialize(...)` supports checkpoint paths, pytrees, and callable initializers:

```python
import jax

net.initialize("./weights.eqx")
net.initialize("./runs/checkpoints/2000::1")   # Orbax checkpoint, optional key suffix
net.initialize(other_model.module)
net.initialize(jax.nn.initializers.xavier_uniform(), key=jax.random.PRNGKey(0))
```

### Dtype

```python
import jax.numpy as jnp

net.dtype(jnp.bfloat16)
```

Casts floating-point parameters before training.

### Tune

`tune(...)` sweeps over combinations of model-control settings:

```python
net.tune(
    freeze=[True, False],
    lora=[(4, 1.0), None],
    optimizer=[optax.adam],
    lr=[lrs(1e-3), lrs(1e-4)],
    dtype=[jnp.float32],
)
```

### Reset

```python
net.reset()
```

Clears all training-time controls: `freeze`, `lora`, `optimizer`, `lr`, `dtype`, `mask`, and init state.

---

## Diagnostics

### Logging

At `solve()` time jNO logs:

- parameter-group summary
- overlap/uncovered diagnostics for groups
- zero-match warnings for empty masks/groups
- detailed path samples in the log file (`quiet` logs)

### Paramax Integration

jNO automatically unwraps Paramax wrappers before each forward evaluation (when `paramax` is installed):

```python
import paramax
import jax.numpy as jnp

scale = paramax.Parameterize(jnp.exp, jnp.log(jnp.ones(3)))
print(paramax.unwrap(("abc", 1, scale)))
# ('abc', 1, Array([1., 1., 1.], dtype=float32))
```

---

## IREE Deployment

After training, a jNO model can be compiled to an **IREE artifact** — a self-contained binary that runs inference without JAX, NumPy, or any Python ML dependency.

IREE supports CPU, CUDA, ROCm, Vulkan, and Metal backends through a single compilation step.

### Compiling a trained model

```python
iree_model = net.to_iree(
    sample_inputs=(jnp.ones((100, 2)),),   # tuple of example inputs matching __call__
)
```

```python
import numpy as np

x = np.random.rand(100, 2).astype(np.float32)
output = iree_model(x)   # returns np.ndarray, no JAX required
```

### Saving and loading

```python
jno.save(iree_model, "deployed_model.pkl")

loaded = jno.load("deployed_model.pkl")
output = loaded(x)
```

### Target backends

```python
iree_model = net.to_iree(sample_inputs, target_backend="llvm-cpu")   # CPU (default)
iree_model = net.to_iree(sample_inputs, target_backend="cuda")        # NVIDIA GPU
iree_model = net.to_iree(sample_inputs, target_backend="rocm")        # AMD GPU
iree_model = net.to_iree(sample_inputs, target_backend="vulkan")      # cross-platform
```

The `iree-compile` binary must be on `PATH`: `pip install iree-compiler iree-runtime`

### Optimization level

```python
iree_model = net.to_iree(
    sample_inputs,
    optimization_level=3,   # 0 = none … 3 = full (default)
)
```

### Compiling a raw JAX function

```python
from jno.utils.iree import IREEModel
import jax.numpy as jnp

def postprocess(u, v):
    return jnp.sqrt(u**2 + v**2)

compiled = IREEModel.compile(
    postprocess,
    sample_inputs=(jnp.ones((100,)), jnp.ones((100,))),
)
```

### Full workflow example

```python
import jno, foundax, jax, optax
import jax.numpy as jnp
import numpy as np

# --- train ---
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y, _ = domain.variable("interior")

net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=64, num_layers=4,
                               key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(1e-3))

u = net(x, y) * x * (1 - x) * y * (1 - y)
pde = jno.np.laplacian(u, [x, y]) + 1.0

crux = jno.core([pde.mse])
crux.solve(5000)

# --- deploy ---
iree_model = net.to_iree(
    sample_inputs=(jnp.ones((1, 2)),),
    target_backend="llvm-cpu",
)
jno.save(iree_model, "poisson_net.pkl")

# --- inference (no JAX needed) ---
model = jno.load("poisson_net.pkl")
pts = np.random.rand(500, 2).astype(np.float32)
predictions = model(pts)
```
