# W&B Integration and Explainability Callbacks

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/09_wandb/wandb_integration.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/09-wandb/">Back to chapter</a>
</div>

This tutorial shows how to connect a jNO training run to [Weights & Biases](https://wandb.ai) and use the built-in explainability callbacks to understand what is happening inside the training loop.

## What gets logged

| Source | W&B keys |
|--------|-----------|
| `GradientNormsCallback` | `explainability/gradient_norm/constraint_0`, `…/constraint_N` |
| `CosSimilarityCallback` | `explainability/cos_sim/0_1`, …, `explainability/cos_sim_matrix` (heatmap) |
| `GradientAlignmentCallback` | `explainability/gradient_alignment` |
| `LossLandscapeCallback` | `explainability/loss_landscape` (heatmap image) |
| `CheckpointCallback` | versioned `checkpoint` artifact with `total_loss`, `individual_losses`, `checkpoint_dir` |

---

## Step 1: Enable W&B in `jno.setup`

```python
dire = jno.setup(__file__, wandb=True)
```

This initialises a W&B run (project name defaults to the script filename stem) and also calls `weave.init("armbrul/jNO")` for Weave tracing if the `weave` package is installed.

Pass a dict to forward any `wandb.init` kwargs:

```python
jno.setup(__file__, wandb={"project": "jNO", "tags": ["poisson", "1d"]})
```

---

## Step 2: Define the Problem

We solve the 1-D Poisson equation with a **soft** boundary condition so the solver has two separate loss terms — necessary to make the explainability metrics meaningful.

```python
domain = jno.domain.line(mesh_size=0.05)
x,  _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

u_net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0))
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 1_000, 1e-5))

u  = u_net(x)
ub = u_net(xb)

pde = -jno.np.grad(jno.np.grad(u, x), x) - jno.np.sin(π * x)
bc  = ub   # u = 0 on boundary
```

---

## Step 3: Create the Explainability Callbacks

All four callbacks share the same interface: `interval` controls how often they run, and the optional `mask` lets you restrict gradient computations to a subset of parameters.

### Gradient norms

```python
cb_norms = jno.callbacks.gradient_norms(interval=50)
```

Tracks $\|\nabla L_i\|_2$ for each constraint $i$. A suddenly large norm usually signals a constraint that is dominating the update.

### Cosine similarity matrix

```python
cb_cos = jno.callbacks.cos_similarity(interval=50)
```

Computes the full $(N \times N)$ pairwise cosine similarity between every pair of constraint gradients. When W&B is active this is uploaded as a heatmap image.

$$\text{sim}_{ij} = \frac{\nabla L_i \cdot \nabla L_j}{\|\nabla L_i\| \|\nabla L_j\|}$$

| Value | Meaning |
|-------|---------|
| $\approx +1$ | Constraints reinforce each other |
| $\approx 0$ | Independent — no interaction |
| $\approx -1$ | Gradient conflict — one constraint hurts the other |

### Total gradient alignment

```python
cb_align = jno.callbacks.gradient_alignment(interval=50)
```

A single scalar in $[-1, 1]$ measuring global agreement across all gradients (Eq. 3.1, [2502.00604]):

$$\text{alignment} \;=\; 2\left\|\frac{1}{N}\sum_{i=1}^{N} \frac{\nabla L_i}{\|\nabla L_i\|}\right\|^2 - 1$$

Near $+1$ means all loss terms pull in the same direction; $0$ means orthogonal; near $-1$ means anti-aligned (destructive interference).

### 2-D loss landscape

```python
cb_landscape = jno.callbacks.loss_landscape(
    interval=200,   # expensive — n_grid² forward passes per call
    n_grid=11,
    alpha_range=0.5,
)
```

Samples two random filter-normalised directions and evaluates the total loss on an $(n\_\text{grid} \times n\_\text{grid})$ grid around the current parameters. Logged as a heatmap image in W&B. A smooth bowl shape is a sign of a well-conditioned optimisation landscape; sharp ridges or irregular bumps indicate ill-conditioning.

!!! tip "Reducing cost"
    Pass `mask=bool_pytree` to restrict perturbations and gradient computations to a small subset of parameters (e.g. only the output layer). This can reduce cost by orders of magnitude while preserving the diagnostic signal.

---

## Step 4: Checkpoint with W&B Artifact

```python
cb_ckpt = jno.callbacks.checkpoint(
    directory=f"{dire}/checkpoints",
    save_interval_epochs=500,
    max_to_keep=3,
    best_fn=lambda m: m["total_loss"],
)
```

Each time a checkpoint is saved, jNO uploads it to W&B as a versioned `checkpoint` artifact. The artifact metadata includes:

```python
{
    "epoch": 500,
    "total_loss": 0.0023,
    "individual_losses": [0.0019, 0.0004],
    "checkpoint_dir": "/path/to/checkpoints/500",
    "timestamp": 1717000000.0,
}
```

---

## Step 5: Solve

```python
crux = jno.core([pde.mse, bc.mse], domain)
crux.solve(
    2_000,
    callbacks=[cb_norms, cb_cos, cb_align, cb_landscape, cb_ckpt],
)
cb_ckpt.close()
```

All callbacks register themselves in `on_solve_begin` (called once after the initial JIT compilation) to pre-compile their JAX functions against the live parameter shapes. This means the first call to a callback at `epoch % interval == 0` runs with a pre-warmed XLA kernel.

---

## Step 6: Read Results Locally

Even without W&B, every callback stores its history as numpy arrays:

```python
# Gradient norms: shape (n_samples, n_constraints)
norms = cb_norms.result["norms"]

# Cosine similarity: shape (n_samples, n_constraints, n_constraints)
cos_mat = cb_cos.result["cos_sim_matrix"]

# Alignment: shape (n_samples,)
alignment = cb_align.result["alignment"]

# Landscapes: shape (n_samples, n_grid, n_grid)
landscapes = cb_landscape.result["landscapes"]
```

---

## What To Notice

- All W&B calls are **no-ops** when `jno.setup` is called without `wandb=True` — no behaviour change in scripts that do not need W&B.
- The explainability callbacks use `jacrev` internally; they are independent of the training step and do not affect the parameter updates.
- For large models, always provide a `mask` to limit which parameters are differentiated. The output-layer weights often give a good proxy signal at a fraction of the cost.
- The gradient alignment scalar dropping during training is a reliable early warning of constraint conflict.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/09_wandb/wandb_integration.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/09-wandb/">Back to 09 W&B</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/09_wandb/wandb_integration.py"
```
