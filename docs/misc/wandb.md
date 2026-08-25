# Weights & Biases

jNO has first-class W&B support. Enabling it adds automatic metric logging, checkpoint artifacts, weight histograms, and [Weave](https://wandb.ai/site/weave) tracing with a single flag in `jno.setup`.

---

## Enabling W&B

Pass `wandb=True` to `jno.setup`:

```python
dire = jno.setup(__file__, wandb=True)
```

This calls `wandb.init` (project name defaults to the script filename stem), logs source code via `run.log_code()`, and initialises Weave tracing via `weave.init("armbrul/jNO")` if the `weave` package is installed.

To forward extra kwargs to `wandb.init`, pass a dict:

```python
jno.setup(__file__, wandb={"project": "jNO", "tags": ["poisson", "1d"], "group": "sweep-01"})
```

Any key not supplied falls back to the default (`project` → script stem, `dir` → run directory).

---

## What gets logged automatically

| Source | W&B keys / type |
|--------|-----------------|
| Training loss (every step) | `loss`, `constraint_0`, `constraint_1`, … |
| `CheckpointCallback` | versioned `checkpoint` artifact |
| Weight histograms | `weights/<model>/<layer>` |
| `GradientNormsCallback` | `explainability/gradient_norm/constraint_N` |
| `CosSimilarityCallback` | `explainability/cos_sim/i_j` + heatmap image |
| `GradientAlignmentCallback` | `explainability/gradient_alignment` |
| `LossLandscapeCallback` | `explainability/loss_landscape` (heatmap image) |

Everything in the table below the first row requires the corresponding callback to be passed to `solve()`. See [Explainability](../training/explainability.md) for details on the explainability callbacks.

---

## Checkpoint artifacts

When `CheckpointCallback` saves a checkpoint and a W&B run is active, it uploads the checkpoint directory as a versioned `checkpoint` artifact. The artifact metadata includes:

```python
{
    "epoch": 500,
    "total_loss": 0.0023,
    "individual_losses": [0.0019, 0.0004],
    "checkpoint_dir": "/path/to/runs/checkpoints/500",
    "timestamp": 1717000000.0,
}
```

```python
cb = jno.callbacks.checkpoint(
    directory=f"{dire}/checkpoints",
    save_interval_epochs=500,
    max_to_keep=3,
    best_fn=lambda m: m["total_loss"],
)
crux.solve(5000, callbacks=[cb])
cb.close()
```

---

## Alerts

Send a W&B alert from anywhere in your script:

```python
from jno.utils.config import wandb_alert

wandb_alert("NaN detected", f"Loss exploded at epoch {epoch}", level="WARN")
```

`level` is one of `"INFO"`, `"WARN"`, `"ERROR"`. The call is a no-op when no W&B run is active.

---

## Helper functions

`jno.utils.config` exposes three thin wrappers used internally; you can call them directly if you need fine-grained control:

```python
from jno.utils.config import get_wandb_run, wandb_log, wandb_log_model

# Check whether a run is active
run = get_wandb_run()   # returns the wandb.Run or None

# Log arbitrary metrics at a specific step
wandb_log({"my_metric": 0.42}, step=1000)

# Upload a model as an artifact
wandb_log_model(my_pytree, name="best_model")
```

All three are no-ops when `get_wandb_run()` returns `None`.

---

## A full run, end to end

The `09_wandb` tutorial script this section used to embed was removed in #84, so the example is
inline here — and, unlike an embed, verified to run:

```python
import jax, optax, foundax, jno
from jno.utils.config import get_wandb_run, wandb_log, wandb_log_model

run = jno.setup(__file__, wandb=True)          # False (default) disables; a dict is passed to wandb.init

dom = jno.Shape.rect(0, 0, 1, 1, size=0.05).domain()
x, y, _ = dom.variable("interior")

net = jno.nn(foundax.mlp(2, hidden_dims=64, num_layers=4, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(1e-3))

u = net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)
crux  = jno.core([(u.dd(x) + u.dd(y) + 1.0).mse])
stats = crux.solve(20_000)                     # returns a `statistics` object

wandb_log({"final_loss": float(stats.total_loss)}, step=20_000)
wandb_log_model(net.module, name="best_model")
jno.save(crux, f"{run}/model.pkl")
jno.wandb_finish()
```

!!! warning "`wandb=True` creates a real run on your account"
    It picks up whatever credentials `wandb login` has stored and syncs to wandb.ai. Leave it at the
    default `False` — or export `WANDB_MODE=disabled` — when you are only testing that a script runs.
    With no active run, `wandb_log`, `wandb_log_model` and `wandb_alert` are all no-ops, so the same
    script works either way.
