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

## Full example

A runnable script that combines all four explainability callbacks, checkpointing, and W&B logging is available in the tutorial examples:

```python
--8<-- "tutorial_examples/09_wandb/wandb_integration.py"
```
