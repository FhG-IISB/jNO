# Callbacks

Callbacks hook into the training loop without modifying the solver itself. Pass a list of callbacks to `solve()`:

```python
crux.solve(10000, callbacks=[cb1, cb2])
```

Each callback's `on_epoch_end` is called after every outer training step and can optionally signal early termination by returning `True`.

---

## Early Stopping

Stop training automatically when a monitored metric stops improving.

```python
cb = jno.callbacks.early_stopping(
    patience=1000,     # epochs with no improvement before stopping
    min_delta=1e-6,    # minimum change to count as improvement
    mode="min",        # "min", "max", or "rel"
)

crux.solve(100_000, callbacks=[cb])

print(cb.stopped_epoch)   # epoch at which training halted (None if not triggered)
print(cb.best_metric)     # best metric value observed
```

### Modes

| `mode` | Stops when |
|--------|-----------|
| `"min"` | metric hasn't dropped by more than `min_delta` for `patience` epochs |
| `"max"` | metric hasn't risen by more than `min_delta` for `patience` epochs |
| `"rel"` | metric hasn't improved by a fraction of `min_delta` relative to best value |

`"rel"` is useful when loss magnitudes vary across runs — a `min_delta=0.01` means "stop if the loss hasn't improved by at least 1%".

### Monitoring a custom metric

By default early stopping watches the total loss. Pass `metric_fn` to monitor anything available at the end of each step:

```python
cb = jno.callbacks.early_stopping(
    patience=500,
    metric_fn=lambda **kw: float(kw["individual_losses"][1]),  # watch constraint #1 only
)
```

The keyword arguments available inside `metric_fn` are: `epoch`, `total_loss`, `individual_losses`, `trainable`, `opt_states`, `rng`, `log`.

### Starting from a baseline

```python
cb = jno.callbacks.early_stopping(
    patience=500,
    baseline=1e-3,   # stops if metric never gets below 1e-3
)
```

---

## Checkpointing

Save model weights, optimizer states, and PRNG key to disk at regular intervals.

```python
cb = jno.callbacks.checkpoint(
    directory="runs/my_experiment/checkpoints",
    save_interval_epochs=500,   # save every 500 outer steps
    max_to_keep=3,              # keep only the 3 most recent checkpoints
)

crux.solve(10000, callbacks=[cb])
```

### Keeping the best checkpoint

Pass `best_fn` to always retain the checkpoint with the lowest returned value, regardless of `max_to_keep`:

```python
cb = jno.callbacks.checkpoint(
    save_interval_epochs=200,
    max_to_keep=2,
    best_fn=lambda m: m["total_loss"],   # keep the checkpoint with lowest total loss
)
```

### Restoring a checkpoint

```python
state = cb.restore()          # latest checkpoint
state = cb.restore(step=2000) # specific step

# state keys: "trainable", "opt_states", "rng", "metadata"
print(state["metadata"])      # {"epoch": 2000, "total_loss": ..., "timestamp": ...}
```

To resume training from a restored checkpoint, reload the solver and re-attach the restored parameters:

```python
crux = jno.core.load("runs/crux.pkl")
crux.set_optimizer(optax.adam(1e-4))
crux.solve(5000)
```

### Async checkpointing

Checkpoints are written in a background thread by default (`async_checkpointing=True`). Set to `False` for synchronous writes if you need guaranteed consistency before the process exits:

```python
cb = jno.callbacks.checkpoint(async_checkpointing=False)
```

---

## Explainability callbacks

jNO also provides callbacks for analysing gradient conflict, cosine similarity, and the loss landscape during training. See [Explainability](../training/explainability.md).

---

## Combining callbacks

```python
crux.solve(
    50_000,
    callbacks=[
        jno.callbacks.checkpoint(save_interval_epochs=1000, max_to_keep=3),
        jno.callbacks.early_stopping(patience=2000, mode="rel", min_delta=1e-3),
    ],
)
```

