# Callbacks

Callbacks hook into the training loop without modifying the solver itself. Pass a list of callbacks to `solve()`:

```python
crux.solve(10000, callbacks=[cb1, cb2])
```

Each callback's `on_epoch_end` is called after every outer training step and can optionally signal early termination by returning `True`.

---

## Build your own callback

Any subclass of `Callback` is a valid callback — override one or more of four hooks:

| Hook | Signature | When it fires |
|------|-----------|---------------|
| `on_solve_begin(**kw)` | returns `None` | Once, after `solve()` finishes JIT setup, before the loop |
| `on_before_update(**kw)` | returns modified grads or `None` | Between grad computation and optimizer update (split path) |
| `on_epoch_end(**kw)` | returns `bool` (`True` to stop training) | After every outer training step |
| `on_training_end(**kw)` | returns `None` | Once, after the loop finishes |

`on_before_update` intercepts the gradient dict before the optimizer applies its update.  Returning a modified dict redirects the optimizer (for example to apply a preconditioner); returning `None` leaves the gradients unchanged.  The hook requires `inner_steps=1` and no Bayesian models; the solver raises a `ValueError` if those constraints are violated.

The `**kw` for each hook is documented in the [base class source](https://github.com/FhG-IISB/jno/blob/main/jno/utils/adaptive/callbacks.py) — the most useful keys inside `on_epoch_end` are `epoch`, `total_loss`, `individual_losses`, `trainable`, `rng`, and `log`.

```python
from jno.utils.adaptive.callbacks import Callback

class LossPrinter(Callback):
    def __init__(self, every: int = 100):
        self.every = every

    def on_epoch_end(self, **kw) -> bool:
        if kw["epoch"] % self.every == 0:
            print(f"epoch {kw['epoch']}: loss = {float(kw['total_loss']):.4e}")
        return False   # never request early stop

crux.solve(10_000, callbacks=[LossPrinter(every=500)])
```

Hooks you don't need can simply be omitted — the base class supplies no-op defaults.

---

## Built-in callbacks

Each is a `jno.callbacks.*` factory that returns a pre-configured `Callback` instance.

### Early Stopping

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

#### Modes

| `mode` | Stops when |
|--------|-----------|
| `"min"` | metric hasn't dropped by more than `min_delta` for `patience` epochs |
| `"max"` | metric hasn't risen by more than `min_delta` for `patience` epochs |
| `"rel"` | metric hasn't improved by a fraction of `min_delta` relative to best value |

`"rel"` is useful when loss magnitudes vary across runs — a `min_delta=0.01` means "stop if the loss hasn't improved by at least 1%".

#### Monitoring a custom metric

By default early stopping watches the total loss. Pass `metric_fn` to monitor anything available at the end of each step:

```python
cb = jno.callbacks.early_stopping(
    patience=500,
    metric_fn=lambda **kw: float(kw["individual_losses"][1]),  # watch constraint #1 only
)
```

The keyword arguments available inside `metric_fn` are: `epoch`, `total_loss`, `individual_losses`, `trainable`, `opt_states`, `rng`, `log`.

#### Starting from a baseline

```python
cb = jno.callbacks.early_stopping(
    patience=500,
    baseline=1e-3,   # stops if metric never gets below 1e-3
)
```

---

### Checkpointing

Save model weights, optimizer states, and PRNG key to disk at regular intervals.

```python
cb = jno.callbacks.checkpoint(
    directory="runs/my_experiment/checkpoints",
    save_interval_epochs=500,   # save every 500 outer steps
    max_to_keep=3,              # keep only the 3 most recent checkpoints
)

crux.solve(10000, callbacks=[cb])
```

#### Resuming from a checkpoint — `jno.core(resume_from=...)`

Checkpointing writes; `resume_from` reads. Point a fresh `jno.core` at the checkpoint directory and
model parameters, optimizer states and the RNG key are restored before the next `solve()`:

```python
crux = jno.core(constraints, domain=dom, resume_from="runs/my_experiment/checkpoints")
crux.solve(10_000)          # continues from the latest checkpoint, not from scratch
```

!!! measured "A crashed run picks up where it stopped"
    Training 200 epochs to a loss of **0.231104**, then resuming in a **new process**: the resumed
    run reports **0.236130** after one step. The same script with no `resume_from` starts at
    **0.814328**.

!!! warning "The resuming process must build the models the same way"
    Optimizer states are keyed by a per-process model counter, so a script that constructs a
    different number of `jno.nn(...)` models — or constructs them in a different order — before
    resuming will fail with an Orbax *"tree structures do not match"* error naming a mismatched
    `opt_states.N`. Re-run the same script; do not resume inside a process that already built other
    models.

    Requires the optional `orbax-checkpoint` package.

#### Keeping the best checkpoint

Pass `best_fn` to always retain the checkpoint with the lowest returned value, regardless of `max_to_keep`:

```python
cb = jno.callbacks.checkpoint(
    save_interval_epochs=200,
    max_to_keep=2,
    best_fn=lambda m: m["total_loss"],   # keep the checkpoint with lowest total loss
)
```

#### Restoring a checkpoint

```python
state = cb.restore()          # latest checkpoint
state = cb.restore(step=2000) # specific step

# state keys: "trainable", "opt_states", "rng", "metadata"
print(state["metadata"])      # {"epoch": 2000, "total_loss": ..., "timestamp": ...}
```

To resume training from a restored checkpoint, reload the solver and re-attach the restored parameters:

```python
crux = jno.load("runs/crux.pkl")
crux.set_optimizer(optax.adam(1e-4))
crux.solve(5000)
```

#### Async checkpointing

Checkpoints are written in a background thread by default (`async_checkpointing=True`). Set to `False` for synchronous writes if you need guaranteed consistency before the process exits:

```python
cb = jno.callbacks.checkpoint(async_checkpointing=False)
```

---

### Energy Natural Gradient Descent (ENGD)

Preconditions parameter gradients with the inverse energy Gram matrix `G⁻¹`, converting gradient descent into an approximate Newton step in the PDE function-space norm.  In practice ENGD can achieve several orders of magnitude lower error than Adam or L-BFGS in far fewer iterations (Zeinhofer, Cakir & Mardal, ICML 2023, Sec 3, arXiv:2302.13163).

**Recommended — `jno.optimizers.engd()` (auto-wires `gram_terms` and the inner `sgd` step):**

```python
import jax, jno
jax.config.update("jax_enable_x64", True)   # float64 for full accuracy

# raw residual expressions (NOT .mse — those are scalar losses)
pde = u.laplacian(x, y) + forcing
bc  = u_bc

net.optimizer(jno.optimizers.engd(line_search=True))   # gram_terms auto-detected
crux = jno.core([pde.mse, bc.mse])
crux.solve(500)
```

**Manual form — `jno.callbacks.engd()` (full control over `gram_terms`):**

```python
import jax, optax
jax.config.update("jax_enable_x64", True)

engd = jno.callbacks.engd(
    gram_terms=[
        (pde.grad(net), 1.0),   # ∫_Ω (Δu_i)(Δu_j) dx
        (bc.grad(net),  1.0),   # ∫_∂Ω u_i u_j ds
    ],
    gram_interval=1,   # recompute G every step (set > 1 to amortise)
)
net.optimizer(optax.sgd(1.0))   # lr=1.0 → G⁻¹∇L is the Newton step

crux = jno.core([pde.mse, bc.mse])
crux.solve(500, callbacks=[engd])
```

**Grid line search (`line_search=True`):** This is the recommended setting for faithful reproduction of §4.1 results: the energy Gram is initially ill-conditioned, making the natural-gradient *direction* correct but its *magnitude* unreliable.  Use `optax.sgd(1.0)` — the selected α is folded into the returned gradient:

```python
engd = jno.callbacks.engd(
    gram_terms=[
        (pde.grad(net), 1.0),
        (bc.grad(net),  1.0),
    ],
    line_search=True,   # 31-point grid search α∈{0.5^k: k=0,…,30} per step
)
net.optimizer(optax.sgd(1.0))   # lr=1 because line search handles step scale
```

**Key constraints:**
- Requires `inner_steps=1` (the hook cannot fire inside the XLA loop).
- Not compatible with `.bayesian()` / `.vi()` models.
- `gram_terms` must all reference the **same** model.
- Pass **raw residual** expressions to `.grad(model)`, not `.mse`-wrapped ones.

**`gram_interval > 1`:** Cache `G` between recomputations (cheap on stable problems):

```python
engd = jno.callbacks.engd(gram_terms=[...], gram_interval=5)
```

### Explainability callbacks

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
