# Evaluation & Debugging

---

## Evaluation

After training, use `crux.eval()` to evaluate any symbolic expression:

```python
# On the training domain
pred = crux.eval(u)    # shape: (B, T, N, out_dim)

# On a different domain (e.g., fine test grid)
test_domain = jno.Shape.rect(0, 0, 1, 1, size=0.01).domain()
pred_fine = crux.eval(u, domain=test_domain)

# Prediction on arbitrary point arrays
import numpy as np
points = np.tile(test_domain.points[None, ...], (B, 1, 1))  # (B, N, 2)
pred = crux.predict(points=points, operation=u, context=test_domain.context)
```

---

## Debugging

### Print Computation Tree

```python
crux.print_tree()              # to stdout
crux.print_tree("tree.txt")    # to file
```

### Print Tensor Shapes

```python
crux.print_shapes()   # per-node shape trace for all constraints and trackers
```

### Debug Print Inside Expressions

Use JAX's debug print inside expressions for verbose intermediate inspection (expensive — use sparingly):

```python
pde = jno.np.laplacian(u, [x, y]) + 1.0
pde.debug._shape = True   # print shape at this node each step
pde.debug._mean = True    # print mean value
```

---

## Profiler

Pass `profile=True` to `solve()` to capture a JAX performance trace:

```python
history = crux.solve(5000, profile=True)
```

jNO skips the first outer step (startup JIT compilation) and then records the next 50 steady-state steps. The traces are written to `<logger.path>/traces/` in **Perfetto format** — open them at [ui.perfetto.dev](https://ui.perfetto.dev) to inspect the timeline.

### What the trace shows

The Perfetto timeline breaks down each training step into the individual XLA ops that make it up. Typical things to look for:

| Symptom | Likely cause |
|---------|-------------|
| A single long "unflatten" span every few steps | Python GC pause — jNO disables cyclic GC during training to suppress these |
| Many short gaps between ops | Host–device sync points; consider fusing ops or using `jax.block_until_ready` less aggressively |
| One constraint takes 10× longer than others | That expression has much higher compute cost — consider finite-difference vs AD trade-off |
| Compile time dominates the first step | Normal for JIT; the skip-first-step logic means this is excluded from the captured trace |

### Controlling the output directory

The trace path follows the jNO logger, which defaults to the current working directory. Set it explicitly with `jno.setup`:

```python
import jno
jno.setup("runs/my_experiment")   # traces go to runs/my_experiment/traces/

crux = jno.core([pde.mse])
history = crux.solve(5000, profile=True)
```

### Notes

- `profile=True` is a **side effect only** — it does not change the return value of `solve()`.
- Profiling adds negligible overhead to the captured steps themselves; the trace serialisation happens asynchronously.
- For very short runs (fewer than 52 epochs) the capture window is clamped: `min(50, epochs - 1)` steps are recorded.

---

## Training Statistics

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

---

## Checkpoints

`solve()` automatically saves a checkpoint (model weights, optimiser state, RNG key) after every call. All checkpoints are accessible via `crux.checkpoints`:

```python
# Inspect available checkpoints
for i, ckpt in enumerate(crux.checkpoints):
    print(i, ckpt["step"], ckpt["time"])

# Restore a specific checkpoint
crux.models = crux.checkpoints[-1]["models"]
```
