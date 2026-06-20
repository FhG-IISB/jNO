# Running Training

---

## Running Training

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

### Memory Optimisations

| Option | Effect | Use When |
|--------|--------|----------|
| `batchsize=N` | Mini-batch gradient estimation | Dataset doesn't fit in GPU memory |
| `checkpoint_gradients=True` | Rematerialise activations during backward pass | Very deep networks or long time sequences |
| `offload_data=True` | Keep dataset on CPU; stream each mini-batch | Very large datasets |

`offload_data` requires `batchsize < total_samples`.

---

## Multi-Phase Training

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
