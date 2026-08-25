# Running Training

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
| `accumulation_steps=N` | Average gradients over `N` micro-batches before one optimizer update | You want a larger effective batch than fits in memory |

`offload_data` requires `batchsize < total_samples`.

### Gradient accumulation

```python
stats = crux.solve(5000, batchsize=8, accumulation_steps=4)
# INFO: Gradient accumulation enabled: 4 micro-batches per update (effective batch = 8 × 4 = 32)
```

Requires `batchsize`. With full-batch training it has nothing to accumulate, and jNO says so and
falls back to `1` rather than pretending. It is also rejected alongside `substeps`.

!!! warning "I could not measure a behavioural difference — treat this as unverified"
    On a small test (`64 *` batched domain, `batchsize=8`, Adam) `accumulation_steps=4` produced a
    **bit-identical** loss history to the default, and trained parameters differing by 5.96e-08 —
    float32 round-off. The log line above *did* appear, so the path is taken.

    That may be legitimate for a degenerate case where the micro-batches carry the same points, but
    it was not possible to observe the documented effect. The two tests covering it
    (`tests/test_integration.py`) assert only that it runs and the loss is finite, so a silent no-op
    would pass. Verify on your own problem before relying on the effective batch size.

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
