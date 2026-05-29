# Training

This section covers every aspect of the jNO training pipeline: constructing the core solver, attaching optimisers, schedules, multi-phase training, parallelism, evaluation, and per-model controls.

---

## Core Solver

`jno.core` is the central training object. It:

1. Builds the symbolic computation graph from your constraints.
2. Performs common sub-expression elimination (CSE).
3. Initialises all neural-network parameters.
4. Compiles a JIT-optimised step function.
5. Runs the training loop and returns training statistics.

```python
crux = jno.core(
    constraints=[pde.mse, boc.mse],   # list of scalar Placeholder expressions
    domain=domain,
    rng_seed=42,                       # optional; also set in .jno.toml → [jno] seed
    mesh=(1, 1),                       # (batch_devices, model_devices)
)
```

---

## Attaching Optimisers

**Every non-frozen model must have an optimiser before calling `solve()`.**

```python
u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
v_net.optimizer(optax.adamw, lr=lrs.warmup_cosine(5000, 500, 1e-3, 1e-4))
```

`model.optimizer()` returns `self` for chaining:

```python
u_net = jno.nn.mlp(2, key=key).optimizer(optax.adam, lr=lrs(1e-3))
```

### After `core.load()`

When loading a saved solver the `Model` references in the expression tree are disconnected from Python variables. Use `set_optimizer` to reassign:

```python
crux = jno.core.load("runs/crux.pkl")
crux.set_optimizer(optax.adam, lr=lrs(1e-4))
crux.solve(1000)
```

---

## Per-Model Controls

Each model is fully independent with respect to its optimiser, trainability, LoRA configuration, and pretrained weight initialisation. See the **[Model Controls](model-controls/index.md)** sub-section for the full API covering freeze, masks, LoRA, dtype conversion, and diagnostics.
