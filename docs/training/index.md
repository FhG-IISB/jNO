# Training

This section covers every aspect of the jNO training pipeline: constructing the core solver, attaching optimisers, schedules, multi-phase training, parallelism, evaluation, and per-model controls.

---

## Core Solver

`jno.core` is the central training object. It:

1. Builds the symbolic computation graph from your constraints.
2. Performs common sub-expression elimination (CSE).
3. Folds a sum of squared partials into one Laplacian node, so `u.xx + u.yy`,
   `u.d2(x) + u.d2(y)` and `u.laplacian(x, y)` all cost the same
   ([details](../operations.md#differentiation)).
4. Initialises all neural-network parameters.
5. Compiles a JIT-optimised step function.
6. Runs the training loop and returns training statistics.

```python
crux = jno.core(
    constraints=[pde.mse, boc.mse], 
    rng_seed=42,                       # optional; also set in .jno.toml → [jno] seed
    mesh=(1, 1),                       # (batch_devices, model_devices)
)
```

---

## Attaching Optimisers

**Every non-frozen model must have an optimiser before calling `solve()`.**

```python
u_net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.9, 2000, 1e-5))
v_net.optimizer(optax.adamw).scale(lrs.warmup_cosine(5000, 500, 1e-3, 1e-4))
```

`model.optimizer()` returns `self` for chaining:

```python
u_net = jno.nn(foundax.mlp(2, key=key)).optimizer(optax.adam(1e-3))
```

### After `core.load()`

When loading a saved solver the `Model` references in the expression tree are disconnected from Python variables. Use `set_optimizer` to reassign:

```python
crux = jno.load("runs/crux.pkl")
crux.set_optimizer(optax.adam, scale=lrs(1e-4))
crux.solve(1000)
```

---

## Per-Model Controls

Each model is fully independent with respect to its optimiser, trainability, LoRA configuration, and pretrained weight initialisation. See the **[Operations → Part B](../operations.md#part-b-operations-that-require-trainable-parameters)** sub-section for the full API covering freeze, masks, LoRA, dtype conversion, and diagnostics.
