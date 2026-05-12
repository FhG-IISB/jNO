# Training

This page covers every aspect of the jNO training pipeline: constructing the core solver, attaching optimisers, learning-rate and weight schedules, multi-phase training, per-model controls (freeze, LoRA, masks), gradient checkpointing, multi-device parallelism, evaluation, and debugging.

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
u_net = jnn.nn.mlp(2, key=key).optimizer(optax.adam, lr=lrs(1e-3))
```

### After `core.load()`

When loading a saved solver the `Model` references in the expression tree are disconnected from Python variables. Use `set_optimizer` to reassign:

```python
crux = jno.core.load("runs/crux.pkl")
crux.set_optimizer(optax.adam, lr=lrs(1e-4))
crux.solve(1000)
```

---

## Learning Rate Schedules

`LearningRateSchedule` is a callable `(epoch, individual_losses) → scalar`.

```python
from jno import LearningRateSchedule as lrs

# Constant
lrs.constant(1e-3)
lrs(1e-3)          # shorthand

# Exponential decay: lr(t) = max(lr_end, lr0 * decay_rate^(t/decay_steps))
lrs.exponential(lr0=1e-3, decay_rate=0.9, decay_steps=1000, lr_end=1e-5)

# Cosine decay
lrs.cosine(total_steps=5000, lr0=1e-3, lr_end=1e-6)

# Linear warm-up then cosine decay
lrs.warmup_cosine(total_steps=5000, warmup_steps=500, lr0=1e-3, lr_end=1e-6)

# Piecewise constant
lrs.piecewise_constant(
    boundaries=[1000, 3000],
    values=[1e-3, 5e-4, 1e-4],   # len(boundaries) + 1 values
)

# Custom function
lrs(lambda epoch, losses: 1e-4 * (0.9 ** (epoch / 500)))
```

All schedules accept `min_lr` and `max_lr` keyword arguments to clamp the output.

---

## Weight Schedules

`WeightSchedule` scales individual constraint losses:  
`total_loss = Σ wᵢ · constraintᵢ`

```python
from jno import WeightSchedule as ws

# Fixed weights (one per constraint)
ws([1.0, 10.0, 1.0])

# Adaptive weights as a function of (epoch, individual_losses)
ws(lambda e, L: [1.0, 10.0 * L[1], 1.0])  # amplify boundary loss when large

# Using losses from previous step to avoid gradient-through-loss issues
ws(lambda e, L: [1.0, jnp.maximum(1.0, L[1] / (L[0] + 1e-8))])
```

Pass to `solve()`:

```python
crux.solve(5000, constraint_weights=ws([1.0, 10.0]))
```

---

## Running Training

```python
stats = crux.solve(
    epochs=5000,
    constraint_weights=ws([1.0, 10.0]),   # optional
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
u_net.optimizer(optax.adam, lr=lrs.warmup_cosine(3000, 300, 1e-3, 1e-5))
crux.solve(3000, constraint_weights=ws([1.0, 10.0])).plot("phase1.png")

# Phase 2: L-BFGS quasi-Newton refinement
u_net.optimizer(optax.lbfgs, lr=lrs(5e-5))
crux.solve(500, constraint_weights=ws([1.0, 5.0])).plot("phase2.png")

# Phase 3: SOAP second-order method
from soap_jax import soap
u_net.optimizer(soap(1), lr=lrs(1e-5))
crux.solve(500).plot("phase3.png")
```

---

## Per-Model Training Controls

Each model in the problem is fully independent with respect to:
- Optimiser and learning rate
- Trainability (frozen vs. active)
- LoRA configuration
- Pretrained weight initialisation

### Freeze / Unfreeze

```python
# Freeze all parameters — model acts as a fixed feature extractor
v_net.freeze()

# Unfreeze by assigning a new optimiser
v_net.optimizer(optax.adam, lr=lrs(1e-4))
```

### Partial Parameter Masks

Train only a subset of layers:

```python
import equinox as eqx

all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
mask = eqx.tree_at(
    lambda m: (m.dense3.weight, m.dense3.bias),
    all_false,
    (True, True),
)
u_net.mask(mask).optimizer(optax.adam, lr=lrs(5e-4))
```

For Flax Linen models navigate the parameter dict with string keys:

```python
all_false_v = jax.tree_util.tree_map(lambda _: False, v_net.module)
mask_v = eqx.tree_at(
    lambda w: w.params["params"]["Dense_0"]["kernel"],
    all_false_v, True,
)
v_net.mask(mask_v).optimizer(optax.adam, lr=lrs(5e-4))
```

### LoRA (Low-Rank Adaptation)

LoRA inserts trainable low-rank adapter matrices into each linear layer while keeping the base weights frozen. After `solve()`, adapters are merged back into the base weights.

```python
v_net.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-4))
```

> `.lora()` takes priority over `.freeze()` — the base weights are frozen by LoRA itself.

### Pretrained Weight Initialisation

```python
# From a Python object (equinox module or pytree)
u_net.initialize(pretrained_module)

# From a file (equinox .eqx format)
u_net.initialize("runs/pretrained.eqx")

# Flax models: from params dict
v_net.initialize(pretrained_flax_params)

# Flax models: from msgpack file
v_net.initialize("runs/pretrained.msgpack")
```

---

## Multi-Device Parallelism

jNO supports data parallelism, model parallelism, and hybrid parallelism via JAX's device mesh:

```python
# No parallelism (single device, default)
crux = jno.core(constraints, domain, mesh=(1, 1))

# Pure data parallelism: split batches across 4 GPUs
crux = jno.core(constraints, domain, mesh=(4, 1))

# Pure model parallelism: shard model weights across 2 GPUs
crux = jno.core(constraints, domain, mesh=(1, 2))

# Hybrid (2 data × 2 model = 4 GPUs total)
crux = jno.core(constraints, domain, mesh=(2, 2))

# Auto-scale to all available devices
n = len(jax.devices())
crux = jno.core(constraints, domain, mesh=(n, 1))
```

**Mesh shape rules:**
- `batch × model` must equal the total number of available devices.
- Data parallelism (`(n, 1)`) maximises throughput when the model fits on a single device.
- Model parallelism (`(1, n)`) allows training models too large for a single device.

---

## Evaluation

After training, use `crux.eval()` to evaluate any symbolic expression:

```python
# On the training domain
pred = crux.eval(u)    # shape: (B, T, N, out_dim)

# On a different domain (e.g., fine test grid)
test_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
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
pde = jnn.laplacian(u, [x, y]) + 1.0
pde.debug._shape = True   # print shape at this node each step
pde.debug._mean = True    # print mean value
```

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
