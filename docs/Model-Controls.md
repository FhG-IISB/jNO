# Model Controls

This page documents the model-level API in jNO: how to configure trainability, optimiser/schedule behavior, targeted operations via `mask(...)`, LoRA, initialisation, dtype casting, and Paramax interaction.

---

## Quick Start

```python
import jax
import jax.numpy as jnp
import optax
import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs

NN = jnn.nn.mlp(2, output_dim=1, hidden_dims=64, num_layers=3, key=jax.random.PRNGKey(0))
NN.dont_show()

# Global optimizer + LR
NN.optimizer(optax.adam)
NN.lr(lrs.exponential(1e-3, 0.8, 5000, 1e-5))

# Targeted LoRA
NN.mask(target="decoder").lora(rank=8, alpha=16)
```

---

## Available Model Methods

`Model` (returned by `jnn.nn.*`) supports:

- `dont_show()`
- `freeze()` / `unfreeze()`
- `mask(param_mask=None, *, target: str=None)`
- `lora(rank=4, alpha=1.0)`
- `optimizer(opt_fn, *, lr=None)`
- `lr(schedule_or_scalar)`
- `initialize(weights_or_path)`
- `dtype(dtype)`
- `tune(...)`
- `reset()`

All methods return `self` and are chainable.

---

## 1. `mask(...)` as Target Selector

`mask(...)` can be used in two ways:

1. Boolean pytree mask:

```python
NN.mask(param_mask)
```

2. Regex target mask (recommended):

```python
NN.mask(target="decoder.*kernel")
# shorthand:
NN.mask("decoder.*kernel")
```

Regex is matched with `re.search` against full parameter paths.

### Zero-match behavior

When a target matches no parameters, jNO emits warnings during `solve()` and writes detailed diagnostics to the log file.

---

## 2. Freeze / Unfreeze

### Global freeze

```python
NN.freeze()
```

Freezes the whole model.

### Targeted freeze

```python
NN.mask(target="encoder").freeze()
```

Freezes only target leaves; non-target leaves remain trainable.

---

## 3. LoRA

```python
NN.lora(rank=8, alpha=16)
```

### Targeted LoRA

```python
NN.mask(target="decoder").lora(rank=8, alpha=16)
```

Current behavior:

- LoRA adapters are created only on matched kernels.
- Matched base leaves are frozen.
- Non-target base leaves remain trainable.

### Freeze all base + targeted adapters

```python
NN.freeze().mask(target="decoder").lora(rank=8, alpha=16)
```

This freezes all base parameters; only LoRA adapters train.

---

## 4. Optimizer and LR

### Global

```python
NN.optimizer(optax.adam)
NN.lr(lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

### Targeted parameter groups

```python
NN.mask("decoder").optimizer(optax.adam).lr(lrs(5e-4))
NN.mask("encoder").optimizer(optax.sgd).lr(lrs(1e-4))
NN.optimizer(optax.adamw)  # global fallback for ungrouped params
```

You can also use shorthand:

```python
NN.mask("decoder").optimizer(optax.adam, lr=lrs(5e-4))
```

During `solve()`, jNO logs group coverage/overlap diagnostics.

---

## 5. Initialize

```python
NN.initialize("/path/to/weights.msgpack")
# or
NN.initialize(pretrained_pytree)
```

### Targeted initialize

```python
NN.mask("decoder").initialize(pretrained_pytree)
```

Loads only matched leaves from pretrained weights; non-matched leaves keep fresh init.

---

## 6. Dtype

```python
NN.dtype(jnp.bfloat16)
```

Casts floating-point parameters before training. Useful for memory/perf experiments.

---

## 7. Paramax Integration

jNO automatically unwraps Paramax wrappers before each forward evaluation in training and tracker paths (when `paramax` is installed).

This means you can attach wrappers directly in the model tree and jNO will evaluate unwrapped values at runtime.

Example pattern:

```python
import paramax
import jax.numpy as jnp

scale = paramax.Parameterize(jnp.exp, jnp.log(jnp.ones(3)))
print(paramax.unwrap(("abc", 1, scale)))
# ('abc', 1, Array([1., 1., 1.], dtype=float32))
```

Note: auto-unwrapping is dependency-aware. If `paramax` is not installed, no unwrapping is attempted.

---

## 8. Precedence and Chaining Rules

- `mask(target=...)` sets the target scope for the next relevant call.
- `mask(...).lora(...)` consumes target for LoRA.
- `mask(...).freeze()` consumes target for partial freeze.
- `mask(...).initialize(...)` consumes target for partial preload.
- `mask(...).optimizer(...).lr(...)` keeps scope so both apply to the same param group.

Recommended explicit style:

```python
NN.mask("decoder").optimizer(optax.adam).lr(lrs(3e-4))
NN.mask("decoder").lora(rank=8, alpha=16)
NN.mask("encoder").freeze()
```

---

## 9. Resetting Model Configuration

```python
NN.reset()
```

Clears training-time configuration (`freeze/lora/optimizer/lr/dtype/masks/initialize staging`) and returns to default model control state.

---

## 10. Logging and Diagnostics

At `solve()` time jNO logs:

- mask target match summary
- zero-match warnings
- parameter-group summary
- overlap/uncovered diagnostics for groups
- detailed path samples in log file (`quiet` logs)

This is designed to make complex model control chains auditable.
