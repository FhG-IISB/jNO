# Model Controls

This page documents the model-control API in jNO 0.2.1.

Important update: architecture factories moved to Foundax. The recommended style is:

- build a model with `foundax` (`fx.mlp`, `fx.fno2d`, `fx.poseidon.T`, ...)
- wrap it with `jno.nn.wrap(...)`
- apply model controls on the wrapped `Model`

---

## Quick Start (Foundax-first)

```python
import optax
import foundax as fx
import jno
from jno import LearningRateSchedule as lrs

net = jno.nn.wrap(
	fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3),
	name="u_net",
)

net.dont_show()
net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 5000, 1e-5))
net.lora(rank=8, alpha=16)
```

Legacy shorthand constructors like `jno.numpy.nn.mlp(...)` are no longer the primary API.

---

## Available Model Methods

`Model` (returned by `jno.nn.wrap(...)`) supports:

- `dont_show()`
- `summary()`
- `freeze()` / `unfreeze()`
- `mask(param_mask=None)`
- `lora(rank=4, alpha=1.0, *, specs=None)`
- `optimizer(opt_fn, *, lr=None)`
- `lr(schedule_or_scalar)`
- `initialize(weights_or_path_or_initializer, *, key=None)`
- `dtype(dtype)`
- `tune(...)`
- `reset()`

All methods return `self` and are chainable.

---

## 1. `mask(...)` Uses Explicit Boolean Pytrees

`mask(...)` now takes a boolean pytree mask only.

```python
import equinox as eqx
import jax

all_true = jax.tree_util.tree_map(lambda _: True, eqx.filter(net.module, eqx.is_array))
net.mask(all_true).optimizer(optax.adam, lr=lrs(1e-4))
```

There is no `target="..."` argument on `mask(...)`.

### Optional helper for regex-like targeting

If you want the old regex-style workflow, build a mask yourself:

```python
import re
import equinox as eqx
import jax

def regex_mask(module, pattern: str):
	arrays = eqx.filter(module, eqx.is_array)
	flat, treedef = jax.tree_util.tree_flatten_with_path(arrays)

	def part(k):
		if hasattr(k, "name"):
			return str(k.name)
		if hasattr(k, "idx"):
			return str(k.idx)
		if hasattr(k, "key"):
			return str(k.key)
		return str(k)

	leaves = []
	for path, _ in flat:
		path_str = "/".join(part(p) for p in path)
		leaves.append(bool(re.search(pattern, path_str)))

	return jax.tree_util.tree_unflatten(treedef, leaves)

decoder_mask = regex_mask(net.module, r"decoder")
net.mask(decoder_mask).optimizer(optax.adam, lr=lrs(3e-4))
```

---

## 2. Freeze / Unfreeze

### Global freeze

```python
net.freeze()
```

Freezes the whole model.

### Masked freeze

```python
net.mask(decoder_mask).freeze()
```

With `mask(...).freeze()`, selected leaves are frozen and non-selected leaves remain trainable.

---

## 3. LoRA

### Uniform LoRA

```python
net.lora(rank=8, alpha=16)
```

### Per-target LoRA specs

```python
net.lora(
	specs=[
		{"target": "encoder", "rank": 4, "alpha": 1.0},
		{"target": "decoder", "rank": 16, "alpha": 4.0},
	]
)
```

`target` in LoRA `specs` is regex-matched against pytree paths of supported linear leaves.

### Combine mask + LoRA for base-trainability control

```python
# Selected base leaves are frozen; non-selected base leaves stay trainable.
net.mask(decoder_mask).lora(rank=8, alpha=16)

# Freeze all base params; train LoRA adapters only.
net.freeze().lora(rank=8, alpha=16)
```

---

## 4. Optimizer and LR

### Global

```python
net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

### Parameter groups via masks

```python
net.optimizer(optax.adamw, lr=lrs(1e-3))   # global fallback for uncovered params
net.mask(decoder_mask).optimizer(optax.adam, lr=lrs(5e-4))
net.mask(encoder_mask).optimizer(optax.sgd, lr=lrs(1e-4))
```

### One-shot mask scope

`mask(...)` is consumed by the next relevant mutator call.

```python
net.mask(decoder_mask).optimizer(optax.adam)
net.lr(lrs(1e-5))  # global, because mask scope was already consumed

net.mask(decoder_mask).lr(lrs(1e-5))  # group-specific LR
```

A bare global `optimizer(...)` call clears previously configured parameter groups.

During `solve()`, jNO logs group coverage, overlap, and uncovered-parameter diagnostics.

---

## 5. Initialize

`initialize(...)` supports:

- checkpoint path (`.eqx` or Orbax checkpoint directory, optionally `"path::model_key"`)
- pytree / module object
- callable initializer

```python
import jax

net.initialize("./weights.eqx")
net.initialize("./runs/checkpoints/2000::1")
net.initialize(other_model.module)
net.initialize(jax.nn.initializers.xavier_uniform(), key=jax.random.PRNGKey(0))
```

Unlike older docs, `mask(...)` does not provide targeted/partial initialize scoping.

---

## 6. Dtype

```python
import jax.numpy as jnp

net.dtype(jnp.bfloat16)
```

Casts floating-point parameters before training.

---

## 7. Tune and Reset

```python
net.tune(
	freeze=[True, False],
	lora=[(4, 1.0), None],
	optimizer=[optax.adam],
	lr=[lrs(1e-3), lrs(1e-4)],
	dtype=[jnp.float32],
)

net.reset()
```

`reset()` clears training-time controls (`freeze/lora/optimizer/lr/dtype/mask/init state`).

---

## 8. Paramax Integration

jNO automatically unwraps Paramax wrappers before each forward evaluation in training and tracker paths (when `paramax` is installed).

```python
import paramax
import jax.numpy as jnp

scale = paramax.Parameterize(jnp.exp, jnp.log(jnp.ones(3)))
print(paramax.unwrap(("abc", 1, scale)))
# ('abc', 1, Array([1., 1., 1.], dtype=float32))
```

If `paramax` is not installed, no unwrapping is attempted.

---

## 9. Logging and Diagnostics

At `solve()` time jNO logs:

- parameter-group summary
- overlap/uncovered diagnostics for groups
- zero-match warnings for empty masks/groups
- detailed path samples in log file (`quiet` logs)

This is designed to make complex model-control chains auditable.
