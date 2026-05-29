# Model Controls

This page documents the model-control API in jNO 0.2.1.

Important update: architecture factories moved to Foundax. The recommended style is:

- build a model with `foundax` (`fx.mlp`, `fx.fno2d`, `fx.poseidon.T`, ...)
- wrap it with `jno.nn.wrap(...)`
- apply model controls on the wrapped `Model`

---

## Quick Start

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

## Available Methods

`Model` (returned by `jno.nn.wrap(...)`) supports:

- `dont_show()`
- `summary()`
- `freeze()` / `unfreeze()`
- `mask(param_mask=None)`
- `constrain(transform)`
- `lora(rank=4, alpha=1.0, *, target=None, wrapper=None, specs=None)`
- `optimizer(opt_fn, *, lr=None)`
- `lr(schedule_or_scalar)`
- `initialize(weights_or_path_or_initializer, *, key=None)`
- `dtype(dtype)`
- `tune(...)`
- `reset()`

All methods return `self` and are chainable.

---

## Mask

`mask(...)` takes a boolean pytree mask only.

```python
import equinox as eqx
import jax

all_true = jax.tree_util.tree_map(lambda _: True, eqx.filter(net.module, eqx.is_array))
net.mask(all_true).optimizer(optax.adam, lr=lrs(1e-4))
```

There is no `target="..."` argument on `mask(...)`.

### Regex-style targeting

If you want to target layers by name, build a boolean mask from a regex:

```python
import re
import equinox as eqx
import jax

def regex_mask(module, pattern: str):
    arrays = eqx.filter(module, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(arrays)

    def part(k):
        if hasattr(k, "name"): return str(k.name)
        if hasattr(k, "idx"):  return str(k.idx)
        if hasattr(k, "key"):  return str(k.key)
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

## Freeze / Unfreeze

```python
net.freeze()                      # freeze entire model
net.mask(decoder_mask).freeze()   # freeze only selected leaves
```

With `mask(...).freeze()`, non-selected leaves remain trainable.

---

## LoRA

LoRA inserts trainable low-rank adapter matrices into matching layers while keeping base weights frozen. By default both linear and conv layers are wrapped automatically.

### Selecting layers

```python
net.lora(rank=8, target="encoder")          # path-regex: only encoder layers
net.mask(encoder_mask).lora(rank=8)         # boolean mask: data-driven selection
net.mask(encoder_mask).lora(rank=8, target="encoder")  # both combined
```

### Uniform and per-target specs

```python
net.lora(rank=8, alpha=16)                  # all layers
net.lora(rank=8, alpha=16, target="encoder")  # restricted to a subset

net.lora(
    specs=[
        {"target": "encoder", "rank": 4,  "alpha": 1.0},
        {"target": "decoder", "rank": 16, "alpha": 4.0},
    ]
)
```

`target` is regex-matched against the slash-joined pytree path. The first matching spec wins.

### Combining mask and freeze with LoRA

```python
net.mask(encoder_mask).lora(rank=8, alpha=16)          # only mask-selected layers get LoRA
net.freeze().lora(rank=8, alpha=16)                    # freeze all; only adapters train
net.freeze().mask(encoder_mask).lora(rank=8, alpha=16) # wrap M-selected only, freeze rest
```

### Default layer types

Without a `wrapper=` argument, jNO wraps Linear and Conv layers (ConvTranspose excluded):

```python
net.lora(rank=4, alpha=1.0)           # Linear + Conv1d/2d/3d
net.lora(rank=4, wrapper=LoRALinear)  # linear only
net.lora(rank=4, wrapper=LoRAConv)    # conv only
```

### Custom adapters

Subclass `LoRAWrapper` to support layer types not in the zoo:

```python
from jno.lora import LoRAWrapper
import equinox as eqx
import jax.numpy as jnp

class MyEmbeddingAdapter(LoRAWrapper):
    adapter_fields = ("delta",)
    base: eqx.Module
    delta: jax.Array
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf):
        return isinstance(leaf, eqx.nn.Embedding) and not isinstance(leaf, LoRAWrapper)

    def __init__(self, base, rank, alpha, *, key):
        self.base, self.rank, self.alpha = base, rank, alpha
        self.delta = jnp.zeros_like(base.weight)

    def __call__(self, x):
        return self.base(x) + self.delta[x] * (self.alpha / self.rank)

    def merge(self):
        w = self.base.weight + self.delta * (self.alpha / self.rank)
        return eqx.tree_at(lambda m: m.weight, self.base, w)

net.lora(rank=4, wrapper=MyEmbeddingAdapter)
net.lora(rank=4, wrapper=[LoRALinear, LoRAConv, MyEmbeddingAdapter])
```

Per-target specs may carry their own `"wrapper"` key:

```python
net.lora(
    specs=[
        {"target": "linear", "rank": 4, "alpha": 1.0},
        {"target": "embed",  "rank": 8, "alpha": 2.0, "wrapper": MyEmbeddingAdapter},
    ]
)
```

### LoRA Zoo — Linear

```python
from jno.lora import (
    LoRALinear,    # standard LoRA (default)
    rsLoRALinear,  # rank-stabilized
    LoRAFALinear,  # frozen A — fewer trainable params
    DoRALinear,    # weight-decomposed
    PiSSALinear,   # SVD init — fastest convergence on pretrained models
    LoRAXSLinear,  # extra-small r×r core
    VeRALinear,    # frozen random A,B; only b,d vectors trained
    MiLoRALinear,  # minor SVD components — preserves pretrained knowledge
    IA3Linear,     # output scaling vector — no low-rank matrices
    LoKrLinear,    # Kronecker product adapter
    OFTLinear,     # block-diagonal orthogonal fine-tuning
)
```

| Class | Trainable params | Key idea |
|-------|-----------------|----------|
| `LoRALinear` | `r·(in + out)` | Standard LoRA; scale = `α/r` |
| `rsLoRALinear` | `r·(in + out)` | Scale = `α/√r` — stable across ranks ([rsLoRA](https://arxiv.org/abs/2312.03732)) |
| `LoRAFALinear` | `r·out` | Frozen A; halves adapter params ([LoRAFA](https://arxiv.org/abs/2308.03303)) |
| `DoRALinear` | `r·(in + out) + out` | Magnitude + direction decomposition ([DoRA](https://arxiv.org/abs/2402.09353)) |
| `PiSSALinear` | `r·(in + out)` | SVD principal-component init ([PiSSA](https://arxiv.org/abs/2404.02948)) |
| `LoRAXSLinear` | `r²` | Frozen A,B; trainable r×r core only ([LoRA-XS](https://arxiv.org/abs/2405.17604)) |
| `VeRALinear` | `out + r` | Seed-based frozen A,B; only b,d vectors trained ([VeRA](https://arxiv.org/abs/2310.11454)) |
| `MiLoRALinear` | `r·(in + out)` | Adapts minor SVD components ([MiLoRA](https://arxiv.org/abs/2405.09913)) |
| `IA3Linear` | `out` | Per-output scale vector; no rank hyperparameter ([IA³](https://arxiv.org/abs/2205.05638)) |
| `LoKrLinear` | `r² + ⌈out/r⌉·⌈in/r⌉` | Kronecker product adapter ([LoKr](https://arxiv.org/abs/2212.10650)) |
| `OFTLinear` | `n_blocks·r²` | Orthogonal fine-tuning via Cayley map ([OFT](https://arxiv.org/abs/2306.07280)) |

### LoRA Zoo — Conv

Matching conv variants for `eqx.nn.Conv1d/2d/3d`. All flatten the weight to `(out_ch, flat_in)`, apply the same adapter logic, and reshape back.

```python
from jno.lora import (
    LoRAConv, rsLoRAConv, LoRAFAConv, DoRAConv, PiSSAConv,
    LoRAXSConv, VeRAConv, MiLoRAConv, IA3Conv, LoKrConv, OFTConv,
)
```

| Class | Trainable params | Key idea |
|-------|-----------------|----------|
| `LoRAConv` | `r·(flat_in + out_ch)` | Standard LoRA on flattened conv weight |
| `rsLoRAConv` | `r·(flat_in + out_ch)` | Rank-stabilized scaling `α/√r` |
| `LoRAFAConv` | `r·out_ch` | Frozen A; only B trained |
| `DoRAConv` | `r·(flat_in + out_ch) + out_ch` | Magnitude + direction decomposition |
| `PiSSAConv` | `r·(flat_in + out_ch)` | SVD principal components init |
| `LoRAXSConv` | `r²` | Frozen A,B from SVD; trainable r×r core |
| `VeRAConv` | `out_ch + r` | Seed-based frozen A,B; only b,d vectors trained |
| `MiLoRAConv` | `r·(flat_in + out_ch)` | SVD minor components |
| `IA3Conv` | `out_ch` | Per-output-channel scale vector |
| `LoKrConv` | `r² + ⌈out_ch/r⌉·⌈flat_in/r⌉` | Kronecker product adapter |
| `OFTConv` | `n_blocks·r²` | Block-diagonal Cayley map on output channels |

Mix linear and conv adapters per layer group:

```python
net.lora(
    specs=[
        {"target": "encoder", "rank": 8,  "alpha": 16,  "wrapper": PiSSALinear},
        {"target": "decoder", "rank": 4,  "alpha": 1.0, "wrapper": rsLoRALinear},
        {"target": "conv",    "rank": 4,  "alpha": 1.0, "wrapper": rsLoRAConv},
    ]
)
```

---

## Optimizer and LR

### Global

```python
net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

### Parameter groups

```python
net.optimizer(optax.adamw, lr=lrs(1e-3))               # global fallback
net.mask(decoder_mask).optimizer(optax.adam, lr=lrs(5e-4))
net.mask(encoder_mask).optimizer(optax.sgd,  lr=lrs(1e-4))
```

`mask(...)` is consumed by the next mutator call. A bare global `optimizer(...)` clears all previously configured groups.

```python
net.mask(decoder_mask).lr(lrs(1e-5))   # group-specific LR only
net.lr(lrs(1e-5))                      # global LR update
```

---

## Initialize

`initialize(...)` supports checkpoint paths, pytrees, and callable initializers:

```python
import jax

net.initialize("./weights.eqx")
net.initialize("./runs/checkpoints/2000::1")
net.initialize(other_model.module)
net.initialize(jax.nn.initializers.xavier_uniform(), key=jax.random.PRNGKey(0))
```

---

## Dtype

```python
import jax.numpy as jnp

net.dtype(jnp.bfloat16)
```

Casts floating-point parameters before training.

---

## Tune and Reset

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

`reset()` clears all training-time controls (`freeze`, `lora`, `optimizer`, `lr`, `dtype`, `mask`, init state).

---

## Constrain

`constrain(transform)` applies a [paramax](https://github.com/danielward27/paramax) reparameterization to trainable parameter leaves. The raw parameter is stored unconstrained; `transform` is applied automatically before every forward pass via `paramax.unwrap()`.

```python
import jax

k_net.constrain(jax.nn.softplus)   # all weights pass through softplus
k_net.constrain(jax.nn.sigmoid)    # all weights projected to (0, 1)
```

When preceded by `mask(...)`, only the selected leaves are wrapped:

```python
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)
k_net.mask(output_mask).constrain(jax.nn.softplus)  # output layer only
```

This is a **hard parameter constraint** — the constraint holds at every step without adding a penalty term to the loss. Use `jno.fn.regularize.nonneg` / `.bounded` for **soft constraints** on the field *output* instead.

---

## Paramax Integration

jNO automatically unwraps paramax wrappers before each forward evaluation. Any `paramax.Parameterize` wrapper inserted directly into the model pytree is also handled:

```python
import paramax
import jax.numpy as jnp
import equinox as eqx

# Direct usage
scale = paramax.Parameterize(jnp.exp, jnp.log(jnp.ones(3)))
print(paramax.unwrap(("abc", 1, scale)))
# ('abc', 1, Array([1., 1., 1.], dtype=float32))

# Via constrain() — preferred
k_net.constrain(jax.nn.softplus)
```

---

## Logging and Diagnostics

At `solve()` time jNO logs:

- parameter-group summary
- overlap/uncovered diagnostics for groups
- zero-match warnings for empty masks/groups
- detailed path samples in log file (`quiet` logs)
