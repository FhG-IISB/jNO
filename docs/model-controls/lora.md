# LoRA

LoRA inserts trainable low-rank adapter matrices into matching layers while keeping base weights frozen. By default both linear and conv layers are wrapped automatically.

## Selecting Layers

```python
net.lora(rank=8, target="encoder")                         # path-regex
net.mask(encoder_mask).lora(rank=8)                        # boolean mask
net.mask(encoder_mask).lora(rank=8, target="encoder")      # both combined
```

| | `target=` / `specs=` | `mask(M).lora()` |
|---|---|---|
| **Selects by** | regex on pytree path string | boolean pytree |
| **Use when** | you know the layer names up front | you have a precomputed mask |

## Uniform and Per-Target Specs

```python
net.lora(rank=8, alpha=16)                    # all layers
net.lora(rank=8, alpha=16, target="encoder")  # restricted to a subset

net.lora(
    specs=[
        {"target": "encoder", "rank": 4,  "alpha": 1.0},
        {"target": "decoder", "rank": 16, "alpha": 4.0},
    ]
)
```

`target` is regex-matched against the slash-joined pytree path. The first matching spec wins.

## Combining with Mask and Freeze

```python
net.mask(encoder_mask).lora(rank=8, alpha=16)           # only mask-selected layers
net.freeze().lora(rank=8, alpha=16)                     # freeze all; only adapters train
net.freeze().mask(encoder_mask).lora(rank=8, alpha=16)  # wrap M-selected, freeze rest
```

## Default Layer Types

Without `wrapper=`, jNO wraps Linear and Conv layers (ConvTranspose excluded):

```python
net.lora(rank=4, alpha=1.0)           # Linear + Conv1d/2d/3d
net.lora(rank=4, wrapper=LoRALinear)  # linear only
net.lora(rank=4, wrapper=LoRAConv)    # conv only
```

## Custom Adapters

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

## LoRA Zoo — Linear

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

**When to use which:**

- **rsLoRALinear** — default upgrade; use higher ranks without numerical issues.
- **LoRAFALinear** — memory-constrained; halves adapter parameter count.
- **DoRALinear** — pretrained models where preserving weight norms matters.
- **PiSSALinear** — pretrained models; adapters start at the most informative directions.
- **LoRAXSLinear** — extreme parameter efficiency.
- **VeRALinear** — fewest trainable params; A, B not stored in checkpoints.
- **MiLoRALinear** — preserves principal directions; adapts the noise subspace.
- **IA3Linear** — no rank hyperparameter; ideal for fast probing.
- **LoKrLinear** — large layers where Kronecker factorisation is more efficient.
- **OFTLinear** — orthogonal weight updates to preserve geometry.

## LoRA Zoo — Conv

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
