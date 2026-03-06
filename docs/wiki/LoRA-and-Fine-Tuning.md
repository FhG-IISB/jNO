# LoRA and Fine-Tuning

jNO supports [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) for parameter-efficient fine-tuning of trained models. Instead of updating all parameters, LoRA freezes the base weights and trains low-rank update matrices, significantly reducing the number of trainable parameters.

## Basic Usage

```python
import jno
import jno.numpy as jnn
import optax

# ... define domain, model, constraints, and train ...

# Create a rank dict from existing parameters
rank_dict = jno.create_rank_dict(crux.params, rank=1, alpha=1.0)

# Fine-tune with LoRA
crux.solve(
    10_000,
    optax.adam,
    jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5),
    lora=rank_dict,
).plot("training_history_lora.png")
```

---

## How It Works

LoRA decomposes weight updates as:

```
W_new = W_base + (alpha / rank) * B @ A
```

Where:
- `W_base` is the frozen original weight matrix
- `A` has shape `(rank, in_features)` — initialized with random normal
- `B` has shape `(out_features, rank)` — initialized to zeros
- `alpha` scales the update magnitude
- Only `A` and `B` are trained

This dramatically reduces the number of trainable parameters. For example, a `Dense(64, 64)` layer has 4,096 parameters. With LoRA rank=8, only `8 × 64 + 64 × 8 = 1,024` parameters are trained.

---

## Creating a Rank Dictionary

### Apply to All Layers

```python
rank_dict = jno.create_rank_dict(crux.params, rank=8, alpha=1.0)
```

### Selective Application

```python
# Only apply to specific layers
rank_dict = jno.create_rank_dict(
    crux.params,
    rank=8,
    include=["Dense_0", "Dense_1"],  # Only these layers
)

# Exclude certain layers
rank_dict = jno.create_rank_dict(
    crux.params,
    rank=8,
    exclude=["Dense_2"],  # Skip this layer
)
```

### Multi-Model Setup

When training coupled systems with multiple networks:

```python
# Apply LoRA only to model 0
rank_dict = jno.create_rank_dict(crux.params, rank=8, model_ids=[0])
```

### Manual Rank Dictionary

For fine-grained control, construct the dictionary manually:

```python
rank_dict = {
    'params': {
        'Dense_0': {'kernel': (8, 1.0)},   # rank=8, alpha=1.0
        'Dense_1': {'kernel': (16, 0.5)},   # rank=16, alpha=0.5
        'Dense_2': {'kernel': float('nan')}, # skip this layer
    }
}
```

For multi-model setups:

```python
rank_dict = {
    0: {  # First model
        'params': {
            'Dense_0': {'kernel': (8, 1.0)},
            'Dense_1': {'kernel': (16, 0.5)},
        }
    },
    1: {  # Second model — no LoRA
        'params': {
            'Dense_0': {'kernel': float('nan')},
        }
    }
}
```

---

## Inspecting Parameters

Use `print_params_structure` to see the parameter tree before creating a rank dict:

```python
from jno.utils.lora import print_params_structure
print_params_structure(crux.params)
```

Output:

```
Params structure:
├── params/
│   ├── Dense_0/
│   │   └── kernel: (2, 64) float32
│   ├── Dense_1/
│   │   └── kernel: (64, 64) float32
│   └── Dense_2/
│       └── kernel: (64, 1) float32
```

---

## Tips

- **Start with a small rank** (1–4) and increase if needed
- **Alpha controls the update scale** — higher alpha = larger updates relative to base weights
- LoRA only applies to 2D weight matrices (kernels), not biases
- After LoRA training, weights are automatically merged back into the base model
- LoRA training is indicated by `[LoRA]` in the training progress output

---

## See Also

- [Training and Solving](Training-and-Solving.md) — the `solve()` method and its `lora` parameter
- [Examples](Examples.md) — the `laplace1D` example demonstrates LoRA fine-tuning
