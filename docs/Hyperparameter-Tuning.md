# Hyperparameter Tuning

jNO integrates [Nevergrad](https://github.com/facebookresearch/nevergrad) to automate the search for optimal neural architecture configurations and training hyperparameters.

---

## Overview

The tuner searches over a user-defined space of:
- **Architecture hyperparameters** — activations, hidden dimensions, number of layers, etc.
- **Training hyperparameters** — epochs, learning rates, batch sizes, optimisers.
- **Per-model options** — freeze, LoRA, optimiser, learning rate per model.

The tuner runs each candidate configuration as an independent training trial and returns the best result.

---

## Defining a Search Space

Create a search space using `jno.numpy.tune.space()`:

```python
import jno.numpy as jnn

space = jnn.tune.space()
```

### Categorical Choice (`unique`)

```python
space.unique("optimizer", [optax.adam, optax.adamw, optax.sgd])
space.unique("activation", [jnp.tanh, jax.nn.gelu, jnp.sin], category="architecture")
space.unique("hidden_dim", [32, 64, 128, 256], category="architecture")
space.unique("num_layers", [2, 3, 4], category="architecture")
space.unique("batchsize", [32, 64, 128])
space.unique("epochs", [500, 1000, 2000])
space.unique("learning_rate", [
    lrs.constant(1e-3),
    lrs.exponential(1e-3, 0.8, 10_000, 1e-5),
    lrs.warmup_cosine(4000, 500, 1e-3, 1e-5),
])
space.unique("weight_schedule", [ws([1.0, 10.0]), ws([1.0, 1.0])])
```

### Continuous Float Range (`float_range`)

```python
space.float_range("dropout", low=0.0, high=0.3)
space.float_range("lr0", low=1e-4, high=1e-2, log_scale=True)
```

### Discrete Integer Range (`int_range`)

```python
space.int_range("hidden_dim", low=16, high=256)
```

### Category Labels

Use the `category` keyword to separate architecture choices from training choices:

- `"architecture"` — model structure (passed to the model's `__init__` via `Arch`)
- `"training"` — training loop parameters (epochs, batchsize, etc.)
- `"optimizer"` — optimiser-related choices

---

## Architecture Search

To search over model architectures, wrap a **class** (not an instance) with `jnn.nn.wrap(ClassName, space=a_space)` and use `jnn.tune.Arch` in the constructor:

```python
import equinox as eqx

a_space = jnn.tune.space()
a_space.unique("act", [jnp.tanh, jax.nn.gelu, jax.nn.selu, jnp.sin], category="architecture")
a_space.unique("hid", [4, 8, 16, 32], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")

class MyMLP(eqx.Module):
    layers: list
    out_layer: eqx.nn.Linear
    act: callable

    def __init__(self, arch: jnn.tune.Arch, *, key):
        depth  = arch("dep")
        hidden = arch("hid")
        self.act = arch("act")
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(2 if i == 0 else hidden, hidden, key=keys[i]) for i in range(depth)]
        self.out_layer = eqx.nn.Linear(hidden, 1, key=keys[depth])

    def __call__(self, x, y):
        h = jnp.concat([x, y], axis=-1)
        for layer in self.layers:
            h = self.act(layer(h))
        return self.out_layer(h)


domain = jno.domain(constructor=jno.domain.disk(mesh_size=0.05))
x, y = domain.variable("interior")

u = jnn.nn.wrap(MyMLP, space=a_space)(x, y)
_u = u * x * (1 - x) * y * (1 - y)

pde = -jnn.laplacian(_u, [x, y]) - 1.0
crux = jno.core([pde.mse], domain)
```

---

## Running a Sweep

```python
import nevergrad as ng

stats = crux.sweep(
    space=space,
    optimizer=ng.optimizers.NGOpt,   # Nevergrad optimiser
    budget=20,                        # number of configurations to evaluate
)
stats.plot("best_history.png")
print(f"Best configuration: {crux.best_config}")
```

### Nevergrad Optimisers

| Optimiser | Description |
|-----------|-------------|
| `ng.optimizers.NGOpt` | Adaptive meta-optimiser (default recommendation) |
| `ng.optimizers.OnePlusOne` | Simple evolutionary strategy |
| `ng.optimizers.CMA` | Covariance Matrix Adaptation ES (good for continuous params) |
| `ng.optimizers.RandomSearch` | Purely random exploration |
| `ng.optimizers.TwoPointsDE` | Differential Evolution |

---

## Per-Model Tuning

Attach tunable options **directly to individual models** with `.tune()`. This allows each model in a multi-model problem to have its own independently-tuned configuration.

```python
backbone = jnn.nn.mlp(2, hidden_dims=32, num_layers=3, key=key2)(x, y)

backbone.tune(
    freeze=[True, False],                # try frozen vs trainable
    lora=[(4, 1.0), (8, 0.5), None],    # try LoRA rank 4, 8, or no LoRA
    optimizer=[optax.adam(1), optax.lbfgs(1)],
    lr=[lrs.constant(1e-3), lrs.constant(1e-4)],
)
```

The tuner automatically enumerates all combinations of per-model options and merges them with the global space.

---

## Accessing the Best Configuration

After a sweep, the best configuration is stored in `crux.best_config`:

```python
print(crux.best_config)
# {'hidden_dim': 64, 'activation': <function tanh>, 'epochs': 1000, 'optimizer': adam, ...}
```

The solver (`crux`) itself is already configured with the best weights and optimiser.

---

## Complete Example

```python
import jax
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs, WeightSchedule as ws
import nevergrad as ng
import jax.numpy as jnp
import equinox as eqx

dire = jno.setup(__file__)
domain = jno.domain(constructor=jno.domain.disk(mesh_size=0.05))
x, y = domain.variable("interior")

# Architecture search space
a_space = jnn.tune.space()
a_space.unique("act", [jnp.tanh, jax.nn.selu, jax.nn.gelu], category="architecture")
a_space.unique("hid", [16, 32, 64], category="architecture")

class TuneMLP(eqx.Module):
    ...  # use arch("hid"), arch("act") in __init__

u = jnn.nn.wrap(TuneMLP, space=a_space)(x, y)
_u = u * x * (1 - x) * y * (1 - y)
pde = -jnn.laplacian(_u, [x, y]) - 1.0

crux = jno.core([pde.mse], domain)

# Training search space
t_space = jnn.tune.space()
t_space.unique("epochs", [500, 1000])
t_space.unique("optimizer", [optax.adam, optax.adamw])
t_space.unique("learning_rate", [lrs.constant(1e-3), lrs.exponential(1e-3, 0.9, 5000, 1e-5)])
t_space.unique("batchsize", [1, 2])

stats = crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=8)
stats.plot(f"{dire}/best_history.png")
print(f"Best config: {crux.best_config}")
crux.save(f"{dire}/crux.pkl")
```
