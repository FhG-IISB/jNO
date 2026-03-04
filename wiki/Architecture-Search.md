# Architecture Search

jNO provides automated hyperparameter and architecture search using [Nevergrad](https://github.com/facebookresearch/nevergrad) for black-box optimization. This allows you to efficiently explore combinations of model architectures, training hyperparameters, and optimizer configurations.

## Overview

Architecture search in jNO works in two parts:

1. **Architecture space** — defines choices for the model itself (activations, widths, depths)
2. **Training space** — defines choices for training (epochs, optimizer, learning rate, etc.)

---

## Defining an Architecture Space

```python
import jno.numpy as jnn
from flax import linen as nn
import jax.numpy as jnp

# Create the search space
a_space = jnn.tune.space()
a_space.unique("act", [nn.tanh, nn.selu, nn.gelu, jnp.sin], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")
```

### Space Types

| Method | Description | Example |
|--------|-------------|---------|
| `unique(name, options)` | Categorical choice | `space.unique("act", [nn.tanh, nn.gelu])` |
| `float_range(name, low, high)` | Continuous float | `space.float_range("dropout", 0.0, 0.5)` |
| `int_range(name, low, high)` | Discrete integer | `space.int_range("n_layers", 2, 6)` |

### Categories

Each parameter belongs to a category:
- `"architecture"` — model structure choices (default for `unique`)
- `"training"` — training hyperparameters (default for `float_range`, `int_range`)
- `"optimizer"` — optimizer choices

---

## Defining an Architecture-Aware Model

Use `jnn.tune.Arch` in your Flax module to access architecture choices:

```python
class MLP(nn.Module):
    arch: jnn.tune.Arch

    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(self.arch("dep")):         # Number of layers
            h = self.arch("act")(nn.Dense(self.arch("hid"))(h))  # Activation and width
        return nn.Dense(1)(h)

# Pass the CLASS (not instance) with the space
u = jnn.nn.wrap(MLP, space=a_space)(x, y)
```

> **Important:** When using architecture search, pass the model **class** to `jnn.nn.wrap()`, not an instance. jNO will instantiate it with different `Arch` configurations during the sweep.

---

## Training Hyperparameter Space

Define a separate space for training parameters:

```python
from jno import LearningRateSchedule as lrs
from jno import WeightSchedule as ws
import optax

t_space = jnn.tune.space()
t_space.unique("epochs", [500, 1000])
t_space.unique("optimizer", [optax.adam(1.0), optax.adamw(1.0)])
t_space.unique("learning_rate", [
    lrs.exponential(1e-3, 0.8, 10000, 1e-5),
    lrs.exponential(1e-2, 0.9, 5000, 1e-5),
    lrs.constant(1e-3)
])
t_space.unique("weight_schedule", [ws([1.0])])
t_space.unique("batchsize", [1, 1])
```

---

## Running a Sweep

Use `crux.sweep()` instead of `crux.solve()`:

```python
import nevergrad as ng

# Create the problem
crux = jno.core([pde], domain)

# Run the sweep
crux.sweep(
    space=t_space,                    # Training hyperparameter space
    optimizer=ng.optimizers.NGOpt,    # Nevergrad optimizer for search
    budget=10                         # Number of configurations to try
).plot("best_training_history.png")
```

### Sweep Parameters

| Parameter | Description |
|-----------|-------------|
| `space` | Training hyperparameter search space |
| `optimizer` | Nevergrad optimizer class (e.g., `NGOpt`, `CMA`, `PSO`) |
| `budget` | Number of trials to run |

### Accessing Results

After a sweep, the best configuration is stored:

```python
print(f"Best configuration: {crux.best_config}")
```

---

## Full Example

```python
import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs, WeightSchedule as ws
import optax
from flax import linen as nn
import jax.numpy as jnp
import nevergrad as ng

# Domain
domain = jno.domain(constructor=jno.domain.disk(mesh_size=0.05))
x, y = domain.variable("interior")

# Architecture search space
a_space = jnn.tune.space()
a_space.unique("act", [nn.tanh, nn.selu, nn.gelu], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")

# Architecture-aware model
class MLP(nn.Module):
    arch: jnn.tune.Arch
    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(self.arch("dep")):
            h = self.arch("act")(nn.Dense(self.arch("hid"))(h))
        return nn.Dense(1)(h)

u = jnn.nn.wrap(MLP, space=a_space)(x, y)
_u = u * x * (1 - x) * y * (1 - y)
pde = -jnn.laplacian(_u, [x, y]) - 1.0

# Problem
crux = jno.core([pde], domain)

# Training search space
t_space = jnn.tune.space()
t_space.unique("epochs", [500, 1000])
t_space.unique("optimizer", [optax.adam(1.0), optax.adamw(1.0)])
t_space.unique("learning_rate", [lrs.exponential(1e-3, 0.8, 10000, 1e-5)])
t_space.unique("weight_schedule", [ws([1.0])])
t_space.unique("batchsize", [1])

# Run sweep
crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=10)
    .plot("best_training_history.png")
```

---

## See Also

- [Custom Models](Custom-Models.md) — defining architecture-aware models
- [Training and Solving](Training-and-Solving.md) — standard training without search
