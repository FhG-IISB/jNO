# Model Controls

This page documents the model-control API in jNO 0.2.1.

The recommended style is:

- build a model with `foundax` (`fx.mlp`, `fx.fno2d`, `fx.poseidon.T`, ...) **or write your own `equinox.Module`**
- wrap it with `jno.nn(...)`
- apply model controls on the wrapped `Model`

`jno.nn` accepts any [Equinox](https://docs.kidger.site/equinox/) module — foundax models are Equinox modules, but so is anything you write yourself using `eqx.Module`. The full set of training controls (optimizer, freeze, LoRA, masks, dtype, diagnostics) is available regardless of where the model comes from.

## Quick Start

```python
import optax
import foundax as fx
import jno
from jno import LearningRateSchedule as lrs

# foundax model
net = jno.nn(
    fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3),
    name="u_net",
)
net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.8, 5000, 1e-5))

# custom equinox model — works identically
import equinox as eqx

class MyNet(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [eqx.nn.Linear(2, 64, key=k1), eqx.nn.Linear(64, 1, key=k2)]

    def __call__(self, x, y):
        h = jax.nn.tanh(self.layers[0](jnp.stack([x, y])))
        return self.layers[1](h)

custom_net = jno.nn(MyNet(jax.random.PRNGKey(0)))
custom_net.optimizer(optax.adam(1e-3))
```

## Available Methods

`Model` (returned by `jno.nn(...)`) supports:

- `dont_show()`
- `summary()`
- `freeze()` / `unfreeze()`
- `mask(param_mask=None)`
- `lora(rank=4, alpha=1.0, *, target=None, wrapper=None, specs=None)`
- `optimizer(opt_fn)`
- `scale(schedule_or_scalar)`
- `initialize(weights_or_path_or_initializer, *, key=None)`
- `dtype(dtype)`
- `tune(...)`
- `reset()`

All methods return `self` and are chainable.
