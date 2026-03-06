# Custom Models

jNO can wrap any [Flax](https://github.com/google/flax) module (both Linen and NNX APIs) for use in the training pipeline. This allows you to define arbitrary neural network architectures while benefiting from jNO's automatic differentiation, domain handling, and training infrastructure.

## Wrapping Flax Linen Modules

```python
import flax.linen as nn
import jax.numpy as jnp
import jno.numpy as jnn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(2):
            h = nn.tanh(nn.Dense(64)(h))
        return nn.Dense(1)(h)

# Wrap an instance of the module
model = jnn.nn.wrap(MLP())

# Use in expressions
u = model(x, y) * x * (1 - x) * y * (1 - y)
```

The wrapped model:
- Accepts jNO `Variable` and `Placeholder` objects as inputs
- Returns a `Placeholder` that can be used in PDE expressions
- Has its parameters automatically managed by `jno.core`

---

## Wrapping Flax NNX Modules

The NNX API uses a more Pythonic, stateful style:

```python
from flax import nnx
import jax.numpy as jnp
import jno.numpy as jnn

class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(in_features=2, out_features=64, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=64, out_features=64, rngs=rngs)
        self.dense3 = nnx.Linear(in_features=64, out_features=1, rngs=rngs)

    def __call__(self, x, y):
        h = jnp.concat([x, y], axis=-1)
        h = jnp.tanh(self.dense1(h))
        h = jnp.tanh(self.dense2(h))
        return self.dense3(h)

# Wrap an instantiated NNX module
model = jnn.nn.wrap(MLP(nnx.Rngs(0)))
u = model(x, y)
```

---

## Custom Module Requirements

Your module's `__call__` method should:

1. Accept the same number of arguments as the variables you pass when calling the wrapped model
2. Return a JAX array (typically of shape `(N, 1)` or `(N,)`)
3. Use only JAX-compatible operations

### Input Signature Examples

```python
# 1D problem: single spatial variable
class Net1D(nn.Module):
    @nn.compact
    def __call__(self, x):          # one input
        ...

u = jnn.nn.wrap(Net1D())(x)

# 2D time-dependent problem
class Net2DT(nn.Module):
    @nn.compact
    def __call__(self, x, y, t):    # three inputs
        ...

u = jnn.nn.wrap(Net2DT())(x, y, t)

# Operator learning with parameter input
class OpNet(nn.Module):
    @nn.compact
    def __call__(self, x, y, theta):  # spatial + parameter
        ...

u = jnn.nn.wrap(OpNet())(x, y, θ)
```

---

## Kolmogorov-Arnold Networks (KAN)

jNO supports [jaxKAN](https://github.com/srigas/jaxKAN) networks. Since KAN expects a single concatenated input, wrap it in an NNX module:

```python
from flax import nnx
from jaxkan.models.KAN import KAN
import jax.numpy as jnp
import jno.numpy as jnn

layer_dims = [3, 12, 12, 1]
req_params = {"D": 5, "flavor": "exact"}

class MyKAN(nnx.Module):
    def __init__(self):
        self.KAN = KAN(
            layer_dims=layer_dims,
            layer_type="chebyshev",
            required_parameters=req_params,
            seed=42,
        )

    def __call__(self, x, y, t):
        h = jnp.concat([x, y, t], axis=-1)
        return self.KAN(h)

u = jnn.nn.wrap(MyKAN())(x, y, t) * x * (1 - x) * y * (1 - y)
```

---

## Hiding Model Summaries

By default, wrapped models print a summary when constructed. To suppress this:

```python
net = jnn.nn.wrap(MLP())
net.dont_show()  # Suppress model summary output
u = net(x, y)
```

---

## Combining Multiple Networks

You can freely combine multiple wrapped networks in expressions:

```python
u_net = jnn.nn.mlp(hidden_dims=64, num_layers=2)
v_net = jnn.nn.wrap(CustomModule())

u = u_net(x, y) * x * (1 - x) * y * (1 - y)
v = v_net(x, y, k)

# Use both in a coupled PDE system
pde1 = -jnn.laplacian(u(x, y), [x, y]) + v(x, y, k) - f(x, y)
pde2 = -jnn.laplacian(v(x, y, k), [x, y]) + u(x, y) - g(x, y)
```

---

## Wrapping for Architecture Search

When using architecture search, pass the **class** (not an instance) along with the search space:

```python
class MLP(nn.Module):
    arch: jnn.tune.Arch

    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(self.arch("dep")):
            h = self.arch("act")(nn.Dense(self.arch("hid"))(h))
        return nn.Dense(1)(h)

a_space = jnn.tune.space()
a_space.unique("act", [nn.tanh, nn.selu, nn.gelu], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")

u = jnn.nn.wrap(MLP, space=a_space)(x, y)  # Pass class, not instance
```

See [Architecture Search](Architecture-Search.md) for the full tuning workflow.

---

## See Also

- [Architecture Guide](Architecture-Guide.md) — built-in architectures
- [Architecture Search](Architecture-Search.md) — automated tuning with custom modules
