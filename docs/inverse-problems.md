# Inverse Problems

Inverse problems identify unknown quantities from observations — e.g. recovering a spatially-varying diffusivity field from sparse temperature measurements. jNO supports two complementary approaches: **soft constraints** via `jno.fn.regularize`, and **hard parameter constraints** via `Model.constrain()`.

---

## Scalar coefficient identification

The simplest case: recover scalar constants from residual constraints. `jno.np.parameter` creates a trainable scalar that participates in the expression tree like any other field.

```python
import jax, jax.numpy as jnp, optax, jno

domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")

# Ground-truth data (synthetic here, replace with measurements)
target = 3.14 * jno.np.sin(jnp.pi * x) - 2.71 * jno.np.cos(jnp.pi * x)

a = jno.np.parameter((1,), name="a")
b = jno.np.parameter((1,), name="b")
for p in [a, b]:
    p.optimizer(optax.adam(1e-2))

residual = (a * jno.np.sin(jnp.pi * x) + b * jno.np.cos(jnp.pi * x)) - target

crux = jno.core([residual.mse])
crux.solve(10_000)

_a, _b = crux.eval([a, b])
print(f"a={float(_a):.3f}  b={float(_b):.3f}")  # → a≈3.14  b≈-2.71
```

---

## Field identification with regularization

When the unknown is a spatially-varying field `k(x,y)`, represent it as a neural network and add regularization to the loss to make the problem well-posed.

### `jno.fn.regularize`

All functions return an **unreduced pointwise** `Placeholder` — apply `.mean` or `.mse` to get a scalar loss term.

#### `smooth(field, *variables)` — H1 seminorm

Penalises rapid spatial variation. Good default for smooth physical fields.

```python
k = jno.nn.wrap(k_net)(x, y)
reg = jno.fn.regularize.smooth(k, x, y)

crux = jno.core([pde.mse, data.mse, reg.mean])
```

#### `tv(field, *variables)` — total variation

Promotes piecewise-constant fields; better when the unknown has sharp interfaces.

```python
reg = jno.fn.regularize.tv(k, x, y)
crux = jno.core([pde.mse, reg.mean])
```

#### `nonneg(field, strength=1.0)` — soft positivity

Zero cost when `field >= 0`; penalises negative values linearly. Useful for physically positive quantities (diffusivity, viscosity).

```python
reg = jno.fn.regularize.nonneg(k)
crux = jno.core([pde.mse, reg.mean])
```

#### `bounded(field, lo, hi)` — two-sided barrier

Penalises values outside `[lo, hi]`.

```python
reg = jno.fn.regularize.bounded(k, lo=0.1, hi=2.0)
crux = jno.core([pde.mse, reg.mean])
```

---

## Inverse problems through a FEM forward (`fem.solve`)

When the forward model is a finite-element solve rather than a neural network, use
`fem.solve()` as the differentiable forward (see [Finite Element Method](fem.md)). Put a
`jno.np.parameter` in the weak form, compare the solve to data, and train through
`crux.solve` — the gradient flows through the assembled solve to the parameter.

```python
# scalar coefficient: recover k in  -k Δu = f
k = jno.np.parameter((1,), name="k")
k.dtype(jnp.float64); k.initialize(jax.nn.initializers.constant(2.0)); k.optimizer(optax.adam(5e-2))
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs)
crux.solve(200)                       # recovers k
```

A **diffusivity field** `k(x)` is `jno.np.parameter(phi)` (one DOF per node); regularize it
with `k.regularize("h1seminorm" | "l2" | "tv" | "nonneg" | "bounded")` — the FE-exact analogue
of `jno.fn.regularize`. For a **transient** weak form, `fem.solve()` returns the trajectory
`u(save_ts)`, so a rate constant is recovered from a time series. Worked examples:
[hidden diffusivity field](tutorials/08-fem-and-varpinns/inverse-diffusivity-field.md) and
[transient rate](tutorials/08-fem-and-varpinns/transient-inverse-heat.md).

---

## Hard parameter constraints via `Model.constrain()`

`constrain(transform)` uses [paramax](https://github.com/danielward27/paramax) to reparameterize the model's **weights** so that `transform(raw_weight)` is used at every forward pass. This is a **hard** constraint on the weights — it holds at every step without adding a penalty.

```python
import equinox as eqx, jax

# Constrain only the output layer weights — e.g. for a monotone final projection
k_net = jno.nn.wrap(k_mlp)
all_false = jax.tree_util.tree_map(lambda _: False, k_net.module)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)
k_net.mask(output_mask).constrain(jax.nn.softplus)   # output layer only
k = k_net(x, y)
```

Call without `.mask(...)` to apply the transform to all trainable weights:

```python
k_net.constrain(jax.nn.sigmoid)   # all weights projected to (0, 1)
```

Supported transforms — any jit-compatible callable works:

| Transform | Effect |
|-----------|--------|
| `jax.nn.softplus` | Weights are always positive |
| `jax.nn.sigmoid` | Weights projected to `(0, 1)` |
| `jnp.abs` | Weights are non-negative |
| custom `lambda w: ...` | Any bijection |

Under the hood, jNO wraps each selected leaf with [`paramax.Parameterize`](https://github.com/danielward27/paramax) and calls `paramax.unwrap()` automatically before every forward pass. You can also insert paramax wrappers directly into a model pytree if you need finer control — see the [paramax docs](https://danielward27.github.io/paramax/) for the full API including `NonTrainable`, `RealToIncreasingOnInterval`, and `WeightNormalization`.

!!! note "Weight constraints ≠ field output constraints"
    `constrain()` shapes the network **weights**, not the field *values* it produces. Applying `softplus` to all weights makes every weight positive but does not generally guarantee a positive output — the interaction of weights and activations across layers can still produce any sign.

    For physically-constrained **field outputs** (diffusivity, viscosity, density), the simplest and most reliable approach is an **output-level transform** using `jno.fn`:

    ```python
    k_raw = jno.nn.wrap(k_mlp)   # unconstrained weights — full network expressivity
    k_raw.optimizer(optax.adam(1e-3))
    k = jno.fn.exp(k_raw(x, y))  # field is always > 0 by construction
    ```

    This is what the full example below uses.

`constrain()` and `jno.fn.regularize` are complementary: use `constrain` for hard weight-space constraints (monotone layers, weight normalization) and `regularize` for soft penalties on the field *output*.

---

## Sensor observations via `jno.domain.from_array`

`jno.domain.from_array` creates a standalone point-cloud domain from in-memory arrays — useful when you want to evaluate or visualise a trained model at specific locations, or build a dedicated observation domain that you pass to `core` instead of the PDE mesh.

```python
import numpy as np

sensor_coords = np.array([[0.1, 0.2], [0.5, 0.5], [0.8, 0.3]])  # shape (N, 2)
sensor_dom = jno.domain.from_array({"obs": sensor_coords})
x_s, y_s, _ = sensor_dom.variable("obs")
```

For the common case where observation points live on the same mesh as the PDE collocation points, just use the collocation variables directly (see the full example below).

---

## Full field-inversion example

Manufactured solution: `k_true = 1`, `u_true = sin(πx)`, PDE `k·u'' = f`.

```python
import jax, jax.numpy as jnp, optax
import foundax, jno

π = jno.np.pi

domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")

# Manufactured source and noiseless observations
f_pde = -(π**2) * jno.np.sin(π * x)   # k_true · u_true'' = −π² sin(πx)
u_obs = jno.np.sin(π * x)

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

# k > 0 enforced via exp output transform (full network expressivity preserved)
k_raw = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=16, num_layers=2, key=k1))
k_raw.optimizer(optax.adam(1e-3))

u_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k2))
u_net.optimizer(optax.adam(1e-3))

k = jno.fn.exp(k_raw(x))        # always > 0 by construction
u = u_net(x) * x * (1 - x)      # hard zero Dirichlet BCs

pde  = k * u.dd(x) - f_pde
data = u - u_obs
reg  = jno.fn.regularize.smooth(k, x)

crux = jno.core([pde.mse, data.mse, reg.mean])
crux.solve(5_000)

_u, _k, _u_obs = crux.eval([u, k, u_obs])

rel_l2_u = float(jnp.linalg.norm(_u - _u_obs) / (jnp.linalg.norm(_u_obs) + 1e-8))
print(f"u  rel-L2 error : {rel_l2_u:.3e}")           # should be < 10 %
print(f"k  min / mean   : {_k.min():.3f} / {_k.mean():.3f}")  # k > 0 from exp; mean ≈ 1.0
```
