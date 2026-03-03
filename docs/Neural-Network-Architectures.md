# Neural Network Architectures

All neural network architectures in jNO are accessible via `jno.numpy.nn` (importable as `jnn.nn`). Every factory method returns a `Model` wrapper that integrates seamlessly with the jNO training pipeline — supporting `.optimizer()`, `.freeze()`, `.lora()`, `.mask()`, and `.initialize()`.

---

## Architecture Overview

| Method | Type | Input | Best For |
|--------|------|-------|----------|
| `nn.mlp` | MLP | pointwise coords | baselines, inverse problems |
| `nn.fno1d/2d/3d` | FNO | grid (structured) | periodic BCs, smooth solutions |
| `nn.cno2d` | CNO | grid (structured) | resolution-invariant operators |
| `nn.unet1d/2d/3d` | U-Net | grid (structured) | multi-scale, encoder-decoder |
| `nn.mgno1d/2d` | MgNO | grid | multi-resolution operator learning |
| `nn.geofno/geofno2d/3d` | GeoFNO | irregular grid | non-rectangular domains |
| `nn.pcno` | PCNO | point cloud | unstructured / scattered data |
| `nn.deeponet` | DeepONet | branch (function) + trunk (coords) | small data, operator learning |
| `nn.pointnet` | PointNet | point cloud | permutation-invariant processing |
| `nn.transformer` | Transformer | sequence | general-purpose attention |
| `nn.pit` | PiT | coords + features | distance-based attention |
| `nn.scot` | ScOT | coords + features | scalable operator transformer |
| `nn.gnot/cgptno/moegnot` | GNOT | mixed | general neural operator |
| `nn.poseidonT/B/L` | Poseidon | 128×128 grid | pretrained foundation model |
| `nn.wrap` | any Equinox/Flax | user-defined | custom architectures |

---

## Common Model API

Every model returned by a factory method supports:

```python
model.optimizer(opt_fn, lr=schedule)   # attach optimizer (required before solve)
model.freeze()                          # freeze all parameters
model.mask(param_mask)                  # freeze/unfreeze individual parameters
model.lora(rank=4, alpha=1.0)          # LoRA fine-tuning
model.initialize(path_or_pytree)       # load pretrained weights
model.dont_show()                      # suppress model summary during compile
```

Methods return `self` for chaining:
```python
model.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-4))
```

---

## MLP (Multi-Layer Perceptron)

A fully-connected feedforward network for point-wise mappings.

```python
u_net = jnn.nn.mlp(
    in_features=2,                      # number of input features (last axis)
    output_dim=1,                       # number of output features
    activation=jnp.tanh,               # hidden activation: relu, gelu, selu, sin, ...
    hidden_dims=64,                     # int or list of ints per layer
    num_layers=3,                       # number of hidden layers
    output_activation=None,            # optional output activation
    use_bias=True,
    dropout_rate=0.0,
    layer_norm=False,
    batch_norm=False,
    key=jax.random.PRNGKey(0),
)
```

**Usage:**
```python
u = u_net(x, y)        # inputs are concatenated along the feature axis
u = u_net(x, y, k)    # additional parameters are also concatenated
```

---

## Fourier Neural Operator (FNO)

FNO learns operators in Fourier space. Particularly effective for problems with smooth solutions or periodic boundary conditions.

### FNO 1D
```python
model = jnn.nn.fno1d(
    in_features=1,
    hidden_channels=64,
    n_modes=16,            # number of Fourier modes to retain
    d_vars=1,              # output channels
    n_layers=4,
    n_steps=1,             # output time steps (autoregressive rollout)
    activation=jax.nn.gelu,
    linear_conv=True,      # False for periodic BCs
    key=key,
)
```

### FNO 2D
```python
model = jnn.nn.fno2d(
    in_features=1,
    hidden_channels=32,
    n_modes=12,
    d_vars=1,
    n_layers=4,
    d_model=(64, 64),      # spatial resolution (for positional encoding)
    use_positions=False,   # prepend coordinate grid to input
    key=key,
)
```

### FNO 3D
```python
model = jnn.nn.fno3d(
    in_features=1,
    hidden_channels=24,
    n_modes=8,
    d_vars=1,
    d_model=(32, 32, 32),
    key=key,
)
```

---

## Continuous Neural Operator (CNO 2D)

U-Net style architecture with bicubic-interpolation continuous activations for resolution-invariant learning.

```python
model = jnn.nn.cno2d(
    in_dim=1,
    out_dim=1,
    size=64,               # spatial size (must be divisible by 2^N_layers)
    N_layers=3,
    N_res=4,               # residual blocks per encoder level
    N_res_neck=4,          # residual blocks in bottleneck
    channel_multiplier=16,
    use_bn=True,
    key=key,
)
```

---

## U-Net

Encoder-decoder with skip connections. Suitable for general PDE operator learning on regular grids.

```python
model = jnn.nn.unet1d(in_dim=1, out_dim=1, hidden_channels=32, n_levels=3, key=key)
model = jnn.nn.unet2d(in_dim=1, out_dim=1, hidden_channels=32, n_levels=3, key=key)
model = jnn.nn.unet3d(in_dim=1, out_dim=1, hidden_channels=16, n_levels=3, key=key)
```

---

## Multigrid Neural Operator (MgNO)

Inspired by algebraic multigrid solvers; efficient for problems with multi-scale features.

```python
model = jnn.nn.mgno1d(in_dim=1, out_dim=1, key=key)
model = jnn.nn.mgno2d(in_dim=1, out_dim=1, key=key)
```

---

## Geometry-Informed FNO (GeoFNO)

FNO adapted for irregular / non-rectangular domains via learned geometric mappings.

```python
from jno.architectures.geofno import compute_Fourier_modes

model = jnn.nn.geofno2d(
    nks=(16, 16),    # Fourier modes per dimension
    Ls=(1.0, 1.0),   # domain physical extents
    in_dim=3,        # input channels
    out_dim=1,
    key=key,
)
```

---

## Point Cloud Neural Operator (PCNO)

Handles unstructured / scattered point clouds using graph convolutions.

```python
model = jnn.nn.pcno(
    in_dim=3,
    out_dim=1,
    hidden_channels=64,
    key=key,
)
```

---

## DeepONet

Separate branch network (encodes input function) and trunk network (encodes query coordinates). Especially effective in the small-data regime.

```python
model = jnn.nn.deeponet(
    n_sensors=10,         # number of sensor points for branch input
    sensor_channels=1,    # channels per sensor measurement
    coord_dim=2,          # spatial dimension for trunk network
    basis_functions=64,   # latent space dimension
    hidden_dim=64,
    n_layers=3,
    key=key,
)

# Usage: u(sensor_data, spatial_coords)
u = model(t, jnn.concat([x, y]))
```

---

## PointNet

Permutation-invariant processing of unordered point clouds.

```python
model = jnn.nn.pointnet(in_dim=3, out_dim=1, key=key)
```

---

## Transformer

Standard multi-head self-attention transformer.

```python
model = jnn.nn.transformer(
    d_model=64,
    n_heads=4,
    n_layers=3,
    dropout_rate=0.0,
    key=key,
)
```

---

## Position-Induced Transformer (PiT)

Attention mechanism where weights depend on pairwise distances between query points. Well-suited for irregular meshes.

```python
model = jnn.nn.pit(in_dim=2, out_dim=1, key=key)
```

---

## Scalable Operator Transformer (ScOT)

Swin-Transformer-based architecture; efficient attention for large resolution inputs.

```python
model = jnn.nn.scot(in_dim=2, out_dim=1, key=key)
```

---

## GNOT (General Neural Operator Transformer)

```python
model = jnn.nn.gnot(...)
model = jnn.nn.cgptno(...)    # Conditional GNOT
model = jnn.nn.moegnot(...)   # Mixture-of-Experts variant
```

---

## Poseidon Foundation Models

Pretrained ScOT models trained on a broad distribution of fluid dynamics PDEs. Suitable for fine-tuning with minimal data.

```python
model = jnn.nn.poseidonT(key=key)   # Tiny  (~4 M params)
model = jnn.nn.poseidonB(key=key)   # Base  (~25 M params)
model = jnn.nn.poseidonL(key=key)   # Large (~80 M params)
```

Expected input: structured 128×128 grids (use `jno.domain.poseidon()`).

---

## Wrapping Custom Models

### Equinox Module

```python
import equinox as eqx
import jno.numpy as jnn

class MyNet(eqx.Module):
    fc: eqx.nn.Linear
    def __init__(self, *, key):
        self.fc = eqx.nn.Linear(2, 1, key=key)
    def __call__(self, x, y):
        return self.fc(jnp.concat([x, y], axis=-1))

my_net = jnn.nn.wrap(MyNet(key=jax.random.PRNGKey(0)))
u = my_net(x, y)   # returns Placeholder for use in constraints
```

### Flax Linen Module

```python
from flax import linen as nn

class FlaxMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        return nn.Dense(1)(x)

my_net = jnn.nn.flaxwrap(FlaxMLP(), input=(dummy_x,), key=key)
```

### Flax NNX Module (auto-detected)

```python
from flax import nnx

class NNXModel(nnx.Module):
    def __init__(self, rngs):
        self.dense = nnx.Linear(2, 1, rngs=rngs)
    def __call__(self, x):
        return self.dense(x)

my_net = jnn.nn.wrap(NNXModel(nnx.Rngs(0)))  # auto-detected as NNX
```

---

## Architecture Search Wrapper

When using the hyperparameter tuner, pass a **class** (not instance) plus an `ArchSpace`:

```python
a_space = jnn.tune.space()
a_space.unique("act", [jnp.tanh, jax.nn.gelu, jnp.sin], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3], category="architecture")

class TunableMLP(eqx.Module):
    def __init__(self, arch: jnn.tune.Arch, *, key):
        hidden = arch("hid")
        depth  = arch("dep")
        self.act = arch("act")
        # build layers...

u = jnn.nn.wrap(TunableMLP, space=a_space)(x, y)
```

---

## Trainable Parameters (Inverse Problems)

To learn scalar or tensor PDE coefficients during training:

```python
# Learnable diffusion coefficient
D = jnn.parameter((1,), key=jax.random.PRNGKey(0), name="D")
d = D()   # symbolic placeholder

# Learnable 2D tensor
K = jnn.parameter((3, 3), key=key, init=jax.nn.initializers.ones)
k = K()

D.optimizer(optax.adam, lr=lrs(1e-3))
```
