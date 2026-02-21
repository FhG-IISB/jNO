# jNO
JAX-native building blocks for training neural operators.

## [Install](https://docs.astral.sh/uv/getting-started/installation/)
`uv` installs and manages environments in your user directory, so you can typically run everything locally **without sudo**.
Local execution policies can be overwritten in windows as follows
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
this will allow you to activate the environment.


```bash
uv sync
```

CUDA extra

```bash
uv sync --extra cuda
```


Install the following i you get an error

```bash
sudo apt-get install libglu1
```



## Architectures

The `jno.numpy.nn` factory provides a unified interface for creating neural operator models. All methods return a `FlaxModule` wrapper compatible with the jNO training pipeline.

### Quick Reference

| Use Case | Recommended Architecture |
|----------|-------------------------|
| Regular grid, periodic BC | `fno1d`, `fno2d`, `fno3d` |
| Regular grid, general BC | `unet1d`, `unet2d`, `unet3d`, `mgno1d`, `mgno2d` |
| Irregular geometry | `geofno`, `geofno2d`, `pcno`, `pit` |
| Point cloud data | `pointnet`, `pcno` |
| Super-resolution | `fno2d`, `unet2d` |
| Time-stepping | `fno2d` (with `n_steps>1`), `scot` |
| Transfer learning | `poseidonT`, `poseidonB`, `poseidonL` |
| Small data regime | `deeponet`, `pit` |
| Multiple input functions | `cgptno`, `gnot` |

---

```python
import jno.numpy as jnn
```

### Fourier Neural Operators

Spectral methods using FFT-based convolutions. Efficient for regular grids with smooth solutions.

```python
# 1D Burgers equation
model = jnn.nn.fno1d(hidden_channels=64, n_modes=16, d_vars=1, n_layers=4)

# 2D Darcy flow
model = jnn.nn.fno2d(hidden_channels=32, n_modes=12, d_vars=1)

# 3D elasticity
model = jnn.nn.fno3d(hidden_channels=24, n_modes=8, d_vars=3)
```

**Key parameters:**
- `hidden_channels`: Width of spectral layers (32–128 typical)
- `n_modes`: Fourier modes retained per dimension
- `linear_conv`: Set `True` for non-periodic boundaries

---

### Geometry-Aware Operators

For irregular domains and unstructured meshes using explicit Fourier bases.

```python
# 2D irregular domain
model = jnn.nn.geofno2d(nks=(16, 16), Ls=(1.0, 1.0), in_dim=3, out_dim=1)

# Point cloud with learnable length scales
model = jnn.nn.pcno(ndims=2, nks=[16, 16], Ls=[1.0, 1.0], in_dim=3, out_dim=1)
```

**Note:** These models require auxiliary inputs (`node_mask`, `nodes`, `node_weights`) during forward pass.

---

### U-Net Architectures

Encoder-decoder with skip connections for multi-scale feature extraction.

```python
# 2D image-to-image
model = jnn.nn.unet2d(in_channels=1, out_channels=1, depth=4, wf=6)

# 1D signal processing
model = jnn.nn.unet1d(in_channels=1, out_channels=1, depth=4)
```

**Key parameters:**
- `depth`: Number of encoder/decoder levels
- `wf`: Width factor (base channels = 2^wf)
- `up_mode`: `'upconv'` (learnable) or `'upsample'` (interpolation)

---

### Multigrid Neural Operators

V-cycle multigrid structure for efficient multi-scale processing.

```python
model = jnn.nn.mgno2d(
    input_shape=(64, 64),
    num_layer=5,
    num_channel_u=24,
    num_channel_f=3,
    output_dim=1
)
```

---

### Attention-Based Operators

#### Position-induced Transformer (PiT)

Uses distance-based attention weights for spatial problems.

```python
model = jnn.nn.pit(
    in_channels=3,
    out_channels=1,
    localities=[100, 50, 50, 50, 100],  # [encoder, *processor, decoder]
    input_res=(64, 64),
    latent_res=(16, 16)
)
```

#### General Neural Operator Transformer (GNOT)

Cross-attention with optional Mixture-of-Experts for multiple input functions.

```python
# Single input function
model = jnn.nn.cgptno(trunk_size=4, branch_sizes=[3], output_size=1, n_layers=3)

# With position-dependent MoE
model = jnn.nn.gnot(trunk_size=4, branch_sizes=[3], space_dim=2, n_experts=4)
```

---

### Branch-Trunk Networks

#### DeepONet

Decomposes operators into basis coefficients (branch) and basis functions (trunk).

```python
# Basic DeepONet
model = jnn.nn.deeponet(n_sensors=100, coord_dim=2, basis_functions=64)

# Advanced with Fourier features
model = jnn.nn.deeponet(
    branch_type='transformer',
    trunk_type='siren',
    coord_embedding='fourier',
    coord_embedding_scale=10.0
)
```

---

### Point Cloud Networks

```python
model = jnn.nn.pointnet(output_dim=3, conv_scale=1.0)
```

Permutation-invariant processing via shared MLPs and max pooling.

---

### Scalable Operator Transformer (ScOT)

Swin Transformer backbone with U-Net skip connections.

```python
model = jnn.nn.scot(
    name="my_scot",
    image_size=128,
    num_channels=3,
    num_out_channels=1,
    embed_dim=96,
    depths=(2, 2, 6, 2)
)
```

---

### Foundation Models (Pretrained)

Poseidon models pretrained on diverse PDE datasets.

```python
# Tiny (~21M params)
model = jnn.nn.poseidonT("weights/poseidon_tiny.msgpack")

# Base (~120M params)
model = jnn.nn.poseidonB("weights/poseidon_base.msgpack")

# Large (~600M params)
model = jnn.nn.poseidonL("weights/poseidon_large.msgpack")
```

**Note:** Poseidon expects 4-channel input. For different configurations, use `nn.scot()` directly.

---

### Custom Modules

Wrap any Flax linen or nnx module for pipeline integration:

```python
import flax.linen as nn
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(2):
            h = nn.tanh(nn.Dense(64)(h))
        return nn.Dense(1)(h)

model = jnn.nn.wrap(MLP())
```


```python
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(in_features=2, out_features=64, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=64, out_features=64, rngs=rngs)
        self.dense3 = nnx.Linear(in_features=64, out_features=1, rngs=rngs)

    def __call__(self, x, y):
        h = jnp.concat([x, y], axis=-1)
        h = jnp.tanh(self.dense1(h))
        h = jnp.tanh(self.dense2(h))
        u = self.dense3(h)
        return u


model = jnn.nn.wrap(MLP(nnx.Rngs(0)))
```


### Kolmogorov Arnold Networks (KAN)

The following repository can be used for KAN networks -> [jaxKAN](https://github.com/srigas/jaxKAN)

Example:

We have to make sure that the input to the KAN is correct. In this instance only one jax array is expected, thus we wrap in a **nnx** class.

```python
from flax import nnx
from jaxkan.models.KAN import KAN
class _KAN(nnx.Module):
    def __init__(self):
        self.KAN = KAN(layer_dims=layer_dims, 
        layer_type="chebyshev", 
        required_parameters=req_params, 
        seed=42,)
    def __call__(self, x, y):
        return self.KAN(jnp.concat([x, y], axis=-1))

u = jnn.nn.wrap(_KAN())(x, y)
```



## Architecture Search

```python
# Architecture space for the model
a_space = jnn.tune.space()
a_space.unique("act", [nn.tanh, nn.selu, nn.gelu, jnp.sin], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")


class MLP(nn.Module):
    arch: jnn.tune.Arch

    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(self.arch("dep")):
            h = self.arch("act")(nn.Dense(self.arch("hid"))(h))
        return nn.Dense(1)(h)

u = jnn.nn.wrap(MLP, space=a_space) #<- Wrap class not instance

# Later on call .sweep not .solve
```



## Dependencies (thank you!)

This project stands on the shoulders of some fantastic open-source JAX ecosystem libraries. Huge thanks to the maintainers and contributors of:

- The Backbone -> [JAX](https://github.com/jax-ml/jax)
- The Optimizers -> [Optax](https://github.com/google-deepmind/optax)
- Specific optimizer package for -> [SOAP (JAX)](https://github.com/haydn-jones/SOAP_JAX)
- Neural Network Library -> [Flax](https://github.com/google/flax)
- Logging Utility -> [lox](https://github.com/huterguier/lox)
- Progress Bar -> [jax-tqdm](https://github.com/jeremiecoullon/jax-tqdm)
