# Foundax Core Architectures

The core neural-operator architecture surface has moved to Foundax.

Use Foundax constructors directly via `foundax` and wrap them in jNO only when you want jNO training/runtime features.

## Architecture Overview

| Family | Constructors | Typical Input | Best For |
|---|---|---|---|
| Linear / MLP | `fx.linear`, `fx.mlp` | pointwise features | baselines, inverse problems |
| FNO | `fx.fno1d`, `fx.fno2d`, `fx.fno3d` | structured grids | smooth operators, periodic-like dynamics |
| CNO | `fx.cno2d` | structured 2-D grids | resolution-aware convolutional operator learning |
| U-Net | `fx.unet1d`, `fx.unet2d`, `fx.unet3d` | structured grids | encoder-decoder multi-scale mappings |
| MgNO | `fx.mgno1d`, `fx.mgno2d` | structured grids | multigrid-inspired operator learning |
| DeepONet | `fx.deeponet` | branch function + query coords | low-data operator learning |
| Transformer | `fx.transformer` | sequence tokens | generic attention baseline |
| Geometry-aware | `fx.geofno`, `fx.pcno`, `fx.pit`, `fx.pointnet` | irregular meshes / points | geometry- or point-cloud-based tasks |
| GNOT family | `fx.cgptno`, `fx.gnot`, `fx.moegptno` | mixed branch/trunk inputs | transformer-style neural operators |

## Common Usage Pattern

```python
import foundax as fx

# Standalone Foundax model
model = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
```

Use inside jNO:

```python
import foundax as fx
import jno
import optax

raw_model = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
u = jno.nn.wrap(raw_model)
u.optimizer(optax.adam, lr=1e-3)
```

After wrapping with `jno.nn.wrap(...)`, jNO model controls (for example optimizer attachment, freezing, masks, and LoRA) are available through the wrapped jNO model object.

## MLP And Linear

```python
import foundax as fx
import jax
import jax.numpy as jnp

linear = fx.linear(in_features=2, out_features=1)
mlp = fx.mlp(
    in_features=2,
    output_dim=1,
    hidden_dims=64,
    num_layers=3,
    activation=jnp.tanh,
    key=jax.random.PRNGKey(0),
)
```

Use these for coordinate-based baselines and lightweight inverse-problem setups.

## Fourier Neural Operators

```python
import foundax as fx

fno_1d = fx.fno1d(in_features=1, hidden_channels=64, n_modes=16)
fno_2d = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
fno_3d = fx.fno3d(in_features=1, hidden_channels=24, n_modes=8)
```

Use for structured-grid operator learning where spectral mixing is a strong inductive bias.

## Continuous Neural Operator (CNO)

```python
import foundax as fx

cno = fx.cno2d(in_dim=1, out_dim=1, size=64, N_layers=3)
```

Use for 2-D continuous neural operator workflows on regular fields.

## U-Net Family

```python
import foundax as fx

unet_1d = fx.unet1d(in_channels=1, out_channels=1, depth=4)
unet_2d = fx.unet2d(in_channels=1, out_channels=1, depth=4)
unet_3d = fx.unet3d(in_channels=1, out_channels=1, depth=4)
```

Use when local multiscale encoder-decoder behavior is preferred over spectral operators.

## Multigrid Neural Operator (MgNO)

```python
import foundax as fx

mgno_1d = fx.mgno1d(input_length=256, num_channel_f=1, output_dim=1)
mgno_2d = fx.mgno2d(input_shape=(64, 64), num_channel_f=1, output_dim=1)
```

Use for multiresolution solver-inspired architectures.

## DeepONet

```python
import foundax as fx

deeponet = fx.deeponet(
    n_sensors=100,
    sensor_channels=1,
    coord_dim=2,
    basis_functions=128,
    hidden_dim=256,
    n_layers=4,
)
```

Use for branch/trunk operator decomposition and coordinate-query prediction.

## Geometry-Aware Architectures

```python
import foundax as fx

geofno = fx.geofno(ndims=2, nks=(64, 64), Ls=(1.0, 1.0), in_dim=3, out_dim=1)
pcno = fx.pcno(ndims=2, nks=(64, 64), Ls=(1.0, 1.0), in_dim=3, out_dim=1)
pit = fx.pit(in_channels=1, out_channels=1, input_res=(64, 64), latent_res=(16, 16), output_res=(64, 64))
pointnet = fx.pointnet(in_features=3, output_dim=1)
```

Use this group for irregular geometries, point clouds, and coordinate-aware attention models.

## Transformer

```python
import foundax as fx

transformer = fx.transformer(
    num_layers=6,
    embed_dim=512,
    num_heads=8,
    mlp_features=2048,
)
```

Generic sequence transformer baseline for tokenized data.

## GNOT Family

```python
import foundax as fx

cgptno = fx.cgptno(trunk_size=2, branch_sizes=[3], output_size=1)
gnot = fx.gnot(trunk_size=2, branch_sizes=[3], output_size=1)
moegptno = fx.moegptno(trunk_size=2, branch_size=3, output_size=1)
```

Use for transformer-based neural operators with branch/trunk style coupling.

## Foundation Wrappers

For pretrained and large-model wrappers such as Poseidon, Morph, MPP, Walrus, BCAT, PDEformer2, DPOT, and PROSE, see [Foundax Overview](Foundation-Models.md).
