# Architecture Guide

The `jno.numpy.nn` factory provides a unified interface for creating neural operator models. All methods return a `FlaxModule` wrapper that is compatible with the jNO training pipeline.

```python
import jno.numpy as jnn
```

## Quick Reference

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

## Fourier Neural Operators

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

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `hidden_channels` | Width of spectral layers | 32–128 |
| `n_modes` | Fourier modes retained per dimension | 8–32 |
| `linear_conv` | Set `True` for non-periodic boundaries | `False` (default) |

---

## Geometry-Aware Operators

For irregular domains and unstructured meshes using explicit Fourier bases.

### GeoFNO

```python
# 2D irregular domain
model = jnn.nn.geofno2d(nks=(16, 16), Ls=(1.0, 1.0), in_dim=3, out_dim=1)
```

### PCNO (Point Cloud Neural Operator)

```python
# Point cloud with learnable length scales
model = jnn.nn.pcno(ndims=2, nks=[16, 16], Ls=[1.0, 1.0], in_dim=3, out_dim=1)
```

> **Note:** These models require auxiliary inputs (`node_mask`, `nodes`, `node_weights`) during the forward pass.

---

## U-Net Architectures

Encoder-decoder networks with skip connections for multi-scale feature extraction.

```python
# 2D image-to-image
model = jnn.nn.unet2d(in_channels=1, out_channels=1, depth=4, wf=6)

# 1D signal processing
model = jnn.nn.unet1d(in_channels=1, out_channels=1, depth=4)

# 3D
model = jnn.nn.unet3d(in_channels=1, out_channels=1, depth=4)
```

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `depth` | Number of encoder/decoder levels | 4 |
| `wf` | Width factor (base channels = 2^wf) | 6 |
| `up_mode` | `'upconv'` (learnable) or `'upsample'` (interpolation) | `'upconv'` |

---

## Multigrid Neural Operators

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

## Attention-Based Operators

### Position-Induced Transformer (PiT)

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

### General Neural Operator Transformer (GNOT)

Cross-attention with optional Mixture-of-Experts for multiple input functions.

```python
# Single input function
model = jnn.nn.cgptno(trunk_size=4, branch_sizes=[3], output_size=1, n_layers=3)

# With position-dependent MoE
model = jnn.nn.gnot(trunk_size=4, branch_sizes=[3], space_dim=2, n_experts=4)
```

---

## Branch-Trunk Networks (DeepONet)

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

## Point Cloud Networks

Permutation-invariant processing via shared MLPs and max pooling.

```python
model = jnn.nn.pointnet(output_dim=3, conv_scale=1.0)
```

---

## Scalable Operator Transformer (ScOT)

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

## Foundation Models (Poseidon)

Pretrained models on diverse PDE datasets. Available in three sizes:

```python
# Tiny (~21M params)
model = jnn.nn.poseidonT("weights/poseidon_tiny.msgpack")

# Base (~120M params)
model = jnn.nn.poseidonB("weights/poseidon_base.msgpack")

# Large (~600M params)
model = jnn.nn.poseidonL("weights/poseidon_large.msgpack")
```

> **Note:** Poseidon expects 4-channel input. For different configurations, use `jnn.nn.scot()` directly.

---

## Simple MLP

For basic physics-informed problems, a fully-connected network is often sufficient:

```python
# Using hidden_dims as an integer (all layers same width)
model = jnn.nn.mlp(hidden_dims=64, num_layers=3)

# Using hidden_dims as a list (varying widths)
model = jnn.nn.mlp(hidden_dims=[64, 64, 32])
```

---

## See Also

- [Custom Models](Custom-Models.md) — wrapping Flax modules and KAN networks
- [Architecture Search](Architecture-Search.md) — automated tuning of architecture parameters
