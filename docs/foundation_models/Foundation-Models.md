# Foundax

This section is Foundax-first: [foundax](https://github.com/FhG-IISB/foundax) is the standalone model repository that provides the architecture and foundation-model APIs used by jNO.

## Foundax Docs Map

- [Core Architectures](Core-Architectures.md): non-foundation model families exposed through `foundax.nn`.
- This page: foundation wrapper families and jNO integration.

## Standalone Foundax Usage

Install Foundax directly when you want to use models outside of jNO:

```bash
uv pip install foundax
# or
pip install foundax
```

Preferred usage style (namespace-based for foundation models):

```python
import foundax as fx

# Core architecture
core_model = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)

# Foundation wrapper
foundation_model = fx.poseidon.T()
```

Foundax still provides older top-level shortcut functions (for example `fx.poseidonT()` and `fx.morph_Ti()`), but namespace-style calls are the recommended and documented API.

## Core Architectures In Foundax

These constructors are exposed from `foundax.nn` and re-exported at package level:

| Family | Constructors |
|---|---|
| Linear / MLP | `fx.linear`, `fx.mlp` |
| FNO | `fx.fno1d`, `fx.fno2d`, `fx.fno3d` |
| U-Net | `fx.unet1d`, `fx.unet2d`, `fx.unet3d` |
| DeepONet | `fx.deeponet` |
| Transformer | `fx.transformer` |
| CNO | `fx.cno2d` |
| MgNO | `fx.mgno1d`, `fx.mgno2d` |
| Geometry-aware operators | `fx.geofno`, `fx.pcno`, `fx.pit`, `fx.pointnet` |
| GNOT family | `fx.cgptno`, `fx.gnot`, `fx.moegptno` |

## Foundation Wrappers In Foundax

| Family | Namespace entry points | Variants / constructors |
|---|---|---|
| Poseidon | `fx.poseidon` | `T`, `B`, `L` |
| Morph | `fx.morph` | `Ti`, `S`, `M`, `L` |
| MPP | `fx.mpp` | `Ti`, `S`, `B`, `L` |
| Walrus | `fx.walrus` | `base` |
| PDEformer2 | `fx.pdeformer2` | `small`, `base`, `fast` |
| BCAT | `fx.bcat` | `base` |
| DPOT | `fx.dpot` | `Ti`, `S`, `M`, `L`, `H` |
| PROSE | `fx.prose` | `fd_1to1`, `fd_2to1`, `ode_2to1`, `pde_2to1` |

For full standalone details:

- [Foundax core architectures](https://github.com/FhG-IISB/foundax/blob/main/docs/core-models.md)
- [Foundax foundation wrappers](https://github.com/FhG-IISB/foundax/blob/main/docs/equinox-architectures.md)

## Using Foundax Inside jNO

Wrap any foundax model with `jno.nn.wrap(...)` before using it in jNO constraints:

```python
import foundax as fx
import jno

u_core = jno.nn.wrap(fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3))
u_fm = jno.nn.wrap(fx.poseidon.T())
```

If jNO is installed from PyPI, `foundax` is already included as a dependency.

### Grid Setup For Poseidon-like Workflows

For structured 2-D workflows, build a matching grid with:

```python
jno.domain.poseidon(nx=128, ny=128)
```

This creates the structured layout typically used with Poseidon-family training and inference in jNO. Physical groups are the same as for `equi_distant_rect`.
