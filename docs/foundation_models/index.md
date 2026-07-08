# Models

jNO ships **no model zoo of its own**. Every model — from a simple MLP to a Fourier Neural Operator
to a large pretrained foundation model — comes from [**foundax**](https://fhg-iisb.github.io/foundax/),
jNO's model library, and is wrapped with `jno.nn(...)` to gain jNO's training controls (optimizer,
schedules, freeze / mask / LoRA, dtype, diagnostics).

**Full architecture reference, constructor signatures, and pretrained checkpoints live in the foundax
docs:** **[https://fhg-iisb.github.io/foundax/](https://fhg-iisb.github.io/foundax/)**

---

## Wrapping a model

`jno.nn` accepts any [Equinox](https://docs.kidger.site/equinox/) module. foundax models are Equinox modules, but you can wrap your own `eqx.Module` in exactly the same way — see [Model Controls](../model-controls/index.md#quick-start) for a custom model example.

```python
import foundax
import jno
import optax

net = jno.nn(foundax.mlp(in_features=2, hidden_dims=64, num_layers=4,
                               key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(1e-3))
```

Once wrapped, all jNO model controls are available: freeze, masks, LoRA, dtype conversion, and diagnostics. See [Model Controls](../model-controls/index.md).

---

## Available architecture families

| Family | foundax constructors |
|--------|----------------------|
| Linear / MLP | `foundax.linear`, `foundax.mlp` |
| DeepONet | `foundax.deeponet` |
| FNO | `foundax.fno1d`, `foundax.fno2d`, `foundax.fno3d` |
| CNO | `foundax.cno2d` |
| U-Net | `foundax.unet1d`, `foundax.unet2d`, `foundax.unet3d` |
| MgNO | `foundax.mgno1d`, `foundax.mgno2d` |
| Geometry-aware | `foundax.geofno`, `foundax.pcno`, `foundax.pit`, `foundax.pointnet` |
| GNOT family | `foundax.cgptno`, `foundax.gnot`, `foundax.moegptno` |
| Transformer | `foundax.transformer` |
| Foundation models | `foundax.poseidon`, `foundax.morph`, `foundax.mpp`, `foundax.walrus`, `foundax.dpot`, `foundax.prose`, … |

For constructor signatures, hyperparameters, and pretrained checkpoints see the [foundax docs](https://fhg-iisb.github.io/foundax/).

---

## Structured-grid domains

For Poseidon-style structured 2D workflows, use the matching domain constructor:

```python
domain = jno.domain.poseidon(nx=128, ny=128)
```
