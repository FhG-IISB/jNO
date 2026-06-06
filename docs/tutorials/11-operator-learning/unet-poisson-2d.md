# U-Net 2D — supervised Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/11_operator_learning/unet_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/11-operator-learning/">Back to chapter</a>
</div>

Same problem and dataset as the [FNO tutorial](fno-poisson-2d.md), different architecture: a convolutional encoder-decoder with skip connections. **Data-driven** supervised learning, just like FNO.

## Problem Setup

Solve `−∇²u = f` on `[0,1]²` with `u = 0` on the boundary, learning the operator `f → u` from precomputed `(f, u)` pairs.

## Step 1: Generate the Dataset

```python
from create_domain import build_domain_from_arrays, generate_poisson_data

GRID, SAMPLES = 16, 20
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")
_u = domain.variable("_u")
```

Identical to the FNO setup — comparing the two architectures cleanly requires the same data.

## Step 2: U-Net Architecture

```python
u = jno.nn.wrap(
    foundax.unet2d(
        in_channels=1,
        out_channels=1,
        depth=2,                  # number of down-sampling stages
        wf=4,                     # width factor: hidden channels = 2^wf
        norm="layer",
        up_mode="upconv",
        padding_mode="reflect",   # mirror boundaries — better than zero-padding for Dirichlet
        key=jax.random.PRNGKey(0),
    )
)
u.optimizer(optax.chain(
    optax.clip_by_global_norm(1e-3),
    optax.adamw(optax.cosine_decay_schedule(5e-4, 50, alpha=1e-7 / 5e-4), weight_decay=1e-6),
))
```

Each encoder stage downsamples and doubles the channel count; each decoder stage upsamples and concatenates the corresponding encoder feature map (the "skip connection"). The output has the same spatial resolution as the input, with `out_channels=1` for a scalar field.

## Step 3: Supervised Loss + Solve

```python
crux = jno.core([(_u - u(_f[0, ...])).mse], domain)
crux.solve(epochs=50, batchsize=10)
```

`_f[0, ...]` drops a leading singleton axis because U-Net expects `(B, H, W, C)` — minor shape plumbing that's worth knowing about when porting between operator architectures.

## What To Notice

- **Multiscale by construction.** The encoder-decoder topology captures features at multiple resolutions, which helps when the solution operator combines local detail (sharp features in `f`) with global structure (Poisson's elliptic smoothing).
- **`padding_mode="reflect"`** is preferred over zero-padding (the default in many CNN libraries) for non-periodic Dirichlet BCs — mirroring keeps the boundary values consistent with the underlying physics. For periodic problems, use `padding_mode="circular"`.
- **Compared to [FNO](fno-poisson-2d.md):** U-Net is convolutional and local; FNO is spectral and global. FNO converges faster on smooth elliptic problems; U-Net generalises better when the solution operator has sharp features or when the input resolution differs from training.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/11_operator_learning/unet_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/11-operator-learning/">Back to 11 Operator Learning</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/11_operator_learning/unet_poisson_2d.py"
```
