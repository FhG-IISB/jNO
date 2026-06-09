"""11 — U-Net 2D for Poisson operator learning

Problem
-------
Learn the solution operator of the 2-D Poisson equation
    −∇²u = f   on [0,1]²,   u = 0 on ∂[0,1]²
from a dataset of (f, u) pairs generated analytically with a truncated
sinusoidal basis.

Architecture
------------
foundax.unet2d is a convolutional encoder-decoder with skip connections.
The encoder progressively downsamples the input through ``depth`` levels;
the decoder upsamples and merges with the corresponding encoder feature
maps via concatenation.  This makes the network naturally multiscale,
which is useful when the solution operator has both local and large-scale
structure.

``padding_mode="reflect"`` mirrors the boundary values rather than
zero-padding — preferred over circular padding for non-periodic
Dirichlet problems like this one.
"""

import foundax
import jax
import optax
from create_domain import build_domain_from_arrays, generate_poisson_data

import jno

KEY = jax.random.PRNGKey(0)
GRID = 16
SAMPLES = 20
EPOCHS = 50
BATCH = 10

# ── Dataset ──────────────────────────────────────────────────────────────────
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")
_u = domain.variable("_u")

# ── Model ────────────────────────────────────────────────────────────────────
u = jno.nn.wrap(
    foundax.unet2d(
        in_channels=1,
        out_channels=1,
        depth=2,
        wf=4,
        norm="layer",
        up_mode="upconv",
        padding_mode="reflect",
        key=KEY,
    )
)
u.optimizer(
    optax.chain(
        optax.clip_by_global_norm(1e-3),
        optax.adamw(optax.cosine_decay_schedule(5e-4, EPOCHS, alpha=1e-7 / 5e-4), weight_decay=1e-6),
    )
)

# ── Supervised loss + solve ─────────────────────────────────────────────────
# u(_f[0, ...]) drops a leading singleton axis because UNet expects (B, H, W, C).
crux = jno.core([(_u - u(_f[0, ...])).mse])
crux.solve(epochs=EPOCHS, batchsize=BATCH)
