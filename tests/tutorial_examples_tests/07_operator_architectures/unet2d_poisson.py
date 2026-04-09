"""
U-Net 2D — Poisson Operator Learning
=====================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

U-Net is an encoder-decoder architecture with skip connections, naturally
capturing multi-scale features.  `padding_mode="reflect"` is preferred
over circular padding for non-periodic boundary conditions.

Run first:
    python create_domain.py

Then:
    python unet2d_poisson.py

Architecture: (B, H, W, C) → Encoder (skip connections) → Decoder → (B, H, W, 1)
"""

import jax
import jno

import optax

KEY = jax.random.PRNGKey(0)
SAMPLES = 200
GRID = 64
EPOCHS = 500
BATCH = 40

# ── Load domain ───────────────────────────────────────────────────────────────
domain = jno.load(f"domain_{SAMPLES}_{GRID}.pkl")
_f = domain.variable("_f")  # (S, 1, 1, H, W, 1)
_u = domain.variable("_u")  # (S, 1, 1, H, W, 1)

# ── Model ─────────────────────────────────────────────────────────────────────
# UNet2D: index with [0, ...] to drop the outer sample wrapper dimension;
# the model receives (1, 1, H, W, 1) which it normalizes internally.
u = jno.np.nn.unet2d(
    in_channels=1,
    out_channels=1,
    depth=4,
    wf=6,  # base channels = 2^6 = 64
    norm="layer",
    up_mode="upconv",
    padding_mode="reflect",  # non-periodic BCs
    key=KEY,
)

# ── Constraint & solver ───────────────────────────────────────────────────────
crux = jno.core([(_u - u(_f[0, ...])).mse], domain)

u.optimizer(
    optax.chain(optax.clip_by_global_norm(1e-3), optax.adamw(1.0, weight_decay=1e-6)),
    lr=jno.schedule.learning_rate.cosine(EPOCHS, 5e-4, 1e-7),
)

crux.solve(
    epochs=EPOCHS,
    constraint_weights=jno.schedule.constraint([1]),
    batchsize=BATCH,
    checkpoint_gradients=False,
    offload_data=False,
)
