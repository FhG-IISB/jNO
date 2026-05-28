"""
U-Net 2D — Poisson Operator Learning
=====================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

U-Net is an encoder-decoder architecture with skip connections, naturally
capturing multi-scale features.  `padding_mode="reflect"` is preferred
over circular padding for non-periodic boundary conditions.

Architecture: (B, H, W, C) → Encoder (skip connections) → Decoder → (B, H, W, 1)
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

# ── Generate small dataset inline ─────────────────────────────────────────────
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")
_u = domain.variable("_u")

# ── Model ─────────────────────────────────────────────────────────────────────
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

# ── Constraint & solver ───────────────────────────────────────────────────────
crux = jno.core([(_u - u(_f[0, ...])).mse], domain)

u.optimizer(
    optax.chain(optax.clip_by_global_norm(1e-3), optax.adamw(1.0, weight_decay=1e-6)),
    lr=jno.schedule.learning_rate.cosine(EPOCHS, 5e-4, 1e-7),
)

crux.solve(
    epochs=EPOCHS,
    batchsize=BATCH,
    checkpoint_gradients=False,
    offload_data=False,
)
