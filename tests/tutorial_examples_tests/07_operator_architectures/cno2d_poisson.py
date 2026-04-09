"""
Continuous Neural Operator 2D — Poisson Operator Learning
==========================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

CNO is a U-Net style architecture that applies activations at 2× resolution
then downsamples (bicubic interpolation), providing continuous and
resolution-invariant operator learning.

Run first:
    python create_domain.py

Then:
    python cno2d_poisson.py

Architecture: (B, H, W, C) → Lift → [Encoder + ResNet]×N → Bottleneck
              → [Decoder + Skip]×N → Project → (B, H, W, 1)

Reference: Raonić et al. "Convolutional Neural Operators" (2023)
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
# size must match the spatial resolution (GRID)
u = jno.np.nn.cno2d(
    in_dim=1,
    out_dim=1,
    size=GRID,
    N_layers=3,
    N_res=4,
    N_res_neck=4,
    channel_multiplier=16,
    use_bn=True,
    key=KEY,
)

# ── Constraint & solver ───────────────────────────────────────────────────────
crux = jno.core([(_u - u(_f)).mse], domain)

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
