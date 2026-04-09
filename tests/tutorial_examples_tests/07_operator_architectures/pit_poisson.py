"""
Position-induced Transformer (PiT) — Poisson Operator Learning
===============================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

PiT replaces the standard softmax(QK^T) attention with distance-based
attention weights, providing built-in spatial inductive bias.  Input is
flattened to a sequence of point features.

Run first:
    python create_domain.py

Then:
    python pit_poisson.py

Architecture: (N, C) sequence → Encoder → Latent → Processor blocks → Decoder → (N, C_out)
              Attention weight A_ij = f(dist(pos_i, pos_j))

Reference: Zhao et al. "Position-induced Transformer" (2023)
"""

import jax
import jno

import optax

KEY = jax.random.PRNGKey(0)
SAMPLES = 200
GRID = 64
N = GRID * GRID  # number of points (flattened grid)
EPOCHS = 500
BATCH = 40

# ── Load domain ───────────────────────────────────────────────────────────────
domain = jno.load(f"domain_{SAMPLES}_{GRID}.pkl")
_f = domain.variable("_f")  # (S, 1, 1, H, W, 1)
_u = domain.variable("_u")  # (S, 1, 1, H, W, 1)

# PiT operates on flattened point sequences: (N, C) without explicit batch.
# Reshape source/target: (1, 1, H, W, 1) → (N, 1) and (1, N, 1) respectively.
_f_flat = _f[0, ...].reshape((N, 1))  # (N, 1)  — traced per sample
_u_flat = _u.reshape((1, N, 1))  # (1, N, 1) — full batch column

# ── Model ─────────────────────────────────────────────────────────────────────
# latent_res is the resolution of the internal coarsened representation.
u = jno.np.nn.pit(
    in_channels=1,
    out_channels=1,
    hid_channels=128,
    n_head=2,
    localities=[100, 50, 50, 50, 100],  # global encoder/decoder, local processor
    input_res=(GRID, GRID),
    latent_res=(GRID // 4, GRID // 4),  # 16 × 16 latent grid
    output_res=(GRID, GRID),
    key=KEY,
)

# ── Constraint & solver ───────────────────────────────────────────────────────
crux = jno.core([(_u_flat - u(_f_flat)).mse], domain)

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
