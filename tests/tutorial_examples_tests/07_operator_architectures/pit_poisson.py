"""
Position-induced Transformer (PiT) — Poisson Operator Learning
===============================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

PiT replaces the standard softmax(QK^T) attention with distance-based
attention weights, providing built-in spatial inductive bias.  Input is
flattened to a sequence of point features.

Architecture: (N, C) sequence → Encoder → Latent → Processor blocks → Decoder → (N, C_out)
              Attention weight A_ij = f(dist(pos_i, pos_j))

Reference: Zhao et al. "Position-induced Transformer" (2023)
"""

import foundax
import jax
import optax
from create_domain import build_domain_from_arrays, generate_poisson_data

import jno

KEY = jax.random.PRNGKey(0)
GRID = 16
SAMPLES = 20
N = GRID * GRID  # number of points (flattened grid)
EPOCHS = 50
BATCH = 10

# ── Generate small dataset inline ─────────────────────────────────────────────
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")
_u = domain.variable("_u")

# PiT operates on flattened point sequences: (N, C) without explicit batch.
_f_flat = _f[0, ...].reshape((N, 1))
_u_flat = _u.reshape((1, N, 1))

# ── Model ─────────────────────────────────────────────────────────────────────
u = jno.nn.wrap(
    foundax.pit(
        in_channels=1,
        out_channels=1,
        hid_channels=16,
        n_head=1,
        localities=[10, 6, 6, 6, 10],
        input_res=(GRID, GRID),
        latent_res=(GRID // 4, GRID // 4),
        output_res=(GRID, GRID),
        key=KEY,
    )
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
