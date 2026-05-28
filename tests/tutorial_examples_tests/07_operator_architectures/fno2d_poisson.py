"""
Fourier Neural Operator 2D — Poisson Operator Learning
=======================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

FNO2D learns the solution operator via global spectral convolutions in
Fourier space, making it highly efficient for problems on regular grids.

Architecture: (B, H, W, C) → Lift → [SpectralConv2D + Conv2D → LayerNorm → GELU]×N
              → Project → (B, H, W, 1)

Reference: Li et al. "Fourier Neural Operator for Parametric PDEs" (2020)
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
    foundax.fno2d(
        in_features=1,
        hidden_channels=16,
        n_modes=6,
        d_vars=1,
        n_layers=2,
        n_steps=1,
        d_model=(GRID, GRID),
        norm="layer",
        linear_conv=True,
        key=KEY,
    )
)

# ── Constraint & solver ───────────────────────────────────────────────────────
crux = jno.core([(_u - u(_f)).mse], domain)

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
