"""
Fourier Neural Operator 2D — Poisson Operator Learning
=======================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

FNO2D learns the solution operator via global spectral convolutions in
Fourier space, making it highly efficient for problems on regular grids.

Run first:
    python create_domain.py

Then:
    python fno2d_poisson.py

Architecture: (B, H, W, C) → Lift → [SpectralConv2D + Conv2D → LayerNorm → GELU]×N
              → Project → (B, H, W, 1)

Reference: Li et al. "Fourier Neural Operator for Parametric PDEs" (2020)
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
u = jno.np.nn.fno2d(
    in_features=1,
    hidden_channels=48,
    n_modes=24,
    d_vars=1,
    n_layers=4,
    n_steps=1,
    d_model=(GRID, GRID),
    norm="layer",
    linear_conv=True,  # non-periodic → suitable for Dirichlet BC
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
    batchsize=BATCH,
    checkpoint_gradients=False,
    offload_data=False,
)
