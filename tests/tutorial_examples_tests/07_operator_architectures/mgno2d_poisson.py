"""
Multigrid Neural Operator 2D — Poisson Operator Learning
=========================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

MgNO implements multigrid V-cycles as neural network layers, providing
efficient multi-scale representations at different grid resolutions.

Run first:
    python create_domain.py

Then:
    python mgno2d_poisson.py

Architecture: Input → [MgConv V-cycle + skip]×num_layer → 1×1-Conv → Output
              Each V-cycle: smooth → restrict → coarser solve → prolongate → smooth
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
# input_shape spatial dims must be divisible by 2^(num_levels-1)
# with num_iteration=[[1,1]]*5, num_levels=5, requires GRID divisible by 16
u = jno.np.nn.mgno2d(
    input_shape=(GRID, GRID),
    num_layer=5,
    num_channel_u=24,
    num_channel_f=1,  # single input channel = source term
    num_iteration=[[1, 1]] * 5,
    output_dim=1,
    activation="gelu",
    padding_mode="SAME",  # SAME padding for Dirichlet BC (non-periodic)
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
