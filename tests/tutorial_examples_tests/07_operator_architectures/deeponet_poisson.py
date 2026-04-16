"""
Deep Operator Network (DeepONet) — Poisson Operator Learning
=============================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u

DeepONet factorises the operator into:
  - Branch network: encodes the input function f sampled at sensor points
  - Trunk network: encodes query coordinates y
  - Output: G(f)(y) = Σᵢ bᵢ(f) · tᵢ(y)

Run first:
    python create_domain.py

Then:
    python deeponet_poisson.py

Reference: Lu et al. "Learning nonlinear operators via DeepONet" (2021)
"""

import jax
import jax.numpy as jnp
import jno

import foundax
import numpy as np
import optax

KEY = jax.random.PRNGKey(0)
SAMPLES = 200
GRID = 64
N = GRID * GRID  # total sensor / query points
EPOCHS = 500
BATCH = 40

# ── Load domain ───────────────────────────────────────────────────────────────
domain = jno.load(f"domain_{SAMPLES}_{GRID}.pkl")
_f = domain.variable("_f")  # (S, 1, 1, H, W, 1)
_u = domain.variable("_u")  # (S, 1, 1, H, W, 1)

# ── Build fixed query coordinates (same for all samples) ─────────────────────
xs = np.linspace(0.0, 1.0, GRID, dtype=np.float32)
ys = np.linspace(0.0, 1.0, GRID, dtype=np.float32)
XG, YG = np.meshgrid(xs, ys, indexing="ij")
coords = jnp.array(np.stack([XG.ravel(), YG.ravel()], axis=-1))  # (N, 2) — constant, shared across samples

# Reshape traced variables to flat sequences
_f_flat = _f[0, ...].reshape((N, 1))  # (N, 1)  branch input per sample
_u_flat = _u.reshape((1, N, 1))  # (1, N, 1) target per sample

# ── Model ─────────────────────────────────────────────────────────────────────
# branch: encodes f sampled at N sensor points (N, 1)
# trunk:  encodes 2D query coordinates (N, 2)
u = jno.nn.wrap(foundax.deeponet(
    branch_type="mlp",
    trunk_type="mlp",
    combination_type="dot",
    n_sensors=N,
    sensor_channels=1,
    coord_dim=2,
    n_outputs=1,
    basis_functions=64,
    hidden_dim=128,
    n_layers=4,
    key=KEY,
))

# ── Constraint & solver ───────────────────────────────────────────────────────
# u(branch_input, trunk_input) → (N, 1) predictions at query points
crux = jno.core([(_u_flat - u(_f_flat, coords)).mse], domain)

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
