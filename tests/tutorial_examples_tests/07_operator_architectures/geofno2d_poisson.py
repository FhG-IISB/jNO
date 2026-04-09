"""
Geometry-aware FNO 2D (GeoFNO) — Poisson Operator Learning
===========================================================

Problem: -∇²u = f  on [0,1]²,  u=0 on boundary (Dirichlet BCs).
Operator: f → u via unstructured-mesh spectral convolutions

GeoFNO extends FNO to work on unstructured meshes by computing global
Fourier integrals via quadrature (node weights) rather than FFT.
All grid points are treated as an unordered point cloud.

Run first:
    python create_domain.py

Then:
    python geofno2d_poisson.py

Architecture: (N, in_dim) + geometry → Lift → [SpectralConvGeo + skip]×L
              → Project → (N, out_dim)

Reference: Li et al. "Geometry-Informed Neural Operator" (2023)
"""

import jax
import jax.numpy as jnp
import jno

import numpy as np
import optax

KEY = jax.random.PRNGKey(0)
SAMPLES = 200
GRID = 64
N = GRID * GRID  # total nodes
EPOCHS = 500
BATCH = 40

# ── Load domain ───────────────────────────────────────────────────────────────
domain = jno.load(f"domain_{SAMPLES}_{GRID}.pkl")
_f = domain.variable("_f")  # (S, 1, 1, H, W, 1)
_u = domain.variable("_u")  # (S, 1, 1, H, W, 1)

# ── Build fixed geometry arrays (constant for all samples) ───────────────────
xs = np.linspace(0.0, 1.0, GRID, dtype=np.float32)
ys = np.linspace(0.0, 1.0, GRID, dtype=np.float32)
XG, YG = np.meshgrid(xs, ys, indexing="ij")

nodes = jnp.array(np.stack([XG.ravel(), YG.ravel()], axis=-1))  # (N, 2)
node_mask = jnp.ones((N, 1), dtype=np.float32)  # (N, 1) all valid
node_weights = jnp.ones((N, 1), dtype=np.float32) / N  # (N, 1) uniform quadrature

# Per-sample input: concat(f, x, y) → (N, 3)
_f_flat = _f[0, ...].reshape((N, 1))  # (N, 1)  — per sample
_x_flat = jnp.array(XG.ravel()[:, None])  # (N, 1)  — constant
_y_flat = jnp.array(YG.ravel()[:, None])  # (N, 1)  — constant
_f_nodes = jno.np.concat([_f_flat, _x_flat, _y_flat], axis=-1)  # (N, 3)  — per sample (f,x,y)

_u_flat = _u.reshape((1, N, 1))  # (1, N, 1) target

# ── Model ─────────────────────────────────────────────────────────────────────
# nks: Fourier modes per spatial dimension
# in_dim=3: input = (f, x, y) at each node
u = jno.np.nn.geofno2d(
    nks=(8, 8),
    Ls=(1.0, 1.0),
    layers=(64, 64, 64, 64),
    fc_dim=128,
    in_dim=3,
    out_dim=1,
    key=KEY,
)

# ── Constraint & solver ───────────────────────────────────────────────────────
# u(x, node_mask, nodes, node_weights) → (N, 1)
crux = jno.core([(_u_flat - u(_f_nodes, node_mask, nodes, node_weights)).mse], domain)

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
