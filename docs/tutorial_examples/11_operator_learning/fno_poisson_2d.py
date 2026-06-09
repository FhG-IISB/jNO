"""11 — Fourier Neural Operator 2D for Poisson operator learning

Problem
-------
Learn the solution operator of the 2-D Poisson equation
    −∇²u = f   on [0,1]²,   u = 0 on ∂[0,1]²
from a dataset of (f, u) pairs generated analytically with a truncated
sinusoidal basis.

Architecture
------------
foundax.fno2d performs global spectral convolutions: at each layer the
input is lifted to a hidden channel, transformed to Fourier space, the
low-mode coefficients are mixed via a learnable linear map, transformed
back, combined with a residual conv, and normalised.  This is well
suited to smooth solution fields with global coupling such as those of
elliptic PDEs.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", ICLR 2021.
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

# ── Dataset ──────────────────────────────────────────────────────────────────
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")
_u = domain.variable("_u")

# ── Model ────────────────────────────────────────────────────────────────────
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
u.optimizer(
    optax.chain(
        optax.clip_by_global_norm(1e-3),
        optax.adamw(optax.cosine_decay_schedule(5e-4, EPOCHS, alpha=1e-7 / 5e-4), weight_decay=1e-6),
    )
)

# ── Supervised loss + solve ─────────────────────────────────────────────────
crux = jno.core([(_u - u(_f)).mse])
crux.solve(epochs=EPOCHS, batchsize=BATCH)
