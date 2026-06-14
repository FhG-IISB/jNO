"""11 — Physics-informed operator learning with FieldView (2-D Poisson)

A Fourier Neural Operator (FNO) learns the solution operator  f ↦ u  of

    −Δu = f      on  Ω = (0, 1)²,      u = 0  on  ∂Ω,

mapping a whole forcing field to the whole solution field in one shot. Because
the operator emits a grid, the coordinates x, y are **not** inputs to the
network — coordinate-based autodiff of the output is identically zero, so PDE
derivatives have to come from finite differences on the grid. That is exactly
what FieldView provides:

    u = net(f).field.bind(x=x, y=y)     # FD derivatives of the operator's grid
    pde = u.xx + u.yy + f               # −Δu = f  ⇒  u_xx + u_yy + f = 0

The FD residual is fully differentiable, so adding it to the loss makes
``crux.solve`` back-propagate through it into the FNO weights (a forward-only
audit would not). A purely data-driven operator fits the solution but quietly
violates the PDE (large residual); the FieldView physics term pulls its output
back onto the physics while the supervised term keeps it accurate.

Contrast with ``deeponet_poisson_2d`` / ``fno_poisson_2d``: those learn the same
operator from a PDE residual (AD, coordinate inputs) or from data alone. This
script combines data with an FD residual on the operator's own grid output.
"""

import foundax
import jax
import numpy as np
import optax
from create_domain import build_domain_from_arrays, generate_poisson_data

import jno

KEY = jax.random.PRNGKey(0)
GRID = 16  # solution grid resolution (raise for higher fidelity)
SAMPLES = 24  # forcing/solution pairs
EPOCHS = 600
BATCH = 8
PHYS_W = 0.02  # weight on the FieldView FD residual; the data term dominates

# ── Operator-learning dataset: forcing fields and their Poisson solutions ─────
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")  # operator input  (forcing)
_u = domain.variable("_u")  # supervised target (analytic solution)
x, y, _ = domain.variable("interior")

# ── Fourier Neural Operator  f ↦ u ────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.fno2d(
        in_features=1,
        hidden_channels=24,
        n_modes=10,
        d_vars=1,
        n_layers=4,
        d_model=(GRID, GRID),
        key=KEY,
    )
)
net.optimizer(optax.adamw(optax.cosine_decay_schedule(2e-3, EPOCHS, alpha=1e-3), weight_decay=1e-6))

# ── Data term + FieldView FD physics term ─────────────────────────────────────
pred = net(_f)  # live operator output (a grid field)
u = pred.field.bind(x=x, y=y)  # FD view: x, y are grid axes, not NN inputs
data = pred - _u  # supervised term
physics = PHYS_W * (u.xx + u.yy + _f)  # FD Poisson residual (gradient-carrying)

crux = jno.core([data.mse, physics.mse], domain=domain)
crux.solve(epochs=EPOCHS, batchsize=BATCH)

# ── Score: data accuracy and physics consistency of the trained operator ──────
p, e = crux.eval([pred, _u])
p, e = np.asarray(p), np.asarray(e)
rel_l2 = float(np.linalg.norm(p - e) / (np.linalg.norm(e) + 1e-8))
res_mse = float(np.mean(np.asarray(crux.eval((u.xx + u.yy + _f).mse))))

print("Physics-informed FNO operator  (f ↦ u  for  −Δu = f)")
print(f"  Relative L2 vs analytic solution : {rel_l2:.3e}")
print(f"  FieldView FD PDE residual (MSE)  : {res_mse:.3e}")

# ── Tolerance checks ──────────────────────────────────────────────────────────
# The data term anchors accuracy (reliable, grid-monotonic); the FieldView FD
# residual is the physics term. Bounds are loose guards calibrated on CPU at this
# scale — final accuracy/timing at larger GRID should be confirmed on GPU.
assert np.isfinite(rel_l2) and rel_l2 < 0.25, f"operator failed to learn (rel_l2={rel_l2:.3e})"
assert np.isfinite(res_mse) and res_mse < 0.15, f"FD physics residual too large (mse={res_mse:.3e})"
