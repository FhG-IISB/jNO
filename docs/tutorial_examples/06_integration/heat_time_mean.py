"""06 — Temporal integral: time-mean heat dose (1-D heat equation)

Problem
-------
    ∂u/∂t = α ∂²u/∂x²,   x ∈ [0, 1],  t ∈ [0, T]
    u(0, t) = u(1, t) = 0          (homogeneous Dirichlet, hard-enforced)
    u(x, 0) = sin(πx)              (initial condition, hard-enforced)

Analytical solution
-------------------
    u(x, t) = exp(−α π² t) sin(πx)

Time-mean temperature ("heat dose")
-------------------------------------
    ū(x) ≡ ∫₀ᵀ u(x, t) dt = sin(πx) · (1 − exp(−α π² T)) / (α π²)

API demonstrated
----------------
    u.integrate(t)   — trapezoidal integral over the temporal axis for a
                        function u(x, t), returning the spatial field ū(x).

Two training losses
-------------------
    1. PDE residual:     ‖∂u/∂t − α ∂²u/∂x²‖²  (pointwise)
    2. Heat-dose loss:   ‖∫₀ᵀ u dt − ū_exact‖²  (global integral constraint)

The heat-dose loss uses `.integrate(t)`, which reduces the temporal axis via
a trapezoidal sum, producing a spatial field of the same shape as u(x, ·).
This is the temporal analogue of the spatial `.integrate()` shown in
`flux_conservation_2d.py`.
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno
from jno import LearningRateSchedule as lrs

π = jno.np.pi

# ── Physical parameters ───────────────────────────────────────────────────────
α = 0.1   # thermal diffusivity
T = 0.5   # final time
N_t = 10  # number of time steps (must be ≥ 2 when using .integrate(t))

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.05, time=(0.0, T, N_t))
x, t = domain.variable("interior")
domain.summary()

# ── Analytical solution and heat dose ─────────────────────────────────────────
u_exact = jno.np.exp(-α * π**2 * t) * jno.np.sin(π * x)

# ū(x) = ∫₀ᵀ e^{−αλt} sin(πx) dt = sin(πx) · (1 − e^{−αλT}) / (αλ),  λ = π²
_αλ = α * float(jnp.pi) ** 2
heat_dose_coeff = (1.0 - float(jnp.exp(-_αλ * T))) / _αλ  # ≈ 0.395
u_mean_exact = jno.np.sin(π * x) * heat_dose_coeff

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=32,
        num_layers=3,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.9, 5_000, 1e-5))

# Hard-enforce IC u(x,0) = sin(πx) and zero Dirichlet BCs u(0,t) = u(1,t) = 0.
u = jno.np.sin(π * x) + t * net(t, x) * x * (1 - x)

# ── Losses ────────────────────────────────────────────────────────────────────
# 1. Pointwise PDE residual: ∂u/∂t − α ∂²u/∂x² = 0
pde = jno.np.grad(u, t) - α * jno.np.grad(jno.np.grad(u, x), x)

# 2. Heat-dose constraint: ∫₀ᵀ u(x,t) dt = ū_exact(x)
#    .integrate(t) reduces over the temporal axis → spatial field (N_x, 1).
#    This global integral loss is the temporal analogue of flux_conservation_2d.
u_mean = u.integrate(t)
heat_dose_residual = u_mean - u_mean_exact

# ── Solve ─────────────────────────────────────────────────────────────────────
# min_consecutive=None uses all N_t time steps each iteration, giving the full
# temporal integral.  This is required for .integrate(t) (min_consecutive ≥ 2).
EPOCHS = 10_000
crux = jno.core([pde.mse, heat_dose_residual.mse], domain).print_shapes()
history = crux.solve(EPOCHS, min_consecutive=None)

# ── Evaluate ──────────────────────────────────────────────────────────────────
u_pred, u_ref = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
print(f"Pointwise rel. L2 error: {rel_l2:.4e}   (exact: u = e^(-αλt) sin(πx))")

# Evaluate the time-mean field explicitly via .integrate(t).
# Pass min_consecutive=None so the full time window is used.
u_mean_pred, u_mean_ref = crux.eval([u_mean, u_mean_exact], min_consecutive=None)
rel_l2_mean = float(
    jnp.linalg.norm(u_mean_pred - u_mean_ref) / (jnp.linalg.norm(u_mean_ref) + 1e-8)
)
print(
    f"Heat-dose    rel. L2 error: {rel_l2_mean:.4e}"
    f"   (exact: ū = sin(πx) · {heat_dose_coeff:.4f})"
)

# ── Record result ─────────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f_out:
    f_out.write(
        f"06_integration/heat_time_mean.py | epochs={EPOCHS}"
        f" | rel_L2_u={rel_l2:.6e} | rel_L2_mean={rel_l2_mean:.6e}\n"
    )

assert rel_l2 < 0.1, f"Pointwise rel. L2 error too large: {rel_l2:.3e}"
assert rel_l2_mean < 0.05, f"Heat-dose rel. L2 error too large: {rel_l2_mean:.3e}"
