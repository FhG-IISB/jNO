"""06 — Heat equation with temporal integral constraint

Equation:  ∂u/∂t = α ∂²u/∂x²,   x ∈ [0,1],  t ∈ [0,T]
           u(0,t) = u(1,t) = 0   (Dirichlet, hard-enforced)
           u(x,0) = sin(πx)      (IC, hard-enforced)

Exact solution:  u*(x,t) = exp(−α π² t) sin(πx)

Time-mean (analytic):  ū*(x) = sin(πx) · (1 − exp(−α π² T)) / (α π²)

Two losses:
  1. PDE residual ‖∂u/∂t − α ∂²u/∂x²‖²
  2. Heat-dose constraint ‖∫₀ᵀ u(x,t) dt − ū*(x)‖²  via .integrate(t)
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno
from jno import LearningRateSchedule as lrs

π = jno.np.pi

# ── Domain ────────────────────────────────────────────────────────────────────
α = 0.1
T = 0.5
N_t = 10

domain = jno.domain.line(mesh_size=0.05, time=(0.0, T, N_t))
x, t = domain.variable("interior")

# ── Exact solution and time mean ──────────────────────────────────────────────
u_exact = jno.np.exp(-α * π**2 * t) * jno.np.sin(π * x)

_αλ = α * float(jnp.pi) ** 2
heat_dose_coeff = (1.0 - float(jnp.exp(-_αλ * T))) / _αλ
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

u = jno.np.sin(π * x) + t * net(t, x) * x * (1 - x)

# ── Losses ────────────────────────────────────────────────────────────────────
pde = jno.np.grad(u, t) - α * jno.np.grad(jno.np.grad(u, x), x)
u_mean = u.integrate(t)
heat_dose_residual = u_mean - u_mean_exact

# ── Solve ─────────────────────────────────────────────────────────────────────
EPOCHS = 10_000
crux = jno.core([pde.mse, heat_dose_residual.mse], domain)
crux.solve(EPOCHS, min_consecutive=None)

# ── Evaluate against exact solution ───────────────────────────────────────────
u_pred, u_ref = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
assert rel_l2 < 0.05, f"Pointwise rel. L2 error too large: {rel_l2:.3e}"

# ── Evaluate time-mean field against analytic formula ─────────────────────────
u_mean_pred, u_mean_ref = crux.eval([u_mean, u_mean_exact], min_consecutive=None)
rel_l2_mean = float(jnp.linalg.norm(u_mean_pred - u_mean_ref) / (jnp.linalg.norm(u_mean_ref) + 1e-8))
assert rel_l2_mean < 0.05, f"Time-mean rel. L2 error too large: {rel_l2_mean:.3e}"
