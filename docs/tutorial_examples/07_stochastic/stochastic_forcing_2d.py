"""07 — 2-D Poisson equation with stochastic forcing"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax
from shapely.geometry import box

import jno

π = jno.np.pi
σ = 0.5  # noise amplitude on the forcing

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain(box(0, 0, 1, 1), mesh_size=0.05)
x, y, _ = domain.variable("interior")

# ── Deterministic forcing and exact solution ───────────────────────────────────
f = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network (hard BCs via ansatz) ──────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=64,
        num_layers=5,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

u = (net(x, y) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y)  # u = 0 on ∂Ω by construction

# ── Stochastic PDE residual ────────────────────────────────────────────────────
# The noise term is resampled every training step.  Its expectation is zero,
# so E[loss] is minimised by the deterministic solution u*(x,y) = sin(πx)sin(πy).
noise = jno.noise.gaussian(std=σ)
pde = -(u.xx + u.yy) - f - noise

# ── Solve ──────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse])
history = crux.solve(40_000)

# ── Evaluate ───────────────────────────────────────────────────────────────────
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(_u - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))

print(f"Noise amplitude σ = {σ}")
print(f"Relative L2 error: {rel_l2:.4e}")

# ── Record ─────────────────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f_out:
    f_out.write(f"07_stochastic/stochastic_forcing_2d.py | epochs=40000 | sigma={σ} | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
