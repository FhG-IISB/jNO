"""07 — 2-D Poisson equation with stochastic forcing

Problem
-------
    −∆u(x,y) = f(x,y) + σ·ξ(x,y),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω

where f(x,y) = 2π² sin(πx) sin(πy) is a known deterministic forcing and
ξ(x,y) ~ N(0,1) is a pointwise Gaussian noise field with amplitude σ = 0.5.

This models a PDE driven by an uncertain or turbulent source term — only the
mean forcing f is known exactly; the fluctuation σ·ξ represents aleatory
uncertainty that is freshly realised at every training step.

Mean-field solution
-------------------
    u*(x,y) = sin(πx) sin(πy)

Why the stochastic PINN recovers the deterministic solution
-----------------------------------------------------------
The MSE loss at step k is

    L_k(θ) = mean_x [ −∆u_θ − f − σ ξ_k ]²

Expanding the square and taking the expectation over the i.i.d. noise ξ_k:

    E[L_k(θ)] = mean_x(−∆u_θ − f)² + σ²

The noise contributes only a constant σ² that is independent of θ.
Therefore the minimiser of E[L_k] is identical to the minimiser of the
deterministic MSE, and the network converges to u*(x,y) = sin(πx) sin(πy)
despite seeing a different noisy residual at every epoch.

Techniques shown
----------------
* jno.noise.gaussian() on the PDE interior residual — noise on the physics,
  not the boundary
* Hard Dirichlet BCs via the x(1−x)y(1−y) ansatz so no boundary loss is needed
* Noise is resampled automatically from the solver's PRNG every step —
  fully reproducible when the seed is fixed via jno.setup(seed=...)
"""

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
crux = jno.core([pde.mse], domain)
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
