"""07 — 2-D Poisson equation with stochastic forcing

Problem
-------
    −∆u = f(x,y) + σ·ξ(x,y),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω

f = 2π² sin(πx) sin(πy),  ξ ~ N(0,1),  σ = 0.5.
Mean-field solution: u*(x,y) = sin(πx) sin(πy).

Smoke-test version: 5 000 epochs with a loose tolerance to keep CI fast.
See docs/tutorial_examples/07_stochastic/stochastic_forcing_2d.py for the full run.
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno
from jno import LearningRateSchedule as lrs

π = jno.np.pi
σ = 0.5

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y, _ = domain.variable("interior")

f       = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network ────────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=64,
        num_layers=5,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = net(x, y) * x * (1 - x) * y * (1 - y)

# ── Stochastic PDE residual ────────────────────────────────────────────────────
# Noise is on the interior forcing, not the boundary.
# E[noise] = 0 so the minimiser is the same as the deterministic solution.
pde = -jno.np.laplacian(u, [x, y]) - f - jno.noise.gaussian(std=σ)

# ── Solve ──────────────────────────────────────────────────────────────────────
crux    = jno.core([pde.mse], domain)
history = crux.solve(5_000)

# ── Evaluate ───────────────────────────────────────────────────────────────────
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(_u - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))

assert rel_l2 < 5e-1, f"relative L2 error too large: {rel_l2:.3e}"
