"""02 — 1-D diffusion-reaction equation  (steady)

Problem
-------
    −u''(x) + σ u(x) = f(x),   x ∈ [0, 1],   u(0) = u(1) = 0

Manufactured solution
---------------------
    u(x) = sin(πx)
    f(x) = (π² + σ) sin(πx)

This tests whether the network can balance diffusion (−u'') and reaction (σu).
Large σ makes the reaction term dominant; try σ ∈ {1, 10, 100}.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
# ── Physical parameter ────────────────────────────────────────────────────────
σ = 10.0  # reaction coefficient — increase to make the problem stiffer

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Manufactured solution and forcing ─────────────────────────────────────────
u_exact = jno.np.sin(π * x)
forcing = (π**2 + σ) * jno.np.sin(π * x)  # f = −u'exact'' + σ u_exact

# ── Network (hard BCs via x(1−x) factor) ─────────────────────────────────────
u_net = jno.np.nn.mlp(
    in_features=1,
    hidden_dims=64,
    num_layers=4,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = u_net(x) * x * (1 - x)

# ── PDE residual:  −u'' + σu − f = 0 ──────────────────────────────────────────
u_xx = jno.np.grad(jno.np.grad(u, x), x)
pde = -u_xx + σ * u - forcing

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"

