"""03 — 1-D heat equation  (parabolic, time-dependent)

Problem
-------
    ∂u/∂t = α ∂²u/∂x²,   x ∈ [0,1],  t ∈ [0, 0.5]
    u(0, t) = u(1, t) = 0          (homogeneous Dirichlet)
    u(x, 0) = sin(πx)              (initial condition)

Analytical solution
-------------------
    u(x, t) = exp(−απ²t) sin(πx)

Network ansatz
--------------
    u ≈ net(t, x) · x (1−x)       — hard-enforces BCs for all t

The initial condition is implemented as a soft constraint evaluated on a
separate "initial" tag.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
# ── Physical parameter ────────────────────────────────────────────────────────
α = 0.1  # thermal diffusivity
T_end = 0.5  # final time

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.01, time=(0, T_end, 10))
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.exp(-α * π**2 * t) * jno.np.sin(π * x)

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.nn.deeponet(
    n_sensors=1,
    coord_dim=1,
    n_outputs=1,
    n_layers=3,
    basis_functions=64,
    hidden_dim=32,
    key=jax.random.PRNGKey(0),
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.9, 10000, 1e-5))

# Hard-enforce both BC and IC:
# u(x,0)=sin(pi x), u(0,t)=u(1,t)=0
u = jno.np.sin(π * x) + t * net(t, x) * x * (1 - x)

# ── Constraints ───────────────────────────────────────────────────────────────
#   PDE:  u_t − α u_xx = 0
pde = jno.np.grad(u, t) - α * jno.np.grad(jno.np.grad(u, x), x)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(10000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
