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
domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.1),
    time=(0, T_end, 4),
)
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.exp(-α * π**2 * t) * jno.np.sin(π * x)

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.np.nn.mlp(
    in_features=2,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(0),
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.7, 10, 1e-5))

tx = jno.np.concat([t, x])
tx0 = jno.np.concat([t0, x0])
u = net(tx) * x * (1 - x)  # hard Dirichlet BCs
u0 = net(tx0) * x0 * (1 - x0)

# ── Constraints ───────────────────────────────────────────────────────────────
#   PDE:  u_t − α u_xx = 0
pde = jno.np.grad(u, t) - α * jno.np.grad(jno.np.grad(u, x), x)
#   IC:   u(x, 0) = sin(πx)
ini = u0 - jno.np.sin(π * x0)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"
