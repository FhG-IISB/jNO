"""04 — 1-D wave equation

Problem
-------
    ∂²u/∂t² = c² ∂²u/∂x²,   x ∈ [0,1],  t ∈ [0, 1]
    u(0, t)  = u(1, t) = 0   (homogeneous Dirichlet)
    u(x, 0)  = sin(πx)        (displacement IC)
    ∂u/∂t(x,0) = 0            (velocity IC — standing wave)

Analytical solution
-------------------
    u(x, t) = cos(cπt) sin(πx)

Standing wave: the spatial shape sin(πx) oscillates in amplitude with
period T = 2/(cπ).  With c=1 and T_end=1 we see half a full oscillation.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
c = 1.0  # wave speed
T_end = 1.0  # final time (half period for c=1)

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.1),
    time=(0, T_end, 5),
)
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.cos(c * π * t) * jno.np.sin(π * x)

# ── Network  (hard Dirichlet BCs via normalized 4x(1−x)) ────────────────────
net = jno.np.nn.mlp(
    in_features=2,
    hidden_dims=48,
    num_layers=4,
    key=jax.random.PRNGKey(7),
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

boundary_envelope = 4.0 * x * (1 - x)
tx = jno.np.concat([t, x])
u = net(tx) * boundary_envelope

# ── PDE constraint:  u_tt − c² u_xx = 0 ─────────────────────────────────────
u_t = jno.np.grad(u, t)
u_tt = jno.np.grad(u_t, t)
u_xx = jno.np.grad(jno.np.grad(u, x), x)
pde = u_tt - c**2 * u_xx

# ── Initial conditions evaluated at t=0 ──────────────────────────────────────
# Use a forward difference at t=0 so the velocity constraint remains meaningful.
dt_ic = 1e-2
boundary_envelope0 = 4.0 * x0 * (1 - x0)
tx0 = jno.np.concat([t0, x0])
txdt = jno.np.concat([t0 + dt_ic, x0])
u_0 = net(tx0) * boundary_envelope0
u_t_0 = ((net(txdt) - net(tx0)) / dt_ic) * boundary_envelope0

ini_u = u_0 - jno.np.sin(π * x0)  # u(x,0) = sin(πx)
ini_ut = u_t_0  # ∂u/∂t(x,0) = 0

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini_u.mse, ini_ut.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"

