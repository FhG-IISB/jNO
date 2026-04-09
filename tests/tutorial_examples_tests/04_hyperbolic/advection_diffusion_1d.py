"""04 — 1-D advection-diffusion equation  (manufactured solution)

Problem
-------
    ∂u/∂t + c ∂u/∂x = ν ∂²u/∂x² + f(x,t),   x ∈ [0,1],  t ∈ [0,1]
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(πx)

Manufactured solution
---------------------
    u_exact(x,t) = e^{−t} sin(πx)

Substituting into the PDE gives the source term:
    f(x,t) = u_t + c u_x − ν u_xx
           = e^{−t} [ (νπ² − 1) sin(πx) + cπ cos(πx) ]

This holds the spatial shape fixed while the amplitude decays exponentially.
The convection term introduces an apparent leftward shift in the forcing.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
c = 1.0  # advection speed
ν = 0.05  # diffusivity (small → convection dominated)
T_end = 1.0

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.1),
    time=(0, T_end, 4),
)
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")

# ── Manufactured solution + source ───────────────────────────────────────────
u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source = jno.np.exp(-t) * ((ν * π**2 - 1) * jno.np.sin(π * x) + c * π * jno.np.cos(π * x))

# ── Network  (hard Dirichlet BCs) ────────────────────────────────────────────
net = jno.np.nn.mlp(
    in_features=2,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(1),
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.6, 10, 1e-5))

tx = jno.np.concat([t, x])
u = net(tx) * x * (1 - x)

# ── PDE residual:  u_t + c u_x − ν u_xx − f = 0 ─────────────────────────────
u_t = jno.np.grad(u, t)
u_x = jno.np.grad(u, x)
u_xx = jno.np.grad(u_x, x)
pde = u_t + c * u_x - ν * u_xx - source

# ── Initial condition:  u(x,0) = sin(πx) ─────────────────────────────────────
u_0 = net(jno.np.concat([t0, x0])) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"
