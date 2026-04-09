"""04 — 1-D viscous Burgers equation  (manufactured solution)

Problem
-------
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f(x,t),   x ∈ [0,1],  t ∈ [0,1]
    u(0,t) = u(1,t) = 0
    u(x,0) = sin(πx)

Manufactured solution
---------------------
    u_exact(x,t) = e^{−t} sin(πx)

Substituting into Burgers gives the source term:
    f = u_t + u u_x − ν u_xx
      = e^{−t} (νπ² − 1) sin(πx)
        + π/2 · e^{−2t} sin(2πx)

The nonlinear term u u_x = e^{−2t} sin(πx)·πcos(πx) = πe^{−2t}/2 · sin(2πx)
creates a higher-frequency component in the forcing that the network must
reproduce, making this a good stress-test of the capacity.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
ν = 0.05  # viscosity — decrease for sharper gradients
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

# f = e^{-t}(νπ² − 1) sin(πx)  +  (π/2) e^{-2t} sin(2πx)
source = jno.np.exp(-t) * (ν * π**2 - 1) * jno.np.sin(π * x) + (π / 2) * jno.np.exp(-2 * t) * jno.np.sin(2 * π * x)

# ── Network  (hard Dirichlet BCs) ────────────────────────────────────────────
net = jno.np.nn.mlp(
    in_features=2,
    hidden_dims=48,
    num_layers=4,
    key=jax.random.PRNGKey(3),
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

tx = jno.np.concat([t, x])
u = net(tx) * x * (1 - x)

# ── PDE residual:  u_t + u u_x − ν u_xx − f = 0 ─────────────────────────────
u_t = jno.np.grad(u, t)
u_x = jno.np.grad(u, x)
u_xx = jno.np.grad(u_x, x)
pde = u_t + u * u_x - ν * u_xx - source

# ── Initial condition ─────────────────────────────────────────────────────────
u_0 = net(jno.np.concat([t0, x0])) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"

