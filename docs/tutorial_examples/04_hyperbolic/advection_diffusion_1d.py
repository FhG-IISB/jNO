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

Pattern shown
-------------
Soft IC + BC pattern (no multiplicative ansatz).  Three explicit
constraints — PDE residual, IC, BC — are passed to ``jno.core``.
"""

import foundax
import jax
import optax

import jno

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
xb, tb = domain.variable("boundary")

# ── Manufactured solution + source ───────────────────────────────────────────
u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source = jno.np.exp(-t) * ((ν * π**2 - 1) * jno.np.sin(π * x) + c * π * jno.np.cos(π * x))

# ── Network ──────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=3,
        basis_functions=64,
        hidden_dim=32,
        key=jax.random.PRNGKey(1),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.6, end_value=1e-5)))

u = net(t, x)

# ── Constraints ──────────────────────────────────────────────────────────────
# PDE residual:  u_t + c u_x − ν u_xx − f = 0
pde = u.d(t) + c * u.d(x) - ν * u.d2(x) - source

# Initial condition:  u(x, 0) = sin(πx)
ic = net(t0, x0) - jno.np.sin(π * x0)

# Spatial boundary:  u(0, t) = u(1, t) = 0
bc = net(tb, xb)

# ── Solve ────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ic.mse, bc.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 2e-1, f"relative L2 error too large: {rel_l2:.3e}"
