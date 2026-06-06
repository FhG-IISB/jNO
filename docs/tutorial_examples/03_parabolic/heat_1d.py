"""03 — 1-D heat equation  (parabolic, time-dependent)

Problem
-------
    ∂u/∂t = α ∂²u/∂x²,   x ∈ [0,1],  t ∈ [0, 0.5]
    u(0, t) = u(1, t) = 0          (homogeneous Dirichlet BCs)
    u(x, 0) = sin(πx)              (initial condition)

Analytical solution
-------------------
    u(x, t) = exp(−απ²t) sin(πx)

Pattern shown
-------------
Soft IC + BC enforcement.  Three constraints compete:
  * PDE residual   on the interior
  * IC residual    on the t=0 slice (the "initial" tag)
  * BC residual    on the x=0 / x=1 spatial boundary at all times
This pattern handles initial- and boundary-value problems uniformly and is
preferred when the geometry isn't a unit interval or when a clean
multiplicative ansatz isn't obvious.  See `laplace_1d.py` for the
hard-enforcement counterpart.
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
α = 0.1  # thermal diffusivity
T_end = 0.5  # final time

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.01, time=(0, T_end, 10))
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")
xb, tb = domain.variable("boundary")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.exp(-α * π**2 * t) * jno.np.sin(π * x)

# ── Network — bare network, no ansatz ────────────────────────────────────────
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=3,
        basis_functions=64,
        hidden_dim=32,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10000, 0.9, end_value=1e-5)))

u = net(t, x)

# ── Constraints ───────────────────────────────────────────────────────────────
#   PDE:  u_t − α u_xx = 0   on the interior
pde = u.d(t) - α * u.d2(x)

#   IC:   u(x, 0) = sin(πx)  on the initial slice
ic = net(t0, x0) - jno.np.sin(π * x0)

#   BC:   u(0, t) = u(1, t) = 0  on the spatial boundary
bc = net(tb, xb)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ic.mse, bc.mse], domain)
history = crux.solve(10000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 2e-1, f"relative L2 error too large: {rel_l2:.3e}"
