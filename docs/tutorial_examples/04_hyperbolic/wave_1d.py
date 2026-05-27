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

import foundax
import jax
import optax

import jno

π = jno.np.pi
c = 1.0  # wave speed
T_end = 1.0  # final time (half period for c=1)

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.01),
    time=(0, T_end, 20),
)
x, t = domain.variable("interior")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.cos(c * π * t) * jno.np.sin(π * x)

# ── Network  ──────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=6,
        basis_functions=128,
        hidden_dim=96,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(7),
    )
)
net.optimizer(
    optax.adam(
        optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-3,
            warmup_steps=200,
            decay_steps=49800,
            end_value=1e-7,
        )
    )
)

# Hard-enforce BC *and* both ICs in the ansatz:
#   u(x,0)  = sin(πx)          [because t²=0]
#   u_t(x,0) = 0               [because d/dt(t²)=2t=0 at t=0]
#   u(0,t)  = u(1,t) = 0       [because sin(0)=sin(π)=0 and x(1-x)=0]
u = jno.np.sin(π * x) + t**2 * net(t, x) * x * (1 - x)

# ── PDE constraint:  u_tt − c² u_xx = 0 ─────────────────────────────────────
u_t = jno.np.grad(u, t)
u_tt = jno.np.grad(u_t, t)
u_xx = jno.np.grad(jno.np.grad(u, x), x)
pde = u_tt - c**2 * u_xx

# ── Solve (single PDE constraint — ICs and BCs are hard-coded) ───────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(50000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
