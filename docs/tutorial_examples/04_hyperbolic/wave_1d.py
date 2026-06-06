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

Pattern shown
-------------
Soft enforcement of all four conditions — the displacement IC, the
velocity IC, and the two spatial BCs — as separate loss terms.  This
generalises to any hyperbolic problem; the matching hard-ansatz pattern
is much more constrained.
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
c = 1.0  # wave speed
T_end = 1.0

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.01),
    time=(0, T_end, 20),
)
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")  # t = 0 slice — for displacement + velocity IC
xb, tb = domain.variable("boundary")  # x = 0 and x = 1 at all t — for the Dirichlet BC

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

u = net(t, x)

# ── PDE constraint:  u_tt − c² u_xx = 0 ─────────────────────────────────────
pde = u.d2(t) - c**2 * u.d2(x)

# ── Initial conditions:  u(x,0) = sin(πx)  and  u_t(x,0) = 0 ────────────────
u0 = net(t0, x0)
ic_disp = u0 - jno.np.sin(π * x0)
ic_vel = u0.d(t0)

# ── Spatial boundary:  u(0, t) = u(1, t) = 0 ───────────────────────────────
bc = net(tb, xb)

# ── Solve ────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ic_disp.mse, ic_vel.mse, bc.mse], domain)
history = crux.solve(50000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 2e-1, f"relative L2 error too large: {rel_l2:.3e}"
