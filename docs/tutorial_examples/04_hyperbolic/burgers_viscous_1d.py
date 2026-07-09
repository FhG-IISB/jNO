"""04 — 1-D viscous Burgers equation  (manufactured solution)"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
ν = 0.05
T_end = 1.0

# ── Domain (1-D space × time) ─────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.1, time=(0, T_end, 4))
# RAD adaptive resampling concentrates collocation points at the steep moving front.
x, t = domain.variable(
    "interior",
    resampling_strategy=jno.sampler.rad(resample_every=200, resample_fraction=0.2, start_epoch=500, k=10),
)
x0, t0 = domain.variable("initial")

# ── Manufactured solution + source term ──────────────────────────────────────
u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source = jno.np.exp(-t) * (ν * π**2 - 1) * jno.np.sin(π * x) + (π / 2) * jno.np.exp(-2 * t) * jno.np.sin(2 * π * x)

# ── Network  (hard Dirichlet BCs via the x(1-x) factor) ──────────────────────
net = jno.nn(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=4,
        basis_functions=64,
        hidden_dim=48,
        key=jax.random.PRNGKey(3),
    )
)
net.optimizer(
    optax.adam(
        optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=1e-3, warmup_steps=10, decay_steps=5000, end_value=1e-5
        )
    )
)

u = net(t, x) * x * (1 - x)

# ── PDE residual:  u_t + u u_x − ν u_xx − f = 0 ──────────────────────────────
u_x = u.d(x)
pde = u.d(t) + u * u_x - ν * u_x.d(x) - source

# ── Initial condition ────────────────────────────────────────────────────────
u_0 = net(t0, x0) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

# ── Solve ────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
