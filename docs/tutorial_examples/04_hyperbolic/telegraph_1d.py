"""04 — 1-D telegraph equation"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
β = 0.5
c = 1.0
T_end = 1.0

domain = jno.domain.line(mesh_size=0.1, time=(0, T_end, 4))
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source = (1 - β + c**2 * π**2) * u_exact

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=3,
        basis_functions=48,
        hidden_dim=32,
        key=jax.random.PRNGKey(22),
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 10, 5000, 1e-5)))

u = (net(t, x) * x * (1 - x)).scalar.bind(x=x, t=t)
u_ic = (net(t0, x0) * x0 * (1 - x0)).scalar.bind(x=x0, t=t0)

# u.tt, u.t and u_ic.t are all AD derivatives of the network through the time
# input — works on any tag without needing a multi-step time window.
pde = u.tt + β * u.t - c**2 * u.xx - source
ini_u = u_ic - jno.np.sin(π * x0)
ini_ut = u_ic.t + jno.np.sin(π * x0)

crux = jno.core([pde.mse, ini_u.mse, ini_ut.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
