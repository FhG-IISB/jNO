"""04 — 1-D wave equation

Problem
-------
    ∂²u/∂t² = c² ∂²u/∂x²   on [0, 1],
    u(0, t) = u(1, t) = 0,    u(x, 0) = sin(πx),    u_t(x, 0) = 0

Analytical solution:  u(x, t) = cos(cπt) sin(πx)  (standing wave).
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
c = 1.0
T_end = 1.0

domain = jno.domain.line(mesh_size=0.05, time=(0, T_end, 8))
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")
xb, tb = domain.variable("boundary")

u_exact = jno.np.cos(c * π * t) * jno.np.sin(π * x)

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=4,
        basis_functions=64,
        hidden_dim=48,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(7),
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(1e-6, 1e-3, 100, 10_000, 1e-6)))

u = net(t, x).scalar.bind(x=x, t=t)
u0 = net(t0, x0).scalar.bind(x=x0, t=t0)

pde = u.tt - c**2 * u.xx
ic_disp = u0 - jno.np.sin(π * x0)
ic_vel = u0.t
bc = net(tb, xb)

crux = jno.core([pde.mse, ic_disp.mse, ic_vel.mse, bc.mse])
crux.solve(10_000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 3e-1, f"relative L2 error too large: {rel_l2:.3e}"
