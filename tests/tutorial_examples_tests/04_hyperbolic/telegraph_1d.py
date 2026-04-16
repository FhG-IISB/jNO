"""04 - 1-D telegraph equation

Problem
-------
    u_tt + beta u_t = c^2 u_xx + f(x, t),   x in [0, 1], t in [0, 1]
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(pi x)
    u_t(x, 0) = -sin(pi x)

Analytical solution
-------------------
    u(x, t) = exp(-t) sin(pi x)
"""

import jax
import jno

import foundax
import optax
from jno import LearningRateSchedule as lrs

pi = jno.np.pi
beta = 0.5
c = 1.0
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.1),
    time=(0, T_end, 4),
)
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(pi * x)
source = (1 - beta + c**2 * pi**2) * u_exact

net = jno.nn.wrap(foundax.deeponet(
    n_sensors=1,
    coord_dim=1,
    n_outputs=1,
    n_layers=4,
    basis_functions=64,
    hidden_dim=48,
    key=jax.random.PRNGKey(22),
))
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

u = net(t, x) * x * (1 - x)
dt_ic = 1e-2
u0 = net(t0, x0) * x0 * (1 - x0)
u_t0 = ((net(t0 + dt_ic, x0) - net(t0, x0)) / dt_ic) * x0 * (1 - x0)

pde = jno.np.grad(jno.np.grad(u, t), t) + beta * jno.np.grad(u, t) - c**2 * jno.np.grad(jno.np.grad(u, x), x) - source
ini_u = u0 - jno.np.sin(pi * x0)
ini_ut = u_t0 + jno.np.sin(pi * x0)

crux = jno.core([pde.mse, ini_u.mse, ini_ut.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
