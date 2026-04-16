"""05 - Coupled parabolic PDE system in 2-D

Problem
-------
    u_t - Delta u + v = f(x, y, t)
    v_t - Delta v + u = g(x, y, t)

on [0, 1]^2 x [0, 1] with homogeneous Dirichlet boundary conditions.

Analytical solution
-------------------
    u(x, y, t) = exp(-t) sin(pi x) sin(pi y)
    v(x, y, t) = exp(-t) sin(2 pi x) sin(pi y)
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

pi = jno.np.pi
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.2),
    time=(0, T_end, 4),
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(pi * x) * jno.np.sin(pi * y)
v_exact = jno.np.exp(-t) * jno.np.sin(2 * pi * x) * jno.np.sin(pi * y)
f = (2 * pi**2 - 1) * u_exact + v_exact
g = (5 * pi**2 - 1) * v_exact + u_exact

u_net = jno.nn.deeponet(
    n_sensors=1,
    coord_dim=2,
    n_outputs=1,
    n_layers=4,
    basis_functions=64,
    hidden_dim=48,
    key=jax.random.PRNGKey(24),
)
v_net = jno.nn.deeponet(
    n_sensors=1,
    coord_dim=2,
    n_outputs=1,
    n_layers=4,
    basis_functions=64,
    hidden_dim=48,
    key=jax.random.PRNGKey(25),
)
for net in [u_net, v_net]:
    net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

xy = jno.np.concat([x, y])
xy0 = jno.np.concat([x0, y0])
u = u_net(t, xy) * x * (1 - x) * y * (1 - y)
v = v_net(t, xy) * x * (1 - x) * y * (1 - y)
u0 = u_net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)
v0 = v_net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)

pde_u = jno.np.grad(u, t) - jno.np.laplacian(u, [x, y]) + v - f
pde_v = jno.np.grad(v, t) - jno.np.laplacian(v, [x, y]) + u - g
ini_u = u0 - jno.np.sin(pi * x0) * jno.np.sin(pi * y0)
ini_v = v0 - jno.np.sin(2 * pi * x0) * jno.np.sin(pi * y0)

crux = jno.core([pde_u.mse, pde_v.mse, ini_u.mse, ini_v.mse], domain)
history = crux.solve(10_000)

_u, _u_exact, _v, _v_exact = crux.eval([u, u_exact, v, v_exact])
rel_l2_u = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
rel_l2_v = float(jax.numpy.linalg.norm(_v - _v_exact) / (jax.numpy.linalg.norm(_v_exact) + 1e-8))
assert rel_l2_u < 1e-1, f"u relative L2 error too large: {rel_l2_u:.3e}"
assert rel_l2_v < 1e-1, f"v relative L2 error too large: {rel_l2_v:.3e}"
