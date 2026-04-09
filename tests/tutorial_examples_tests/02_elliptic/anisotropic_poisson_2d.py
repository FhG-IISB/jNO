"""02 - 2-D anisotropic Poisson equation

Problem
-------
    -(a u_xx + b u_yy) = f(x, y),   (x, y) in [0, 1]^2
    u = 0 on the boundary

Analytical solution
-------------------
    u(x, y) = sin(pi x) sin(pi y)

which gives

    f(x, y) = (a + b) pi^2 sin(pi x) sin(pi y)
"""

import jax
import jno
import optax

pi = jno.np.pi
a = 1.0
b = 3.0

domain = jno.domain.rect(mesh_size=0.2)
x, y, _ = domain.variable("interior")

u_exact = jno.np.sin(pi * x) * jno.np.sin(pi * y)
forcing = (a + b) * pi**2 * u_exact

net = jno.np.nn.mlp(in_features=2, hidden_dims=32, num_layers=4, key=jax.random.PRNGKey(12))
net.optimizer(optax.adam(1), lr=jno.schedule.learning_rate.exponential(1e-3, 0.5, 10, 1e-5))

u = net(x, y) * x * (1 - x) * y * (1 - y)
pde = -(a * u.d2(x) + b * u.d2(y)) - forcing

crux = jno.core([pde.mse], domain)
history = crux.solve(10_000)

_u, _u_exact = crux.eval([u, u_exact])

rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"
