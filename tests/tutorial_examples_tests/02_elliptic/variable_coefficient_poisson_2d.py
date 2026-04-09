"""02 - 2-D variable-coefficient elliptic equation

Problem
-------
    -div(kappa(x, y) grad u(x, y)) = f(x, y),   (x, y) in [0, 1]^2
    u = 0 on the boundary

Analytical solution
-------------------
    u(x, y) = sin(pi x) sin(pi y)
    kappa(x, y) = 1 + x + y
"""
import jax
import jno

import optax
from jno import LearningRateSchedule as lrs
pi = jno.np.pi
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.2))
x, y, _ = domain.variable("interior")

kappa = 1 + x + y
u_exact = jno.np.sin(pi * x) * jno.np.sin(pi * y)
forcing = 2 * pi**2 * kappa * u_exact - pi * jno.np.cos(pi * x) * jno.np.sin(pi * y) - pi * jno.np.sin(pi * x) * jno.np.cos(pi * y)

net = jno.np.nn.mlp(in_features=2, hidden_dims=48, num_layers=4, key=jax.random.PRNGKey(13))
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = net(x, y) * x * (1 - x) * y * (1 - y)
flux_x = kappa * jno.np.grad(u, x)
flux_y = kappa * jno.np.grad(u, y)
pde = -jno.np.divergence([flux_x, flux_y], [x, y]) - forcing

crux = jno.core([pde.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"

