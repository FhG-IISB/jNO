"""02 - 2-D Poisson equation with mixed boundary conditions

Problem
-------
    -Delta u = f(x, y),   (x, y) in [0, 1]^2

    u = 0              on x = 0 and x = 1
    du/dy = 0          on y = 0 and y = 1

Analytical solution
-------------------
    u(x, y) = sin(pi x) cos(pi y)

which gives

    f(x, y) = 2 pi^2 sin(pi x) cos(pi y)
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

pi = jno.np.pi
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.2))
x, y, _ = domain.variable("interior")
xt, yt, _ = domain.variable("top")
xb, yb, _ = domain.variable("bottom")

u_exact = jno.np.sin(pi * x) * jno.np.cos(pi * y)
forcing = 2 * pi**2 * u_exact

net = jno.nn.mlp(in_features=2, hidden_dims=48, num_layers=4, key=jax.random.PRNGKey(14))
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = net(x, y) * x * (1 - x)
u_top = net(xt, yt) * xt * (1 - xt)
u_bottom = net(xb, yb) * xb * (1 - xb)

pde = -jno.np.laplacian(u, [x, y]) - forcing
neumann_top = jno.np.grad(u_top, yt)
neumann_bottom = jno.np.grad(u_bottom, yb)

crux = jno.core([pde.mse, neumann_top.mse, neumann_bottom.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
