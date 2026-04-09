"""01 - 1-D biharmonic equation (beam-like fourth-order problem)

Problem
-------
    u''''(x) = 24,   x in [0, 1]

    u(0) = u(1) = 0,
    u'(0) = u'(1) = 0

Analytical solution
-------------------
    u(x) = x^2 (1 - x)^2

The x^2 (1-x)^2 ansatz hard-enforces the clamped boundary conditions, making
this a compact fourth-order derivative example.
"""
import jax
import jno

import optax
from jno import LearningRateSchedule as lrs
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
x, _ = domain.variable("interior")

u_exact = x**2 * (1 - x) ** 2

net = jno.np.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(11),
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.6, 10, 1e-5))

u = net(x) * x**2 * (1 - x) ** 2
u_xxxx = jno.np.grad(jno.np.grad(jno.np.grad(jno.np.grad(u, x), x), x), x)

pde = u_xxxx - 24.0

crux = jno.core([pde.mse], domain)
history = crux.solve(10, profile=True)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"

