"""01 - 1-D biharmonic equation (beam-like fourth-order problem)

Problem
-------
    u''''(x) = 24,   x in [0, 1]

    u(0) = u(1) = 0,
    u'(0) = u'(1) = 0   (clamped)

Analytical solution
-------------------
    u(x) = x²(1 − x)²

The hard-enforced ansatz ``u = net(x) · x²(1 − x)²`` exactly satisfies the
clamped BCs (both u and u' vanish at the endpoints), reducing the problem to
learning a smooth correction factor on top of the boundary-respecting polynomial.
"""

import foundax
import jax
import optax

import jno

domain = jno.domain.line(mesh_size=0.05)
x, _ = domain.variable("interior")

u_exact = x**2 * (1 - x) ** 2

net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=32,
        num_layers=3,
        key=jax.random.PRNGKey(11),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.6, end_value=1e-5)))

u = net(x) * x**2 * (1 - x) ** 2
u_xxxx = u.d2(x).d2(x)

pde = u_xxxx - 24.0

crux = jno.core([pde.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
