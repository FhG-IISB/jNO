"""01 - 1-D biharmonic equation (beam-like fourth-order problem)

Problem
-------
    u''''(x) = sin(π x),   x in [0, 1]

    u(0) = u(1) = 0,
    u'(0) = u'(1) = 0   (clamped)

Analytical solution
-------------------
    u(x) = sin(π x) / π⁴

The hard-enforced ansatz `u = net(x) · x²(1−x)²` exactly satisfies the
clamped BCs (both u and u' vanish at the endpoints), but the *interior*
shape is non-trivial: the network must learn `sin(πx) / [π⁴ · x²(1−x)²]`,
which is well-behaved interior but has nominal 0/0 limits at the
endpoints — exactly the deviation the clamped ansatz is designed to
absorb.
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi

domain = jno.domain.line(mesh_size=0.05)
x, _ = domain.variable("interior")

u_exact = jno.np.sin(π * x) / π**4

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

pde = u_xxxx - jno.np.sin(π * x)

crux = jno.core([pde.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 2e-1, f"relative L2 error too large: {rel_l2:.3e}"
