"""02 — 1-D diffusion-reaction equation (steady)

Problem
-------
    −u''(x) + σ u(x) = f(x)   on [0, 1],    u(0) = u(1) = 0

Manufactured solution:  u(x) = sin(πx)
Forcing:                f(x) = (π² + σ) sin(πx)
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
σ = 10.0  # reaction coefficient

domain = jno.domain.line(mesh_size=0.1)
x, _ = domain.variable("interior")

u_exact = jno.np.sin(π * x)
forcing = (π**2 + σ) * u_exact

net = jno.nn.wrap(foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

u = (net(x) * x * (1 - x)).scalar.bind(x=x)
pde = -u.xx + σ * u - forcing

crux = jno.core([pde.mse], domain)
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
