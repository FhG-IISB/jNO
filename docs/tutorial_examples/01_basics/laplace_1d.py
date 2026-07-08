"""01 — 1-D Laplace equation (simplest possible PINN)"""

import foundax
import jax
import optax

import jno

domain = jno.domain.line(mesh_size=0.1)
x, _ = domain.variable("interior")

u_exact = x

net = jno.nn(foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

u = (x + x * (1 - x) * net(x)).scalar.bind(x=x)
pde = u.xx  # Laplace: u'' = 0

crux = jno.core([pde.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
