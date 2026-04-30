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

import foundax
import optax
from pathlib import Path

pi = jno.np.pi
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y, _ = domain.variable("interior")

kappa = 1 + x + y
u_exact = jno.np.sin(pi * x) * jno.np.sin(pi * y)
forcing = 2 * pi**2 * kappa * u_exact - pi * jno.np.cos(pi * x) * jno.np.sin(pi * y) - pi * jno.np.sin(pi * x) * jno.np.cos(pi * y)

net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=80, num_layers=5, key=jax.random.PRNGKey(13)))
net.optimizer(optax.adam(optax.exponential_decay(init_value=1e-3, transition_steps=80, decay_rate=0.5, end_value=1e-5)))

u = net(x, y) * x * (1 - x) * y * (1 - y)
flux_x = kappa * jno.np.grad(u, x)
flux_y = kappa * jno.np.grad(u, y)
pde = -jno.np.divergence([flux_x, flux_y], [x, y]) - forcing

crux = jno.core([pde.mse], domain)
history = crux.solve(40000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))

# Write result to tracking file
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/variable_coefficient_poisson_2d.py | epochs=40000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
