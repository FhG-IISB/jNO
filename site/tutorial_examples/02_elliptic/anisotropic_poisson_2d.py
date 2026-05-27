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

from pathlib import Path

import foundax
import jax
import optax

import jno

pi = jno.np.pi
a = 1.0
b = 3.0

domain = jno.domain.rect(mesh_size=0.1)
x, y, _ = domain.variable("interior")

u_exact = jno.np.sin(pi * x) * jno.np.sin(pi * y)
forcing = (a + b) * pi**2 * u_exact

net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=64,
        num_layers=5,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(12),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(init_value=1e-3, transition_steps=80, decay_rate=0.5, end_value=1e-5)))

u = net(x, y) * x * (1 - x) * y * (1 - y)
pde = -(a * u.d2(x) + b * u.d2(y)) - forcing

crux = jno.core([pde.mse], domain)
history = crux.solve(40_000)

_u, _u_exact = crux.eval([u, u_exact])

rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))

# Write result to tracking file
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/anisotropic_poisson_2d.py | epochs=40000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
