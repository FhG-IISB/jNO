"""02 — 2-D anisotropic Poisson equation"""

from pathlib import Path

import foundax
import jax
import optax
from shapely.geometry import box

import jno

π = jno.np.pi
a, b = 1.0, 3.0

domain = jno.domain(box(0, 0, 1, 1), mesh_size=0.1)
x, y, _ = domain.variable("interior")

u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
forcing = (a + b) * π**2 * u_exact

net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(12)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

u = (net(x, y) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y)
pde = -(a * u.xx + b * u.yy) - forcing

crux = jno.core([pde.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/anisotropic_poisson_2d.py | epochs=5000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
