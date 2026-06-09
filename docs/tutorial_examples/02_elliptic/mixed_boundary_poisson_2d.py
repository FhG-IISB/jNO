"""02 — 2-D Poisson equation with mixed boundary conditions

Problem
-------
    −∇²u = f(x, y)   on [0, 1]²
    u = 0            on x = 0 and x = 1   (hard Dirichlet via ansatz)
    ∂u/∂y = 0        on y = 0 and y = 1   (soft Neumann loss)

Manufactured solution:  u(x, y) = sin(πx) cos(πy)
Forcing:                f(x, y) = 2π² sin(πx) cos(πy)
"""

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi
# jno.domain.rect gives left/right/top/bottom edge tags out of the box.
domain = jno.domain.rect(mesh_size=0.05)
x, y, _ = domain.variable("interior")
xt, yt, _ = domain.variable("top")
xb, yb, _ = domain.variable("bottom")

u_exact = jno.np.sin(π * x) * jno.np.cos(π * y)
forcing = 2 * π**2 * u_exact

net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=48, num_layers=4, key=jax.random.PRNGKey(14)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

# Hard Dirichlet on x = 0, 1 via the x(1-x) factor; Neumann on y boundaries is soft.
u = (net(x, y) * x * (1 - x)).scalar.bind(x=x, y=y)
u_top = (net(xt, yt) * xt * (1 - xt)).scalar.bind(x=xt, y=yt)
u_bot = (net(xb, yb) * xb * (1 - xb)).scalar.bind(x=xb, y=yb)

pde = -(u.xx + u.yy) - forcing
neumann_top = u_top.y  # ∂u/∂y at y = 1
neumann_bot = u_bot.y  # ∂u/∂y at y = 0

crux = jno.core([pde.mse, neumann_top.mse, neumann_bot.mse], domain)
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/mixed_boundary_poisson_2d.py | epochs=5000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
