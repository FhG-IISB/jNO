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
import jno.jnp_ops as jnn
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import optax
from jno import LearningRateSchedule as lrs

pi = jnn.pi
dire = jno.setup(__file__)

domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.03))
x, y, _ = domain.variable("interior")
xt, yt, _ = domain.variable("top")
xb, yb, _ = domain.variable("bottom")

u_exact = jnn.sin(pi * x) * jnn.cos(pi * y)
forcing = 2 * pi**2 * u_exact

net = jnn.nn.mlp(in_features=2, hidden_dims=48, num_layers=4, key=jax.random.PRNGKey(14))
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10_000, 1e-5))

u = net(x, y) * x * (1 - x)
u_top = net(xt, yt) * xt * (1 - xt)
u_bottom = net(xb, yb) * xb * (1 - xb)

pde = -jnn.laplacian(u, [x, y]) - forcing
neumann_top = jnn.grad(u_top, yt)
neumann_bottom = jnn.grad(u_bottom, yb)
error = jnn.tracker((u - u_exact).mse, interval=200)

crux = jno.core([pde.mse, neumann_top.mse, neumann_bottom.mse, error], domain)
history = crux.solve(10_000)
history.plot(f"{dire}/training_history.png")

pts = np.array(domain.context["interior"][0, 0])
xs, ys = pts[:, 0], pts[:, 1]
pred = np.array(crux.eval(u)).reshape(xs.shape[0], 1)[:, 0]
true = np.array(crux.eval(u_exact)).reshape(xs.shape[0], 1)[:, 0]
err = np.abs(pred - true)

triang = tri.Triangulation(xs, ys)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, data, title in [
    (axes[0], true, "Exact"),
    (axes[1], pred, "PINN"),
    (axes[2], err, "|error|"),
]:
    tc = ax.tripcolor(triang, data, shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{dire}/solution.png", dpi=150)
print(f"Saved to {dire}/")
