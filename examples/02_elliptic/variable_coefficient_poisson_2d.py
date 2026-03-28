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

kappa = 1 + x + y
u_exact = jnn.sin(pi * x) * jnn.sin(pi * y)
forcing = 2 * pi**2 * kappa * u_exact - pi * jnn.cos(pi * x) * jnn.sin(pi * y) - pi * jnn.sin(pi * x) * jnn.cos(pi * y)

net = jno.nn.mlp(in_features=2, hidden_dims=48, num_layers=4, key=jax.random.PRNGKey(13))
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10_000, 1e-5))

u = net(x, y) * x * (1 - x) * y * (1 - y)
flux_x = kappa * jnn.grad(u, x)
flux_y = kappa * jnn.grad(u, y)
pde = -jnn.divergence([flux_x, flux_y], [x, y]) - forcing
error = jnn.tracker((u - u_exact).mse, interval=200)

crux = jno.core([pde.mse, error], domain)
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
