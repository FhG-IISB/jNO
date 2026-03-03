"""1-D Poisson equation — minimal example with a tracker.

Problem:
    -u''(x) = sin(πx),  x ∈ [0, 1],  u(0) = u(1) = 0

Analytical solution:
    u(x) = sin(πx) / π²

The network enforces the homogeneous Dirichlet BCs exactly via the hard
constraint u = net(x) * x * (1 - x).  A tracker logs the mean absolute
error against the analytical solution every 100 epochs so you can watch
convergence in the log alongside the PDE residual.
"""

import jax
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs

π = jnn.pi
dire = jno.setup(__file__)

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")
# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jnn.sin(π * x) / π**2

# ── Network (BCs enforced via hard constraint) ────────────────────────────────
u_net = jnn.nn.mlp(
    in_features=1,
    hidden_dims=64,
    num_layers=4,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10_000, 1e-5))

u = u_net(x)  # * x * (1 - x)

# ── Constraints ───────────────────────────────────────────────────────────────
# PDE residual: -u'' - sin(πx) = 0
pde = -u.d2(x, scheme="finite_difference") - jnn.sin(π * x)
bc = u_net(xb)  # Enforces u(0) = u(1) = 0 via the hard constraint above.

# Tracker: mean absolute error vs analytical, evaluated every 100 epochs
error = jnn.tracker((u - u_exact).mse, interval=100)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse, error], domain)
crux.solve(10_000).plot(f"{dire}/training_history.png")

# ── Quick plot ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

# context['interior'] shape: (batch, time, n_points, n_features)
xs = np.array(crux.domain_data.context["interior"][0, 0, :, 0])
sort_idx = np.argsort(xs)
xs = xs[sort_idx]

pred = np.array(crux.eval(u)[0, 0, :, 0])[sort_idx]
true = np.array(crux.eval(u_exact)[0, 0, :, 0])[sort_idx]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].set_title("Solution")
axes[0].plot(xs, pred, label="network")
axes[0].plot(xs, true, "--", label="exact")
axes[0].set_xlabel("x")
axes[0].legend()

axes[1].set_title("Pointwise error")
axes[1].plot(xs, np.abs(pred - true))
axes[1].set_xlabel("x")

plt.tight_layout()
plt.savefig(f"{dire}/solution.png")
print(f"Plots saved to {dire}/")
