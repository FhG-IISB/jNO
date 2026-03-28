"""01 — 1-D Poisson equation  (soft boundary constraints + FD Laplacian)

Problem
-------
    −u''(x) = sin(πx),   x ∈ [0, 1],   u(0) = u(1) = 0

Analytical solution
-------------------
    u(x) = sin(πx) / π²

Compared to laplace_1d.py this example uses:
* Finite-difference second derivative  (u.d2)
* Soft boundary constraints  (separate boundary tag)
* A tracker that logs the MSE error every 100 epochs
"""

import jax
import jno
import jno.jnp_ops as jnn
import optax
import numpy as np
import matplotlib.pyplot as plt
from jno import LearningRateSchedule as lrs

π = jnn.pi
dire = jno.setup(__file__)

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jnn.sin(π * x) / π**2

# ── Network ───────────────────────────────────────────────────────────────────
u_net = jno.nn.mlp(
    in_features=1,
    hidden_dims=64,
    num_layers=4,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10_000, 1e-5))

u = u_net(x)

# ── Constraints ───────────────────────────────────────────────────────────────
pde = -u.d2(x, scheme="finite_difference") - jnn.sin(π * x)
bc = u_net(xb)  # soft: u(0) = u(1) = 0
error = jnn.tracker((u - u_exact).mse, interval=100)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse, error], domain)
history = crux.solve(10_000)
history.plot(f"{dire}/training_history.png")

# ── Plot ──────────────────────────────────────────────────────────────────────
pts = np.array(crux.domain_data.context["interior"][0, 0, :, 0])
idx = np.argsort(pts)
xs = pts[idx]
pred = np.array(crux.eval(u)).reshape(xs.shape[0], 1)[:, 0][idx]
true = np.array(crux.eval(u_exact)).reshape(xs.shape[0], 1)[:, 0][idx]

mae = np.abs(pred - true).mean()
print(f"Mean absolute error vs exact: {mae:.6e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_title("Solution")
ax1.plot(xs, pred, label="PINN")
ax1.plot(xs, true, "--", label="exact")
ax1.set_xlabel("x")
ax1.legend()

ax2.set_title("Pointwise |error|")
ax2.plot(xs, np.abs(pred - true))
ax2.set_xlabel("x")

plt.tight_layout()
plt.savefig(f"{dire}/solution.png", dpi=150)
print(f"Saved to {dire}/")
