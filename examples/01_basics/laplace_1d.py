"""01 — 1-D Laplace / Poisson equation  (simplest possible PINN)

Problem
-------
    −u''(x) = sin(πx),   x ∈ [0, 1],   u(0) = u(1) = 0

Analytical solution
-------------------
    u(x) = sin(πx) / π²

Techniques shown
----------------
* Homogeneous Dirichlet BCs via hard constraint:  u = net(x) · x (1−x)
* Automatic-differentiation gradient  (jnn.grad)
* Tracker to log the L²-error against the exact solution
* Single-phase Adam with exponential LR decay
"""

import jax
import jno
import jno.jnp_ops as jnn
import optax
from jno import LearningRateSchedule as lrs
import matplotlib.pyplot as plt
import numpy as np

π = jnn.pi
dire = jno.setup(__file__)

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jnn.sin(π * x) / π**2

# ── Network with hard-enforced BCs ────────────────────────────────────────────
u_net = jno.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 5_000, 1e-5))

u = u_net(x) * x * (1 - x)  # hard BC: u(0) = u(1) = 0

# ── Constraints ───────────────────────────────────────────────────────────────
pde = -jnn.grad(jnn.grad(u, x), x) - jnn.sin(π * x)  # should be 0
error = jnn.tracker((u - u_exact).mse, interval=100)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, error], domain)
history = crux.solve(5_000)
history.plot(f"{dire}/training_history.png")

# ── Plot ──────────────────────────────────────────────────────────────────────
pts = np.array(crux.domain_data.context["interior"][0, 0, :, 0])
idx = np.argsort(pts)
xs = pts[idx]
pred = np.array(crux.eval(u)).reshape(xs.shape[0], 1)[:, 0][idx]
true = np.array(crux.eval(u_exact)).reshape(xs.shape[0], 1)[:, 0][idx]

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
