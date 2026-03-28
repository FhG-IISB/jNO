"""02 — 1-D diffusion-reaction equation  (steady)

Problem
-------
    −u''(x) + σ u(x) = f(x),   x ∈ [0, 1],   u(0) = u(1) = 0

Manufactured solution
---------------------
    u(x) = sin(πx)
    f(x) = (π² + σ) sin(πx)

This tests whether the network can balance diffusion (−u'') and reaction (σu).
Large σ makes the reaction term dominant; try σ ∈ {1, 10, 100}.
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

# ── Physical parameter ────────────────────────────────────────────────────────
σ = 10.0  # reaction coefficient — increase to make the problem stiffer

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.005))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Manufactured solution and forcing ─────────────────────────────────────────
u_exact = jnn.sin(π * x)
forcing = (π**2 + σ) * jnn.sin(π * x)  # f = −u'exact'' + σ u_exact

# ── Network (hard BCs via x(1−x) factor) ─────────────────────────────────────
u_net = jno.nn.mlp(
    in_features=1,
    hidden_dims=64,
    num_layers=4,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10_000, 1e-5))

u = u_net(x) * x * (1 - x)

# ── PDE residual:  −u'' + σu − f = 0 ──────────────────────────────────────────
u_xx = jnn.grad(jnn.grad(u, x), x)
pde = -u_xx + σ * u - forcing
error = jnn.tracker((u - u_exact).mse, interval=200)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, error], domain)
history = crux.solve(10_000)
history.plot(f"{dire}/training_history.png")

# ── Evaluate ──────────────────────────────────────────────────────────────────
pts = np.array(crux.domain_data.context["interior"][0, 0, :, 0])
idx = np.argsort(pts)
xs = pts[idx]
pred = np.array(crux.eval(u)).reshape(xs.shape[0], 1)[:, 0][idx]
true = np.array(crux.eval(u_exact)).reshape(xs.shape[0], 1)[:, 0][idx]

mae = np.abs(pred - true).mean()
print(f"σ = {σ}  |  Mean absolute error: {mae:.6e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_title(f"Diffusion-reaction  σ={σ}")
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
