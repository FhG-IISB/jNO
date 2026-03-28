"""02 — 2-D Helmholtz equation

Problem
-------
    ∇²u(x,y) + k² u(x,y) = −f(x,y),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω

Manufactured solution
---------------------
    u(x,y) = sin(πx) sin(πy)

Substituting gives the source term:
    f(x,y) = (2π² − k²) sin(πx) sin(πy)

Note: the problem becomes resonant when k = π√2 ≈ 4.44.
Try different values of k (e.g. 1, 2, 4) to see the effect on convergence.
"""

import jax
import jno
import jno.jnp_ops as jnn
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from jno import LearningRateSchedule as lrs

π = jnn.pi
dire = jno.setup(__file__)

# ── Parameter ─────────────────────────────────────────────────────────────────
k = 2.0  # wave number — change to test different regimes

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.04))
x, y, _ = domain.variable("interior")
xb, yb, _ = domain.variable("boundary")

# ── Manufactured solution and forcing ─────────────────────────────────────────
u_exact = jnn.sin(π * x) * jnn.sin(π * y)
forcing = (2 * π**2 - k**2) * jnn.sin(π * x) * jnn.sin(π * y)

# ── Network ───────────────────────────────────────────────────────────────────
u_net = jnn.nn.mlp(
    in_features=2,
    hidden_dims=64,
    num_layers=5,  # slightly deeper for the oscillatory problem
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10_000, 1e-5))

u = u_net(x, y)

# ── PDE residual:  ∇²u + k²u + f = 0 ────────────────────────────────────────
pde = u.laplacian(x, y, scheme="automatic_differentiation") + k**2 * u + forcing
bc = u_net(xb, yb)
error = jnn.tracker((u - u_exact).mse, interval=200)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse, error], domain)
history = crux.solve(100)
history.plot(f"{dire}/training_history.png")

# ── Evaluate ──────────────────────────────────────────────────────────────────
pts = np.array(domain.context["interior"][0, 0])
xs, ys = pts[:, 0], pts[:, 1]

pred = np.array(crux.eval([u])).reshape(xs.shape[0], 1)[:, 0]
true = np.array(crux.eval(u_exact)).reshape(xs.shape[0], 1)[:, 0]
err = np.abs(pred - true)

l2_rel = np.sqrt(np.mean((pred - true) ** 2)) / (np.sqrt(np.mean(true**2)) + 1e-12)
print(f"k = {k}  |  Relative L² error: {l2_rel:.4e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
triang = tri.Triangulation(xs, ys)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, data, title in [
    (axes[0], true, "Exact"),
    (axes[1], pred, f"PINN (k={k})"),
    (axes[2], err, "|error|"),
]:
    tc = ax.tripcolor(triang, data, shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{dire}/solution.png", dpi=150)
print(f"Saved to {dire}/")
