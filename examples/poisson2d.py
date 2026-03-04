"""2-D Poisson equation on a rectangle — AD vs finite-difference comparison.

Problem:
    -∇²u(x,y) = 2π² sin(πx) sin(πy),   (x,y) ∈ [0,1]²,   u = 0  on ∂Ω

Analytical solution:
    u(x,y) = sin(πx) sin(πy)

We solve the same PDE twice — once with automatic differentiation and once
with finite-difference stencils — then compare both against the analytical
solution and each other.
"""

import jax
import jno
import jno.numpy as jnn
import optax
import numpy as np
import matplotlib.pyplot as plt
from jno import LearningRateSchedule as lrs

π = jnn.pi
dire = jno.setup(__file__)

# ── Shared domain (rectangle, fine mesh for FD accuracy) ──────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.04))
x, y, _ = domain.variable("interior")
xb, yb, _ = domain.variable("boundary")

# Analytical solution and forcing
u_exact = jnn.sin(π * x) * jnn.sin(π * y)
forcing = 2 * π**2 * jnn.sin(π * x) * jnn.sin(π * y)


# ── Helper: build, train, and return a solver ────────────────────────────────
def make_solver(scheme, label, epochs=10_000):
    """Create an MLP, set up the PDE with the given derivative scheme, and solve."""
    net = jnn.nn.mlp(
        in_features=2,
        hidden_dims=64,
        num_layers=4,
        key=jax.random.PRNGKey(0),
    ).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, epochs, 1e-5))

    u = net(x, y)

    # PDE residual:  -∇²u - f = 0
    pde = -u.laplacian(x, y, scheme=scheme) - forcing
    bc = net(xb, yb)

    error = jnn.tracker((u - u_exact).mse, interval=100)

    crux = jno.core([pde.mse, bc.mse, error], domain)
    crux.solve(epochs).plot(f"{dire}/{label}_history.png")
    return crux, u


# ── Solve with AD ─────────────────────────────────────────────────────────────
print("═" * 60)
print("  Solving with AUTOMATIC DIFFERENTIATION")
print("═" * 60)
crux_ad, u_ad = make_solver("automatic_differentiation", "ad")

# ── Solve with FD ─────────────────────────────────────────────────────────────
print("═" * 60)
print("  Solving with FINITE DIFFERENCES")
print("═" * 60)
crux_fd, u_fd = make_solver("finite_difference", "fd")

# ── Compare ───────────────────────────────────────────────────────────────────
pts = np.array(domain.context["interior"][0, 0])  # (N, 2)
xs, ys = pts[:, 0], pts[:, 1]

pred_ad = np.array(crux_ad.eval(u_ad)[0, 0, :, 0])
pred_fd = np.array(crux_fd.eval(u_fd)[0, 0, :, 0])
true = np.array(crux_ad.eval(u_exact)[0, 0, :, 0])

err_ad = np.abs(pred_ad - true)
err_fd = np.abs(pred_fd - true)
err_diff = np.abs(pred_ad - pred_fd)

print(f"\n{'Metric':<30} {'AD':>12} {'FD':>12}")
print("─" * 56)
print(f"{'Mean abs error vs exact':<30} {err_ad.mean():>12.6f} {err_fd.mean():>12.6f}")
print(f"{'Max  abs error vs exact':<30} {err_ad.max():>12.6f} {err_fd.max():>12.6f}")
print(f"{'Mean |AD − FD|':<30} {err_diff.mean():>12.6f}")
print(f"{'Max  |AD − FD|':<30} {err_diff.max():>12.6f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib.tri as tri

triang = tri.Triangulation(xs, ys)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, data, title in [
    (axes[0, 0], true, "Exact"),
    (axes[0, 1], pred_ad, "AD solution"),
    (axes[0, 2], pred_fd, "FD solution"),
    (axes[1, 0], err_ad, "|AD − exact|"),
    (axes[1, 1], err_fd, "|FD − exact|"),
    (axes[1, 2], err_diff, "|AD − FD|"),
]:
    tc = ax.tripcolor(triang, data, shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{dire}/comparison.png", dpi=150)
print(f"\nPlots saved to {dire}/")
