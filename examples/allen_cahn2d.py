"""2-D Allen–Cahn equation — manufactured-solution verification.

Problem (Allen–Cahn with source):
    ∂u/∂t = ε² ∇²u + u − u³ + f(x,y,t),   (x,y) ∈ [0,1]²,  t ∈ [0,1]

Manufactured (analytical) solution:
    u(x,y,t) = e^{-t} sin(πx) sin(πy)

This satisfies homogeneous Dirichlet BCs on ∂[0,1]² automatically.
The source term f is obtained by substituting u into the PDE:

    f = u_t − ε² ∇²u − u + u³
      = e^{-t} sin(πx) sin(πy) (2ε²π² − 2)
        + e^{-3t} sin³(πx) sin³(πy)

Parameters:
    ε = 0.1   (interface width — controls sharpness of nonlinearity)

We solve with a DeepONet and compare against the exact solution.
"""

import jax
import jno
import jno.numpy as jnn
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from jno import LearningRateSchedule as lrs

π = jnn.pi
sin = jnn.sin
exp = jnn.exp
dire = jno.setup(__file__)

# ── Physical parameters ───────────────────────────────────────────────────────
eps = 0.1  # interface width
T_end = 1.0  # final time

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, 10),
)

x, y, t = domain.variable("interior")

# ── Analytical (manufactured) solution ────────────────────────────────────────
S = sin(π * x) * sin(π * y)  # spatial shape
u_exact = exp(-t) * S(x, y)  # full solution

# Source term from MMS:
#   f = e^{-t} S (2ε²π² − 2) + e^{-3t} S³
coeff = 2 * eps**2 * π**2 - 2
source = exp(-t) * S(x, y) * coeff + exp(-3 * t) * S(x, y) ** 3

# ── Network ───────────────────────────────────────────────────────────────────
net = jnn.nn.deeponet(
    n_sensors=1,
    sensor_channels=1,
    coord_dim=2,
    basis_functions=40,
    hidden_dim=40,
    n_layers=3,
    key=jax.random.PRNGKey(42),
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(2_500, 500, 1e-3, 1e-5))

# Hard-enforce homogeneous Dirichlet BCs via the ansatz: u = N(t,x,y)·x(1−x)y(1−y)
u = net(t, jnn.concat([x, y])) * x * (1 - x) * y * (1 - y)

# ── PDE residual ──────────────────────────────────────────────────────────────
#   ∂u/∂t − ε² ∇²u − u + u³ − f = 0
pde = jnn.grad(u, t) - eps**2 * jnn.laplacian(u, [x, y]) - u + u**3 - source

# ── Initial condition residual ────────────────────────────────────────────────
# Evaluate the network at t=0 using the interior spatial points.
# Using 0*t ensures t=0 while keeping the expression on the interior tag
# (avoids the "initial" tag which has T=1 and would collapse the time scan).
u_at_0 = net(0 * t, jnn.concat([x, y])) * x * (1 - x) * y * (1 - y)
ini = u_at_0 - sin(π * x) * sin(π * y)

# ── Trackers ──────────────────────────────────────────────────────────────────
error = jnn.tracker((u - u_exact).mse, interval=200)

# ── Assemble and solve ────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse, error], domain)

print("═" * 60)
print("  2-D Allen–Cahn (MMS verification)  ε = {}".format(eps))
print("═" * 60)

crux.solve(2_500).plot(f"{dire}/training_history.png")
jno.save(crux, f"{dire}/allen_cahn2d.pkl")

# ── Comparison against exact solution ─────────────────────────────────────────
pts = np.array(domain.context["interior"][0, 0])  # (N, 2)
xs, ys = pts[:, 0], pts[:, 1]
triang = tri.Triangulation(xs, ys)

n_times = domain.context["interior"].shape[1]  # T axis (B, T, N, D)
snap_indices = list(range(n_times))  # all time slices
n_snaps = len(snap_indices)

# Pre-evaluate once — shape is (B, T, N, 1)
pred_all = np.array(crux.eval(u))
true_all = np.array(crux.eval(u_exact))
print(pred_all.shape, true_all.shape)

# Per-snapshot error table
print(f"\n{'t':>6}  {'L² rel':>12}  {'Max |err|':>12}  {'Mean |err|':>12}")
print("─" * 48)
for ti in snap_indices:
    t_val = ti / max(n_times - 1, 1) * T_end
    p = pred_all[0, ti, :, 0]  # (B=0, T=ti, N, D=0)
    r = true_all[0, ti, :, 0]
    ae = np.abs(p - r)
    l2_rel = np.sqrt(np.mean((p - r) ** 2)) / (np.sqrt(np.mean(r**2)) + 1e-12)
    print(f"{t_val:6.3f}  {l2_rel:12.6e}  {ae.max():12.6e}  {ae.mean():12.6e}")

fig, axes = plt.subplots(3, n_snaps, figsize=(3.2 * n_snaps, 9), squeeze=False)

for col, ti in enumerate(snap_indices):
    pred = pred_all[0, ti, :, 0]
    true = true_all[0, ti, :, 0]
    err = np.abs(pred - true)
    t_val = ti / max(n_times - 1, 1) * T_end

    vmin, vmax = true.min(), true.max()

    # Row 0: exact
    tc0 = axes[0, col].tripcolor(triang, true, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tc0, ax=axes[0, col], shrink=0.8)
    axes[0, col].set_title(f"Exact  t={t_val:.2f}", fontsize=8)
    axes[0, col].set_aspect("equal")

    # Row 1: predicted
    tc1 = axes[1, col].tripcolor(triang, pred, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tc1, ax=axes[1, col], shrink=0.8)
    axes[1, col].set_title(f"PINN  t={t_val:.2f}", fontsize=8)
    axes[1, col].set_aspect("equal")

    # Row 2: pointwise error
    tc2 = axes[2, col].tripcolor(triang, err, shading="gouraud", cmap="hot")
    fig.colorbar(tc2, ax=axes[2, col], shrink=0.8)
    axes[2, col].set_title(f"|err|  t={t_val:.2f}", fontsize=8)
    axes[2, col].set_aspect("equal")

# Remove tick labels on interior plots to save space
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Row labels
for ax, label in zip(axes[:, 0], ["Exact", "PINN", "|Error|"]):
    ax.set_ylabel(label, fontsize=10, fontweight="bold")

# Global error summary
l2_err = np.sqrt(np.mean((pred_all - true_all) ** 2))
linf_err = np.max(np.abs(pred_all - true_all))
print(f"\nGlobal relative L² error:  {l2_err / (np.sqrt(np.mean(true_all**2)) + 1e-12):.6e}")
print(f"Global max pointwise error: {linf_err:.6e}")

plt.suptitle(f"Allen–Cahn 2-D  (ε = {eps})  —  Exact vs PINN", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{dire}/comparison.png", dpi=150, bbox_inches="tight")
print(f"\nPlots saved to {dire}/")
