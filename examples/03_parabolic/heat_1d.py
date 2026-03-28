"""03 — 1-D heat equation  (parabolic, time-dependent)

Problem
-------
    ∂u/∂t = α ∂²u/∂x²,   x ∈ [0,1],  t ∈ [0, 0.5]
    u(0, t) = u(1, t) = 0          (homogeneous Dirichlet)
    u(x, 0) = sin(πx)              (initial condition)

Analytical solution
-------------------
    u(x, t) = exp(−απ²t) sin(πx)

Network ansatz
--------------
    u ≈ net(t, x) · x (1−x)       — hard-enforces BCs for all t

The initial condition is implemented as a soft constraint evaluated on a
separate "initial" tag.
"""

import copy
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
α = 0.1  # thermal diffusivity
T_end = 0.5  # final time

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.02),
    time=(0, T_end, 10),
)
x, t = domain.variable("interior")
x0, _ = domain.variable("initial")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jnn.exp(-α * π**2 * t) * jnn.sin(π * x)

# ── Network ───────────────────────────────────────────────────────────────────
net = jnn.nn.deeponet(
    n_sensors=1,
    sensor_channels=1,
    coord_dim=1,
    basis_functions=32,
    hidden_dim=32,
    n_layers=3,
    key=jax.random.PRNGKey(0),
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.7, 10_000, 1e-5))

u = net(t, x) * x * (1 - x)  # hard Dirichlet BCs
u0 = net(0.0, x0) * x0 * (1 - x0)

# ── Constraints ───────────────────────────────────────────────────────────────
#   PDE:  u_t − α u_xx = 0
pde = jnn.grad(u, t) - α * jnn.grad(jnn.grad(u, x), x)
#   IC:   u(x, 0) = sin(πx)
ini = u0 - jnn.sin(π * x0)
error = jnn.tracker((u - u_exact).mse, interval=200)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse, error], domain)
history = crux.solve(10_000)
history.plot(f"{dire}/training_history.png")

# ── Plot: solution at selected time slices ────────────────────────────────────
pts = np.array(domain.context["interior"][0, 0, :, 0])  # spatial coords
idx = np.argsort(pts)
xs = pts[idx]
time_values = np.array(domain.context["__time__"]).reshape(-1)
n_t = time_values.shape[0]


def eval_snapshots(expr):
    values = []
    for ti in range(n_t):
        sub_domain = copy.deepcopy(domain)
        sub_domain.context["__time__"] = np.asarray(domain.context["__time__"])[ti : ti + 1]
        sub_domain.context["interior"] = np.asarray(domain.context["interior"])[:, ti : ti + 1, :, :]
        values.append(np.array(crux.eval(expr, domain=sub_domain))[0, :, 0])
    return np.stack(values, axis=0)


pred_all = eval_snapshots(u)
true_all = eval_snapshots(u_exact)

snap_times = [0, n_t // 4, n_t // 2, n_t - 1]
fig, axes = plt.subplots(1, len(snap_times), figsize=(14, 4), sharey=True)

for ax, ti in zip(axes, snap_times):
    t_val = float(time_values[ti])
    p = pred_all[ti, :][idx]
    r = true_all[ti, :][idx]
    ax.plot(xs, r, "--", label="exact")
    ax.plot(xs, p, label="PINN")
    ax.set_title(f"t = {t_val:.3f}")
    ax.set_xlabel("x")
    if ax is axes[0]:
        ax.set_ylabel("u")
        ax.legend()

plt.suptitle(f"1-D heat equation  α={α}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{dire}/solution.png", dpi=150)
print(f"Saved to {dire}/")
