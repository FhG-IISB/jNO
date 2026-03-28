"""03 — 2-D Allen–Cahn equation  (manufactured-solution verification)

Problem (Allen–Cahn with source)
---------------------------------
    ∂u/∂t = ε² ∇²u + u − u³ + f(x,y,t),   (x,y) ∈ [0,1]²,  t ∈ [0,1]

Manufactured solution
---------------------
    u(x,y,t) = e^{−t} sin(πx) sin(πy)

This automatically satisfies homogeneous Dirichlet BCs on ∂[0,1]².
The source term is computed by substitution:

    f = u_t − ε² ∇²u − u + u³
      = e^{−t} sin(πx) sin(πy) (2ε²π² − 2)
        + e^{−3t} sin³(πx) sin³(πy)

Parameters: ε = 0.1  (interface width)
"""

import copy
import jax
import jno
import jno.jnp_ops as jnn
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from jno import LearningRateSchedule as lrs

π = jnn.pi
sin = jnn.sin
exp = jnn.exp
dire = jno.setup(__file__)

eps = 0.1
T_end = 1.0

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, 10),
)
x, y, t = domain.variable("interior")

# ── Manufactured solution + source ───────────────────────────────────────────
S = sin(π * x) * sin(π * y)
u_exact = exp(-t) * S

coeff = 2 * eps**2 * π**2 - 2
source = exp(-t) * S * coeff + exp(-3 * t) * S**3

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.nn.deeponet(
    n_sensors=1,
    sensor_channels=1,
    coord_dim=2,
    basis_functions=40,
    hidden_dim=40,
    n_layers=3,
    key=jax.random.PRNGKey(42),
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(5_000, 300, 1e-3, 1e-5))

u = net(t, jnn.concat([x, y])) * x * (1 - x) * y * (1 - y)

# ── PDE residual ──────────────────────────────────────────────────────────────
pde = jnn.grad(u, t) - eps**2 * jnn.laplacian(u, [x, y]) - u + u**3 - source

# ── Initial condition  (t=0 via 0*t trick) ──────────────────────────────────
u_at_0 = net(0 * t, jnn.concat([x, y])) * x * (1 - x) * y * (1 - y)
ini = u_at_0 - sin(π * x) * sin(π * y)

error = jnn.tracker((u - u_exact).mse, interval=200)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse, error], domain)

print(f"Allen–Cahn 2-D  (ε={eps})")

history = crux.solve(5_000)
history.plot(f"{dire}/training_history.png")

# ── Compare vs exact  ─────────────────────────────────────────────────────────
pts = np.array(domain.context["interior"][0, 0])
xs, ys = pts[:, 0], pts[:, 1]
triang = tri.Triangulation(xs, ys)
time_values = np.array(domain.context["__time__"]).reshape(-1)


def eval_snapshots(expr):
    values = []
    for ti in range(len(time_values)):
        sub_domain = copy.deepcopy(domain)
        sub_domain.context["__time__"] = np.asarray(domain.context["__time__"])[ti : ti + 1]
        sub_domain.context["interior"] = np.asarray(domain.context["interior"])[:, ti : ti + 1, :, :]
        values.append(np.array(crux.eval(expr, domain=sub_domain))[0, :, 0])
    return np.stack(values, axis=0)


pred_all = eval_snapshots(u)
true_all = eval_snapshots(u_exact)
n_times = len(time_values)

print(f"\n{'t':>6}  {'rel L²':>12}  {'max |err|':>12}")
print("─" * 36)
for ti in range(n_times):
    t_val = float(time_values[ti])
    p = pred_all[ti, :]
    r = true_all[ti, :]
    l2_rel = np.sqrt(np.mean((p - r) ** 2)) / (np.sqrt(np.mean(r**2)) + 1e-12)
    print(f"{t_val:6.3f}  {l2_rel:12.4e}  {np.abs(p - r).max():12.4e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
snap_idx = [0, n_times // 2, n_times - 1]
fig, axes = plt.subplots(3, len(snap_idx), figsize=(4 * len(snap_idx), 9))

for col, ti in enumerate(snap_idx):
    t_val = float(time_values[ti])
    p = pred_all[ti, :]
    r = true_all[ti, :]
    e = np.abs(p - r)
    vmin, vmax = r.min(), r.max()

    for row, (data, cmap, ttl) in enumerate(
        [
            (r, "viridis", f"Exact  t={t_val:.2f}"),
            (p, "viridis", f"PINN  t={t_val:.2f}"),
            (e, "hot", f"|err|  t={t_val:.2f}"),
        ]
    ):
        kw = dict(shading="gouraud", cmap=cmap)
        if row < 2:
            kw.update(vmin=vmin, vmax=vmax)
        tc = axes[row, col].tripcolor(triang, data, **kw)
        fig.colorbar(tc, ax=axes[row, col], shrink=0.8)
        axes[row, col].set_title(ttl, fontsize=8)
        axes[row, col].set_aspect("equal")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

for ax, lbl in zip(axes[:, 0], ["Exact", "PINN", "|Error|"]):
    ax.set_ylabel(lbl, fontsize=10, fontweight="bold")

plt.suptitle(f"Allen–Cahn 2-D  ε={eps}", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{dire}/comparison.png", dpi=150, bbox_inches="tight")
print(f"\nSaved to {dire}/")
