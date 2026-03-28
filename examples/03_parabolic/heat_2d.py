"""03 — 2-D heat equation  (parabolic, time-dependent)

Problem
-------
    ∂u/∂t = α ∇²u,   (x,y) ∈ [0,1]²,  t ∈ [0, 0.5]
    u = 0 on ∂Ω  (homogeneous Dirichlet BCs)
    u(x,y,0) = sin(πx) sin(πy)

Analytical solution
-------------------
    u(x,y,t) = exp(−2απ²t) sin(πx) sin(πy)

The x(1−x)y(1−y) factor in the ansatz hard-enforces the Dirichlet BCs on the
unit-square boundary for all times.  The initial condition is a soft constraint
evaluated on the "initial" domain tag.
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
dire = jno.setup(__file__)

α = 0.1  # thermal diffusivity
T_end = 0.5  # final time
N_t = 8  # number of time slices

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=(0.05)),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t = domain.variable("interior")
x0, y0, _ = domain.variable("initial")
domain.summary()
# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jnn.exp(-2 * α * π**2 * t) * jnn.sin(π * x) * jnn.sin(π * y)

# ── Network ───────────────────────────────────────────────────────────────────
net = jnn.nn.deeponet(
    n_sensors=1,
    sensor_channels=1,
    coord_dim=2,
    basis_functions=40,
    hidden_dim=40,
    n_layers=3,
    key=jax.random.PRNGKey(0),
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10_000, 1_000, 1e-3, 1e-5))
net.summary()
xy = jnn.concat([x, y])
xy0 = jnn.concat([x0, y0])

u = net(t, xy) * x * (1 - x) * y * (1 - y)
u0 = net(0.0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)

# ── Constraints ───────────────────────────────────────────────────────────────
pde = jnn.grad(u, t) - α * jnn.laplacian(u, [x, y])
ini = u0 - jnn.sin(π * x0) * jnn.sin(π * y0)
error = jnn.tracker((u - u_exact).mse, interval=200)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse, error], domain).print_shapes()
history = crux.solve(10_000)
history.plot(f"{dire}/training_history.png")

# ── Plot ──────────────────────────────────────────────────────────────────────
spacetime_pts = np.array(domain.context["interior"][0])
time_values = np.array(domain.context["__time__"]).reshape(-1)

pts = spacetime_pts[0]
xs, ys = pts[:, 0], pts[:, 1]
triang = tri.Triangulation(xs, ys)

n_t = time_values.shape[0]

snap_idx = [0, n_t // 2, n_t - 1]
fig, axes = plt.subplots(3, len(snap_idx), figsize=(4 * len(snap_idx), 9))

for col, ti in enumerate(snap_idx):
    t_val = float(time_values[ti])

    sub_domain = copy.deepcopy(domain)
    sub_domain.context["__time__"] = np.asarray(domain.context["__time__"])[ti : ti + 1]
    sub_domain.context["interior"] = np.asarray(domain.context["interior"])[:, ti : ti + 1, :, :]

    p = np.array(crux.eval(u, domain=sub_domain))[0, :, 0]
    r = np.array(crux.eval(u_exact, domain=sub_domain))[0, :, 0]
    e = np.abs(p - r)
    vmin, vmax = r.min(), r.max()

    for row, (data, cmap, title) in enumerate(
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
        axes[row, col].set_title(title, fontsize=8)
        axes[row, col].set_aspect("equal")

for ax, lbl in zip(axes[:, 0], ["Exact", "PINN", "|Error|"]):
    ax.set_ylabel(lbl, fontsize=10, fontweight="bold")

plt.suptitle(f"2-D heat equation  α={α}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{dire}/comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved to {dire}/")
