"""03 - 2-D reaction-diffusion equation

Problem
-------
    u_t - nu Delta u + lambda u = f(x, y, t),   (x, y) in [0, 1]^2, t in [0, 1]
    u = 0 on the boundary
    u(x, y, 0) = sin(pi x) sin(pi y)

Analytical solution
-------------------
    u(x, y, t) = exp(-t) sin(pi x) sin(pi y)
"""

import copy
import os

import jax
import jno
import jno.jnp_ops as jnn
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import optax
from jno import LearningRateSchedule as lrs


def pick(default, test):



pi = jnn.pi
dire = jno.setup(__file__)

nu = 0.1
lam = 0.5
T_end = 1.0
N_t = pick(8, 4)

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=pick(0.05, 0.2)),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t = domain.variable("interior")
x0, y0, _ = domain.variable("initial")
t_like = 0 * x + t
zero0 = 0 * x0

u_exact = jnn.exp(-t) * jnn.sin(pi * x) * jnn.sin(pi * y)
source = (-1 + 2 * nu * pi**2 + lam) * u_exact

net = jnn.nn.mlp(in_features=3, hidden_dims=48, num_layers=4, key=jax.random.PRNGKey(21))
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(pick(12_000, 10), pick(500, 1), 1e-3, 1e-5))

u = net(x, y, t_like) * x * (1 - x) * y * (1 - y)
u0 = net(x0, y0, zero0) * x0 * (1 - x0) * y0 * (1 - y0)

pde = jnn.grad(u, t) - nu * jnn.laplacian(u, [x, y]) + lam * u - source
ini = u0 - jnn.sin(pi * x0) * jnn.sin(pi * y0)
error = jnn.tracker((u - u_exact).mse, interval=pick(200, 1))

crux = jno.core([pde.mse, ini.mse, error], domain)
history = crux.solve(pick(12_000, 10))

    # Always run full mode
    print("Smoke test completed for reaction_diffusion_2d.py")
else:
    history.plot(f"{dire}/training_history.png")

    pts = np.array(domain.context["interior"][0, 0])
    xs, ys = pts[:, 0], pts[:, 1]
    triang = tri.Triangulation(xs, ys)
    time_values = np.array(domain.context["__time__"]).reshape(-1)
    snap_idx = [0, len(time_values) // 2, len(time_values) - 1]

    fig, axes = plt.subplots(3, len(snap_idx), figsize=(4 * len(snap_idx), 9))
    for col, ti in enumerate(snap_idx):
        sub_domain = copy.deepcopy(domain)
        sub_domain.context["__time__"] = np.asarray(domain.context["__time__"])[ti : ti + 1]
        sub_domain.context["interior"] = np.asarray(domain.context["interior"])[:, ti : ti + 1, :, :]
        pred = np.array(crux.eval(u, domain=sub_domain))[0, :, 0]
        true = np.array(crux.eval(u_exact, domain=sub_domain))[0, :, 0]
        err = np.abs(pred - true)
        t_val = float(time_values[ti])
        vmin, vmax = true.min(), true.max()

        for row, (data, cmap, title) in enumerate(
            [
                (true, "viridis", f"Exact t={t_val:.2f}"),
                (pred, "viridis", f"PINN t={t_val:.2f}"),
                (err, "hot", f"|err| t={t_val:.2f}"),
            ]
        ):
            kwargs = dict(shading="gouraud", cmap=cmap)
            if row < 2:
                kwargs.update(vmin=vmin, vmax=vmax)
            tc = axes[row, col].tripcolor(triang, data, **kwargs)
            fig.colorbar(tc, ax=axes[row, col], shrink=0.8)
            axes[row, col].set_aspect("equal")
            axes[row, col].set_title(title, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{dire}/comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {dire}/")
