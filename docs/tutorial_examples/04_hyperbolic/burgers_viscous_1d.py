# --8<-- [start:code]
"""04 — 1-D viscous Burgers equation  (manufactured solution)"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
ν = 0.05
T_end = 1.0

# ── Domain (1-D space × time) ─────────────────────────────────────────────────
domain = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.1).domain(time=(0, T_end, 4))
# RAD adaptive resampling concentrates collocation points at the steep moving front.
x, t = domain.variable(
    "interior",
    resampling_strategy=jno.sampler.rad(resample_every=200, resample_fraction=0.2, start_epoch=500, k=10),
)
x0, t0 = domain.variable("initial")

# ── Manufactured solution + source term ──────────────────────────────────────
u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source = jno.np.exp(-t) * (ν * π**2 - 1) * jno.np.sin(π * x) + (π / 2) * jno.np.exp(-2 * t) * jno.np.sin(2 * π * x)

# ── Network  (hard Dirichlet BCs via the x(1-x) factor) ──────────────────────
net = jno.nn(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=4,
        basis_functions=64,
        hidden_dim=48,
        key=jax.random.PRNGKey(3),
    )
)
net.optimizer(
    optax.adam(
        optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=1e-3, warmup_steps=10, decay_steps=5000, end_value=1e-5
        )
    )
)

u = net(t, x) * x * (1 - x)

# ── PDE residual:  u_t + u u_x − ν u_xx − f = 0 ──────────────────────────────
u_x = u.d(x)
pde = u.d(t) + u * u_x - ν * u_x.d(x) - source

# ── Initial condition ────────────────────────────────────────────────────────
u_0 = net(t0, x0) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

# ── Solve ────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:code]
print(f"Relative L2 error: {rel_l2:.4e}   (manufactured u = e^-t sin(pi x))")

# ── Figure: predicted vs manufactured solution at the 4 time levels ───────────
# Each time level is one DeepONet input sample (n_sensors=1, branch input = t).
# Evaluate the trained network on a finer x-grid via a domain override.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

fine = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.02).domain(time=(0, T_end, 4))
_xg, _tg, _ug, _ueg = crux.eval([x, t, u, u_exact], domain=fine, min_consecutive=4)
_xg = np.asarray(_xg).reshape(4, -1)  # (n_t, n_x)
_ug = np.asarray(_ug).reshape(4, -1)
_ueg = np.asarray(_ueg).reshape(4, -1)
_tlvl = np.asarray(_tg).reshape(4)
order = np.argsort(_xg[0])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, 4))
for k in range(4):
    xs = _xg[k][order]
    axes[0].plot(xs, _ug[k][order], color=colors[k], label=f"t={_tlvl[k]:.2f}")
    axes[0].plot(xs, _ueg[k][order], "--", color=colors[k], lw=1)
    axes[1].plot(xs, np.abs(_ug[k][order] - _ueg[k][order]), color=colors[k], label=f"t={_tlvl[k]:.2f}")
axes[0].plot([], [], "k-", label="jNO")
axes[0].plot([], [], "k--", label="exact")
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x, t)")
axes[0].set_title("predicted vs manufactured solution")
axes[0].legend(fontsize=8, ncol=2)
axes[1].set_xlabel("x")
axes[1].set_ylabel("|u − u_exact|")
axes[1].set_title(f"pointwise error (rel-L2 = {rel_l2:.2e})")
axes[1].set_yscale("log")
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", alpha=0.3)
fig.savefig(Path(__file__).parents[2] / "assets" / "burgers_viscous_1d.png")
