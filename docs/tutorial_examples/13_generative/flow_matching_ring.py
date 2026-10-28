# --8<-- [start:code]
"""13 — Conditional flow matching: train a velocity, march it as an ODE

The target is a ring of eight Gaussian modes. It is chosen because it fails *loudly* in two
independent ways: a collapsed velocity puts the sample radius somewhere other than 2, and a
mode-dropping one leaves visible gaps around the circle.

Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023 (§3, the conditional objective).
"""

from pathlib import Path

import foundax
import jax
import numpy as np
import optax

import jno

KEY = jax.random.PRNGKey(0)
D = 2  # data dimension
N = 4096  # dataset size
EPOCHS = 4000
POINTS = 512  # collocation points per step — the point axis, not the sample axis

# ── Data: eight modes on a circle of radius 2 ────────────────────────────────
rng = np.random.default_rng(0)
angle = 2 * np.pi * rng.integers(0, 8, N) / 8
data = (np.stack([np.cos(angle), np.sin(angle)], -1) * 2.0 + 0.15 * rng.normal(size=(N, D))).astype(np.float32)

domain = jno.domain.from_array({"data": data})
d0, d1, _ = domain.variable("data")
x1 = jno.np.concatenate([d0, d1])  # the data sample

# ── The probability path, written as the formula it is ───────────────────────
# `like=x1` takes the draw's shape from x1 itself, so the prior tracks solve(points=...)
# automatically. A shape written down by hand would go stale and fail by broadcasting.
x0 = jno.noise.gaussian(std=1.0, like=x1)  # prior draw
t = jno.noise.uniform(low=0.0, high=1.0, like=d0)  # path time, one per point

x_t = (1 - t) * x0 + t * x1  # the linear (optimal-transport) path
target = x1 - x0  # its velocity, constant along the path

# ── The velocity network ─────────────────────────────────────────────────────
v = jno.nn(
    foundax.mlp(
        D + 1,  # state, then time
        output_dim=D,
        hidden_dims=128,
        num_layers=3,
        activation=jax.nn.swish,
        key=KEY,
    )
)
v.optimizer(optax.adam(3e-3))

# ── Train: the whole objective is one term ───────────────────────────────────
crux = jno.core([(v(x_t, t) - target).mse])
stats = crux.solve(epochs=EPOCHS, points=POINTS)

final_loss = float(stats.training_logs[-1]["total_loss"][-1])
print(f"final CFM loss: {final_loss:.4f}   (an O(1) plateau is correct — the objective is irreducible)")

# ── Sample: march the trained velocity as an ODE ─────────────────────────────
z = jax.random.normal(jax.random.PRNGKey(1), (2000, D))
path = np.asarray(v.flow(z, time=(0.0, 1.0, 101), save=True))  # (101, 2000, D)
samples = path[-1]

radius = np.linalg.norm(samples, axis=-1)
occupancy = np.histogram((np.arctan2(samples[:, 1], samples[:, 0]) % (2 * np.pi)) / (2 * np.pi / 8), bins=8, range=(0, 8))[
    0
]
print(f"generated radius: {radius.mean():.3f} ± {radius.std():.3f}   (data: 2.0)")
print(f"mode occupancy:   {occupancy}   (min {occupancy.min()})")

# ── Figure ───────────────────────────────────────────────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "figure.dpi": 120,
    }
)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
lim = 3.2

ax[0].scatter(data[:2000, 0], data[:2000, 1], s=3, c="#1A202C", alpha=0.35, linewidths=0)
ax[0].set_title("data")

# The integrator's own trajectories, every 40th path so the bundle stays legible; the grey dots are
# where those paths started. Nothing here is drawn by hand -- it is `save=True` output.
shown = path[:, ::40, :]
ax[1].plot(shown[..., 0], shown[..., 1], c="#0D9488", lw=0.5, alpha=0.55)
ax[1].scatter(shown[0, :, 0], shown[0, :, 1], s=5, c="#94A3B8", alpha=0.9, linewidths=0, zorder=3)
ax[1].scatter(shown[-1, :, 0], shown[-1, :, 1], s=5, c="#1A202C", alpha=0.9, linewidths=0, zorder=3)
ax[1].set_title("prior (grey) → data (dark), 100 Heun steps")

ax[2].scatter(samples[:, 0], samples[:, 1], s=3, c="#0D9488", alpha=0.35, linewidths=0)
ax[2].set_title(f"generated  (radius {radius.mean():.2f}, all 8 modes)")

for a in ax:
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_axis_off()

fig.savefig(Path(__file__).parents[2] / "assets" / "flow_matching_ring.png")
# --8<-- [end:code]
