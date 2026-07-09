# --8<-- [start:code]
"""11 — Fourier Neural Operator 2D for Poisson operator learning"""

import foundax
import jax
import numpy as np
import optax
from create_domain import build_domain_from_arrays, generate_poisson_data

import jno

KEY = jax.random.PRNGKey(0)
GRID = 16
SAMPLES = 256
EPOCHS = 600
BATCH = 32

# ── Dataset ──────────────────────────────────────────────────────────────────
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")
_u = domain.variable("_u")

# ── Model ────────────────────────────────────────────────────────────────────
u = jno.nn(
    foundax.fno2d(
        in_features=1,
        hidden_channels=16,
        n_modes=6,
        d_vars=1,
        n_layers=2,
        n_steps=1,
        d_model=(GRID, GRID),
        norm="layer",
        linear_conv=True,
        key=KEY,
    )
)
u.optimizer(
    optax.chain(
        optax.clip_by_global_norm(1e-3),
        optax.adamw(optax.cosine_decay_schedule(5e-4, EPOCHS, alpha=1e-7 / 5e-4), weight_decay=1e-6),
    )
)

# ── Supervised loss + solve ─────────────────────────────────────────────────
crux = jno.core([(_u - u(_f)).mse])
crux.solve(epochs=EPOCHS, batchsize=BATCH)

# ── Held-out evaluation on unseen forcings (different RNG seed) ───────────────
N_TEST = 8
f_test, u_test = generate_poisson_data(N_TEST, GRID, n_modes=5, alpha=1.5, seed=123)
test_dom = build_domain_from_arrays(f_test, u_test, GRID)

# the trained operator's OWN prediction on the held-out forcings
pred = np.asarray(crux.eval([u(test_dom.variable("_f"))], domain=test_dom)).reshape(N_TEST, GRID, GRID)
truth = np.asarray(u_test).reshape(N_TEST, GRID, GRID)

rel = np.linalg.norm((pred - truth).reshape(N_TEST, -1), axis=1) / np.linalg.norm(truth.reshape(N_TEST, -1), axis=1)
sample = int(np.argmin(np.abs(rel - rel.mean())))  # a representative (near-mean) sample
print(f"held-out per-sample rel-L2: {np.array2string(rel, precision=3)}")
print(f"held-out mean rel-L2 = {float(rel.mean()):.4f}  (representative sample {sample}: {float(rel[sample]):.4f})")
assert rel.mean() < 0.2, float(rel.mean())
# --8<-- [end:code]

# ── Figure ────────────────────────────────────────────────────────────────────
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "figure.dpi": 120,
    }
)

p, t = pred[sample], truth[sample]
err = p - t
extent = [0.0, 1.0, 0.0, 1.0]
vmin = float(min(p.min(), t.min()))
vmax = float(max(p.max(), t.max()))
m = float(np.abs(err).max())

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
im0 = ax[0].imshow(p, origin="lower", extent=extent, cmap="cividis", vmin=vmin, vmax=vmax, aspect="equal")
ax[0].set_title("FNO prediction")
fig.colorbar(im0, ax=ax[0], shrink=0.8)
im1 = ax[1].imshow(t, origin="lower", extent=extent, cmap="cividis", vmin=vmin, vmax=vmax, aspect="equal")
ax[1].set_title("ground-truth solution")
fig.colorbar(im1, ax=ax[1], shrink=0.8)
im2 = ax[2].imshow(err, origin="lower", extent=extent, cmap="RdBu_r", vmin=-m, vmax=m, aspect="equal")
ax[2].set_title(f"pointwise error  (rel-$L^2$ = {float(rel[sample]):.3f})")
fig.colorbar(im2, ax=ax[2], shrink=0.8)
for a in ax:
    a.set_axis_off()

fig.savefig(Path(__file__).parents[2] / "assets" / "fno_poisson_2d.png")
