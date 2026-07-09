# --8<-- [start:code]
"""11 — DeepONet 2D for parametric Poisson"""

import foundax
import jax
import numpy as np
import optax
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jno

KEY = jax.random.PRNGKey(0)
N_SAMPLES = 50
EPOCHS = 2_000

# ── Parametric domain — replicate one mesh across N_SAMPLES random k values ──
dom = N_SAMPLES * jno.Shape.rect(0, 0, 2, 1, size=0.05).domain()
x, y, _ = dom.variable("interior")

k_values = jax.random.uniform(KEY, shape=(N_SAMPLES, 1, 1), minval=0.5, maxval=1.5)
k = dom.variable("k", k_values)

# ── Network ──────────────────────────────────────────────────────────────────
net = jno.nn(
    foundax.deeponet(
        n_sensors=1,  # branch input is the scalar k
        coord_dim=2,  # trunk input is (x, y)
        basis_functions=32,
        hidden_dim=128,
        activation=jax.numpy.tanh,
        key=KEY,
    )
)
net.optimizer(optax.adam(optax.cosine_decay_schedule(1e-3, EPOCHS, alpha=1e-5 / 1e-3)))

# ── Hard BC ansatz + PDE residual ────────────────────────────────────────────
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.d2(x) + u.d2(y)) + 1.0

# ── Solve ────────────────────────────────────────────────────────────────────
crux = jno.core(constraints=[pde.mse])
crux.solve(epochs=EPOCHS, batchsize=32)

# ── Held-out evaluation: query the trained operator at an unseen k, compare to
#    a finite-difference reference solution of  k Δu + 1 = 0,  u = 0 on ∂Ω. ────
K_TEST = 1.234  # a coefficient NOT in the training draw
GX, GY = 80, 40  # interior grid on [0, 2] x [0, 1]
xs = np.linspace(0.0, 2.0, GX + 1)[1:-1]
ys = np.linspace(0.0, 1.0, GY + 1)[1:-1]
XX, YY = np.meshgrid(xs, ys, indexing="xy")
coords = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)

# the trained operator's OWN output at the held-out k (hard-BC ansatz applied)
raw = net.module(jax.numpy.array([K_TEST], dtype=jax.numpy.float32), jax.numpy.asarray(coords))
ansatz = coords[:, 0] * (2 - coords[:, 0]) * coords[:, 1] * (1 - coords[:, 1])
u_pred = (np.asarray(raw) * ansatz).reshape(XX.shape)

# finite-difference reference (5-point Laplacian, homogeneous Dirichlet)
nx, ny = len(xs), len(ys)
hx, hy = xs[1] - xs[0], ys[1] - ys[0]
Lx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nx, nx)) / hx**2
Ly = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(ny, ny)) / hy**2
A = sp.kron(Ly, sp.identity(nx)) + sp.kron(sp.identity(ny), Lx)
u_ref = spla.spsolve(A.tocsr(), np.full(nx * ny, -1.0 / K_TEST)).reshape(XX.shape)

rel_l2 = float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
print(f"held-out k={K_TEST}: rel-L2 vs finite-difference reference = {rel_l2:.4f}")
assert rel_l2 < 0.15, rel_l2
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

err = u_pred - u_ref
extent = [0.0, 2.0, 0.0, 1.0]
vmax = float(max(u_pred.max(), u_ref.max()))
m = float(np.abs(err).max())

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
im0 = ax[0].imshow(u_pred, origin="lower", extent=extent, cmap="cividis", vmin=0.0, vmax=vmax, aspect="equal")
ax[0].set_title(f"DeepONet prediction  (k = {K_TEST})")
fig.colorbar(im0, ax=ax[0], shrink=0.8)
im1 = ax[1].imshow(u_ref, origin="lower", extent=extent, cmap="cividis", vmin=0.0, vmax=vmax, aspect="equal")
ax[1].set_title("finite-difference reference")
fig.colorbar(im1, ax=ax[1], shrink=0.8)
im2 = ax[2].imshow(err, origin="lower", extent=extent, cmap="RdBu_r", vmin=-m, vmax=m, aspect="equal")
ax[2].set_title(f"pointwise error  (rel-$L^2$ = {rel_l2:.3f})")
fig.colorbar(im2, ax=ax[2], shrink=0.8)
for a in ax:
    a.set_axis_off()

fig.savefig(Path(__file__).parents[2] / "assets" / "deeponet_poisson_2d.png")
