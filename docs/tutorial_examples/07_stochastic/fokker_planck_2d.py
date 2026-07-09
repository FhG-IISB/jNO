# --8<-- [start:code]
"""07 — 2-D Fokker–Planck on a disc  (Shape + RAD resampling + residual tracker)"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# --8<-- [start:setup]
# Disc of radius 3 centred at the origin — captures the Gaussian's effective support.
domain = jno.Shape.disk(0, 0, 3.0, size=0.25).domain()
x, y, _ = domain.variable("interior")
xb, yb, _ = domain.variable("boundary")

p_exact = jno.np.exp(-(x**2 + y**2)) / π
p_exact_bc = jno.np.exp(-(xb**2 + yb**2)) / π
# --8<-- [end:setup]

# --8<-- [start:residual]
net = jno.nn(foundax.mlp(in_features=2, hidden_dims=64, num_layers=5, key=jax.random.PRNGKey(0)))
# optax.chain: global-norm clipping in front of Adam — the stiff advection–diffusion
# residual produces occasional huge gradient spikes; clipping caps their global norm so
# a single bad batch can't blow up the Adam step and destabilise training.
net.optimizer(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.exponential_decay(1e-3, 2000, 0.5, end_value=1e-5)),
    )
)

p = net(x, y).scalar.bind(x=x, y=y)
prob_flux = jno.np.vector(x * p, y * p)  # OU drift flux as a VectorView
drift = prob_flux.div(x, y)  # ∇·(b·p)
diff = 0.5 * (p.xx + p.yy)  # ½ ∆p
fp = drift + diff  # residual = 0
# --8<-- [end:residual]

# --8<-- [start:constraints]
norm = p.integrate() - 1.0  # ∬ p dx dy = 1
p_bc = net(xb, yb) - (p_exact_bc + jno.noise.gaussian(std=1e-4))
# --8<-- [end:constraints]

# --8<-- [start:solve]
residuals = jno.trackers.residual_stats(interval=1000)
crux = jno.core([fp.mse, norm.mse, p_bc.mse])
crux.solve(15_000, callbacks=[residuals])
# --8<-- [end:solve]

# --8<-- [start:eval]
_p, _p_exact = crux.eval([p, p_exact])
rel_l2 = float(jnp.linalg.norm(_p - _p_exact) / (jnp.linalg.norm(_p_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
if residuals.value is not None:
    print(f"Per-constraint max residuals: {residuals.value['maxes']}")
# --8<-- [end:eval]

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"07_stochastic/fokker_planck_2d.py | epochs=15000 | rel_L2={rel_l2:.6e}\n")

# --8<-- [start:assert]
assert rel_l2 < 3e-1, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:assert]
# --8<-- [end:code]

# ── Figure: computed density vs analytic Gaussian + error, on the disc ────────
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

_xn, _yn, _pn, _pe = crux.eval([x, y, p, p_exact])
_xn = np.asarray(_xn).ravel()
_yn = np.asarray(_yn).ravel()
_pn = np.asarray(_pn).ravel()
_pe = np.asarray(_pe).ravel()
_err = _pn - _pe

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
vmax = max(float(_pn.max()), float(_pe.max()))
im0 = axes[0].tripcolor(_xn, _yn, np.clip(_pn, 0.0, None), cmap="cividis", vmin=0.0, vmax=vmax, shading="gouraud")
axes[0].set_title("computed density  p(x, y)")
fig.colorbar(im0, ax=axes[0], shrink=0.8)
im1 = axes[1].tripcolor(_xn, _yn, _pe, cmap="cividis", vmin=0.0, vmax=vmax, shading="gouraud")
axes[1].set_title(r"analytic  $e^{-(x^2+y^2)}/\pi$")
fig.colorbar(im1, ax=axes[1], shrink=0.8)
elim = float(np.abs(_err).max())
im2 = axes[2].tripcolor(_xn, _yn, _err, cmap="RdBu_r", vmin=-elim, vmax=elim, shading="gouraud")
axes[2].set_title(f"error  (rel-L2 = {rel_l2:.2e})")
fig.colorbar(im2, ax=axes[2], shrink=0.8)
for ax in axes:
    ax.set_aspect("equal")
    ax.set_axis_off()

fig.savefig(Path(__file__).parents[2] / "assets" / "fokker_planck_2d.png")
