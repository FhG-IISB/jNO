"""07 — 2-D Fokker–Planck on a disc  (shapely + RAD resampling + residual tracker)

Stochastic process (Itô SDE)
    dX = −X dt + dW₁ ,    dY = −Y dt + dW₂

Steady-state Fokker–Planck PDE
    ∂(x p)/∂x + ∂(y p)/∂y + ½ ∆p = 0   on the disc of radius 3,    p ≈ 0 on ∂Ω

Analytical stationary distribution
    p∞(x, y) = (1/π) exp(−x² − y²)

Showcases
---------
* ``jno.domain(Point(...).buffer(r))`` — non-rectangular shapely disc, meshed
  inline via ``mesh_size=``.
* ``jno.np.vector(...).div(x, y)`` — drift flux as a typed VectorView.
* ``jno.trackers.residual_stats`` — per-constraint mean/std/max residuals so
  the user can spot loss imbalance during training.
* ``jno.noise.gaussian`` — fresh boundary-observation noise each epoch.
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax
from shapely.geometry import Point

import jno

π = jno.np.pi

# --8<-- [start:setup]
# Disc of radius 3 centred at the origin — captures the Gaussian's effective support.
domain = jno.domain(Point(0, 0).buffer(3.0), mesh_size=0.25)
x, y, _ = domain.variable("interior")
xb, yb, _ = domain.variable("boundary")

p_exact = jno.np.exp(-(x**2 + y**2)) / π
p_exact_bc = jno.np.exp(-(xb**2 + yb**2)) / π
# --8<-- [end:setup]

# --8<-- [start:residual]
net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=64, num_layers=5, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 2000, 0.5, end_value=1e-5)))

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
crux = jno.core([fp.mse, norm.mse, p_bc.mse], domain)
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
