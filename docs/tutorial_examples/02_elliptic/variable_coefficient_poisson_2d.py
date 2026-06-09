"""02 — 2-D variable-coefficient Poisson  (shapely domain + named partials + tracker)

Problem
-------
    −∇·(κ(x, y) ∇u(x, y)) = f(x, y)   on [0, 1]²,    u = 0 on ∂Ω

with κ(x, y) = 1 + x + y and exact solution sin(πx) sin(πy).

Showcases
---------
* ``jno.domain(box(...))`` — shapely-backed geometry, ``mesh_size=`` triggers
  meshing inline.
* ``.scalar.coords(x=x, y=y)`` — registered partial-derivative names; ``.grad()``
  uses them implicitly so ``κ * u.grad()`` reads as ``κ ∇u``.
* ``jno.trackers.gradient_norms`` — per-loss ∇L₂ norms during training,
  exposing the relative loss budgets for free.
"""

from pathlib import Path

import foundax
import jax
import optax
from shapely.geometry import box

import jno

π = jno.np.pi

# --8<-- [start:setup]
domain = jno.domain(box(0, 0, 1, 1), mesh_size=0.05)
x, y, _ = domain.variable("interior")

κ = 1 + x + y
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
# --8<-- [end:setup]

# --8<-- [start:residual]
net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(13)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

# Multiplicative hard BC — the x(1-x)y(1-y) factor's derivatives flow through.
u = (net(x, y) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y)
# −∂_x(κ ∂_x u) − ∂_y(κ ∂_y u) = forcing.  Named-partial syntax reads
# component-by-component as if from the math, including the product rule.
pde = -((κ * u.x).d(x) + (κ * u.y).d(y)) - (
    2 * π**2 * κ * u_exact - π * jno.np.cos(π * x) * jno.np.sin(π * y) - π * jno.np.sin(π * x) * jno.np.cos(π * y)
)
# --8<-- [end:residual]

# --8<-- [start:solve]
grad_norms = jno.trackers.gradient_norms(interval=500)
crux = jno.core([pde.mse])
crux.solve(5_000, callbacks=[grad_norms])
# --8<-- [end:solve]

# --8<-- [start:eval]
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
if grad_norms.value is not None:
    print(f"Final ∇L norm: {grad_norms.value['norms']}")
# --8<-- [end:eval]

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/variable_coefficient_poisson_2d.py | epochs=5000 | rel_L2={rel_l2:.6e}\n")

# --8<-- [start:assert]
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:assert]
