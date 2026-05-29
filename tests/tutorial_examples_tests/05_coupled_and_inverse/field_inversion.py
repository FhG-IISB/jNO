"""05 — Field inversion: recover a spatially-varying coefficient from data

Problem
-------
Identify a coefficient field k(x) > 0 from observations of u(x).

Manufactured solution
---------------------
    k_true(x) = 1.0             (constant, always positive)
    u_true(x) = sin(πx)

    PDE:  k · u'' = f,  where f = k_true · u_true'' = −π² sin(πx)
    BCs:  u = 0  on ∂[0,1]

Technique
---------
Positivity of k is enforced via an exp output transform at the field level:

    k = jno.fn.exp(k_raw(x))     # always > 0, no weight-space distortion

u is a neural network with hard zero BCs via boundary factor.

Loss:
  1. PDE residual   k·u'' − f = 0
  2. Data misfit    u − u_obs = 0
  3. H1 smoothness  |dk/dx|²   (regularisation for the ill-posed inversion)

After training we check:
  - u relative L2 error < 10 %
  - k is positive everywhere (guaranteed by exp)
  - k is close to 1.0 on average (mean absolute deviation < 0.5)
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")

# ── Manufactured source term and observations ─────────────────────────────────
f_pde = -(π**2) * jno.np.sin(π * x)    # k_true · u_true'' = 1 · (−π² sin(πx))
u_obs = jno.np.sin(π * x)              # noiseless observations

# ── Networks ──────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

k_raw = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=16, num_layers=2, key=k1))
k_raw.optimizer(optax.adam(1e-3))

u_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k2))
u_net.optimizer(optax.adam(1e-3))

# ── Fields ────────────────────────────────────────────────────────────────────
k = jno.fn.exp(k_raw(x))          # exp output transform: k > 0 by construction
u = u_net(x) * x * (1 - x)        # hard zero Dirichlet BCs at x=0 and x=1

# ── Losses ────────────────────────────────────────────────────────────────────
pde  = k * u.dd(x) - f_pde        # PDE residual
data = u - u_obs                   # data misfit
reg  = jno.fn.regularize.smooth(k, x)  # H1 smoothness on k

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, data.mse, reg.mean], domain=domain)
crux.solve(5_000)

# ── Evaluation ────────────────────────────────────────────────────────────────
_u, _k, _u_obs = crux.eval([u, k, u_obs])

rel_l2_u = float(jnp.linalg.norm(_u - _u_obs) / (jnp.linalg.norm(_u_obs) + 1e-8))
k_min    = float(_k.min())
k_mean   = float(_k.mean())

print(f"u  rel-L2 error : {rel_l2_u:.3e}")
print(f"k  min / mean   : {k_min:.3f} / {k_mean:.3f}  (true: >0 / 1.0)")

# ── Assertions ────────────────────────────────────────────────────────────────
assert rel_l2_u < 1e-1, f"u relative L2 error too large: {rel_l2_u:.3e}"
assert k_min > 0,        f"k must be positive everywhere (got min={k_min:.4f})"
assert abs(k_mean - 1.0) < 0.5, f"k mean should be near 1.0 (got {k_mean:.3f})"

# ── Result tracking ───────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"05_coupled_and_inverse/field_inversion.py"
        f" | epochs=5000"
        f" | rel_L2_u={rel_l2_u:.6e}"
        f" | k_min={k_min:.4f}"
        f" | k_mean={k_mean:.4f}\n"
    )
