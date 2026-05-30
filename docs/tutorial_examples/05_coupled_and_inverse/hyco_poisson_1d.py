"""05 — HyCo: Hybrid-Cooperative Learning for PINNs (1D Poisson)

Based on: Liverani, Steynberg & Zuazua (2025) — arXiv:2509.14123

Idea
----
Train two networks that cooperate via a shared interaction loss:

  u_phy  — physical model: enforces the PDE residual
  u_syn  — synthetic model: fits sparse, noisy sensor observations

Both models are encouraged to agree at interior collocation points through
a mutual alignment term.  ``jno.fn.stop_gradient`` ensures that, when
optimising u_phy via the interaction, gradients do NOT flow into u_syn
(and vice versa), so each model only updates its own parameters.

Loss decomposition
------------------
  Physical model optimises:   L_pde  +  β · L_int_phy
  Synthetic model optimises:  α · L_data  +  β · L_int_syn

  L_pde      = MSE of PDE residual at interior points          (u_phy only)
  L_data     = MSE of u_syn vs. observations at sensor points  (u_syn only)
  L_int_phy  = MSE(u_phy, stop_gradient(u_syn))  at interior  (u_phy only)
  L_int_syn  = MSE(u_syn, stop_gradient(u_phy))  at interior  (u_syn only)

Because jno.core sums all four terms, gradients flow as:
  L_pde      → u_phy params          (u_syn not in expression)
  L_int_phy  → u_phy params only     (u_syn path blocked by stop_gradient)
  L_data     → u_syn params          (u_phy not in expression)
  L_int_syn  → u_syn params only     (u_phy path blocked by stop_gradient)

Problem
-------
    u'' + π² sin(πx) = 0,   u(0) = u(1) = 0   on [0, 1]
    Exact solution: u(x) = sin(πx)

    Sensors: 7 observations with additive Gaussian noise (σ = 0.05)
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

π = jno.np.pi

# ── Reproducible noise ────────────────────────────────────────────────────────
rng = np.random.default_rng(0)

# ── Domain ────────────────────────────────────────────────────────────────────
# Standard 1D line domain — provides the dense interior collocation points
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.02))
x, _ = domain.variable("interior")

# Sparse sensor observations — noisy samples of the true solution
x_sen = np.linspace(0.1, 0.9, 7).reshape(-1, 1)
u_sen = np.sin(np.pi * x_sen) + rng.normal(0, 0.05, x_sen.shape)

# Register sensor coordinates on the same domain as a named point set
(x_s,) = domain.variable("sensors", sample=x_sen, split=True, point_data=True)

# ── Networks ──────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

u_phy_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k1))
u_syn_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k2))
for net in [u_phy_net, u_syn_net]:
    net.optimizer(optax.adam(1e-3))

# Fields — boundary factor enforces u(0) = u(1) = 0 exactly
u_phy = u_phy_net(x) * x * (1 - x)  # physical model at collocation pts
u_syn = u_syn_net(x) * x * (1 - x)  # synthetic model at collocation pts

# ── Losses ────────────────────────────────────────────────────────────────────

# 1. PDE residual — drives u_phy to satisfy the Poisson equation
L_pde = (u_phy.dd(x) + π**2 * jno.np.sin(π * x)).mse

# 2. Data fidelity — drives u_syn to match the noisy sensor readings
u_syn_s = u_syn_net(x_s) * x_s * (1 - x_s)  # at sensor pts
u_obs = jno.np.array(u_sen)  # Constant: noisy observations
L_data = (u_syn_s - u_obs).mse

# 3. Interaction — mutual alignment at collocation points
#    stop_gradient blocks gradient flow into the "reference" model so that
#    each interaction term only updates the "student" model's parameters.
L_int_phy = (u_phy - jno.fn.stop_gradient(u_syn)).mse  # u_phy learns from u_syn
L_int_syn = (u_syn - jno.fn.stop_gradient(u_phy)).mse  # u_syn learns from u_phy

# ── Solve — both models update simultaneously in each step ────────────────────
α, β = 1.0, 1.0  # weighting: interaction vs. primary objectives

crux = jno.core(
    [L_pde, β * L_int_phy, α * L_data, β * L_int_syn],
    domain,
)
crux.solve(3_000)

# ── Evaluation ────────────────────────────────────────────────────────────────
u_exact_expr = jno.np.sin(π * x)
_u_phy, _u_syn, _u_exact = crux.eval([u_phy, u_syn, u_exact_expr])

rel_phy = float(jnp.linalg.norm(_u_phy - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))
rel_syn = float(jnp.linalg.norm(_u_syn - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))

print(f"u_phy rel-L2 error : {rel_phy:.3e}  (physics model)")
print(f"u_syn rel-L2 error : {rel_syn:.3e}  (synthetic model)")

# ── Assertions ────────────────────────────────────────────────────────────────
assert rel_phy < 0.05, f"Physical model error too large: {rel_phy:.3e}"
assert rel_syn < 0.10, f"Synthetic model error too large: {rel_syn:.3e}"

# ── Result tracking ───────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"05_coupled_and_inverse/hyco_poisson_1d.py"
        f" | epochs=3000 | alpha={α} | beta={β}"
        f" | rel_L2_phy={rel_phy:.6e}"
        f" | rel_L2_syn={rel_syn:.6e}\n"
    )
