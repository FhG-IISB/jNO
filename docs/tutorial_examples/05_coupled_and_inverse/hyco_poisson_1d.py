"""05 — HyCo: Hybrid-Cooperative Learning for PINNs (1-D Poisson)

Based on Liverani, Steynberg & Zuazua (2025) — arXiv:2509.14123.

Idea
----
Train two networks that cooperate via a shared interaction loss:
  * **u_phy** — physical model, enforces the PDE residual
  * **u_syn** — synthetic model, fits sparse noisy sensor observations

Both models are encouraged to agree at interior collocation points via a
mutual alignment term, with ``.stop_gradient`` blocking cross-talk in
the optimiser updates. Alternating updates are expressed through ``substeps``:
each outer epoch runs one ``u_phy`` step followed by one ``u_syn`` step,
each with its own Adam state.

Problem
-------
    u'' + π² sin(πx) = 0,    u(0) = u(1) = 0   on [0, 1]
    Exact solution: u(x) = sin(πx)
    Sensors: 7 noisy observations  (σ = 0.05)
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

π = jno.np.pi
rng = np.random.default_rng(0)

domain = jno.domain.line(mesh_size=0.02)
x, _ = domain.variable("interior")

# Sparse noisy sensors, registered as a named point set on the same domain.
x_sen = np.linspace(0.1, 0.9, 7).reshape(-1, 1)
u_sen = np.sin(np.pi * x_sen) + rng.normal(0, 0.05, x_sen.shape)
(x_s,) = domain.variable("sensors", sample=x_sen, split=True, point_data=True)

# Two networks, one optimiser each.
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
u_phy_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k1))
u_syn_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k2))
for net in (u_phy_net, u_syn_net):
    net.optimizer(optax.adam(1e-3))

# Hard Dirichlet ansatz; .bind(x=x) so the PDE residual reads as u.xx.
u_phy = (u_phy_net(x) * x * (1 - x)).scalar.bind(x=x)
u_syn = (u_syn_net(x) * x * (1 - x)).scalar.bind(x=x)

# Loss components
L_pde = (u_phy.xx + π**2 * jno.np.sin(π * x)).mse
u_syn_at_sensors = u_syn_net(x_s) * x_s * (1 - x_s)
L_data = (u_syn_at_sensors - jno.np.array(u_sen)).mse
L_int_phy = (u_phy - u_syn.stop_gradient).mse
L_int_syn = (u_syn - u_phy.stop_gradient).mse

α, β = 1.0, 1.0
crux = jno.core([L_pde, β * L_int_phy, α * L_data, β * L_int_syn])
# Alternating updates: first u_phy (constraints 0, 1), then u_syn (constraints 2, 3).
crux.solve(3_000, substeps=[[0, 1], [2, 3]])

u_exact_expr = jno.np.sin(π * x)
_u_phy, _u_syn, _u_exact = crux.eval([u_phy, u_syn, u_exact_expr])
rel_phy = float(jnp.linalg.norm(_u_phy - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))
rel_syn = float(jnp.linalg.norm(_u_syn - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))

print(f"u_phy rel-L2 : {rel_phy:.3e}    u_syn rel-L2 : {rel_syn:.3e}")
assert rel_phy < 0.05, f"Physical model error too large: {rel_phy:.3e}"
assert rel_syn < 0.10, f"Synthetic model error too large: {rel_syn:.3e}"

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"05_coupled_and_inverse/hyco_poisson_1d.py | epochs=3000 | alpha={α} | beta={β}"
        f" | rel_L2_phy={rel_phy:.6e} | rel_L2_syn={rel_syn:.6e}\n"
    )
