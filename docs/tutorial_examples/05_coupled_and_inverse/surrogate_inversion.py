"""05 — Surrogate inversion: recover a hidden input from a frozen PINN"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

π = jno.np.pi

# ── Keys ─────────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Forward solve
# ══════════════════════════════════════════════════════════════════════════════

domain = jno.domain.line(mesh_size=0.01)
x, _ = domain.variable("interior")

u_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k1))
u_net.optimizer(optax.adam(1e-3))

u = u_net(x) * x * (1 - x)  # hard zero Dirichlet BCs
pde = u.dd(x) + π**2 * jno.np.sin(π * x)  # residual: u'' + π²sin(πx) = 0

crux_fwd = jno.core([pde.mse])
crux_fwd.solve(3_000)

# Verify forward accuracy before inverting
_u, _u_exact = crux_fwd.eval([u, jno.np.sin(π * x)])
fwd_err = float(jnp.linalg.norm(_u - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))
print(f"[forward] rel-L2 error: {fwd_err:.3e}")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Inverse: recover x* from u(x*) = u_obs
# ══════════════════════════════════════════════════════════════════════════════

# Freeze the trained surrogate — its weights will not change during inversion
u_net.freeze()

x_true = 0.3
u_obs = float(jnp.sin(jnp.pi * x_true))  # "measured" value ≈ 0.809
print(f"[inverse] target u_obs = {u_obs:.4f}  (x_true = {x_true})")

# x_query is the variable we optimise — it starts as an unknown scalar in [0, 1]
x_query = jno.np.parameter((1,), name="x_query")
x_query.initialize(jax.nn.initializers.constant(0.1))  # initial guess: 0.1
x_query.optimizer(optax.adam(5e-3))

# Evaluate the frozen surrogate at the query point (same BC factor as training)
u_at_query = u_net(x_query) * x_query * (1 - x_query)

# Loss: drive u_net(x_query) towards the observation
inv_loss = (u_at_query - u_obs) ** 2

# Single-point domain — the loss has no spatial dependence on the mesh
inv_domain = jno.domain.from_array({"pt": np.zeros((1, 1))})

crux_inv = jno.core([inv_loss.mean], domain=inv_domain)
crux_inv.solve(500)

# ── Results ──────────────────────────────────────────────────────────────────
_x_q = crux_inv.eval([x_query])[0]
x_recovered = float(_x_q)
abs_err = abs(x_recovered - x_true)

print(f"[inverse] recovered x = {x_recovered:.4f}  (true = {x_true:.4f})")
print(f"[inverse] absolute error: {abs_err:.4f}")

# ── Assertions ────────────────────────────────────────────────────────────────
assert fwd_err < 0.05, f"Forward solve inaccurate: rel-L2 = {fwd_err:.3e}"
assert abs_err < 0.05, f"Inversion error too large: |x_recovered − x_true| = {abs_err:.4f}"

# ── Result tracking ───────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"05_coupled_and_inverse/surrogate_inversion.py"
        f" | fwd_epochs=3000 | inv_epochs=500"
        f" | fwd_rel_L2={fwd_err:.6e}"
        f" | x_recovered={x_recovered:.6f}"
        f" | abs_err={abs_err:.6f}\n"
    )
