"""05 — Bayesian inverse surrogate: posterior over an inverted input"""

from pathlib import Path

import blackjax
import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

π = jno.np.pi

# ── Keys ─────────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(42)
k_fwd, k_inv = jax.random.split(key)

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Forward solve (deterministic PINN)
# ══════════════════════════════════════════════════════════════════════════════

domain = jno.domain.line(mesh_size=0.01)
x, _ = domain.variable("interior")

u_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k_fwd))
u_net.optimizer(optax.adam(1e-3))

u = u_net(x) * x * (1 - x)  # hard zero Dirichlet BCs
pde = u.dd(x) + π**2 * jno.np.sin(π * x)  # u'' + π² sin(πx) = 0

crux_fwd = jno.core([pde.mse])
crux_fwd.solve(3_000)

# Sanity-check forward accuracy before inverting.
_u, _u_exact = crux_fwd.eval([u, jno.np.sin(π * x)])
fwd_err = float(jnp.linalg.norm(_u - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))
print(f"[forward] rel-L2 error: {fwd_err:.3e}")
assert fwd_err < 0.05, f"Forward solve inaccurate: rel-L2 = {fwd_err:.3e}"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Inverse: posterior over x_query given u_obs
# ══════════════════════════════════════════════════════════════════════════════

# Freeze the trained surrogate — its weights stay constant under NUTS.
u_net.freeze()

x_true = 0.3
sigma_obs = 0.02  # likelihood scale; observation itself is noiseless here.
u_obs = float(jnp.sin(jnp.pi * x_true))  # ≈ 0.809
print(f"[inverse] u_obs = {u_obs:.4f}  (x_true = {x_true})")

# Bayesian input parameter.  Initialise near x_true so the chain
# discovers the left mode (see the non-identifiability caveat above).
x_query = jno.np.parameter((1,), name="x_query")
x_query.initialize(jax.nn.initializers.constant(0.2))
x_query.bayesian(
    blackjax.nuts,
    step_size=5e-3,
    warmup=300,
    keep=500,
    max_num_doublings=4,
    # adapt=True default — pure Bayesian inference against a frozen surrogate
    # gives a fixed-target logdensity, so window adaptation is well-defined.
)

# Surrogate evaluated at x_query (same hard-BC factor as Phase 1).
u_at_query = u_net(x_query) * x_query * (1 - x_query)

# Gaussian-noise likelihood with known σ.
residual = (u_at_query - u_obs) / sigma_obs

# Single-point inverse domain — loss has no spatial dependence on a mesh.
inv_domain = jno.domain.from_array({"pt": np.zeros((1, 1))})
crux_inv = jno.core([residual.mse], domain=inv_domain)
crux_inv.solve(800)

# ── Posterior summary ────────────────────────────────────────────────────────
x_chain = x_query.posterior_samples  # (500, 1)
x_mean = float(jnp.mean(x_chain))
x_std = float(jnp.std(x_chain))
x_lo, x_hi = (float(v) for v in jnp.quantile(x_chain, jnp.array([0.05, 0.95])))

print(f"[inverse] x_query = {x_mean:.4f} ± {x_std:.4f}")
print(f"          90% CI = [{x_lo:.4f}, {x_hi:.4f}]   (left mode of bimodal posterior)")

abs_err = abs(x_mean - x_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/05_inverse_surrogate_uncertainty.py | "
        f"fwd_epochs=3000 | inv_epochs=800 | fwd_rel_L2={fwd_err:.6e} | "
        f"x_mean={x_mean:.4f} | abs_err={abs_err:.4f} | CI_width={x_hi - x_lo:.4f}\n"
    )

assert abs_err < 0.1, f"posterior-mean x off by {abs_err:.4f} (truth {x_true})"
