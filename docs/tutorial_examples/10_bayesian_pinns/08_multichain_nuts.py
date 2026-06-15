"""08 — Multi-chain NUTS with R-hat / ESS convergence diagnostics"""

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp

import jno

# ── Domain & target data ──────────────────────────────────────────────────────
π = jno.np.pi
domain = jno.domain.line(mesh_size=0.02)
x, _ = domain.variable("interior")

A_true, B_true = 3.14, -2.71
target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x)

# ── Per-parameter NUTS with K=4 parallel chains + over-dispersed init ────────
K_CHAINS = 4
k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")

for p in (a, b):
    p.bayesian(
        blackjax.nuts,
        step_size=1e-2,
        warmup=300,
        keep=400,
        num_chains=K_CHAINS,
        init_jitter=0.1,
    )

# ── Residual + solve ──────────────────────────────────────────────────────────
residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
crux = jno.core([residual.mse])
crux.solve(700)

# ── Per-chain summaries + cross-chain diagnostics ────────────────────────────
a_chain = a.posterior_samples  # (K, N, 1)
b_chain = b.posterior_samples

# Posterior mean / 90 % CI over the union of (K * N) draws.
A_mean = float(jnp.mean(a_chain))
A_lo, A_hi = (float(v) for v in jnp.quantile(a_chain, jnp.array([0.05, 0.95])))
B_mean = float(jnp.mean(b_chain))
B_lo, B_hi = (float(v) for v in jnp.quantile(b_chain, jnp.array([0.05, 0.95])))

# Per-chain means so we can see whether the chains agree.
A_per_chain = [float(jnp.mean(a_chain[k])) for k in range(K_CHAINS)]
B_per_chain = [float(jnp.mean(b_chain[k])) for k in range(K_CHAINS)]

# R-hat and ESS — pure-JAX, no arviz needed.
A_rhat = float(jno.bayesian.rhat(a_chain)[0])
B_rhat = float(jno.bayesian.rhat(b_chain)[0])
A_ess = float(jno.bayesian.ess(a_chain)[0])
B_ess = float(jno.bayesian.ess(b_chain)[0])

print(f"A = {A_mean:.3f}  90% CI = [{A_lo:.3f}, {A_hi:.3f}]   truth = {A_true}")
print(f"   per-chain means = {[f'{v:.3f}' for v in A_per_chain]}")
print(f"   R-hat = {A_rhat:.4f}    ESS = {A_ess:.1f}  (of {K_CHAINS * a_chain.shape[1]} draws)")
print(f"B = {B_mean:.3f}  90% CI = [{B_lo:.3f}, {B_hi:.3f}]   truth = {B_true}")
print(f"   per-chain means = {[f'{v:.3f}' for v in B_per_chain]}")
print(f"   R-hat = {B_rhat:.4f}    ESS = {B_ess:.1f}  (of {K_CHAINS * b_chain.shape[1]} draws)")

rel_A = abs(A_mean - A_true) / abs(A_true)
rel_B = abs(B_mean - B_true) / abs(B_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/08_multichain_nuts.py | epochs=700 | K={K_CHAINS} | "
        f"rel_A={rel_A:.4f} | rel_B={rel_B:.4f} | "
        f"rhat_A={A_rhat:.4f} | rhat_B={B_rhat:.4f} | "
        f"ess_A={A_ess:.1f} | ess_B={B_ess:.1f}\n"
    )

# Loose convergence asserts.  R-hat ≤ 1.1 is the community heuristic
# for converged chains (Vehtari et al. 2021 suggests 1.01 as a stricter
# threshold).  We use 1.1 here to be robust to the short chain length
# we run on CPU.
assert rel_A < 0.3, f"A posterior mean off by {rel_A:.2%}"
assert rel_B < 0.3, f"B posterior mean off by {rel_B:.2%}"
assert A_rhat < 1.1, f"A R-hat too high: {A_rhat:.4f}"
assert B_rhat < 1.1, f"B R-hat too high: {B_rhat:.4f}"
assert A_ess > 30.0, f"A ESS too low: {A_ess:.1f}"
assert B_ess > 30.0, f"B ESS too low: {B_ess:.1f}"
