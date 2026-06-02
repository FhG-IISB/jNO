"""05 — Bayesian inverse parameter identification with credible intervals

Problem
-------
Same setup as ``inverse_parameter.py`` (low-dim coefficient recovery), but
each scalar parameter is sampled with NUTS rather than point-optimised, so
we get both a posterior mean *and* a credible interval.

    d(x) = A·sin(πx) + B·cos(πx)        ground truth  A = 3.14,  B = -2.71

Technique
---------
Two scalar parameters (a, b) are declared as trainable jno parameters.
Each is configured with ``.bayesian(blackjax.nuts, ...)``.  ``solve()``
runs warmup + sampling outer epochs; the post-warmup chains are stacked
into ``a.posterior_samples`` / ``b.posterior_samples``.

References
----------
Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo.*  JMLR 15(1), 1593-1623.
"""

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp

import jno

# ── Domain & "measured" data ──────────────────────────────────────────────────
π = jno.np.pi
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.02))
x, _ = domain.variable("interior")

A_true, B_true = 3.14, -2.71
target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x)

# ── Trainable scalar parameters with per-parameter NUTS samplers ──────────────
k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")

for p in [a, b]:
    # adapt=True (default) runs blackjax.window_adaptation for `warmup`
    # steps and tunes step_size + inverse_mass_matrix automatically — the
    # `step_size` given here is just the adapter's initial guess.
    p.bayesian(
        blackjax.nuts,
        step_size=1e-2,
        warmup=300,
        keep=500,
    )

# ── Residual + solve ──────────────────────────────────────────────────────────
residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
crux = jno.core([residual.mse], domain)
crux.solve(800)

# ── Posterior summary — raw chain → user post-processes ───────────────────────
a_chain = a.posterior_samples  # shape (500, 1)
b_chain = b.posterior_samples

A_mean = float(jnp.mean(a_chain))
A_lo, A_hi = (float(v) for v in jnp.quantile(a_chain, jnp.array([0.05, 0.95])))
B_mean = float(jnp.mean(b_chain))
B_lo, B_hi = (float(v) for v in jnp.quantile(b_chain, jnp.array([0.05, 0.95])))

print(f"A = {A_mean:.3f}  90% CI = [{A_lo:.3f}, {A_hi:.3f}]   truth = {A_true}")
print(f"B = {B_mean:.3f}  90% CI = [{B_lo:.3f}, {B_hi:.3f}]   truth = {B_true}")

rel_A = abs(A_mean - A_true) / abs(A_true)
rel_B = abs(B_mean - B_true) / abs(B_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"05_coupled_and_inverse/bayesian_inverse_parameter.py | epochs=800 | rel_A={rel_A:.4f} | rel_B={rel_B:.4f}\n")

assert rel_A < 0.3, f"A posterior mean off by {rel_A:.2%}"
assert rel_B < 0.3, f"B posterior mean off by {rel_B:.2%}"
