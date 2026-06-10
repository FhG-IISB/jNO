"""09 — BNN regression via Variational Inference (mean-field)"""

from pathlib import Path

import blackjax
import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

# ── Synthetic training data — same 32 points + gap as Tutorial 07 ───────────
sigma_obs = 0.1
rng_np = np.random.default_rng(0)

x_left = np.linspace(-0.8, -0.2, 16)
x_right = np.linspace(0.2, 0.8, 16)
x_train_np = np.concatenate([x_left, x_right]).astype(np.float32)
y_clean = np.sin(6.0 * x_train_np) ** 3
y_train_np = (y_clean + sigma_obs * rng_np.normal(size=x_train_np.shape)).astype(np.float32)

train_dom = jno.domain.from_array({"train": x_train_np.reshape(-1, 1)})
x_train_var, _ = train_dom.variable("train")
if "train" in train_dom.context:
    train_dom.context["train"] = train_dom.context["train"].astype(np.float32)
y_train_const = jnp.asarray(y_train_np).reshape(-1, 1)

# ── Bayesian MLP — every weight fit via mean-field VI ────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=16,
        num_layers=2,
        key=jax.random.PRNGKey(0),
    )
)
# Mean-field VI: q(θ) is a product of independent Gaussians, one per
# weight.  blackjax optimises the ELBO via the supplied optax
# optimiser; after solve, ``posterior_draws`` i.i.d. samples are
# drawn from the fitted q and stored on net.posterior_samples in the
# same (1, N, *param) layout as the MCMC path.
u_net.vi(
    blackjax.meanfield_vi,
    optimizer=optax.adam(5e-3),
    num_samples=8,
    posterior_draws=600,
)

# For VI on a BNN the likelihood scaling matters: ``residual.mse`` is
# ``mean_i (y_pred_i - y_i)² / σ²``, but the canonical Gaussian-noise
# log-likelihood is the **sum** over data points.  We rescale by
# ``sqrt(N)`` so that ``residual.mse`` equals the sum-of-squared-
# standardised-residuals — this puts the right magnitude on the
# likelihood term and gives VI a strong gradient signal.
N_obs = float(x_train_np.shape[0])
y_pred = u_net(x_train_var)
residual = (y_pred - y_train_const) / sigma_obs * jnp.sqrt(N_obs)

crux = jno.core([residual.mse])
crux.solve(6000)  # 6000 ELBO optimisation steps

# ── Predict on a dense eval grid (same as T07) ──────────────────────────────
eval_dom = jno.domain.line(x_range=(-1.0, 1.0), mesh_size=0.02)
x_eval, _ = eval_dom.variable("interior")

# auto-chain: u_net carries posterior_samples from the fitted q.
u_chain = crux.eval([u_net(x_eval)], domain=eval_dom)  # (1, 600, n_eval, 1)
u_mean = jnp.mean(u_chain, axis=(0, 1))
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=(0, 1))

band_width = np.asarray(u_hi - u_lo).reshape(-1)
x_eval_np = np.asarray(eval_dom.context["interior"]).reshape(-1)

u_truth = np.sin(6.0 * x_eval_np) ** 3
u_mean_np = np.asarray(u_mean.reshape(-1))
in_data = (np.abs(x_eval_np) >= 0.2) & (np.abs(x_eval_np) <= 0.8)
in_gap = np.abs(x_eval_np) < 0.2

rel_l2_in_data = float(np.linalg.norm(u_mean_np[in_data] - u_truth[in_data]) / (np.linalg.norm(u_truth[in_data]) + 1e-8))
band_in_data = float(np.median(band_width[in_data]))
band_in_gap = float(np.median(band_width[in_gap]))
band_ratio = band_in_gap / max(band_in_data, 1e-12)

print(f"[vi-bnn] in-data rel-L2 of posterior mean : {rel_l2_in_data:.4f}")
print(f"         band width  in-data (median)     : {band_in_data:.4f}")
print(f"         band width  in-gap  (median)     : {band_in_gap:.4f}")
print(f"         gap / data band ratio            : {band_ratio:.2f}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/09_vi_bnn_regression.py | epochs=2000 | "
        f"rel_L2_in_data={rel_l2_in_data:.4f} | band_in_data={band_in_data:.4f} | "
        f"band_in_gap={band_in_gap:.4f} | band_ratio={band_ratio:.2f}\n"
    )

# Loose asserts — VI is faster to converge than SGLD on this problem,
# so the in-data band should be tighter while the gap-vs-data ratio
# stays positive (uncertainty still grows in the data gap).
assert rel_l2_in_data < 0.5, f"posterior mean in-data rel-L2 too high: {rel_l2_in_data:.3e}"
assert band_in_data > 1e-4, f"in-data band collapsed: {band_in_data:.3e}"
assert band_in_gap > band_in_data, (
    f"VI predictive band did not widen in the gap (in-data {band_in_data:.4f}, gap {band_in_gap:.4f})"
)
