"""09 — BNN regression via Variational Inference (mean-field)

Same regression problem as Tutorial 07 — approximate

    u(x) = sin³(6x),     x ∈ [-1, 1],

from 32 noisy observations placed in ``[-0.8, -0.2] ∪ [0.2, 0.8]`` —
a deliberate gap around ``x = 0`` — but trained via **mean-field
Variational Inference** (``blackjax.meanfield_vi``) instead of SGLD.

Compared with T07's vanilla SGLD result on the same dataset, VI is:

* **optimisation-based** — fits a Gaussian product ``q(θ)`` to the
  posterior by maximising the evidence lower bound (ELBO), then draws
  i.i.d. samples from the fitted ``q``.  Each ELBO step is one optax
  update — feels just like training a regular network.
* **much faster** at producing usable predictive bands — typically a
  few hundred steps for a small MLP versus the thousands-of-steps
  budget SGLD needs to wander far enough through the posterior.
* **tighter but less calibrated** in the data-dense region —
  mean-field assumes per-weight independence and can underestimate
  posterior variance for correlated weights.  Trade-off vs SGLD's
  wider but better-mixed chain.

Both methods produce the same headline B-PINN behaviour: the
predictive band **widens in the data gap** because the posterior
weight distribution disagrees more strongly there.

Wiring
------
Same as T07 — 32 training inputs as a 1-D point cloud, ``y_train`` as
a plain ``jnp`` constant in the residual, inference on a dense
``jno.domain.line(...)`` grid via ``crux.eval``.  The only change is
swapping ``.bayesian(blackjax.sgld, ...)`` for ``.vi(blackjax.meanfield_vi,
optimizer=optax.adam(1e-3), ...)`` — same downstream API.

References
----------
Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M.
(2017).  *Automatic Differentiation Variational Inference.*  JMLR
18(1), 430-474.

Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013).
*Stochastic Variational Inference.*  JMLR 14(1), 1303-1347.

Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913
(see §3.2 for VI on BNN-PINNs).
"""

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

crux = jno.core([residual.mse], train_dom)
crux.solve(6000)  # 6000 ELBO optimisation steps

# ── Predict on a dense eval grid (same as T07) ──────────────────────────────
eval_dom = jno.domain(constructor=jno.domain.line(x_range=(-1.0, 1.0), mesh_size=0.02))
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
