"""07 — Bayesian Neural Network regression (no PDE, full MLP via SGLD)

Problem
-------
Approximate the function

    u(x) = sin³(6x),     x ∈ [-1, 1],

from 32 noisy observations placed in ``[-0.8, -0.2] ∪ [0.2, 0.8]`` —
a deliberate **gap around x = 0** and outside ``±0.8`` — with Gaussian
noise ``σ = 0.1``.

This is the textbook Bayesian-Neural-Network regression demo of
Yang, Meng, & Karniadakis (2021, §3.1).  Compared with Tutorial 01
(which trains a BNN as a *PINN* — MLP weights are sampled, PDE residual
constrains them), this tutorial trains an MLP purely as a **regressor**:

* no PDE residual,
* the only constraint is the data-fit MSE,
* every weight in the MLP is sampled with SGLD.

The headline B-PINN behaviour is the predictive band:

* **narrow** in the data-dense regions (``|x| ∈ [0.2, 0.8]``),
* **widens sharply** in the data gap (``|x| < 0.2``) and outside the
  data range (``|x| > 0.8``).

That uncertainty growing where data don't constrain the model is the
core value proposition of BNNs that point-estimate networks can't
provide.

Wiring
------
The training set is packed as ``(N, 2)`` "coordinates" (`x_train` and
`y_train` per node) and attached to a ``jno.domain.from_array`` domain;
``.variable("obs", split=True)`` returns per-node Variables for the
two columns.  A separate dense eval domain
``jno.domain.line(x_range=(-1, 1), ...)`` provides the prediction
grid.  The whole loop runs through ``crux.solve()`` — no manual
blackjax loop.

Reference
---------
Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913
(see §3.1 "Function regression").

Welling, M., & Teh, Y. W. (2011).  *Bayesian Learning via Stochastic
Gradient Langevin Dynamics.*  ICML 2011, 681-688.
"""

from pathlib import Path

import blackjax
import foundax
import jax
import jax.numpy as jnp
import numpy as np

import jno

# ── Synthetic training data — 32 points with a gap in [-0.2, 0.2] ───────────
sigma_obs = 0.1
rng_np = np.random.default_rng(0)

x_left = np.linspace(-0.8, -0.2, 16)
x_right = np.linspace(0.2, 0.8, 16)
x_train_np = np.concatenate([x_left, x_right]).astype(np.float32)
y_clean = np.sin(6.0 * x_train_np) ** 3
y_train_np = (y_clean + sigma_obs * rng_np.normal(size=x_train_np.shape)).astype(np.float32)

# Pack (x, y) as per-node "coordinates" so a single Variable returns both.
train_data = np.stack([x_train_np, y_train_np], axis=1)  # (32, 2)
train_dom = jno.domain.from_array({"obs": train_data})
x_train_var, y_train_var, _ = train_dom.variable("obs", split=True)

# `from_array` round-trips through .npz which widens float32 → float64.
# SGLD has no Metropolis cond so this dtype isn't fatal, but matching
# the network's float32 weights keeps the loss / gradients in one dtype.
for _tag in ("obs", "obs_0", "obs_1"):
    if _tag in train_dom.context:
        train_dom.context[_tag] = train_dom.context[_tag].astype(np.float32)

# ── Bayesian MLP — every weight sampled via SGLD ─────────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=16,
        num_layers=2,
        key=jax.random.PRNGKey(0),
    )
)
u_net.bayesian(
    blackjax.sgld,
    step_size=1e-3,
    warmup=3000,
    keep=600,
    thin=2,
)

# Data-fit residual, scaled by σ for a proper Gaussian-noise likelihood.
y_pred = u_net(x_train_var)
residual = (y_pred - y_train_var) / sigma_obs

# ── Solve through crux.solve — no manual blackjax loop ──────────────────────
crux = jno.core([residual.mse], train_dom)
crux.solve(4200)

# ── Predict on a dense eval grid ─────────────────────────────────────────────
eval_dom = jno.domain(constructor=jno.domain.line(x_range=(-1.0, 1.0), mesh_size=0.02))
x_eval, _ = eval_dom.variable("interior")

# auto-chain default: u_net is Bayesian, so the chain is vmapped through.
u_chain = crux.eval([u_net(x_eval)], domain=eval_dom)  # (n_kept, n_eval, 1)
u_mean = jnp.mean(u_chain, axis=0)
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=0)

# Per-eval-point band width.  eval_dom.context["interior"] has shape
# (1, 1, n_eval, 1) — flatten to (n_eval,) for plotting / asserts.
band_width = np.asarray(u_hi - u_lo).reshape(-1)
x_eval_np = np.asarray(eval_dom.context["interior"]).reshape(-1)

# Reference function and rel-L2 of the posterior mean against truth.
u_truth = np.sin(6.0 * x_eval_np) ** 3
u_mean_np = np.asarray(u_mean.reshape(-1))
in_data = (np.abs(x_eval_np) >= 0.2) & (np.abs(x_eval_np) <= 0.8)
in_gap = np.abs(x_eval_np) < 0.2

rel_l2_in_data = float(np.linalg.norm(u_mean_np[in_data] - u_truth[in_data]) / (np.linalg.norm(u_truth[in_data]) + 1e-8))
band_in_data = float(np.median(band_width[in_data]))
band_in_gap = float(np.median(band_width[in_gap]))
band_ratio = band_in_gap / max(band_in_data, 1e-12)

print(f"[bnn] in-data rel-L2 of posterior mean : {rel_l2_in_data:.4f}")
print(f"      band width  in-data (median)     : {band_in_data:.4f}")
print(f"      band width  in-gap  (median)     : {band_in_gap:.4f}")
print(f"      gap / data band ratio            : {band_ratio:.2f}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/07_bnn_regression.py | epochs=4200 | "
        f"rel_L2_in_data={rel_l2_in_data:.4f} | band_in_data={band_in_data:.4f} | "
        f"band_in_gap={band_in_gap:.4f} | band_ratio={band_ratio:.2f}\n"
    )

# Loose asserts — SGLD without explicit adaptation is noisy.
assert rel_l2_in_data < 1.0, f"posterior mean in-data rel-L2 too high: {rel_l2_in_data:.3e}"
assert band_in_data > 1e-4, f"in-data band collapsed: {band_in_data:.3e}"
assert band_in_gap > band_in_data, (
    f"BNN band did not widen in the gap (in-data {band_in_data:.4f}, gap {band_in_gap:.4f}); check chain length / step_size"
)
