"""10 — Head-only Bayesian regression (``.mask().bayesian()``)

Demonstrates jno's **per-mask backend dispatch**: ``.mask(M).bayesian(...)``
restricts MCMC sampling to a chosen subset of a model's parameter
pytree.  Unmasked leaves keep their initial values throughout
``solve()`` — they are *not* updated.

Concretely: a 2-layer MLP whose output linear layer ("head", 17 params)
is sampled with SGLD while the hidden body (304 params) stays at its
random initialisation.

The pattern this v1 release ships
---------------------------------
This tutorial exercises **Pattern A** from the design plan:

* ``.mask(head_mask).bayesian(...)`` on a model with **no** global
  ``.optimizer(...)`` set.
* Body leaves stay at their random init; head leaves get a posterior.

A more practically useful pattern — **Pattern B**, body Adam-trained +
head Bayesian on the same model — is currently blocked at ``solve()``
time with a clear ``NotImplementedError("state-storage refactor")``.
Supporting it requires the internal ``opt_states`` mapping to hold
*both* an optax state and a kernel state under a single layer id; the
necessary refactor (composite keys or tuple-state encoding) is
documented as the v2 follow-up.  See ``docs/training/bayesian.md`` for
the v2 plan and the rationale.

Why exercise Pattern A anyway?  Because:

* Every public API surface — the configurator (``.mask().bayesian()``),
  the partition / reassembly logic in ``init_state``/``step``, the
  buffer flush that returns a full-pytree ``posterior_samples`` — is
  exactly the same code path Pattern B will reuse once unblocked.  This
  tutorial pins those surfaces today.
* It demonstrates the v1 narrowing semantic explicitly: chain-along-N
  variance is non-zero on the masked head leaves and machine-precision-
  small on the unmasked body leaves.  This is the headline contract.

Problem
-------
Recover a smooth 1-D target from 32 noisy observations:

    u(x) = sin(π x),     x ∈ [-1, 1],     y = u(x) + ε,   ε ~ N(0, σ²),

σ = 0.05.  Because the body is frozen at random init, the effective
"feature map" is a fixed random Gaussian-process-like basis; the head
SGLD then performs Bayesian linear regression on those features.  The
sin(π x) target is deliberately low-frequency so even a randomly-
initialised 16-feature body can plausibly represent it.

Wiring
------
* 32 training x's packed as a 1-D point-cloud domain via
  ``jno.domain.from_array({"train": ...})`` — same pattern T07 / T09
  use.
* Mask construction: ``eqx.tree_at`` replaces the ``output_layer``
  subtree of an all-False template with an all-True copy, giving a
  pytree-of-bools that selects both ``output_layer.weight`` and
  ``output_layer.bias``.
* The whole loop runs through ``crux.solve()`` — no manual blackjax
  calls.

References
----------
This tutorial does not lean on a specific paper; the closest classical
peg is **last-layer Bayesian neural networks** (Bayesian linear
regression on neural-network-induced features), but the typical
treatment in that literature pretrains the body deterministically —
which is exactly the Pattern B v2 case this v1 release cannot yet do.
"""

from pathlib import Path

import blackjax
import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import numpy as np

import jno

# ── Synthetic data — 32 evenly-spaced noisy observations of sin(π x) ──
sigma_obs = 0.05
rng_np = np.random.default_rng(0)

x_train_np = np.linspace(-1.0, 1.0, 32).astype(np.float32)
y_clean = np.sin(np.pi * x_train_np)
y_train_np = (y_clean + sigma_obs * rng_np.normal(size=x_train_np.shape)).astype(np.float32)

train_dom = jno.domain.from_array({"train": x_train_np.reshape(-1, 1)})
x_train_var, _ = train_dom.variable("train")
if "train" in train_dom.context:
    train_dom.context["train"] = train_dom.context["train"].astype(np.float32)
y_train_const = jnp.asarray(y_train_np).reshape(-1, 1)

# ── MLP with a frozen random body and a Bayesian head ───────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=16,
        num_layers=2,
        key=jax.random.PRNGKey(0),
    )
)

# Build the head mask: True only for the leaves under
# ``module.output_layer`` (weight + bias), False everywhere else.
# eqx.tree_at takes (where, pytree, replace=...) and rebuilds the
# pytree with the selected subtree replaced.
all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
head_all_true = jax.tree_util.tree_map(lambda _: True, u_net.module.output_layer)
head_mask = eqx.tree_at(lambda m: m.output_layer, all_false, replace=head_all_true)

# Sanity-print the structure once: 2 head leaves, 4 body leaves frozen.
_n_masked = sum(int(b) for b in jax.tree_util.tree_leaves(head_mask))
_n_total = len(jax.tree_util.tree_leaves(head_mask))
print(f"[masked-bnn-head] mask selects {_n_masked} / {_n_total} leaves (head)")

# Phase-11 configurator: .mask(M) primes the next .bayesian() call to
# create a per-mask Bayesian group instead of setting the global config.
u_net.mask(head_mask).bayesian(
    blackjax.sgld,
    step_size=1e-3,
    warmup=1500,
    keep=400,
    thin=2,
)

# Likelihood: scale by σ for a proper Gaussian-noise log-likelihood
# magnitude (no √N factor — SGLD's gradient direction is fine without
# the sum-vs-mean rescaling VI needs).
y_pred = u_net(x_train_var)
residual = (y_pred - y_train_const) / sigma_obs

crux = jno.core([residual.mse], train_dom)
crux.solve(2300)

# ── Diagnostics: confirm Pattern A's contract ───────────────────────
# Body leaves should have variance-along-N == 0 (frozen at init);
# head leaves should have variance-along-N > 0 (sampled by SGLD).
chain = u_net.posterior_samples
_var_along_n = jax.tree_util.tree_map(lambda leaf: float(jnp.mean(jnp.var(leaf, axis=1))), chain)
head_var = jax.tree_util.tree_leaves(
    eqx.tree_at(lambda m: m.output_layer, _var_along_n, replace=jax.tree_util.tree_map(lambda _: 1.0, head_all_true))
)
all_var_leaves = jax.tree_util.tree_leaves(_var_along_n)
head_mask_leaves = jax.tree_util.tree_leaves(head_mask)
body_var_max = max(v for v, m in zip(all_var_leaves, head_mask_leaves) if not m)
head_var_min = min(v for v, m in zip(all_var_leaves, head_mask_leaves) if m)
print(f"[masked-bnn-head] body  max var-along-N : {body_var_max:.3e}  (should be ~0)")
print(f"[masked-bnn-head] head  min var-along-N : {head_var_min:.3e}  (should be > 0)")

# ── Predictive bands at a dense eval grid ───────────────────────────
eval_dom = jno.domain.line(x_range=(-1.0, 1.0), mesh_size=0.02)
x_eval, _ = eval_dom.variable("interior")

# auto-chain: u_net carries posterior_samples on its head; bodies are
# constant, so vmapping the full chain through the model is correct.
u_chain = crux.eval([u_net(x_eval)], domain=eval_dom)  # (1, keep, n_eval, 1)
u_mean = jnp.mean(u_chain, axis=(0, 1))
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=(0, 1))

band_width = np.asarray(u_hi - u_lo).reshape(-1)
x_eval_np = np.asarray(eval_dom.context["interior"]).reshape(-1)
u_truth = np.sin(np.pi * x_eval_np)
u_mean_np = np.asarray(u_mean.reshape(-1))

rel_l2 = float(np.linalg.norm(u_mean_np - u_truth) / (np.linalg.norm(u_truth) + 1e-8))
band_median = float(np.median(band_width))
band_max = float(np.max(band_width))

print(f"[masked-bnn-head] posterior-mean rel-L2 vs sin(πx) : {rel_l2:.4f}")
print(f"[masked-bnn-head] band width  median              : {band_median:.4f}")
print(f"[masked-bnn-head] band width  max                 : {band_max:.4f}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/10_masked_bnn_head.py | epochs=2300 | "
        f"rel_L2={rel_l2:.4f} | band_median={band_median:.4f} | band_max={band_max:.4f} | "
        f"body_var_max={body_var_max:.3e} | head_var_min={head_var_min:.3e}\n"
    )

# Asserts: the v1 contract has to hold, and SGLD on 17 head params with
# a random body should at least produce a sane envelope around the data.
# The headline contract — body frozen, head varies — is verified
# numerically below; rel-L2 / band targets are loose since the random
# body limits how well the head can fit a continuous target.
assert body_var_max < 1e-10, f"v1 contract violated: body leaves should be frozen (max var {body_var_max:.3e})"
assert head_var_min > 1e-8, f"v1 contract violated: head leaves should vary across the chain (min var {head_var_min:.3e})"
assert rel_l2 < 2.0, f"posterior-mean rel-L2 too high: {rel_l2:.3e}"
assert band_median > 1e-4, f"posterior band degenerate (median {band_median:.3e})"
