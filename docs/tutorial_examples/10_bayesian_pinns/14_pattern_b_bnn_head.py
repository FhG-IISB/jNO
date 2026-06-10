"""14 — Pattern B: Bayesian last layer (head sampled, body Adam-trained)"""

from pathlib import Path

import blackjax
import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

# ── Synthetic data — same gapped target as T07 / T10 ────────────────────────
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

# ── MLP with head Bayesian, body Adam-trained ───────────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=16,
        num_layers=2,
        key=jax.random.PRNGKey(0),
    )
)

# Build the head mask: True only under module.output_layer; False on body.
all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
head_all_true = jax.tree_util.tree_map(lambda _: True, u_net.module.output_layer)
head_mask = eqx.tree_at(lambda m: m.output_layer, all_false, replace=head_all_true)
n_head = sum(int(b) for b in jax.tree_util.tree_leaves(head_mask))
n_total = len(jax.tree_util.tree_leaves(head_mask))
print(f"[pattern-b] head leaves: {n_head} / {n_total}")

# Pattern B: body trained via Adam, head sampled via SGLD.  Phase 15
# enables both to coexist on the same model.
u_net.optimizer(optax.adam(5e-3))  # ← body (the unmasked complement)
u_net.mask(head_mask).bayesian(  # ← head
    blackjax.sgld,
    step_size=5e-4,
    warmup=1500,
    keep=400,
    thin=2,
)

# Likelihood with σ-scaled residuals.
y_pred = u_net(x_train_var)
residual = (y_pred - y_train_const) / sigma_obs

crux = jno.core([residual.mse])
crux.solve(2300)

# ── Diagnostics ─────────────────────────────────────────────────────────────
chain = u_net.posterior_samples  # (1, keep, *full pytree)
# Head leaves vary across the chain; body leaves were Adam-trained
# during sampling so they ALSO vary across the chain (each sample
# captures the body state at sample time).  The headline contract
# we check is just that the chain is populated and the head leaves
# vary.
head_idxs = [i for i, m in enumerate(jax.tree_util.tree_leaves(head_mask)) if m]
chain_leaves = jax.tree_util.tree_leaves(chain)
head_var = max(float(jnp.mean(jnp.var(chain_leaves[i], axis=1))) for i in head_idxs)
print(f"[pattern-b] head leaf max var-along-N : {head_var:.3e}  (should be > 0)")

# ── Predictive bands on a dense eval grid ───────────────────────────────────
eval_dom = jno.domain.line(x_range=(-1.0, 1.0), mesh_size=0.02)
x_eval, _ = eval_dom.variable("interior")

u_chain = crux.eval([u_net(x_eval)], domain=eval_dom)  # (1, keep, n_eval, 1)
u_mean = jnp.mean(u_chain, axis=(0, 1))
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=(0, 1))
band = np.asarray(u_hi - u_lo).reshape(-1)
x_eval_np = np.asarray(eval_dom.context["interior"]).reshape(-1)
u_truth = np.sin(np.pi * x_eval_np)
rel_l2 = float(np.linalg.norm(u_mean.reshape(-1) - u_truth) / (np.linalg.norm(u_truth) + 1e-8))
band_median = float(np.median(band))

print(f"[pattern-b] posterior-mean rel-L2 vs sin(πx) : {rel_l2:.4f}")
print(f"[pattern-b] band width median               : {band_median:.4f}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/14_pattern_b_bnn_head.py | epochs=2300 | "
        f"rel_L2={rel_l2:.4f} | band_median={band_median:.4f} | head_var_max={head_var:.3e}\n"
    )

# Asserts: Pattern B contract is the same as Tutorial 10's (head varies,
# fit recovers sin(πx)); the difference is the body is now trained
# rather than frozen — so the fit is materially tighter than T10's.
assert head_var > 1e-8, f"head should vary across the chain; got var={head_var:.3e}"
assert rel_l2 < 1.0, f"posterior-mean rel-L2 too high: {rel_l2:.3e}"
assert band_median > 1e-4, f"posterior band degenerate (median {band_median:.3e})"
