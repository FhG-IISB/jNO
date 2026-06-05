# Head-only Bayesian regression (`.mask().bayesian()`)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/10_masked_bnn_head.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Restricts MCMC sampling to a subset of a model's parameter pytree.**
A 2-layer MLP regresses `sin(πx)` from 32 noisy observations; only the
**output linear layer ("head", 17 parameters)** is SGLD-sampled, while
the hidden body (304 parameters) stays at its random initialisation
throughout `solve()`.

## What you get

For the configuration in the script (32 obs, σ = 0.05, 2300 epochs,
`keep=400`, `thin=2`):

| Metric | Value | Meaning |
|---|---|---|
| **body** max variance-along-chain | ≈ 5 × 10⁻¹⁶ | Body leaves are genuinely frozen — no chain motion. |
| **head** min variance-along-chain | ≈ 6.5 × 10⁻² | Head leaves move across SGLD samples. |
| posterior-mean rel-L2 vs `sin(πx)` | ≈ 0.39 | Mediocre — the random body limits expressiveness. |
| posterior band median (90 %) | ≈ 0.24 | Non-degenerate predictive uncertainty. |

The body's "var-along-chain ≈ machine precision" is the **headline
v1 contract**: the masked subset gets a posterior; the rest stays put.

## The v1 release ships Pattern A only

The design plan describes six composable patterns of
`.mask().bayesian()` / `.mask().vi()` against optax /
no-optax / multi-mask / LoRA / mixed-VI-MCMC backbones.  This v1
release ships **Pattern A**:

* `.mask(M).bayesian(...)` (or `.vi(...)`) on a model with **no**
  global `.optimizer(...)`.
* Body leaves outside `M` stay at their initial values; head leaves
  inside `M` get a posterior.

The patterns that need a state-storage refactor — most notably
**Pattern B**, body Adam-trained *and* head Bayesian on the same model —
raise a clear `NotImplementedError("state-storage refactor")` at
`solve()` time.  The full set of blocked patterns and the v2 plan are
documented in [Training → Bayesian Sampling](../../training/bayesian.md#composable-per-mask-backends-v1).

## Why exercise Pattern A even when Pattern B is more useful?

* Every public API surface — `.mask(M).bayesian(...)` configuration,
  the partition/reassembly logic in
  `init_state`/`step`, the buffer flush that returns a full-pytree
  `posterior_samples` — is exactly the code path Pattern B reuses
  once unblocked.  This tutorial pins those surfaces today.
* The "what's masked = posterior, the rest = constant" contract is
  exercised end-to-end: the assertions verify it numerically on every
  CI run.

## Mask construction

The mask is a pytree of bools with the **same structure** as
`net.module`.  `eqx.tree_at` builds it by replacing one subtree of an
all-False template with an all-True copy of the target subtree:

```python
all_false      = jax.tree_util.tree_map(lambda _: False, u_net.module)
head_all_true  = jax.tree_util.tree_map(lambda _: True,  u_net.module.output_layer)
head_mask      = eqx.tree_at(lambda m: m.output_layer, all_false,
                             replace=head_all_true)

u_net.mask(head_mask).bayesian(blackjax.sgld, step_size=1e-3,
                                warmup=1500, keep=400, thin=2)
```

The `where=lambda m: m.output_layer` lambda must point at the **same**
subtree on both sides; jno verifies the resulting mask has the same
pytree structure as the model.

## How `posterior_samples` is laid out

The chain stores the **full** model pytree (both masked and unmasked
leaves) at every sample.  Unmasked leaves are constant along the
chain axis; masked leaves vary.  This keeps the downstream surface
uniform:

```python
chain = u_net.posterior_samples       # full pytree, leading axis (K, N, ...)
u_chain = crux.eval([u_net(x_eval)], domain=eval_dom)
                                       # (1, 400, n_eval, 1) — auto-vmap
u_mean = jnp.mean(u_chain, axis=(0, 1))
```

`crux.eval(samples="auto")`, `jno.bayesian.{rhat, ess}`, and wandb
posterior stats all work transparently — no special case for masked
solves.  The full-pytree storage is a memory cost: for very narrow
masks on wide models, sparse storage (only varying leaves + an init
snapshot) is documented as a v2 follow-up.

## Caveats specific to this v1 demo

* **Random body = random features.**  With the body frozen at its
  initial weights, the head is doing Bayesian linear regression on
  whatever feature map the random body happens to provide.  For a
  smooth target like `sin(πx)` this is mildly OK; for higher-frequency
  targets (e.g. the `sin³(6x)` problem of T07/T09) it would be
  noticeably worse.  The pattern that combines a *learned* feature map
  with a Bayesian head — Pattern B — is the practically useful one and
  is the v2 priority.
* **Multi-chain + masks** is also blocked in v1 (`num_chains > 1`
  raises with masks).  Single-chain SGLD with a thin schedule is what
  this tutorial uses.
* **Window adaptation** (NUTS/HMC with `adapt=True`) is not exercised
  against masks in v1 either — SGLD has no adaptation hook so the
  fixed-step pattern composes cleanly with the masked dispatch.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/10_masked_bnn_head.py"
```
