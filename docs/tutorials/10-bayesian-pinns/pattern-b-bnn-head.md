# Pattern B — Bayesian Last Layer (head sampled, body Adam-trained)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/14_pattern_b_bnn_head.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**The practical Bayesian Last Layer recipe.**  A feature-extractor MLP
body is trained deterministically with Adam, while the output layer
("head") is MCMC-sampled to quantify predictive uncertainty.  Compared
with [Tutorial 10](./masked-bnn-head.md) (head Bayesian, body **frozen
at random init**), Pattern B trains the body simultaneously so the
head sees a *learned* feature map rather than a random one — and the
posterior bands tighten dramatically.

## What changed: from Pattern A to Pattern B

```python
# Tutorial 10 — Pattern A: body frozen at init, head sampled.
net.mask(head_mask).bayesian(blackjax.sgld, step_size=1e-3, ...)

# Tutorial 14 — Pattern B: body Adam-trained, head sampled.
net.optimizer(optax.adam(5e-3))                      # NEW — body trained
net.mask(head_mask).bayesian(blackjax.sgld, ...)     # head sampled
```

One line — `net.optimizer(...)` — added on top of the Pattern A recipe.
The masked Bayesian configurator was previously blocked from coexisting
with a global optimiser on the same model; Phase 15 lifts that block via
a `_MixedState` wrapper that carries both states under the same layer
key.

## How it works under the hood

At each step:

1. **Compute the full-loss gradient.**  Single `jax.value_and_grad`
   over the entire trainable pytree.
2. **Optax update — masked to the body.**  The optax chain is wrapped
   in `optax.masked` with the complement of the Bayesian mask, so only
   the body leaves receive updates.
3. **MCMC kernel step — masked to the head.**  The kernel's
   log-density closure reassembles the current body (just updated)
   with each candidate head, evaluates the loss, and samples the head.

This is a natural Metropolis-within-Gibbs / stochastic-approximation
EM scheme.  For K>1 chains the body's gradient is computed at the
chain-0 head sample (SAEM simplification — proper averaging across
chains would cost K forward passes per step).

## Numbers from the tutorial

Same `sin(πx)` regression problem as T07 / T10 (32 noisy observations,
σ = 0.05).  Trained for 2300 epochs (warmup 1500, keep 400, thin 2).

| Pattern | Body | Head | rel-L2 vs `sin(πx)` | Band median (90 %) | Head leaf var |
|---|---|---|---|---|---|
| A (T10) | **frozen** at init | SGLD | 0.394 | 0.240 | 6.5 × 10⁻² |
| **B (T14)** | **Adam-trained** | SGLD | **0.063** | 0.269 | 2.3 × 10⁻¹ |

Pattern B gives a **6× tighter posterior mean** at essentially the
same band width — the body's Adam updates pull the feature map toward
something useful, and the head's posterior tightens around it.

## Composition with the rest of the API

Pattern B composes cleanly with every existing feature:

* **Multi-chain (`num_chains=K`)** — kernel state is K-leading-masked;
  the body is a single shared point estimate; the body's gradient is
  computed at the chain-0 representative head sample.
* **Initializers** — `.initialize(jno.bayesian.pathfinder(...))`
  warm-starts the masked head subset; pathfinder runs only on the
  head's log-density, the body's optax loop then continues with the
  warm-started head.
* **R-hat / ESS diagnostics** — work unchanged on the masked head
  chain.
* **Auto-IMM injection** — fires on the masked subset only (the head
  dimension `D`).

## What's the same as Pattern A

The mask construction, the chain shape `(K, N, *full_param)`, the
`.posterior_samples` accessor, and `crux.eval([net(x)], samples="auto")`
all behave identically to Pattern A.  Drop-in: rename
`.mask(head_mask).bayesian(...)` → keep that line, just add
`net.optimizer(...)` before it.

## When to use Pattern B vs full-net Bayesian

* **Full-net Bayesian** (Tutorial 07) — every weight sampled by SGLD.
  Honest uncertainty propagation through the entire model, but
  long mixing time for high-dimensional posteriors.
* **Pattern A** (Tutorial 10) — random-feature Bayesian regression
  with a frozen body.  Useful as a baseline / demonstration; rarely
  optimal in practice.
* **Pattern B** (this tutorial) — Bayesian Last Layer: body
  Adam-trained for fast convergence on the feature map; head sampled
  for predictive uncertainty.  Often the best speed/quality tradeoff
  for B-PINNs and Bayesian regression on neural-network-induced
  features.

## References

* Snoek, J., Rippel, O., Swersky, K., Kiros, R., Satish, N.,
  Sundaram, N., Patwary, M. M. A., Prabhat, & Adams, R. P. (2015).
  *Scalable Bayesian Optimization Using Deep Neural Networks.*
  §3 (Adaptive basis regression with Bayesian linear regression on
  the last layer).  ICML 2015.
  [arXiv:1502.05700](https://arxiv.org/abs/1502.05700)
* Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M.,
  & Hennig, P. (2021).  *Laplace Redux — Effortless Bayesian Deep
  Learning.*  §3 (last-layer Laplace).  NeurIPS 2021.
  [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
* Cappé, O., & Moulines, E. (2009).  *On-line Expectation-Maximization
  Algorithm for Latent Data Models.*  §3 (SAEM convergence theory).
  Journal of the Royal Statistical Society, Series B, 71(3), 593-613.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/14_pattern_b_bnn_head.py"
```
