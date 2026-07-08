# Bayesian PINNs

A worked example of Bayesian Physics-Informed Neural Networks
(B-PINNs) in jNO: recovering an unknown PDE coefficient from noisy data
with a **calibrated** posterior.  Training is driven through
`crux.solve()` using jNO's per-parameter `.bayesian(...)` (MCMC) or
`.vi(...)` (variational) configurator, which attaches a blackjax
inference algorithm to scalar PDE coefficients, model weights, or
inverted inputs.

[Tutorial 01](./bayesian-inverse-coefficient.md) fits a neural
surrogate to noisy field observations, then **freezes** it and runs
NUTS on a single reaction coefficient — a fixed-target posterior whose
90% credible interval covers the truth, with R-hat ≈ 1 across four
chains.

All chains are built on the [blackjax MCMC
library](https://github.com/blackjax-devs/blackjax) — NUTS, HMC,
MALA, SGLD, SGHMC, plus window adaptation.  Background and API
reference for the `.bayesian()` integration live in
[Training → Bayesian Sampling](../../training/bayesian.md).

## Examples

| # | File | What it shows | Reference |
|---|---|---|---|
| 01 | [`bayesian_inverse_coefficient`](./bayesian-inverse-coefficient.md) | Bayesian inverse PINN: freeze a neural surrogate, then NUTS on an unknown reaction coefficient `k` — a calibrated posterior (90% CI covers the truth, R-hat ≈ 1, ESS ≈ 600). | Yang et al. 2021; Hoffman & Gelman 2014 |

## When to use a neural surrogate (and when not to)

A naïve mixed-mode B-PINN — optax on the surrogate, NUTS on the
coefficient simultaneously — samples the coefficient against a
**moving target** because the surrogate shifts every step.  That isn't
proper MCMC on a fixed posterior, and the resulting credible interval
becomes brittle to hyperparameter choices (different step sizes,
warmup lengths, surrogate sizes can all give materially different
posteriors).

Two patterns avoid the moving-target problem:

* **No surrogate at all.**  If the forward model has a closed form
  (e.g. `exp(-kt)` for first-order decay, `sin(πx)/π²` for the
  analytical Poisson reference), plug it directly into the likelihood
  and let NUTS sample a **fixed-target** posterior over the coefficient.
  Hyperparameters then affect chain efficiency only, not the target.
* **Train, then freeze.**  Fit the surrogate to convergence with optax,
  call `u_net.freeze()`, and sample the coefficient against the now-**fixed**
  forward map — the fixed target makes window adaptation well-defined and the
  posterior cleanly calibrated (this is what Tutorial 01 does).  For a surrogate
  that must keep training *alongside* sampling, jNO's
  `substeps=[([surrogate], n_train), ([residual], 1)]` interleaves the two —
  substep 0 trains the surrogate, substep 1 runs one NUTS proposal with the
  surrogate `stop_gradient`-ed.

## How to read the chain output

For every Bayesian model the chain is on `model.posterior_samples`
(shape `(n_kept, *param_shape)`).  Through `crux.eval([expr])`, jNO
auto-detects any Bayesian dependency in `expr` and vmap-pushes the chain
through, giving you posterior predictive arrays at any spatial point.
See [training/bayesian.md](../../training/bayesian.md) for the full
API.

A "good" B-PINN result looks like:

1. **Posterior mean close to truth.**  How close depends on noise level,
   chain length, and (for mixed-mode runs) how well the optax surrogate
   converges before sampling starts.
2. **Truth inside the credible interval.**  If truth lies *outside* the
   90 % CI, the posterior is mis-calibrated — usually a sign that the
   step size is too large or the warmup too short.
3. **CI width that's neither zero nor enormous.**  Zero width means the
   chain hasn't mixed; very wide intervals mean the data + physics don't
   pin down the parameter (an honest answer for some inverse problems!).

## References

- Yang, L., Meng, X., & Karniadakis, G. E. (2021).
  *B-PINNs: Bayesian physics-informed neural networks for forward and
  inverse PDE problems with noisy data.*
  Journal of Computational Physics, 425, 109913.
- Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler.*
  Journal of Machine Learning Research, 15(1), 1593-1623.
