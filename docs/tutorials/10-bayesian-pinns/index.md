# Bayesian PINNs

Twelve worked examples of Bayesian Physics-Informed Neural Networks
(B-PINNs) in jNO.  Every tutorial drives training through
`crux.solve()` and uses jNO's per-parameter `.bayesian(...)` (MCMC) or
`.vi(...)` (variational) configurator to attach a blackjax inference
algorithm to scalar PDE coefficients, model weights, or inverted
inputs.  [Tutorial 08](./multichain-nuts.md) adds multi-chain sampling
with Gelman-Rubin R-hat and effective sample size diagnostics;
[Tutorial 09](./vi-bnn-regression.md) trains the same BNN regressor
as T07 via mean-field variational inference for tighter, faster
posterior bands; [Tutorial 10](./masked-bnn-head.md) demonstrates the
`.mask(M).bayesian(...)` per-mask backend dispatch — sampling only a
chosen subset of a model's parameter pytree; [Tutorials 11](./pathfinder-init.md)
and [12](./laplace-init.md) expose the *logdensity-aware initializer*
hook on `.initialize()`, landing **pathfinder** (Zhang et al. 2022)
and **Laplace** (MacKay 1992; Daxberger et al. 2021) as warm-starts
for any HMC-family chain — the same extension point future SVGD /
MAP initializers will plug into.

Two tutorials demonstrate training an **entire MLP via Bayesian
sampling** (no optax): [Tutorial 01](./forward-noisy-poisson-1d.md)
treats the MLP as a *PINN* with a PDE residual constraint;
[Tutorial 07](./bnn-regression.md) treats it as a pure *regressor*
with only a data-fit constraint — same `.bayesian()` plumbing, no PDE
machinery.

All chains are built on the [blackjax MCMC
library](https://github.com/blackjax-devs/blackjax) — NUTS, HMC,
MALA, SGLD, SGHMC, plus window adaptation.  Background and API
reference for the `.bayesian()` integration live in
[Training → Bayesian Sampling](../../training/bayesian.md).

## Examples

| # | File | What it shows | Reference |
|---|---|---|---|
| 01 | [`forward_noisy_poisson_1d`](./forward-noisy-poisson-1d.md) | Forward-UQ B-PINN: SGLD over MLP weights for the 1-D Poisson with noisy boundary data — prediction bands that widen in data-sparse regions. | Yang et al. 2021 §3.2.1 |
| 02 | [`inverse_multi_coefficient`](./inverse-multi-coefficient.md) | Per-parameter NUTS on (A, B) of a harmonic-regression target — purest demonstration of `.bayesian()` for each scalar. | Yang et al. 2021 |
| 03 | [`inverse_reaction_coefficient`](./inverse-reaction-coefficient.md) | NUTS on the scalar `k` in `λ u'' + k tanh(u) = f` using the closed-form `u` — a fixed-target posterior. | Yang et al. 2021 §3.3.1 |
| 04 | [`inverse_ode_decay`](./inverse-ode-decay.md) | First-order decay ODE `du/dt = -k u`; recovers `k` from noisy observations using the closed-form `exp(-kt)` — no surrogate, fixed-target posterior. | Linka et al. 2022 |
| 05 | [`inverse_surrogate_uncertainty`](./inverse-surrogate-uncertainty.md) | Forward-then-freeze: train a PINN surrogate of `sin(πx)`, then NUTS samples the **inverted input** `x_query` given a noisy observation — calibrated uncertainty on the inverse output. | — |
| 06 | [`inverse_fem_diffusivity`](./inverse-fem-diffusivity.md) | Bayesian inference with jNO's **FEM solver** as the differentiable forward: recover the scalar diffusivity `α` in `-α Δu = f` from noisy nodal observations.  Pattern for any numerical (non-closed-form) forward. | — |
| 07 | [`bnn_regression`](./bnn-regression.md) | **Full BNN regression, no PDE.**  SGLD over MLP weights against a 32-point gapped training set of `sin³(6x)`; predictive band widens ~2.5× in the data gap.  Canonical "uncertainty grows where data don't constrain" demonstration. | Yang et al. 2021 §3.1 |
| 08 | [`multichain_nuts`](./multichain-nuts.md) | **Four parallel chains** per parameter with R-hat / ESS convergence diagnostics.  Same recovery problem as T02 but with `num_chains=4` + `init_jitter`; uses `jno.bayesian.{rhat, ess}` (pure-JAX, no arviz). | Gelman & Rubin 1992; Vehtari et al. 2021 |
| 09 | [`vi_bnn_regression`](./vi-bnn-regression.md) | **Mean-field Variational Inference** on the same gapped BNN regression problem as T07.  Optimisation-based alternative to SGLD: faster convergence, tighter in-data bands, smaller gap-vs-data ratio.  Demonstrates `Model.vi(blackjax.meanfield_vi, ...)` with the residual-by-√N scaling needed for sum-likelihood VI. | Kucukelbir et al. 2017 |
| 10 | [`masked_bnn_head`](./masked-bnn-head.md) | **`.mask(M).bayesian(...)` per-mask backend dispatch.**  2-layer MLP with the output linear layer ("head", 17 params) SGLD-sampled while the hidden body (304 params) stays at random init.  Demonstrates the v1 contract: chain-variance is non-zero on the masked head and machine-precision-small on the unmasked body.  Pattern B (head Bayesian + body Adam-trained) is the v2 plan — see [Training → Bayesian Sampling](../../training/bayesian.md#composable-per-mask-backends-v1). | — |
| 11 | [`pathfinder_init`](./pathfinder-init.md) | **`.initialize(jno.bayesian.pathfinder(...))` warm-start.**  Logdensity-aware initializer hook on `Model.initialize` — pathfinder runs L-BFGS on the loss-derived log-density and returns a warm starting position + a diagonal `inverse_mass_matrix` estimate.  Three side-by-side runs (baseline / pathfinder-only / pathfinder + window chain) on T02's harmonic-regression problem.  Demonstrates the extensible `_BayesianInitializer` protocol that future Laplace / SVGD / MAP initializers will plug into. | Zhang et al. 2022 |
| 12 | [`laplace_init`](./laplace-init.md) | **`.initialize(jno.bayesian.laplace(...))` warm-start.**  Second logdensity-aware initializer on the same `.initialize()` hook: finds the MAP via optax (Adam by default) and forms `N(MAP, H⁻¹)` with an exact Hessian at the MAP.  Diagonal-Hessian strategy scales to BNN-size pytrees; full-Hessian strategy gives clean correlated posterior covariance for small models.  Two side-by-side runs (baseline / laplace) on T02's problem. | MacKay 1992 §6; Daxberger et al. 2021 §2 |

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
  Tutorials 03 and 04 use this approach.
* **Two-stage via `substeps=`.**  When the surrogate is genuinely
  needed (the PDE has no tractable closed form), use jNO's
  `substeps=[([surrogate-constraints], n_train), ([pde-constraint], 1)]`
  with `.stop_gradient()` on the surrogate in the PDE-residual term.
  Substep 0 trains the surrogate (`n_train` steps); substep 1 runs one
  NUTS proposal with the surrogate frozen.  The 20:1 ratio (or higher)
  approximates an idealised two-stage where the surrogate fully
  converges before sampling.  None of the tutorials in this section
  *require* this pattern (we picked problems with closed-form forward
  models for clarity), but the substep machinery is wired and tested
  for it.

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
- Linka, K., Schäfer, A., Meng, X., Zou, Z., Karniadakis, G. E., & Kuhl,
  E. (2022).  *Bayesian Physics-Informed Neural Networks for real-world
  nonlinear dynamical systems.*  Computer Methods in Applied Mechanics
  and Engineering, 402, 115346.
- Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler.*
  Journal of Machine Learning Research, 15(1), 1593-1623.
- Welling, M., & Teh, Y. W. (2011).  *Bayesian Learning via Stochastic
  Gradient Langevin Dynamics.*  ICML 2011, 681-688.
