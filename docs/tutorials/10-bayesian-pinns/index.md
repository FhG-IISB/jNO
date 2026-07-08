# Bayesian PINNs

A worked example of Bayesian Physics-Informed Neural Networks
(B-PINNs) in jNO.  Training is driven through `crux.solve()` using
jNO's per-parameter `.bayesian(...)` (MCMC) or `.vi(...)` (variational)
configurator, which attaches a blackjax inference algorithm to scalar
PDE coefficients, model weights, or inverted inputs.

The tutorial trains an **entire MLP via Bayesian sampling** (no optax):
[Tutorial 01](./forward-noisy-poisson-1d.md) treats the MLP as a
*PINN* with a PDE residual constraint.

All chains are built on the [blackjax MCMC
library](https://github.com/blackjax-devs/blackjax) — NUTS, HMC,
MALA, SGLD, SGHMC, plus window adaptation.  Background and API
reference for the `.bayesian()` integration live in
[Training → Bayesian Sampling](../../training/bayesian.md).

## Examples

| # | File | What it shows | Reference |
|---|---|---|---|
| 01 | [`forward_noisy_poisson_1d`](./forward-noisy-poisson-1d.md) | Forward-UQ B-PINN: SGLD over MLP weights for the 1-D Poisson with noisy boundary data — prediction bands that widen in data-sparse regions. | Yang et al. 2021 §3.2.1 |

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
* **Two-stage via `substeps=`.**  When the surrogate is genuinely
  needed (the PDE has no tractable closed form), use jNO's
  `substeps=[([surrogate-constraints], n_train), ([pde-constraint], 1)]`
  with `.stop_gradient` on the surrogate in the PDE-residual term.
  Substep 0 trains the surrogate (`n_train` steps); substep 1 runs one
  NUTS proposal with the surrogate frozen.  The 20:1 ratio (or higher)
  approximates an idealised two-stage where the surrogate fully
  converges before sampling.  The substep machinery is wired and
  tested for this pattern.

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
