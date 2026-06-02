# Bayesian PINNs

Five worked examples of Bayesian Physics-Informed Neural Networks
(B-PINNs) in jNO.  Each script wires one or more blackjax kernels into
`solve()` via `.bayesian(...)` and returns posterior chains over either
neural-network weights, scalar PDE coefficients, or both.

Background and API reference live in
[Training → Bayesian Sampling](../../training/bayesian.md).

## Examples

| # | File | What it shows | Reference |
|---|---|---|---|
| 01 | [`forward_noisy_poisson_1d`](./forward-noisy-poisson-1d.md) | Forward-UQ B-PINN: SGLD over MLP weights for the 1-D Poisson with noisy boundary data — prediction bands that widen in data-sparse regions. | Yang et al. 2021 §3.2.1 |
| 02 | [`inverse_multi_coefficient`](./inverse-multi-coefficient.md) | Per-parameter NUTS on (A, B) of a noisy harmonic-regression target — purest demonstration of `.bayesian()` for each scalar. | Yang et al. 2021 |
| 03 | [`inverse_source_steady_state`](./inverse-source-steady-state.md) | Mixed-mode: NUTS over an unknown source amplitude in `α u'' + A sin(πx) = 0`, with an optax-trained surrogate. | Yang et al. 2021 §3.3 |
| 04 | [`inverse_reaction_coefficient`](./inverse-reaction-coefficient.md) | Mixed-mode NUTS on the scalar `k` in the nonlinear PDE `λ u'' + k tanh(u) = f`. | Yang et al. 2021 §3.3.1 |
| 05 | [`inverse_ode_decay`](./inverse-ode-decay.md) | First-order decay ODE `du/dt = -k u`; recovers the rate constant `k` from noisy observations — same recipe Linka et al. 2022 use for COVID-19 SIR models. | Linka et al. 2022 |

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
