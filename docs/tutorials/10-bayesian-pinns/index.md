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

**Train, then freeze, in full.** The whole pattern is two `jno.core` calls with a
`freeze()` between them — the first fits the surrogate, the second samples the
coefficient against the now-fixed forward map:

```python
import jax, jax.numpy as jnp, numpy as np, optax, blackjax, foundax, jno

π, LAMBDA, K_TRUE = jnp.pi, 0.1, 2.0

d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.05).domain()
x, _ = d.variable("interior")
x_np = np.asarray(d.context["interior"]).reshape(-1, 1)
u_obs = jnp.asarray(np.sin(π * x_np) + 0.02 * np.random.default_rng(0).normal(size=x_np.shape))
f_known = (LAMBDA * π**2 + K_TRUE) * jno.np.sin(π * x)     # a known physical input

# ── phase 1: fit the surrogate to the noisy data ─────────────────────────────
u_net = jno.nn(foundax.mlp(1, output_dim=1, hidden_dims=32, num_layers=3,
                           key=jax.random.PRNGKey(0)))
u_net.optimizer(optax.adam(2e-3))
u = u_net(x) * x * (1 - x)                        # hard u(0) = u(1) = 0
jno.core([(u - u_obs).mse]).solve(1500)

# ── phase 2: freeze it, then sample k against the now-FIXED forward map ──────
u_net.freeze()          # ⇒ fixed-target posterior ⇒ window adaptation is well-defined
k = jno.np.parameter((1,), name="k").initialize(jax.nn.initializers.constant(1.0))
k.bayesian(blackjax.nuts, step_size=1e-2, warmup=150, keep=300,
           num_chains=4,        # 4 chains ⇒ a meaningful R-hat
           init_jitter=0.5)     # over-disperse the starts so R-hat stays conservative

uf = u_net(x) * x * (1 - x)
residual = (-LAMBDA * uf.dd(x) + k * uf - f_known) / 0.05   # −λu″ + k u − f = 0
jno.core([residual.mse]).solve(450)

# ── read the chain ──────────────────────────────────────────────────────────
ch = k.posterior_samples                            # (num_chains, keep, 1)
lo, hi = jnp.quantile(ch.reshape(-1), jnp.array([0.05, 0.95]))
print(jnp.mean(ch), lo, hi, jno.bayesian.rhat(ch))
```

Run as written this gives `k = 2.028`, a 90 % CI of `[1.943, 2.112]` covering the truth
`2.0`, and R-hat `1.004`. [Tutorial 01](./bayesian-inverse-coefficient.md) is the same
pattern with a longer fit and a longer chain, plus the calibration diagnostics.

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
