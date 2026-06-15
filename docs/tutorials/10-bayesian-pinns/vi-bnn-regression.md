# Mean-field Variational Inference on a BNN regressor

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/09_vi_bnn_regression.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Same gapped regression problem as [Tutorial 07](./bnn-regression.md),
but trained via mean-field Variational Inference instead of SGLD.**
The 32 noisy observations of `u(x) = sin³(6x)` are placed in
`[-0.8, -0.2] ∪ [0.2, 0.8]` — a deliberate gap around `x = 0`.  Each
weight in a small MLP is given a Gaussian variational marginal
`q(θ_i) = N(μ_i, σ_i)`; the joint `q(θ) = ∏_i q(θ_i)` is fit by
maximising the evidence lower bound (ELBO), then 600 i.i.d. samples
are drawn from the fitted `q` for posterior summaries.

## VI vs SGLD on the same problem

| Metric | SGLD (T07) | Mean-field VI (T09) |
|---|---|---|
| In-data rel-L2 of mean | ≈ 0.19 | ≈ 0.24 |
| In-data band (90 %) | ≈ 1.53 | ≈ 0.32 |
| In-gap band (90 %) | ≈ 3.74 | ≈ 0.44 |
| Gap / in-data ratio | ≈ 2.45× | ≈ 1.38× |
| Wall-clock | similar (CPU, < 30 s) | similar (CPU, < 30 s) |

**VI gives much tighter bands**, but the gap-vs-data ratio is smaller
because mean-field's per-weight independence assumption typically
underestimates correlated weight uncertainty.  Both methods reproduce
the qualitative BNN behaviour (band widens in the gap) — VI just does
so with less dramatic absolute widths.

## Why scale the residual by √N?

Look at line 95 of the script::

    residual = (y_pred - y_train_const) / sigma_obs * jnp.sqrt(N_obs)

The canonical Gaussian-noise log-likelihood is the **sum** over data
points, but ``residual.mse`` returns the **mean**.  Multiplying the
residual by ``√N`` makes ``residual.mse`` equal to the sum of squared
standardised residuals, which is the right magnitude for the Bayesian
log-likelihood.  Without this rescaling the likelihood term is N
times too small and the prior dominates, leaving VI stuck near
initialisation.

(For MCMC tutorials in this section the same scaling would tighten
the posterior; it's not strictly required since MCMC's gradient signal
is more robust to magnitude than VI's stochastic ELBO gradient.)

## How jno's mean-field VI is initialised

jno overrides two of blackjax's defaults at solve-start (see
``jno/bayesian.py:init_state``):

1. ``state.mu = position`` (the model's initial weights), rather than
   blackjax's zeros — gives VI a sensible starting point on
   non-trivial architectures.
2. ``state.rho = -3.0`` everywhere (initial std ≈ 0.05), rather than
   blackjax's larger default — keeps initial MC ELBO samples close to
   the mean so the gradient estimator is low-variance from the start.
   The optimiser then *grows* rho where the posterior is genuinely
   wide.

## API used

```python
import blackjax, optax

u_net.vi(
    blackjax.meanfield_vi,
    optimizer=optax.adam(5e-3),
    num_samples=8,
    posterior_draws=600,
)
crux.solve(6000)              # 6000 ELBO optimisation steps
```

After solve, ``u_net.posterior_samples`` has shape ``(1, 600, *param)``
— exact same arviz layout as MCMC posteriors, so
``crux.eval(samples="auto")``, ``jno.bayesian.{rhat, ess}``, and wandb
stats work identically.

## References

* Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M.
  (2017).  *Automatic Differentiation Variational Inference.*  JMLR
  18(1), 430-474.
* Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013).
  *Stochastic Variational Inference.*  JMLR 14(1), 1303-1347.
* Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs:
  Bayesian physics-informed neural networks for forward and inverse
  PDE problems with noisy data.*  JCP 425, 109913 (uses VI on BNN
  PINN in §3.2).

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/09_vi_bnn_regression.py"
```
