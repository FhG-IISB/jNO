# Multi-chain NUTS with R-hat and ESS

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/08_multichain_nuts.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Four parallel chains per Bayesian parameter, with Gelman-Rubin R-hat
and effective sample size (ESS) confirming convergence.**  Same
inverse problem as [Tutorial 02](./inverse-multi-coefficient.md)
(recover `(A, B)` in `d(x) = A sin(πx) + B cos(πx)`), but using
``.bayesian(blackjax.nuts, num_chains=4, init_jitter=0.1, ...)`` per
parameter and `jno.bayesian.{rhat, ess}` for diagnostics.

## Why multi-chain?

A single MCMC chain can look stationary yet still be stuck in a local
mode.  **R-hat** (Gelman & Rubin 1992; Vehtari et al. 2021) compares
between-chain to within-chain variance: values close to 1.0 (≤ 1.05
by community convention) indicate the chains explore the same
posterior; values much larger flag non-convergence.

**ESS** quantifies how many *independent* draws a chain is worth given
its autocorrelation.  An ESS of ≈ 100+ per parameter is typically
considered sufficient.  Both helpers in jNO are pure JAX (no arviz
dep) and operate directly on `posterior_samples` shape `(K, N, *)`.

## What the tutorial reports

| Metric | A | B |
|---|---|---|
| Posterior mean | 3.09 | −2.62 |
| 90 % CI | [1.66, 4.70] | [−4.18, −1.06] |
| Truth | 3.14 | −2.71 |
| R-hat | 1.0004 | 1.0052 |
| ESS | ~89 / 1600 draws | ~88 / 1600 draws |

R-hat ≪ 1.01 → chains have converged to the same distribution.  ESS
~89 (vs 1600 raw draws) is modest — that's the cost of strong sample
autocorrelation in NUTS on a short chain.

## API used

```python
a.bayesian(
    blackjax.nuts,
    step_size=1e-2,
    warmup=300,
    keep=400,
    num_chains=4,
    init_jitter=0.1,   # over-disperses chain starts → conservative R-hat
)

# After solve, posterior_samples has shape (K, N, *param):
a.posterior_samples.shape  # (4, 400, 1)

# Diagnostics:
jno.bayesian.rhat(a.posterior_samples)  # → (1,)
jno.bayesian.ess(a.posterior_samples)   # → (1,)
```

## References

- Gelman, A., & Rubin, D. B. (1992).  *Inference from iterative
  simulation using multiple sequences.*  Statistical Science 7(4),
  457-511.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C.
  (2021).  *Rank-Normalization, Folding, and Localization: An
  Improved R̂ for Assessing Convergence of MCMC.*  Bayesian Analysis
  16(2), 667-718.
- Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler.*
  JMLR 15(1), 1593-1623.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/08_multichain_nuts.py"
```
