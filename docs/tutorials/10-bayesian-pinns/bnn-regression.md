# BNN regression: full MLP via SGLD, no PDE

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/07_bnn_regression.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Pure Bayesian-neural-network regression with calibrated uncertainty in
data-sparse regions.**  Approximate `u(x) = sin³(6x)` from 32 noisy
observations placed in `[-0.8, -0.2] ∪ [0.2, 0.8]` — a deliberate gap
around `x = 0`.  Every weight in a small MLP is sampled with SGLD; the
only constraint is the data-fit MSE.

Compared with [Tutorial 01](./forward-noisy-poisson-1d.md) (which
trains a BNN as a *PINN* — MLP weights sampled, PDE residual
constraining them), this tutorial is the **pure regression**
complement: no PDE residual, the MLP is judged purely on whether it
fits the data.

## Why this matters

A deterministic MLP returns one weight vector and one curve through
the data.  In the gap around `x = 0` it would extrapolate confidently
with no signal that the prediction there is essentially a guess.

A Bayesian MLP samples *many* networks that all fit the observed data
roughly equally well.  In the gap region those networks **disagree**
strongly — the credible band widens.  That uncertainty growing where
data don't constrain the model is the core BNN value proposition:

| Region | Posterior band (90 %) |
|---|---|
| In-data (`|x| ∈ [0.2, 0.8]`) | width ≈ 1.4 |
| In-gap (`|x| < 0.2`) | width ≈ 3.7 |
| Gap / in-data ratio | ≈ 2.6 × |

## Reference

Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913
(see §3.1 "Function regression").

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/07_bnn_regression.py"
```
