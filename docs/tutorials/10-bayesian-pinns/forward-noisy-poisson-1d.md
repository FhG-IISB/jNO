# Forward B-PINN: 1-D Poisson with noisy data

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/01_forward_noisy_poisson_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Forward uncertainty quantification.**  Standard 1-D Poisson
`u''(x) = -sin(πx)` with two **noisy** Dirichlet boundary observations.
Instead of optax, the MLP weights are sampled via SGLD; after training,
`crux.eval([u])` auto-vmaps the evaluator over the chain to give a
posterior mean and a credible band.

## Why this matters

A deterministic PINN returns one weight vector and one prediction; you
can't tell *where* the model is uncertain.  A B-PINN's predictive band
widens in data-sparse regions and tightens near observations — the
mechanism Yang et al. 2021 illustrated as the central advantage of the
Bayesian framework.

## Reference

Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/01_forward_noisy_poisson_1d.py"
```
