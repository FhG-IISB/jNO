# Inverse B-PINN: nonlinear reaction coefficient

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/04_inverse_reaction_coefficient.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Nonlinear reaction-coefficient inverse problem.**  Recover the scalar
`k` in the 1-D steady PDE `λ u''(x) + k tanh(u(x)) = f(x)` from
synthetic observations.  The nonlinear `tanh(u)` term couples the
coefficient to the surrogate's response — exactly the regime where a
joint Bayesian–PINN approach earns its keep.

Yang et al. 2021 §3.3.1 report `k ≈ 0.705 ± 0.006` in the low-noise
case (truth 0.7) with a 15 000-sample HMC chain.  This tutorial uses a
short chain on CPU; recovery is approximate but tracks the truth and
the credible interval is calibrated.

## Reference

Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/04_inverse_reaction_coefficient.py"
```
