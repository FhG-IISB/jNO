# Inverse: nonlinear reaction coefficient (closed-form)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/04_inverse_reaction_coefficient.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Reaction-coefficient inverse problem with a fixed-target posterior.**
Recover the scalar `k` in `λ u''(x) + k tanh(u(x)) = f(x)` from noisy
observations of the forcing `f`.  Because the analytical solution
`u_exact = sin(πx)/π²` is known, we plug it directly into the PDE
residual instead of training a surrogate.  NUTS samples a fixed-target
posterior over `k` with calibrated, hyperparameter-stable credible
intervals — matching Yang et al. 2021's textbook B-PINN HMC result
(`k ≈ 0.705 ± 0.006` for the small-noise case, our short chain hits
`k ≈ 0.703 ± 0.043`).

If `u` were unknown (a real PDE solve), the surrogate approach via
two-stage `substeps=[([0, 1], 20), ([2], 1)]` is the recommended
pattern — see [the source-recovery
tutorial](./inverse-source-steady-state.md) for that variant.

## Reference

Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/04_inverse_reaction_coefficient.py"
```
