# Inverse: nonlinear reaction coefficient (closed-form)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/03_inverse_reaction_coefficient.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
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

If `u` were unknown (a real PDE solve), the two-stage surrogate
pattern via `substeps=[([surrogate_constraints], n_train), ([pde], 1)]`
with `.stop_gradient` on the surrogate in the PDE residual is the
recommended alternative.  Tutorial 06 ([Inverse FEM
Diffusivity](./inverse-fem-diffusivity.md)) shows the more general
case where the forward model is fully numerical (FEM).

## Reference

Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/03_inverse_reaction_coefficient.py"
```
