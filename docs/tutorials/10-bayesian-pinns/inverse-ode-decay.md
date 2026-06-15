# Inverse: ODE rate constant (no surrogate)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/04_inverse_ode_decay.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Bayesian rate-constant inference for a first-order ODE with a fixed
posterior.**  The model `du/dt = -k u(t)` has the closed-form solution
`u(t) = exp(-k t)`.  We plug this analytical expression directly into
the likelihood — no neural surrogate is involved — and let NUTS sample
a fixed-target posterior over `k` from sparse noisy observations.

This is the simplest member of a broad family of real-world
rate-constant inverse problems (radioactive decay, first-order
pharmacokinetic elimination, single-compartment epidemic dynamics).
Linka et al. 2022 use the same Bayesian inference recipe for COVID-19
SIR modelling; the only difference is the dimensionality of the state
vector and the noise model.

**Why no neural surrogate?**  When the forward model has a closed form
(or a cheap numerical integrator), wrapping it in a PINN introduces
mixed-mode noise that makes the posterior brittle to hyperparameters.
A direct analytical likelihood gives a properly-defined Bayesian
inference where hyperparameters affect *chain efficiency* only — not
the *target*.  See [Inverse FEM
Diffusivity](./inverse-fem-diffusivity.md) for the pattern when no
closed form is available: jNO's FEM solver provides the differentiable
forward and blackjax samples the posterior directly.

## References

Linka, K., Schäfer, A., Meng, X., Zou, Z., Karniadakis, G. E., & Kuhl,
E. (2022).  *Bayesian Physics-Informed Neural Networks for real-world
nonlinear dynamical systems.*  Computer Methods in Applied Mechanics
and Engineering, 402, 115346.

Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler.*
Journal of Machine Learning Research, 15(1), 1593-1623.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/04_inverse_ode_decay.py"
```
