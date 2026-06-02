# Inverse B-PINN: ODE rate constant

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/05_inverse_ode_decay.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Bayesian rate-constant inference for a first-order ODE.**  The model
`du/dt = -k u(t)` with `u(0) = 1` is the simplest member of a wide
family of real-world rate-constant inverse problems: radioactive decay,
first-order pharmacokinetic elimination, single-compartment epidemic
dynamics.  We sample `k` with NUTS while an MLP surrogate is trained
with Adam.

Linka et al. 2022 use the same Bayesian PINN recipe on the COVID-19 SIR
model; the only change is the dimensionality of the state vector and
the noise model on the observations.  The setup here is a tractable
1-D variant that runs in under a minute on CPU and demonstrates the
key pattern: long warmup so the optax surrogate converges before NUTS
begins sampling, then a `keep`-length chain to estimate the posterior.

## Reference

Linka, K., Schäfer, A., Meng, X., Zou, Z., Karniadakis, G. E., & Kuhl,
E. (2022).  *Bayesian Physics-Informed Neural Networks for real-world
nonlinear dynamical systems.*  Computer Methods in Applied Mechanics
and Engineering, 402, 115346.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/05_inverse_ode_decay.py"
```
