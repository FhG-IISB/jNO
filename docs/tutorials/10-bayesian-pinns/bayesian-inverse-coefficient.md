# Bayesian inverse PINN: recover a PDE coefficient

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/01_bayesian_inverse_coefficient.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Inverse uncertainty quantification.**  A steady reaction–diffusion system
`-λ u''(x) + k u(x) = f(x)` on the unit line with a **known** forcing `f` and an
**unknown** reaction coefficient `k`.  We observe the field `u` at noisy sensors
and want the *posterior* over `k` — a value **and** an honest credible interval,
not just a point estimate.

![Left: the neural surrogate fitting noisy sensor observations of the field and tracking the true sin(πx). Right: the NUTS posterior histogram over the reaction coefficient k, with the 90% credible interval shaded, the true value k=2 inside it, and R-hat≈1.](/jNO/assets/bayesian_inverse_coefficient.png)

## The two-stage recipe

Sampling the full ~1 000-weight network posterior on a stiff PINN loss is hard to
calibrate.  The robust pattern samples only the *tractable* unknown instead:

1. **Fit a neural surrogate** `u_net` to the noisy sensor data with optax
   (deterministic).  Hard `u·x·(1-x)` factors impose the `u(0)=u(1)=0` BCs.
2. **`u_net.freeze()`** turns the surrogate into a *fixed* forward map, so NUTS
   samples a well-defined, **fixed-target** posterior over the single scalar `k`
   through the physics residual `-λ u'' + k u - f`.  A fixed target is what makes
   window adaptation well-defined and the posterior cheap and cleanly calibrated.

```python
u_net.freeze()                                   # fixed forward map
k = jno.np.parameter((1,), name="k")
k.bayesian(blackjax.nuts, step_size=1e-2, warmup=300, keep=600,
           num_chains=4, init_jitter=0.5)        # 4 chains ⇒ a meaningful R-hat

uf = u_net(x) * x * (1 - x)
residual = (-LAMBDA * uf.dd(x) + k * uf - f_known) / SIGMA_PHYS
jno.core([residual.mse]).solve(900)              # recovers the posterior over k
```

## What to notice

- **The credible interval covers the truth.**  `k = 1.94 ± 0.05`, 90% CI
  `[1.87, 2.02]`, which contains the true `k = 2` — with `R-hat ≈ 1.00` and
  `ESS ≈ 600` across four chains confirming the chains mixed.
- **`sigma_phys` budgets for *model* error, not just observation noise.**  A
  differentiated neural surrogate carries a small `u''` bias; setting the physics
  likelihood scale to cover it is what keeps the posterior honest (truth inside
  the interval) rather than over-confident.
- **Read the chain, not a point estimate.**  A narrow interval away from the truth
  would signal an over-tight `sigma_phys` or a poor surrogate; a very wide interval
  means the data and physics simply don't pin `k` down.

## Reference

Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems with noisy
data.*  Journal of Computational Physics, 425, 109913.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/01_bayesian_inverse_coefficient.py:code"
```
