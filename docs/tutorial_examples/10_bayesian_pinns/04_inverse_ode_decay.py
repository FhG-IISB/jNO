"""04 — Bayesian inverse for ODE rate constant (no surrogate, closed-form)

Problem
-------
A first-order decay ODE

    du/dt = -k · u(t),   t ∈ [0, T_end],   u(0) = 1,

has the closed-form solution ``u_exact(t) = exp(-k_true · t)``.  Given
sparse noisy observations of ``u``, recover the posterior over ``k``.

Why no neural surrogate?
------------------------
A mixed-mode B-PINN (optax-trained surrogate + NUTS-sampled coefficient)
samples ``k`` against a moving-target logdensity — the kernel's
distribution shifts each step as the surrogate updates.  That isn't
proper MCMC on a fixed posterior, and the resulting credible interval
becomes brittle to hyperparameter choices.

When the forward model has a closed form (as here: ``exp(-kt)``), the
right thing is **not** to PINN-approximate it — it's to plug the
analytical solution directly into the likelihood and let NUTS sample a
**fixed** posterior.  Hyperparameters then affect chain efficiency, not
the target.  When a closed form is *not* available (a real PDE with no
analytical solution), the recommended pattern is the **two-stage**
approach via ``substeps=[([surrogate_constraints], n_train), ([pde], 1)]``
with ``surrogate.stop_gradient()`` applied to the surrogate output in the
PDE residual — substep 0 trains the surrogate, substep 1 samples the
coefficient with the surrogate frozen.  See
``docs/training/bayesian.md`` for the trade-offs.

Technique
---------
* ``k`` is a scalar trainable parameter, sampled with NUTS.
* The "surrogate" is the closed-form expression ``exp(-k · t)``; no
  network, no optax.
* Loss = sum of squared residuals against synthetic noisy observations
  (Gaussian-noise likelihood with known ``σ``).

References
----------
Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo.*  JMLR 15(1), 1593-1623.

Linka, K., Schäfer, A., Meng, X., Zou, Z., Karniadakis, G. E., & Kuhl, E.
(2022).  *Bayesian Physics-Informed Neural Networks for real-world
nonlinear dynamical systems.*  Computer Methods in Applied Mechanics
and Engineering, 402, 115346.  (Linka et al. use a surrogate because
their SIR system has no closed form; here we don't need one.)
"""

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp

import jno

# ── Physical setup ────────────────────────────────────────────────────────────
k_true = 0.5
T_end = 4.0
sigma_obs = 0.05  # observation-noise standard deviation

# ── Domain (1-D line over the time axis: x ≡ t) ───────────────────────────────
domain = jno.domain.line(x_range=(0.0, T_end), mesh_size=0.1)
t, _ = domain.variable("interior")

# ── Synthetic noisy observations of u(t) under k_true ─────────────────────────
# jno.noise.gaussian is redrawn every step from the solver's PRNG key;
# with a fixed global seed the realisation is deterministic across the
# whole chain, so the likelihood NUTS sees is consistent.
u_obs = jno.np.exp(-k_true * t) + jno.noise.gaussian(std=sigma_obs)

# ── Bayesian rate constant — NUTS with closed-form forward model ─────────────
k = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="k")
k.bayesian(
    blackjax.nuts,
    step_size=1e-1,  # initial guess; window adaptation tunes it
    warmup=400,
    keep=600,
    # adapt=True default: window_adaptation runs for `warmup` steps and
    # tunes step_size + inverse_mass_matrix.  This is pure Bayesian
    # inference (no mixed mode), so adaptation is well-defined.
)

# ── Closed-form forward model: u(t; k) = exp(-k · t) ──────────────────────────
u_model = jno.np.exp(-k * t)

# ── Likelihood: Gaussian-noise residual scaled by σ ───────────────────────────
# The 1/σ² factor is the proper log-likelihood weighting; jno's `.mse`
# averages the squared residual.  Multiplying by 1/σ² gives the correct
# scale of the Gaussian log-density (up to a constant).
residual = (u_model - u_obs) / sigma_obs

# ── Solve — pure Bayesian, no surrogate, no substeps needed ──────────────────
crux = jno.core([residual.mse], domain)
crux.solve(1000)

# ── Posterior summary ────────────────────────────────────────────────────────
k_chain = k.posterior_samples  # shape (600, 1)
k_mean = float(jnp.mean(k_chain))
k_std = float(jnp.std(k_chain))
k_lo, k_hi = (float(v) for v in jnp.quantile(k_chain, jnp.array([0.05, 0.95])))

print(f"k = {k_mean:.4f} ± {k_std:.4f}")
print(f"   90% CI = [{k_lo:.4f}, {k_hi:.4f}]   truth = {k_true}")

rel_k = abs(k_mean - k_true) / abs(k_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"10_bayesian_pinns/04_inverse_ode_decay.py | epochs=1000 | rel_k={rel_k:.4f} | CI_width={k_hi - k_lo:.4f}\n")

assert rel_k < 0.1, f"posterior-mean k off by {rel_k:.2%}"
