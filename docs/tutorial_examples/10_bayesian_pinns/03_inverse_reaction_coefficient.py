"""03 — Bayesian inverse: recover nonlinear reaction coefficient k

Problem (Yang et al. 2021, §3.3.1, steady-state form)
-----------------------------------------------------
    λ u''(x) + k · tanh(u(x)) = f(x),   x ∈ [-0.7, 0.7],   λ = 0.01,

with Dirichlet boundary conditions matching the analytical reference

    u_exact(x) = sin(πx) / π² ,

from which ``f`` is generated under the **true** value ``k_true = 0.7``.
Given noisy observations of the forcing ``f(x)`` plus knowledge of the
solution ``u(x)``, recover the posterior over ``k``.

Why no neural surrogate?
------------------------
This tutorial deliberately avoids a PINN surrogate.  In a mixed-mode
B-PINN (optax-trained surrogate + NUTS-sampled coefficient), the kernel
sees a moving-target logdensity because the surrogate shifts every
step.  That isn't proper MCMC on a fixed posterior, and the chain
becomes brittle to hyperparameters.

Here ``u(x)`` is known analytically, so we plug ``u_exact`` directly
into the PDE-residual likelihood.  NUTS samples a **fixed-target**
posterior; hyperparameters affect chain efficiency only.

When ``u`` is **not** known analytically (e.g. a real PDE solve), use
the *two-stage* pattern via ``substeps=[([0, 1], 20), ([2], 1)]``:
substep 0 trains a surrogate from data, substep 1 samples ``k`` against
the PDE residual with ``surrogate.stop_gradient``.  See
``docs/training/bayesian.md`` for the trade-offs.

Reference
---------
Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913.
"""

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp

import jno

π = jno.np.pi

# ── Physical setup ────────────────────────────────────────────────────────────
λ = 0.01
k_true = 0.7
sigma_obs = 0.005  # noise on observed f(x)

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.line(x_range=(-0.7, 0.7), mesh_size=0.05)
x, _ = domain.variable("interior")

# ── Analytical solution + noisy forcing observations under k_true ────────────
u = jno.np.sin(π * x) / π**2  # exact solution
u_xx = -jno.np.sin(π * x)  # analytical 2nd derivative (no FD noise)

# Synthetic noisy forcing observations.  jno.noise.gaussian is redrawn
# each step from the solver's PRNG key — with a fixed global seed the
# realisation is consistent across the whole chain.
f_obs = λ * u_xx + k_true * jno.np.tanh(u) + jno.noise.gaussian(std=sigma_obs)

# ── Bayesian reaction coefficient — NUTS with closed-form forward ────────────
k = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="k")
k.bayesian(
    blackjax.nuts,
    step_size=1e-2,  # initial guess; window adaptation tunes it
    warmup=200,
    keep=400,
    # adapt=True default — pure Bayesian inference, adaptation is well-defined.
)

# ── Likelihood: PDE-residual scaled by σ ──────────────────────────────────────
residual = (λ * u_xx + k * jno.np.tanh(u) - f_obs) / sigma_obs

# ── Solve — pure Bayesian, no surrogate, no substeps needed ──────────────────
crux = jno.core([residual.mse], domain)
crux.solve(600)

# ── Posterior summary ────────────────────────────────────────────────────────
k_chain = k.posterior_samples
k_mean = float(jnp.mean(k_chain))
k_std = float(jnp.std(k_chain))
k_lo, k_hi = (float(v) for v in jnp.quantile(k_chain, jnp.array([0.05, 0.95])))

print(f"k = {k_mean:.4f} ± {k_std:.4f}")
print(f"   90% CI = [{k_lo:.4f}, {k_hi:.4f}]   truth = {k_true}")

rel_k = abs(k_mean - k_true) / abs(k_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/03_inverse_reaction_coefficient.py | epochs=600 | "
        f"rel_k={rel_k:.4f} | CI_width={k_hi - k_lo:.4f}\n"
    )

assert rel_k < 0.1, f"posterior-mean k off by {rel_k:.2%}"
