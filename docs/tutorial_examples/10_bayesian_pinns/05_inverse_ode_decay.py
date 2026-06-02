"""05 — Bayesian PINN inverse: rate constant of a 1-D decay ODE

Problem
-------
A first-order decay process governed by

    du/dt = -k · u(t),   t ∈ [0, T_end],   u(0) = 1,

with the closed-form solution ``u_exact(t) = exp(-k_true · t)`` used to
generate synthetic noisy observations at every interior grid point.

The exponential-decay ODE is the simplest member of a broad family of
real-world rate-constant inverse problems (radioactive decay,
first-order pharmacokinetic elimination, single-compartment epidemic
dynamics).  Linka et al. (2022) — *Bayesian PINNs for real-world
nonlinear dynamical systems* — use the same recipe for COVID-19 SIR
modelling; the only difference is the dimensionality of the state
vector and the noise model on the observations.

Technique
---------
* ``k`` is a scalar trainable parameter, sampled with NUTS (``adapt=False``
  — mixed-mode caveat).
* ``net(t)`` is a small MLP, *optimised* with Adam — surrogate ``u(t)``.
* IC ``u(0) = 1`` is enforced via a boundary term on the ``left`` tag of
  ``jno.domain.line``.
* Synthetic noisy observations of ``u(t)`` form the data-fit term.

References
----------
Linka, K., Schäfer, A., Meng, X., Zou, Z., Karniadakis, G. E., & Kuhl, E.
(2022).  *Bayesian Physics-Informed Neural Networks for real-world
nonlinear dynamical systems.*  Computer Methods in Applied Mechanics
and Engineering, 402, 115346.

Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo.*  JMLR 15(1), 1593-1623.
"""

from pathlib import Path

import blackjax
import foundax
import jax
import jax.numpy as jnp
import optax

import jno

# ── Physical setup ────────────────────────────────────────────────────────────
k_true = 0.5
T_end = 4.0

# ── Domain (1-D line over the time axis: x ≡ t) ───────────────────────────────
domain = jno.domain(constructor=jno.domain.line(x_range=(0.0, T_end), mesh_size=0.2))
t, _ = domain.variable("interior")
t0, _ = domain.variable("left")  # left endpoint = initial time

# ── Analytical solution (noiseless target — see note below) ───────────────────
# We use a noiseless target here because the same `jno.noise.gaussian`
# realisation would be redrawn every optimiser step, making the likelihood
# inconsistent across the Bayesian chain.  For a fully noisy observation
# model, synthesise the data array once outside the trace and feed it as a
# constant.
u_obs = jno.np.exp(-k_true * t)

# ── Trainable rate constant — Bayesian ───────────────────────────────────────
k = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="k")


def _prior_k(p, _scale=2.0):
    """Wide Gaussian prior centred at 0 to keep the chain bounded while
    the surrogate trains.  Width 2.0 is non-informative on the scale of
    typical rate constants."""
    leaves = jax.tree_util.tree_leaves(p)
    sq = sum(jnp.sum(leaf**2) for leaf in leaves if hasattr(leaf, "dtype"))
    return -0.5 * sq / (_scale * _scale)


k.bayesian(
    blackjax.nuts,
    step_size=5e-3,
    warmup=800,  # long warmup → optax surrogate converges before sampling
    keep=400,  # post-warmup samples
    max_num_doublings=4,
    prior=_prior_k,
    adapt=False,  # mixed mode
)

# ── Surrogate u(t) — deterministic optax ──────────────────────────────────────
net = jno.nn.wrap(foundax.mlp(in_features=1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(1)))
net.optimizer(optax.adam(1e-3))

u = net(t)

# ── Constraints ───────────────────────────────────────────────────────────────
# ODE residual: du/dt + k u = 0
ode = u.d(t, scheme="finite_difference") + k * u

# Initial condition u(0) = 1 via the left-boundary tag
ic = net(t0) - 1.0

# Data fit: surrogate matches noisy observations at interior grid points
data = u - u_obs

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([ode.mse, ic.mse, data.mse], domain)
crux.solve(1200)

# ── Posterior summary ────────────────────────────────────────────────────────
k_chain = k.posterior_samples  # (400, 1)
k_mean = float(jnp.mean(k_chain))
k_lo, k_hi = (float(v) for v in jnp.quantile(k_chain, jnp.array([0.05, 0.95])))

print(f"k = {k_mean:.3f}  90% CI = [{k_lo:.3f}, {k_hi:.3f}]   truth = {k_true}")

rel_k = abs(k_mean - k_true) / abs(k_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"10_bayesian_pinns/05_inverse_ode_decay.py | epochs=1200 | rel_k={rel_k:.4f} | CI_width={k_hi - k_lo:.4f}\n")

assert rel_k < 0.4, f"posterior-mean k off by {rel_k:.2%}"
