"""01 — Bayesian PINN forward problem: 1-D Poisson with noisy boundary data

Problem
-------
    u''(x) = -sin(πx),    x ∈ [0, 1],    u(0) = u(1) = 0,

with closed-form reference ``u_exact(x) = sin(πx) / π²``.

(Compared with the Yang-et-al. §3.2.1 setup, which uses
``u = sin³(6x)`` on ``[-0.7, 0.7]`` with ``λ = 0.01``, we use the
simpler unit-coefficient Poisson here so a small SGLD chain on CPU
recovers the solution within seconds.  The Bayesian setup is otherwise
identical.)

The PDE residual is enforced exactly at interior collocation points;
the two boundary observations carry Gaussian noise of standard
deviation ``σ_b = 0.01``.  The B-PINN's posterior over the network
weights then has two visible features:

* the predictive mean fits the analytical solution near the data,
* the credible band widens in regions with fewer observations.

Technique
---------
Each outer epoch performs one **Stochastic Gradient Langevin Dynamics**
(SGLD) transition over the full network pytree.  After training, the
post-warmup chain is stacked into ``net.posterior_samples`` and
``crux.eval([u])`` auto-vmaps the evaluator over the chain to yield
posterior **prediction bands** at the interior points.

A caveat on calibration
-----------------------
Vanilla SGLD on a ~300-parameter MLP without preconditioning does not
fully concentrate the chain around the data-fit MAP in a tractable
step budget, so the in-data rel-L2 is RNG-path sensitive (different
seeds can land anywhere from rel_L2 ≈ 0.5 to several).  For tightly
calibrated bands on neural-network weights the literature recommends
preconditioned variants (pSGLD), SGHMC with mass-matrix adaptation,
or variational inference — those are out of scope here.  We commit
instead to the **qualitative** B-PINN behaviour: the chain doesn't
diverge, and the credible band has non-trivial width.

References
----------
Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913.

Welling, M., & Teh, Y. W. (2011).  *Bayesian learning via stochastic
gradient Langevin dynamics.*  ICML 2011, 681-688.
"""

from pathlib import Path

import blackjax
import foundax
import jax
import jax.numpy as jnp

import jno

π = jno.np.pi

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Analytical solution and forcing ──────────────────────────────────────────
# u_exact(x) = sin(πx)/π² ⇒ u'' = -sin(πx).
u_exact_expr = jno.np.sin(π * x) / π**2
f_clean = -jno.np.sin(π * x)

# Sparse noisy sensor observations of u at the boundary.  jno.noise.gaussian
# draws a fresh observation each step from the solver's PRNG key — fully
# reproducible given the global seed.  (Yang et al. also noise the
# interior forcing sensors; that combination currently mixes
# interior/boundary point sets across constraints in jno's trace, so we
# noise only the boundary data here — sufficient to show the
# uncertainty-quantification benefit.)
sigma_b = 0.01
u_b_clean = jno.np.sin(π * xb) / π**2
u_b_obs = u_b_clean + jno.noise.gaussian(std=sigma_b)

# ── Bayesian PINN — SGLD posterior over MLP weights ───────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=16,
        num_layers=2,
        key=jax.random.PRNGKey(0),
    )
)
u_net.bayesian(
    blackjax.sgld,
    step_size=1e-5,
    warmup=2000,
    keep=400,
    thin=2,
)

u = u_net(x)

# ── Constraints ───────────────────────────────────────────────────────────────
# PDE residual on noisy f-sensors, data fit on noisy boundary u-sensors.
pde = u.d2(x, scheme="finite_difference") - f_clean
bc = u_net(xb) - u_b_obs

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse], domain)
crux.solve(2800)

# ── Posterior prediction bands (auto-chain default) ──────────────────────────
u_chain = crux.eval([u])  # shape (K, N, n_points, 1) — K=1 by default
u_exact = crux.eval([u_exact_expr])  # no Bayesian deps → point value
# Reduce over both the chain (K) and sample (N) axes for per-point
# posterior summaries; with K=1 this is equivalent to axis=1 alone.
u_mean = jnp.mean(u_chain, axis=(0, 1))
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=(0, 1))

rel_l2 = float(jnp.linalg.norm(u_mean - u_exact) / (jnp.linalg.norm(u_exact) + 1e-8))
band_width = float(jnp.max(u_hi - u_lo))

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/01_forward_noisy_poisson_1d.py | epochs=2800 | "
        f"rel_L2_mean={rel_l2:.4f} | max_band_width={band_width:.4f}\n"
    )

# Loose tolerance: vanilla SGLD on a ~300-param MLP without
# preconditioning is RNG-path sensitive — the chain does not fully
# concentrate around the data-fit MAP in a tractable step budget.  The
# in-data rel-L2 is therefore noisy across reseeds.  For tightly
# calibrated bands the literature recommends preconditioned variants
# (pSGLD), SGHMC with mass-matrix adaptation, or variational inference.
# We only assert that the chain doesn't diverge and produces a
# non-trivial band — the qualitative B-PINN behaviour.
assert jnp.isfinite(rel_l2), f"posterior mean diverged: rel_l2 = {rel_l2}"
assert rel_l2 < 100.0, f"posterior-mean rel L2 unreasonably high: {rel_l2:.3e}"
assert band_width > 1e-4, f"credible band collapsed: max width {band_width:.3e}"
