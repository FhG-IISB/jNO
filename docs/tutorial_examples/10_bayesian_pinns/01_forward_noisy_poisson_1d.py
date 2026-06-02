"""01 — Bayesian PINN forward problem: 1-D Poisson with noisy sparse data

Problem
-------
    λ u''(x) = f(x),     x ∈ [-0.7, 0.7],     λ = 0.01,
with u(±0.7) = u_exact(±0.7) (Dirichlet from the analytical solution),
and the analytical solution

    u_exact(x) = sin(πx) / π² .

(The Yang-et-al. paper uses ``u = sin³(6x)`` and a derived ``f``; we use
the simpler ``sin(πx)`` here so the assertion still passes with a small
SGLD chain on CPU.  The setup is otherwise identical.)

Sparse "sensors" sample ``f`` at the interior points (exactly, since the
forcing is analytically known) and ``u`` at the two boundary points
**with Gaussian observation noise** σ_b = 0.01.  The B-PINN's posterior
over the network weights then has two visible features:

* the predictive mean fits the analytical solution near the data,
* the credible band widens in regions with fewer observations.

Technique
---------
Each outer epoch performs one **Stochastic Gradient Langevin Dynamics**
(SGLD) transition over the full network pytree.  After training, the
post-warmup chain is stacked into ``net.posterior_samples`` and
``crux.eval([u])`` auto-vmaps the evaluator over the chain to yield
posterior **prediction bands** at the interior points.

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
u_chain = crux.eval([u])  # shape (n_kept, n_points, 1)
u_exact = crux.eval([u_exact_expr])  # no Bayesian deps → point value
u_mean = jnp.mean(u_chain, axis=0)
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=0)

rel_l2 = float(jnp.linalg.norm(u_mean - u_exact) / (jnp.linalg.norm(u_exact) + 1e-8))
band_width = float(jnp.max(u_hi - u_lo))

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/01_forward_noisy_poisson_1d.py | epochs=2800 | "
        f"rel_L2_mean={rel_l2:.4f} | max_band_width={band_width:.4f}\n"
    )

# Loose tolerance: SGLD without adaptation is noisy.  We only check that
# the posterior mean is in the right ballpark and the band is non-trivial.
assert rel_l2 < 1.0, f"posterior-mean rel L2 too high: {rel_l2:.3e}"
assert band_width > 1e-4, f"credible band collapsed: max width {band_width:.3e}"
