"""01 — Bayesian 1-D Poisson PINN  (SGLD posterior over MLP weights)

Problem
-------
    −u''(x) = sin(πx),   x ∈ [0, 1],   u(0) = u(1) = 0

Analytical solution
-------------------
    u(x) = sin(πx) / π²

Technique
---------
Same PINN setup as ``poisson_1d.py``, but the MLP weights are *sampled*
rather than optimised.  Each outer epoch performs one Stochastic Gradient
Langevin Dynamics (SGLD) transition over the full network pytree.  After
training, the post-warmup chain is stacked into ``net.posterior_samples``
and ``crux.eval([u], samples="chain")`` vmaps the evaluator over the
chain to yield posterior **prediction bands** at the interior points.

Reference
---------
Welling, M., & Teh, Y. W. (2011).  *Bayesian Learning via Stochastic
Gradient Langevin Dynamics.*  ICML 2011, 681-688.
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

# ── Analytical solution (used for assertions only) ────────────────────────────
u_exact_expr = jno.np.sin(π * x) / π**2

# ── Network — SGLD instead of optax ───────────────────────────────────────────
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
pde = -u.d2(x, scheme="finite_difference") - jno.np.sin(π * x)
bc = u_net(xb)  # soft: u(0) = u(1) = 0

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse], domain)
crux.solve(2800)

# ── Posterior prediction bands ────────────────────────────────────────────────
u_chain = crux.eval([u], samples="chain")  # shape (200, n_points, 1)
u_exact = crux.eval([u_exact_expr])
u_mean = jnp.mean(u_chain, axis=0)
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=0)

# Posterior-mean fit quality (loose — SGLD without adaptation is noisy).
rel_l2 = float(jnp.linalg.norm(u_mean - u_exact) / (jnp.linalg.norm(u_exact) + 1e-8))
# Credible band has non-degenerate width somewhere on the interval.
band_width = float(jnp.max(u_hi - u_lo))

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"01_basics/bayesian_poisson_1d.py | epochs=2800 | rel_L2_mean={rel_l2:.4f} | max_band_width={band_width:.4f}\n"
    )

# Loose tolerance: SGLD without step-size adaptation is noisy; the assert
# only catches catastrophic regressions.  For tighter accuracy use
# blackjax.window_adaptation externally and pass the adapted parameters.
assert rel_l2 < 0.8, f"posterior-mean rel L2 too high: {rel_l2:.3e}"
assert band_width > 1e-4, f"credible band collapsed: max width {band_width:.3e}"
