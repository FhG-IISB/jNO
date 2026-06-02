"""04 — Bayesian PINN inverse: recover nonlinear reaction coefficient k

Problem (Yang et al. 2021, §3.3.1, steady-state form)
-----------------------------------------------------
    λ u''(x) + k · tanh(u(x)) = f(x),   x ∈ [-0.7, 0.7],   λ = 0.01,

with Dirichlet boundary conditions imposed via the analytical solution

    u_exact(x) = sin(πx) / π² ,

from which ``f`` is derived under the **true** value ``k_true = 0.7``.
The objective is to recover ``k`` given noisy interior observations of
``u`` and ``f`` plus boundary data.

Yang et al. report B-PINN-HMC posterior ``k = 0.705 ± 6 × 10⁻³`` for the
small-noise case; here, with a tinier mesh, a smaller surrogate net, and
a short chain, we use a looser tolerance (``|k_mean − 0.7| < 0.2``).

Technique
---------
* ``k`` is a scalar trainable parameter, sampled with NUTS (``adapt=False``
  — the mixed-mode caveat in ``docs/training/bayesian.md`` applies).
* ``net(x)`` is a small MLP, *optimised* with Adam — surrogate ``u(x)``.
* Loss = PDE-residual MSE + boundary-data MSE.

References
----------
Yang, L., Meng, X., & Karniadakis, G. E. (2021).  *B-PINNs: Bayesian
physics-informed neural networks for forward and inverse PDE problems
with noisy data.*  Journal of Computational Physics, 425, 109913.
"""

from pathlib import Path

import blackjax
import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Physical setup ────────────────────────────────────────────────────────────
λ = 0.01
k_true = 0.7

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(x_range=(-0.7, 0.7), mesh_size=0.1))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Analytical solution and forcing under k_true ──────────────────────────────
u_exact = jno.np.sin(π * x) / π**2
# u'' = -sin(πx) ⇒ λ u'' = -λ sin(πx); tanh(u_exact) is just tanh(sin(πx)/π²).
f_exact = λ * (-jno.np.sin(π * x)) + k_true * jno.np.tanh(u_exact)

u_b_exact = jno.np.sin(π * xb) / π**2

# ── Trainable reaction coefficient — Bayesian ────────────────────────────────
k = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="k")
k.bayesian(
    blackjax.nuts,
    step_size=5e-3,
    warmup=200,
    keep=200,
    max_num_doublings=4,
    # Mixed mode (k is Bayesian, net is optax) — adapt=True would tune
    # against the untrained surrogate.  Hand-pick step_size instead.
    adapt=False,
)

# ── Surrogate u(x) — deterministic optax ─────────────────────────────────────
net = jno.nn.wrap(foundax.mlp(in_features=1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(1)))
net.optimizer(optax.adam(1e-3))

u = net(x)

# ── Constraints ───────────────────────────────────────────────────────────────
# PDE residual: λ u'' + k tanh(u) - f_exact = 0
pde = λ * u.d2(x, scheme="finite_difference") + k * jno.np.tanh(u) - f_exact

# Boundary: net(xb) ≈ u_b_exact
bc = net(xb) - u_b_exact

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse], domain)
crux.solve(400)

# ── Posterior summary ────────────────────────────────────────────────────────
k_chain = k.posterior_samples  # (200, 1)
k_mean = float(jnp.mean(k_chain))
k_lo, k_hi = (float(v) for v in jnp.quantile(k_chain, jnp.array([0.05, 0.95])))

print(f"k = {k_mean:.3f}  90% CI = [{k_lo:.3f}, {k_hi:.3f}]   truth = {k_true}")

rel_k = abs(k_mean - k_true) / abs(k_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/04_inverse_reaction_coefficient.py | epochs=400 | "
        f"rel_k={rel_k:.4f} | CI_width={k_hi - k_lo:.4f}\n"
    )

assert rel_k < 0.4, f"posterior-mean k off by {rel_k:.2%}"
