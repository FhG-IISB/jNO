"""06 — Bayesian inverse with jNO-FEM as the differentiable forward

Problem
-------
2-D Poisson with an unknown scalar diffusivity ``α``:

    -α Δu = f       in Ω = [0, 1]²
        u = 0      on ∂Ω

Manufactured ground truth:

    u_exact(x, y) = x(1 - x) y(1 - y)
    f(x, y)       = 2·α_true · [x(1 - x) + y(1 - y)]

Given noisy observations of ``u`` at the FEM mesh nodes (under the true
``α = 1``), recover the posterior over ``α``.

Why this matters
----------------
The Bayesian inverse tutorials up to this point (`03`, `04`) used
**closed-form** forward models (`exp(-kt)`, `sin(πx)/π²`).  Real
engineering inverse problems rarely have closed forms — they have
numerical PDE solvers.  This tutorial shows the pattern when the
forward is the **FEAX-backed FEM solver** that jNO already exposes.

Architecture
------------
* jNO's ``domain.init_fem`` + ``weak.assemble`` build a JAX-traceable
  stiffness matrix ``A`` and load vector ``b`` for the α = 1 problem.
* Because the diffusion term is **linear in α**, we exploit the
  scaling identity ``u(α) = u(α=1) / α`` rather than re-assembling each
  NUTS step (saves a lot of compile time without losing generality —
  the same pattern works with a per-step re-assembly when the PDE has
  α-dependent boundary terms or nonlinear couplings, just slower).
* The likelihood ``logdensity(α) = -‖u(α) - u_obs‖² / (2σ²) + log_prior``
  is a plain JAX function of ``α``; we pass it directly to
  ``blackjax.window_adaptation`` and ``blackjax.nuts`` — jNO's
  ``.bayesian()`` API is currently scoped to problems whose forward is
  expressible as a jNO Placeholder expression, so we drop one level
  for this FEM-backed setup.

This is the right pattern whenever your forward model lives outside
jNO's tracer (FEM, finite volume, an external solver, ODE integrator):
use jNO for the differentiable forward, blackjax for the chain.
"""

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

import jno


# ── Manufactured exact solution ──────────────────────────────────────────────
def exact_u(x, y):
    return x * (1.0 - x) * y * (1.0 - y)


def source_f(x, y, alpha_true=1.0):
    # -α Δu = f  ⇒  f(x, y) = 2 α [x(1-x) + y(1-y)]
    return 2.0 * alpha_true * (x * (1.0 - x) + y * (1.0 - y))


def to_dense(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


# ── FEM setup (jNO does the differentiable assembly) ─────────────────────────
α_true = 1.0
sigma_obs = 0.005  # observation noise

domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.15))
domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[domain.dirichlet(["left", "right", "bottom", "top"], 0.0)],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss", split=True)

du_dx = jno.np.grad(u, xg)
du_dy = jno.np.grad(u, yg)
phi_x = jno.np.grad(phi, xg)
phi_y = jno.np.grad(phi, yg)

# Weak form for α = 1.  Diffusion contribution is linear in α, so
# A(α) = α · A_base; we'll exploit this below.
weak_base = du_dx * phi_x + du_dy * phi_y - source_f(xg, yg, alpha_true=1.0) * phi
A_base, b = weak_base.assemble(domain, target="fem_system")
A_base_dense = to_dense(A_base)
b_dense = jnp.asarray(b)

# Baseline FEM solution at α = 1.
u_baseline = jnp.linalg.solve(A_base_dense, b_dense).reshape(-1)

# ── Sanity-check the FEM forward against the manufactured solution ───────────
coords = np.asarray(domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])
u_exact_nodes = exact_u(x_nodes, y_nodes).reshape(-1)
fwd_err = float(jnp.linalg.norm(u_baseline - u_exact_nodes) / (jnp.linalg.norm(u_exact_nodes) + 1e-12))
print(f"[forward] FEM rel-L2 vs manufactured: {fwd_err:.4e}")
assert fwd_err < 1e-1, f"FEM forward inaccurate: rel-L2 = {fwd_err:.3e}"

# ── Synthetic noisy observations under α_true = 1 ────────────────────────────
key = jax.random.PRNGKey(0)
key, key_noise = jax.random.split(key)
u_obs = u_baseline + sigma_obs * jax.random.normal(key_noise, u_baseline.shape)

# ── Likelihood + prior on α ──────────────────────────────────────────────────
# Forward exploiting linearity: A(α) = α · A_base  ⇒  u(α) = u_baseline / α.
# (For nonlinear/coupled PDEs replace this with a per-call assemble + solve —
# the rest of the pattern is identical, just slower.)


def neg_log_posterior(alpha):
    α_val = alpha[0]
    u_alpha = u_baseline / α_val
    log_lik = -0.5 * jnp.sum((u_alpha - u_obs) ** 2) / sigma_obs**2
    log_prior = -0.5 * (α_val - 1.0) ** 2 / 4.0  # weakly informative prior
    return -(log_lik + log_prior)


def logdensity_fn(alpha):
    return -neg_log_posterior(alpha)


# ── Window adaptation + NUTS via blackjax directly ───────────────────────────
warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn, initial_step_size=0.1)
key, key_adapt = jax.random.split(key)
adapt_result, _info = warmup.run(key_adapt, jnp.array([2.0]), num_steps=300)

kernel = blackjax.nuts(logdensity_fn, **adapt_result.parameters)
state = adapt_result.state


@jax.jit
def one_step(carry, _):
    state, key = carry
    key, sub = jax.random.split(key)
    new_state, _info = kernel.step(sub, state)
    return (new_state, key), new_state.position


keep = 1000
key, key_chain = jax.random.split(key)
(_, _), positions = jax.lax.scan(one_step, (state, key_chain), None, length=keep)
chain = positions[:, 0]  # shape (keep,)

# ── Posterior summary ────────────────────────────────────────────────────────
α_mean = float(jnp.mean(chain))
α_std = float(jnp.std(chain))
α_lo, α_hi = (float(v) for v in jnp.quantile(chain, jnp.array([0.05, 0.95])))

print(f"[inverse] α = {α_mean:.4f} ± {α_std:.4f}")
print(f"          90% CI = [{α_lo:.4f}, {α_hi:.4f}]   truth = {α_true}")

rel_α = abs(α_mean - α_true) / abs(α_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/06_inverse_fem_diffusivity.py | warmup=300 | keep={keep} | "
        f"fwd_rel_L2={fwd_err:.4e} | rel_alpha={rel_α:.4f} | CI_width={α_hi - α_lo:.4f}\n"
    )

assert rel_α < 0.05, f"posterior-mean α off by {rel_α:.2%}"
