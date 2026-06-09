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
* jNO's ``domain.init_fem`` + ``weak.assemble`` build the JAX-traceable
  stiffness matrix ``A`` and load vector ``b``.  We solve the α = 1
  problem once to get ``u_baseline``.
* Because the diffusion term is **linear in α**, ``A(α) = α · A_base``
  and therefore ``u(α) = u_baseline / α``.  We express the forward as a
  jNO expression of the trainable ``α`` and a constant per-node
  ``u_baseline`` array — so the whole loss flows through
  ``crux.solve()`` with NUTS attached via ``.bayesian()``.
* For nonlinear PDEs the scaling identity fails; you'd then need to
  wrap the per-step ``assemble + linalg.solve`` in a jNO FunctionCall
  placeholder.  Same architecture, just slower.
"""

from pathlib import Path

import jax

# FEAX's FEM assembly produces float64.  Enable x64 once at the top so
# the FEM solve, the inverse domain, and the NUTS kernel state all live
# in a single dtype — otherwise the Metropolis cond in NUTS errors out
# on a mixed-precision pytree.
jax.config.update("jax_enable_x64", True)

import blackjax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402


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

fem_domain = jno.domain.rect(mesh_size=0.15)
fem_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[fem_domain.dirichlet(["left", "right", "bottom", "top"], 0.0)],
    fem_solver=True,
)

u_sym, phi_sym = fem_domain.fem_symbols()
xg, yg, _ = fem_domain.variable("fem_gauss", split=True)

du_dx = u_sym.d(xg)
du_dy = u_sym.d(yg)
phi_x = phi_sym.d(xg)
phi_y = phi_sym.d(yg)

# Weak form for α = 1; A(α=1) = A_base.
weak_base = du_dx * phi_x + du_dy * phi_y - source_f(xg, yg, alpha_true=1.0) * phi_sym
A_base, b = weak_base.assemble(fem_domain, target="fem_system")
A_base_dense = to_dense(A_base)
b_dense = jnp.asarray(b)

u_baseline = jnp.linalg.solve(A_base_dense, b_dense).reshape(-1)

# ── Sanity-check the FEM forward against the manufactured solution ───────────
coords = np.asarray(fem_domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])
u_exact_nodes = exact_u(x_nodes, y_nodes).reshape(-1)
fwd_err = float(jnp.linalg.norm(u_baseline - u_exact_nodes) / (jnp.linalg.norm(u_exact_nodes) + 1e-12))
print(f"[forward] FEM rel-L2 vs manufactured: {fwd_err:.4e}")
assert fwd_err < 1e-1, f"FEM forward inaccurate: rel-L2 = {fwd_err:.3e}"

# ── Synthetic noisy observations under α_true = 1 ────────────────────────────
key_obs = jax.random.PRNGKey(0)
u_obs = u_baseline + sigma_obs * jax.random.normal(key_obs, u_baseline.shape)

# ── Per-node data domain: pack (u_baseline, u_obs) as 2-D "coordinates" ──────
# `from_array` makes each node a single sample whose Variable returns its
# coordinate.  We split the (N, 2) array into per-node u_base / u_meas
# Variables that the constraint compiler treats as spatial inputs of
# shape (B, N, 1) each.
node_data = np.stack([np.asarray(u_baseline), np.asarray(u_obs)], axis=1)  # (N, 2)
inv_domain = jno.domain.from_array({"nodes": node_data})
u_base, u_meas, _ = inv_domain.variable("nodes", split=True)

# ── Bayesian diffusivity — NUTS through jno.core.solve ───────────────────────
α = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="α")
# Start at α = 2.0 (deliberately far from truth = 1) so the chain has
# something to discover.  jno.np.parameter hard-codes float32 internally;
# with x64 enabled (above) we also promote α to float64 so its dtype
# matches the FEM forward.
α.initialize(jax.nn.initializers.constant(2.0))
α.dtype(jnp.float64)
α.bayesian(
    blackjax.nuts,
    step_size=0.1,  # initial guess; window adaptation refines it
    warmup=300,
    keep=1000,
    # adapt=True default — fixed-target posterior, adaptation well-defined.
)

# Forward in the trace: u(α) = u_baseline / α.  Loss is the Gaussian-noise
# residual averaged over the FEM nodes.
residual = (u_base / α - u_meas) / sigma_obs

# ── Solve — pure Bayesian via crux.solve, no manual blackjax loop ────────────
crux = jno.core([residual.mse])
crux.solve(1300)

# ── Posterior summary ────────────────────────────────────────────────────────
α_chain = α.posterior_samples  # shape (1000, 1)
α_mean = float(jnp.mean(α_chain))
α_std = float(jnp.std(α_chain))
α_lo, α_hi = (float(v) for v in jnp.quantile(α_chain, jnp.array([0.05, 0.95])))

print(f"[inverse] α = {α_mean:.4f} ± {α_std:.4f}")
print(f"          90% CI = [{α_lo:.4f}, {α_hi:.4f}]   truth = {α_true}")

rel_α = abs(α_mean - α_true) / abs(α_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/06_inverse_fem_diffusivity.py | warmup=300 | keep=1000 | "
        f"fwd_rel_L2={fwd_err:.4e} | rel_alpha={rel_α:.4f} | CI_width={α_hi - α_lo:.4f}\n"
    )

assert rel_α < 0.1, f"posterior-mean α off by {rel_α:.2%}"
