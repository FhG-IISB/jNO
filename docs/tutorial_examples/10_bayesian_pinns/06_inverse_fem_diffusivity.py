"""06 - Bayesian inverse with jno.fem as the differentiable forward.

Recover a posterior over the scalar diffusivity alpha in  -alpha Delta u = f  from noisy
observations of u, with NUTS driven through ``crux.solve``. jno.fem assembles the forward
operator; because the operator is alpha * (stiffness) with a fixed load, the solution scales
as ``u(alpha) = u_baseline / alpha``, so the per-sample forward is a cheap closed form.
"""

from pathlib import Path

import jax

# feax assembles in float64; enable x64 once so the FEM solve, the inverse domain, and the
# NUTS kernel state share one dtype (NUTS errors on a mixed-precision pytree).
jax.config.update("jax_enable_x64", True)

import blackjax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

alpha_true, sigma_obs = 1.0, 0.005
exact_u = lambda x, y: x * (1 - x) * y * (1 - y)  # noqa: E731
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

# ── FEM forward via jno.fem (alpha = 1; the operator is alpha * A_base) ───────────────────
d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))  # -Delta u = f at alpha = 1
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)
u_baseline = jnp.linalg.solve(dense(fem.A), jnp.asarray(fem.b).reshape(-1))

pts = np.asarray(fem.points)
fwd_err = float(
    jnp.linalg.norm(u_baseline - exact_u(pts[:, 0], pts[:, 1])) / (jnp.linalg.norm(exact_u(pts[:, 0], pts[:, 1])) + 1e-12)
)
print(f"[forward] FEM rel-L2 vs manufactured: {fwd_err:.4e}")
assert fwd_err < 1e-1

# ── Noisy observations under alpha_true, packed as a per-node data domain ─────────────────
u_obs = u_baseline + sigma_obs * jax.random.normal(jax.random.PRNGKey(0), u_baseline.shape)
inv_domain = jno.domain.from_array({"nodes": np.stack([np.asarray(u_baseline), np.asarray(u_obs)], axis=1)})
u_base, u_meas, _ = inv_domain.variable("nodes", split=True)

# ── Bayesian diffusivity: NUTS through crux.solve (forward u(alpha) = u_baseline / alpha) ──
alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="alpha")
alpha.initialize(jax.nn.initializers.constant(2.0))  # start far from truth = 1
alpha.dtype(jnp.float64)
alpha.bayesian(blackjax.nuts, step_size=0.1, warmup=300, keep=1000)
crux = jno.core([((u_base / alpha - u_meas) / sigma_obs).mse])
crux.solve(1300)

chain = alpha.posterior_samples
mean, std = float(jnp.mean(chain)), float(jnp.std(chain))
lo, hi = (float(v) for v in jnp.quantile(chain, jnp.array([0.05, 0.95])))
print(f"[inverse] alpha = {mean:.4f} +/- {std:.4f}   90% CI = [{lo:.4f}, {hi:.4f}]   truth = {alpha_true}")

rel_alpha = abs(mean - alpha_true) / abs(alpha_true)
with open(Path(__file__).parent.parent.parent / "tutorial_results.txt", "a") as fh:
    fh.write(
        f"10_bayesian_pinns/06_inverse_fem_diffusivity.py | warmup=300 | keep=1000 | fwd_rel_L2={fwd_err:.4e} | rel_alpha={rel_alpha:.4f} | CI_width={hi - lo:.4f}\n"
    )
assert rel_alpha < 0.1
