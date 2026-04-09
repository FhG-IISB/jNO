"""01 — 1-D Poisson equation  (soft boundary constraints + FD Laplacian)

Problem
-------
    −u''(x) = sin(πx),   x ∈ [0, 1],   u(0) = u(1) = 0

Analytical solution
-------------------
    u(x) = sin(πx) / π²

Compared to laplace_1d.py this example uses:
* Finite-difference second derivative  (u.d2)
* Soft boundary constraints  (separate boundary tag)
* A final relative-L² check against the exact solution
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.sin(π * x) / π**2

# ── Network ───────────────────────────────────────────────────────────────────
u_net = jno.np.nn.mlp(
    in_features=1,
    hidden_dims=64,
    num_layers=4,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 1000, 1e-5))

u = u_net(x)

# ── Constraints ───────────────────────────────────────────────────────────────
pde = -u.d2(x, scheme="finite_difference") - jno.np.sin(π * x)
bc = u_net(xb)  # soft: u(0) = u(1) = 0

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse], domain)
history = crux.solve(1000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 0.05, f"relative L2 error too large: {rel_l2:.3e}"
