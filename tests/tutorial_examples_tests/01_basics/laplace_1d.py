"""01 — 1-D Laplace / Poisson equation  (simplest possible PINN)

Problem
-------
    −u''(x) = sin(πx),   x ∈ [0, 1],   u(0) = u(1) = 0

Analytical solution
-------------------
    u(x) = sin(πx) / π²

Techniques shown
----------------
* Homogeneous Dirichlet BCs via hard constraint:  u = net(x) · x (1−x)
* Automatic-differentiation gradient  (jno.np.grad)
* Final relative-L² check against the exact solution
* Single-phase Adam with exponential LR decay
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
x, _ = domain.variable("interior")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.sin(π * x) / π**2

# ── Network with hard-enforced BCs ────────────────────────────────────────────
u_net = jno.np.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = u_net(x) * x * (1 - x)  # hard BC: u(0) = u(1) = 0

# ── Constraints ───────────────────────────────────────────────────────────────
pde = -jno.np.grad(jno.np.grad(u, x), x) - jno.np.sin(π * x)  # should be 0

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"
