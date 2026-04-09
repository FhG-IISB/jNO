"""02 — 2-D Helmholtz equation

Problem
-------
    ∇²u(x,y) + k² u(x,y) = −f(x,y),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω

Manufactured solution
---------------------
    u(x,y) = sin(πx) sin(πy)

Substituting gives the source term:
    f(x,y) = (2π² − k²) sin(πx) sin(πy)

Note: the problem becomes resonant when k = π√2 ≈ 4.44.
Try different values of k (e.g. 1, 2, 4) to see the effect on convergence.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
# ── Parameter ─────────────────────────────────────────────────────────────────
k = 2.0  # wave number — change to test different regimes

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.2))
x, y, _ = domain.variable("interior")

# ── Manufactured solution and forcing ─────────────────────────────────────────
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
forcing = (2 * π**2 - k**2) * jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network ───────────────────────────────────────────────────────────────────
u_net = jno.np.nn.mlp(
    in_features=2,
    hidden_dims=64,
    num_layers=5,  # slightly deeper for the oscillatory problem
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = u_net(x, y) * x * (1 - x) * y * (1 - y)

# ── PDE residual:  ∇²u + k²u + f = 0 ────────────────────────────────────────
pde = u.laplacian(x, y, scheme="automatic_differentiation") + k**2 * u + forcing

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"

