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
import os
import jax
import jno

import foundax
import optax
from jno import LearningRateSchedule as lrs

TEST_MODE = os.getenv("JNO_TUTORIAL_TEST_MODE", "").lower() in {"1", "true", "yes"}

def pick(full, test):
    return test if TEST_MODE else full

π = jno.np.pi
# ── Parameter ─────────────────────────────────────────────────────────────────
k = 2.0  # wave number — change to test different regimes

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=pick(0.05, 0.3)))
x, y, _ = domain.variable("interior")

# ── Manufactured solution and forcing ─────────────────────────────────────────
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
forcing = (2 * π**2 - k**2) * jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network ───────────────────────────────────────────────────────────────────
u_net = jno.nn.wrap(foundax.mlp(
    in_features=2,
    hidden_dims=pick(64, 24),
    num_layers=pick(5, 3),  # slightly deeper for the oscillatory problem
    key=jax.random.PRNGKey(0),
)).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = u_net(x, y) * x * (1 - x) * y * (1 - y)

# ── PDE residual:  ∇²u + k²u + f = 0 ────────────────────────────────────────
pde = u.laplacian(x, y, scheme="automatic_differentiation") + k**2 * u + forcing

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(pick(5000, 1000))

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
if TEST_MODE:
    assert jax.numpy.isfinite(rel_l2), f"non-finite relative L2 error: {rel_l2}"
else:
    assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"