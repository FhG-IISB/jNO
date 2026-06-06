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

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi
# ── Parameter ─────────────────────────────────────────────────────────────────
k = 2.0  # wave number — change to test different regimes

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.rect(mesh_size=0.05)
x, y, _ = domain.variable("interior")

# ── Manufactured solution and forcing ─────────────────────────────────────────
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
forcing = (2 * π**2 - k**2) * jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network ───────────────────────────────────────────────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=64,
        num_layers=5,  # slightly deeper for the oscillatory problem
        key=jax.random.PRNGKey(0),
    )
).optimizer(optax.adam(optax.exponential_decay(init_value=1e-3, transition_steps=80, decay_rate=0.5, end_value=1e-5)))

u = u_net(x, y) * x * (1 - x) * y * (1 - y)

# ── PDE residual:  ∇²u + k²u + f = 0 ────────────────────────────────────────
pde = u.laplacian(x, y, scheme="automatic_differentiation") + k**2 * u + forcing

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(40000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))

# Write result to tracking file
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/helmholtz_2d.py | epochs=40000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
