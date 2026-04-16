"""05 — Coupled elliptic PDE system  (manufactured solution)

Problem
-------
    −∇²u + v = f(x,y)   on Ω = [0,1]²
    −∇²v + u = g(x,y)

    u = v = 0 on ∂Ω

Manufactured solution
---------------------
    u(x,y) = sin(πx) sin(πy)
    v(x,y) = sin(2πx) sin(πy)

Source terms:
    f(x,y) = 2π² sin(πx) sin(πy)  + sin(2πx) sin(πy)
    g(x,y) = 5π² sin(2πx) sin(πy) + sin(πx)  sin(πy)

Two separate MLPs are used — one for u, one for v — and trained jointly.
Both fields enforce the homogeneous Dirichlet boundary conditions through the
same hard-constraint ansatz.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
sin = jno.np.sin
# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.2))
x, y, _ = domain.variable("interior")

# ── Manufactured solutions and source terms ───────────────────────────────────
u_exact = sin(π * x) * sin(π * y)
v_exact = sin(2 * π * x) * sin(π * y)

f = 2 * π**2 * sin(π * x) * sin(π * y) + sin(2 * π * x) * sin(π * y)
g = 5 * π**2 * sin(2 * π * x) * sin(π * y) + sin(π * x) * sin(π * y)

# ── Networks ──────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

u_net = jno.nn.mlp(in_features=2, hidden_dims=64, num_layers=4, key=k1)
v_net = jno.nn.mlp(in_features=2, hidden_dims=64, num_layers=4, key=k2)

for net in [u_net, v_net]:
    net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

# Hard BCs for both fields.
# The factor 16 normalizes the interior peak of x(1-x)y(1-y) to 1,
# which keeps the learned field on the same order of magnitude as the exact solution.
boundary_envelope = 16.0 * x * (1 - x) * y * (1 - y)
u = u_net(x, y) * boundary_envelope
v = v_net(x, y) * boundary_envelope

# ── PDE residuals ─────────────────────────────────────────────────────────────
Δu = jno.np.laplacian(u, [x, y])
Δv = jno.np.laplacian(v, [x, y])

pde1 = -Δu + v - f
pde2 = -Δv + u - g

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde1.mse, pde2.mse], domain)
history = crux.solve(10000)

_u, _u_exact, _v, _v_exact = crux.eval([u, u_exact, v, v_exact])
rel_l2_u = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
rel_l2_v = float(jax.numpy.linalg.norm(_v - _v_exact) / (jax.numpy.linalg.norm(_v_exact) + 1e-8))
assert rel_l2_u < 1e-1, f"u relative L2 error too large: {rel_l2_u:.3e}"
assert rel_l2_v < 1e-1, f"v relative L2 error too large: {rel_l2_v:.3e}"
