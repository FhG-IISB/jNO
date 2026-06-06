"""01 — 1-D Laplace equation  (simplest possible PINN)

Problem
-------
    u''(x) = 0,   x ∈ [0, 1],   u(0) = 0,  u(1) = 1

Analytical solution
-------------------
    u(x) = x   (the linear interpolant between the two boundary values)

Techniques shown
----------------
* Non-homogeneous Dirichlet BCs via a hard-enforced ansatz:
    u = x + x(1−x)·net(x)
  The linear part x exactly satisfies u(0)=0 and u(1)=1; the
  x(1−x) factor vanishes at both endpoints so the network only
  learns the *deviation* from the linear interpolant.
* `.d2()` shortcut for the second derivative (autodiff)
* Single-phase Adam with exponential LR decay
"""

import foundax
import jax
import optax

import jno

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.1)
x, _ = domain.variable("interior")

# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = x  # linear interpolant between u(0)=0 and u(1)=1

# ── Network with non-homogeneous hard-enforced BCs ────────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=32,
        num_layers=3,
        key=jax.random.PRNGKey(0),
    )
).optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

# Linear interpolant + network correction that vanishes at the endpoints.
u = x + x * (1 - x) * u_net(x)

# ── Constraints ───────────────────────────────────────────────────────────────
pde = u.d2(x)  # Laplace: u'' = 0

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
