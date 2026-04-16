"""02 — 2-D Poisson equation  (AD vs finite-difference comparison)

Problem
-------
    −∇²u(x,y) = 2π² sin(πx) sin(πy),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω

Analytical solution
-------------------
    u(x,y) = sin(πx) sin(πy)

Both automatic differentiation (AD) and finite-difference (FD) Laplacians are
used for the same network architecture so the results can be compared directly.
"""

import jax
import jno

import optax

π = jno.np.pi
# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.2))
x, y, _ = domain.variable("interior")

domain.summary()

u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
layer_dims = [2, 10, 10, 1]
req_params = {"D": 5, "flavor": "exact"}


def make_solver(scheme: str, label: str, epochs: int = 5000) -> float:
    net = jno.nn.mlp(in_features=2, hidden_dims=64, num_layers=4, key=jax.random.PRNGKey(0))
    net.optimizer(optax.adam(1))
    net.lr(jno.schedule.learning_rate.exponential(1e-3, 0.5, epochs, 1e-5))

    u = net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)
    pde = -u.laplacian(x, y, scheme=scheme) - forcing

    crux = jno.core([pde.mse], domain).print_shapes()
    history = crux.solve(epochs)

    _u, _u_exact = crux.eval([u, u_exact])
    rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
    return rel_l2

    # net.optimizer(optax.lbfgs(None, 20))

    # history = crux.solve(int(epochs * 0.25))


rel_l2_ad = make_solver("automatic_differentiation", "ad")
rel_l2_fd = make_solver("finite_difference", "fd")

assert rel_l2_ad < 1e-1, f"AD relative L2 error too large: {rel_l2_ad:.3e}"
assert rel_l2_fd < 1e-1, f"FD relative L2 error too large: {rel_l2_fd:.3e}"
