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

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi
# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.rect(mesh_size=0.1)
x, y, _ = domain.variable("interior")

domain.summary()

u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
layer_dims = [2, 10, 10, 1]
req_params = {"D": 5, "flavor": "exact"}


def make_solver(scheme: str, label: str, epochs: int = 40000) -> float:
    net = jno.nn.wrap(
        foundax.mlp(
            in_features=2,
            hidden_dims=96,
            num_layers=5,
            activation=jax.nn.tanh,
            key=jax.random.PRNGKey(0),
        )
    )
    net.optimizer(
        optax.adam(
            optax.exponential_decay(
                init_value=1e-3,
                transition_steps=epochs // 10,
                decay_rate=0.5,
                end_value=1e-5,
            )
        )
    )

    u = net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)
    pde = -u.laplacian(x, y, scheme=scheme) - forcing

    crux = jno.core([pde.mse], domain).print_shapes()
    _history = crux.solve(epochs)

    _u, _u_exact = crux.eval([u, u_exact])
    rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
    return rel_l2

    # net.optimizer(optax.lbfgs(None, 20))

    # history = crux.solve(int(epochs * 0.25))


rel_l2_ad = make_solver("automatic_differentiation", "ad")
rel_l2_fd = make_solver("finite_difference", "fd")

# Write result to tracking file
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/poisson_2d.py | epochs=40000 | AD_rel_L2={rel_l2_ad:.6e} | FD_rel_L2={rel_l2_fd:.6e}\n")

assert rel_l2_ad < 1e-1, f"AD relative L2 error too large: {rel_l2_ad:.3e}"
assert rel_l2_fd < 1e-1, f"FD relative L2 error too large: {rel_l2_fd:.3e}"
