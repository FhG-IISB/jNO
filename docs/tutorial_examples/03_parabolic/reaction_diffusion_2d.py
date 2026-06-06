"""03 - 2-D reaction-diffusion equation

Problem
-------
    u_t - nu Delta u + lambda u = f(x, y, t),   (x, y) in [0, 1]^2, t in [0, 1]
    u = 0 on the boundary
    u(x, y, 0) = sin(pi x) sin(pi y)

Analytical solution
-------------------
    u(x, y, t) = exp(-t) sin(pi x) sin(pi y)
"""

from pathlib import Path

import foundax
import jax
import optax

import jno

pi = jno.np.pi
nu = 0.1
lam = 0.5
T_end = 1.0
N_t = 4

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(pi * x) * jno.np.sin(pi * y)
source = (-1 + 2 * nu * pi**2 + lam) * u_exact

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=2,
        n_outputs=1,
        n_layers=5,
        basis_functions=96,
        hidden_dim=64,
        key=jax.random.PRNGKey(21),
    )
)
net.optimizer(
    optax.adam(
        optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-3,
            warmup_steps=40,
            decay_steps=40000 - 40,
            end_value=1e-5,
        )
    )
)

xy = jno.np.concat([x, y])
xy0 = jno.np.concat([x0, y0])
u = net(t, xy) * x * (1 - x) * y * (1 - y)
u0 = net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)

pde = u.d(t) - nu * jno.np.laplacian(u, [x, y]) + lam * u - source
ini = u0 - jno.np.sin(pi * x0) * jno.np.sin(pi * y0)

crux = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(40000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))

# Write result to tracking file
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"03_parabolic/reaction_diffusion_2d.py | epochs=40000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
