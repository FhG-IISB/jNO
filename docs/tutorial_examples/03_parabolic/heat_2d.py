"""03 — 2-D heat equation (parabolic, time-dependent)"""

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi
α = 0.1
T_end = 0.5

domain = jno.domain.rect(mesh_size=0.05, time=(0, T_end, 4))
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-2 * α * π**2 * t) * jno.np.sin(π * x) * jno.np.sin(π * y)

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=2,
        n_outputs=1,
        n_layers=3,
        basis_functions=48,
        hidden_dim=32,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 50, 5000, 1e-5)))

xy = jno.np.concat([x, y])
xy0 = jno.np.concat([x0, y0])

# Hard Dirichlet BCs; partial-derivative names bound for math-like residuals.
u = (net(t, xy) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y, t=t)
u0 = net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)

pde = u.t - α * (u.xx + u.yy)
ini = u0 - jno.np.sin(π * x0) * jno.np.sin(π * y0)

residuals = jno.trackers.residual_stats(interval=1000)
crux = jno.core([pde.mse, ini.mse])
crux.solve(5_000, callbacks=[residuals])

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"03_parabolic/heat_2d.py | epochs=5000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
