"""05 — Coupled parabolic system in 2-D

Problem
-------
    u_t − ∇²u + v = f(x, y, t)
    v_t − ∇²v + u = g(x, y, t)

on [0, 1]² with homogeneous Dirichlet BCs.

Analytical
    u(x, y, t) = exp(−t) sin(πx) sin(πy)
    v(x, y, t) = exp(−t) sin(2πx) sin(πy)
"""

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi
T_end = 1.0

domain = jno.domain.rect(mesh_size=0.05, time=(0, T_end, 4))
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(π * x) * jno.np.sin(π * y)
v_exact = jno.np.exp(-t) * jno.np.sin(2 * π * x) * jno.np.sin(π * y)
f = (2 * π**2 - 1) * u_exact + v_exact
g = (5 * π**2 - 1) * v_exact + u_exact


def _net(key: int):
    n = jno.nn.wrap(
        foundax.deeponet(
            n_sensors=1,
            coord_dim=2,
            n_outputs=1,
            n_layers=3,
            basis_functions=48,
            hidden_dim=32,
            key=jax.random.PRNGKey(key),
        )
    )
    n.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 50, 5000, 1e-5)))
    return n


u_net, v_net = _net(24), _net(25)

xy = jno.np.concat([x, y])
xy0 = jno.np.concat([x0, y0])
ansatz = lambda raw, xa, ya: raw * xa * (1 - xa) * ya * (1 - ya)  # noqa: E731
u = ansatz(u_net(t, xy), x, y).scalar.bind(x=x, y=y, t=t)
v = ansatz(v_net(t, xy), x, y).scalar.bind(x=x, y=y, t=t)
u0 = ansatz(u_net(t0, xy0), x0, y0)
v0 = ansatz(v_net(t0, xy0), x0, y0)

pde_u = u.t - (u.xx + u.yy) + v - f
pde_v = v.t - (v.xx + v.yy) + u - g
ini_u = u0 - jno.np.sin(π * x0) * jno.np.sin(π * y0)
ini_v = v0 - jno.np.sin(2 * π * x0) * jno.np.sin(π * y0)

crux = jno.core([pde_u.mse, pde_v.mse, ini_u.mse, ini_v.mse])
crux.solve(5_000)

_u, _u_exact, _v, _v_exact = crux.eval([u, u_exact, v, v_exact])
rel_l2_u = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
rel_l2_v = float(jax.numpy.linalg.norm(_v - _v_exact) / (jax.numpy.linalg.norm(_v_exact) + 1e-8))
print(f"u rel_L2 = {rel_l2_u:.4e}    v rel_L2 = {rel_l2_v:.4e}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as out:
    out.write(
        f"05_coupled_and_inverse/coupled_parabolic_2d.py | epochs=5000"
        f" | rel_L2_u={rel_l2_u:.6e} | rel_L2_v={rel_l2_v:.6e}\n"
    )

assert rel_l2_u < 2e-1, f"u relative L2 error too large: {rel_l2_u:.3e}"
assert rel_l2_v < 2e-1, f"v relative L2 error too large: {rel_l2_v:.3e}"
