"""03 — 2-D heat equation with a PDEformer-2 backbone

Problem
-------
    ∂u/∂t = α ∇²u   on [0, 1]²,    u = 0 on ∂Ω,    u(x, y, 0) = sin(πx) sin(πy)

Analytical: u(x, y, t) = exp(−2απ²t) sin(πx) sin(πy)

PDEformer-2 was pre-trained on canonical PDE graphs. The standard PINN
multiplicative ansatz (``net * x(1-x)y(1-y)``) is incompatible with the
graph encoder (raw Variable(x), Variable(y) aren't in its vocabulary), so
this tutorial uses **soft BCs** — the boundary residual is a separate loss
term that the trainer minimises alongside the PDE residual.
"""

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
xb, yb, tb = domain.variable("boundary")

u_exact = jno.np.exp(-2 * α * π**2 * t) * jno.np.sin(π * x) * jno.np.sin(π * y)

net = jno.nn.wrap(
    foundax.pdeformer2.small(
        num_encoder_layers=2,
        embed_dim=64,
        ffn_embed_dim=128,
        num_heads=4,
        inr_dim_hidden=64,
        inr_num_layers=3,
        hyper_num_layers=2,
        scalar_num_layers=2,
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 100, 2000, 1e-5)))

u = net(t, x, y).scalar.bind(x=x, y=y, t=t)
u0 = net(t0, x0, y0)
ub = net(tb, xb, yb)

# Soft BCs — the canonical PDE graph (no ansatz) is what PDEformer-2 expects.
pde = u.t - α * (u.xx + u.yy)
ini = u0 - jno.np.sin(π * x0) * jno.np.sin(π * y0)
bc = ub

crux = jno.core([pde.mse, ini.mse, bc.mse])
crux.solve(2_000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Heat 2D (PDEformer-2): rel_L2 = {rel_l2:.4e}")

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"03_parabolic/heat_2d_pdeformer2.py | epochs=2000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 5e-1, f"relative L2 error too large: {rel_l2:.3e}"
