"""03 — 2-D reaction-diffusion with PDEformer-2 backbone (smoke test)

Tiny configuration of
``docs/tutorial_examples/03_parabolic/reaction_diffusion_2d_pdeformer2.py``
to validate that the bridge handles linear reaction terms (+ λu) in addition
to dt and ∇² in a few seconds.
"""

from pathlib import Path

import foundax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi
sin = jno.np.sin

α = 0.05
λ = 1.0
T_end = 1.0
N_t = 3

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.2),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")
xb, yb, tb = domain.variable("boundary")

# ── Analytical solution (verification only) ───────────────────────────────────
u_exact = jno.np.exp(-(2 * α * π**2 + λ) * t) * sin(π * x) * sin(π * y)

# ── Network: tiny PDEformer-2 ────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.pdeformer2.small(
        num_encoder_layers=1,
        embed_dim=32,
        ffn_embed_dim=64,
        num_heads=4,
        inr_dim_hidden=32,
        inr_num_layers=2,
        hyper_num_layers=1,
        scalar_num_layers=1,
    )
)
net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))

u = net(t, x, y)
u0 = net(t0, x0, y0)
ub = net(tb, xb, yb)

pde = jno.np.grad(u, t) - α * jno.np.laplacian(u, [x, y]) + λ * u
ini = u0 - sin(π * x0) * sin(π * y0)
bc = ub

crux = jno.core([pde.mse, ini.mse, bc.mse], domain)
stats = crux.solve(20)

final_loss = float(stats.training_logs[-1]["total_loss"][-1])

# Evaluate against analytical solution (exp is fine — used by jno.eval, not the bridge).
_u, _u_exact = crux.eval([u, u_exact])
import numpy as np

rel_l2 = float(np.linalg.norm(np.asarray(_u) - np.asarray(_u_exact)) / (np.linalg.norm(np.asarray(_u_exact)) + 1e-8))

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"03_parabolic/reaction_diffusion_2d_pdeformer2.py | epochs=20 "
        f"| final_loss={final_loss:.6e} | rel_L2={rel_l2:.6e}\n"
    )

assert jnp.isfinite(final_loss), f"final loss is not finite: {final_loss}"
