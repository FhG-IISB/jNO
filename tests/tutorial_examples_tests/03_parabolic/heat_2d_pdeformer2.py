"""03 — 2-D heat equation with PDEformer-2 backbone (smoke test)

Same problem as the tutorial ``docs/tutorial_examples/03_parabolic/heat_2d_pdeformer2.py``
but uses a tiny PDEformer-2 config and few epochs so it runs as a smoke test
in a few seconds.

Verifies: the bridge auto-attaches, training proceeds without error, and the
final loss is finite.
"""

from pathlib import Path

import foundax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi
α = 0.1
T_end = 0.5
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

pde = jno.np.grad(u, t) - α * jno.np.laplacian(u, [x, y])
ini = u0 - jno.np.sin(π * x0) * jno.np.sin(π * y0)
bc = ub  # u = 0 on ∂Ω (soft)

crux = jno.core([pde.mse, ini.mse, bc.mse], domain)
stats = crux.solve(20)

final_loss = float(stats.training_logs[-1]["total_loss"][-1])

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"03_parabolic/heat_2d_pdeformer2.py | epochs=20 | final_loss={final_loss:.6e}\n")

assert jnp.isfinite(final_loss), f"final loss is not finite: {final_loss}"
