"""06 — Integral constraints and flux monitoring (2-D Poisson)

Problem
-------
    −∇²u(x,y) = 2π² sin(πx) sin(πy),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω

Analytical solution
-------------------
    u(x,y) = sin(πx) sin(πy)

This tutorial demonstrates how to use .integrate() for two distinct purposes:

1. As a **tracker** — monitor the volume mean ∫_Ω u dA during training.
   The exact value is 4/π² ≈ 0.405.  A network that only satisfies the zero
   Dirichlet BC would give ∫ u = 0, so this metric reveals interior accuracy.

2. As a **soft constraint** — enforce a prescribed integral value in the loss,
   in addition to the PDE residual and boundary condition.

Both uses are JIT-compatible and fully differentiable w.r.t. model parameters.
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax
from jno.numpy import tracker
from shapely.geometry import box

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain(box(0, 0, 1, 1), mesh_size=0.05)

x, y, _ = domain.variable("interior")
x_b, y_b, _ = domain.variable("boundary")

domain.summary()

# ── Analytic forcing and boundary data ─────────────────────────────────────────
forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact_b = jno.np.sin(π * x_b) * jno.np.sin(π * y_b)  # = 0 on ∂Ω

# Exact volume integral: ∫₀¹∫₀¹ sin(πx)sin(πy) dxdy = (2/π)² = 4/π²
TARGET_INTEGRAL = 4.0 / float(jnp.pi) ** 2  # ≈ 0.4053

# ── Model ──────────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=64,
        num_layers=4,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(
    optax.adam(
        optax.exponential_decay(
            init_value=1e-3,
            transition_steps=3000,
            decay_rate=0.5,
            end_value=1e-5,
        )
    )
)

# Hard-enforce u=0 on ∂Ω by multiplying by x(1-x)y(1-y).
# The network then only needs to learn the interior shape.
u = (net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y)

# ── Losses ─────────────────────────────────────────────────────────────────────
# Standard PDE residual
pde = -(u.xx + u.yy) - forcing

# Volume-mean tracker — logged every 200 epochs, does not enter the gradient.
# After convergence this should approach TARGET_INTEGRAL ≈ 0.405.
vol_mean = tracker(u.integrate(), interval=200)

# Optional soft integral constraint — uncomment to add it to the loss.
# This can accelerate convergence when the PDE residual alone is slow to
# pin the global magnitude of the solution.
#
# integral_loss = (u.integrate() - TARGET_INTEGRAL).square()
# losses = [pde.mse, integral_loss, vol_mean]

losses = [pde.mse, vol_mean]

# ── Solve ──────────────────────────────────────────────────────────────────────
EPOCHS = 30_000
crux = jno.core(losses).print_shapes()
_history = crux.solve(EPOCHS)

# ── Evaluate ───────────────────────────────────────────────────────────────────
u_pred, u_ref = crux.eval([u, jno.np.sin(π * x) * jno.np.sin(π * y)])

rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
print(f"Target integral:   {TARGET_INTEGRAL:.6f}")

# ── Record result ──────────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"06_integration/flux_conservation_2d.py | epochs={EPOCHS} "
        f"| rel_L2={rel_l2:.6e} | target_integral={TARGET_INTEGRAL:.6f}\n"
    )

assert rel_l2 < 0.15, f"Relative L2 error too large: {rel_l2:.3e}"
