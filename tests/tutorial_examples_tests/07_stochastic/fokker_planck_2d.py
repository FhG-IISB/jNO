"""07 — 2-D Fokker-Planck equation for the Ornstein-Uhlenbeck process

Stochastic process (Itô SDE)
-----------------------------
    dX = −X dt + dW₁
    dY = −Y dt + dW₂        (2-D Ornstein-Uhlenbeck, unit restoring rate and diffusion)

Steady-state Fokker-Planck PDE
--------------------------------
    ∂(x p)/∂x + ∂(y p)/∂y + ½ (∂²p/∂x² + ∂²p/∂y²) = 0,   (x,y) ∈ Ω = [−3, 3]²

Analytical stationary solution
--------------------------------
    p∞(x, y) = (1/π) exp(−x² − y²)

Smoke-test version: 3 000 epochs with a loose tolerance to keep CI fast.
See docs/tutorial_examples/07_stochastic/fokker_planck_2d.py for the full run.
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno
from jno import LearningRateSchedule as lrs

π = jno.np.pi

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), mesh_size=0.15))
x, y, _ = domain.variable("interior")
xb, yb, _ = domain.variable("boundary")

# ── Analytical steady-state distribution ─────────────────────────────────────
p_exact = jno.np.exp(-(x**2 + y**2)) / π
p_exact_bc = jno.np.exp(-(xb**2 + yb**2)) / π

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=64,
        num_layers=5,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

p = net(x, y)

# ── Fokker-Planck residual: ∂(xp)/∂x + ∂(yp)/∂y + ½∆p = 0 ──────────────────
drift = jno.np.grad(x * p, x) + jno.np.grad(y * p, y)
diff = 0.5 * jno.np.laplacian(p, [x, y])
fp = drift + diff

# ── Normalization: ∫∫ p dx dy = 1 ────────────────────────────────────────────
norm = p.integrate() - 1.0

# ── Boundary condition with stochastic measurement noise ─────────────────────
# jno.noise.gaussian() is a lazy Placeholder — a fresh realisation is drawn
# each step from the solver's PRNG key.  Reproducible via jno.setup(seed=...).
p_bc = net(xb, yb) - (p_exact_bc + jno.noise.gaussian(std=1e-4))

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([fp.mse, norm.mse, p_bc.mse], domain)
history = crux.solve(50_000)

# ── Evaluate ─────────────────────────────────────────────────────────────────
_p, _p_exact = crux.eval([p, p_exact])
rel_l2 = float(jnp.linalg.norm(_p - _p_exact) / (jnp.linalg.norm(_p_exact) + 1e-8))

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
