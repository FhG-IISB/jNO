"""07 — 2-D Fokker-Planck equation for the Ornstein-Uhlenbeck process

Stochastic process (Itô SDE)
-----------------------------
    dX = −X dt + dW₁
    dY = −Y dt + dW₂        (2-D Ornstein-Uhlenbeck, unit restoring rate and diffusion)

Steady-state Fokker-Planck PDE
--------------------------------
    ∂(x p)/∂x + ∂(y p)/∂y + ½ (∂²p/∂x² + ∂²p/∂y²) = 0,   (x,y) ∈ Ω = [−3, 3]²

The first two terms are the *drift* divergence from the OU restoring forces −X, −Y.
The third term is the *diffusion* Laplacian with σ² = 1.

Soft Dirichlet BC:  p ≈ 0 on ∂Ω
    (the Gaussian decays to exp(−9)/π ≈ 4 × 10⁻⁵ at the domain edges, so
    the boundary condition is effectively zero yet still constrains the scale)

Analytical stationary solution
--------------------------------
    p∞(x, y) = (1/π) exp(−x² − y²)

Techniques shown
-----------------
* Fokker-Planck (forward Kolmogorov) PINN for a 2-D SDE
* Normalization constraint via the integration operator: ∫∫_Ω p dx dy = 1
* jno.noise.gaussian() adds stochastic measurement noise to the boundary
  observations — a fresh realisation is drawn each training step automatically
  through the solver's split PRNG key, without any user-side key management
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain (centred at origin so the stationary Gaussian is symmetric) ────────
domain = jno.domain.rect(x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), mesh_size=0.15)
x, y, _ = domain.variable("interior")
xb, yb, _ = domain.variable("boundary")

# ── Analytical steady-state distribution ─────────────────────────────────────
p_exact = jno.np.exp(-(x**2 + y**2)) / π
p_exact_bc = jno.np.exp(-(xb**2 + yb**2)) / π  # ≈ 4e-5 at the corners

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
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

p = net(x, y)  # probability density field

# ── Fokker-Planck residual ─────────────────────────────────────────────────────
# drift term:  ∂(xp)/∂x + ∂(yp)/∂y
drift = (x * p).d(x) + (y * p).d(y)
# diffusion term:  ½ ∆p
diff = 0.5 * jno.np.laplacian(p, [x, y])
fp = drift + diff  # residual = 0

# ── Normalization constraint:  ∫∫_Ω p dx dy = 1 ──────────────────────────────
norm = p.integrate() - 1.0

# ── Boundary condition with stochastic measurement noise ─────────────────────
# In practice the boundary values would come from noisy SDE path statistics.
# Here we simulate that by adding Gaussian noise (std = 1e-4) to the exact
# (near-zero) boundary values.  jno.noise.gaussian() is a lazy Placeholder —
# the solver resamples it every epoch so the network sees fresh noise without
# any explicit key management in user code.
p_bc = net(xb, yb) - (p_exact_bc + jno.noise.gaussian(std=1e-4))

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([fp.mse, norm.mse, p_bc.mse], domain)
history = crux.solve(50_000)

# ── Evaluate ─────────────────────────────────────────────────────────────────
_p, _p_exact = crux.eval([p, p_exact])
rel_l2 = float(jnp.linalg.norm(_p - _p_exact) / (jnp.linalg.norm(_p_exact) + 1e-8))

print(f"Relative L2 error: {rel_l2:.4e}")

# ── Record ────────────────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"07_stochastic/fokker_planck_2d.py | epochs=50000 | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 2e-1, f"relative L2 error too large: {rel_l2:.3e}"
