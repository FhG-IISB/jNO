"""Energy Natural Gradient Descent (ENGD) on 2-D Poisson — comparison vs Adam and GD.

Reproduces §4.1 of Zeinhofer, Cakir & Mardal (ICML 2023, arXiv:2302.13163):

    −Δu(x, y) = f(x, y) = 2π² sin(πx) sin(πy),  (x,y) ∈ [0,1]²
    u = 0  on  ∂[0,1]²
    u*(x,y) = sin(πx) sin(πy)

Matching the paper's exact collocation setup (§4.1):
    Interior: 30×30 = 900 equidistant points
    Boundary: 30 per edge × 4 = 120 equidistant points
    Network:  tanh MLP, width=32, depth=1  →  129 trainable parameters
    Init:     weights/biases scaled to std≈0.1 (reference: scale=0.1 in §4.1)

The PINN loss uses soft boundary conditions (explicit BC loss terms), matching
the paper's metric  G = ∫_Ω(Δu_i)(Δu_j)dx + ∫_∂Ω u_i u_j ds  exactly.

**Key requirement: P < N** (129 params vs 1020 points) keeps the Gram matrix
full-rank.  Using fewer points or more parameters would make G rank-deficient.

**Grid line search (§4.1).** ENGD uses a 31-point grid line search
α ∈ {0.5^0, …, 0.5^30} per iteration to find the optimal step size.
This is essential: the energy Gram matrix is initially ill-conditioned
(cond ≫ 1e10), making the natural gradient direction correct but its
magnitude unreliable.  The line search is enabled via ``line_search=True``
in ``jno.callbacks.engd()`` and is the primary reason ENGD converges faster
than Adam/GD without explicit step-size tuning.

Paper median errors at 500 iterations (Table 1 / §4.1):
    GD      8.2e-3
    Adam    1.1e-3 (at 200 000 iterations)
    BFGS    4.4e-4
    E-NGD   2.4e-7

Run on CPU for reliable float64:

    JAX_PLATFORMS=cpu pixi run python whitepaper/engd_poisson2d.py
"""

from __future__ import annotations

import os
from pathlib import Path

import jax

# float64 must be enabled before any JAX computation.
jax.config.update("jax_enable_x64", True)
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import equinox as eqx
import foundax
import jax.numpy as jnp
import numpy as np
import optax

import jno
import jno.jnp_ops as jnn

# ── Hyper-parameters (§4.1) ───────────────────────────────────────────────────
N_INT = 30        # 30×30 = 900 interior points
N_SIDE = 30       # 30 per edge = 120 boundary points total
N_ENGD = 500      # ENGD / GD iterations (paper §4.1)
N_ADAM = 500      # Adam iterations for this demo (paper uses 200 000)
LR_ADAM = 1e-3    # Adam LR
LR_GD = 1e-2      # GD LR
LR_ENGD = 1.0     # ENGD with line search: lr=1.0, step size from grid search
INIT_SCALE = 0.1  # match reference's scale=0.1 weight init (§4.1, ngrad/models.py)
SEED = 0

# ── Exact collocation grids (§4.1) ───────────────────────────────────────────
k = np.linspace(1 / (N_INT + 1), N_INT / (N_INT + 1), N_INT)
X, Y = np.meshgrid(k, k, indexing="ij")
int_pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)  # (900, 2)

t = np.linspace(0, 1, N_SIDE + 1)[:-1]
bdy_pts = np.concatenate(
    [
        np.stack([t, np.zeros(N_SIDE)], 1),   # bottom  y=0
        np.stack([t, np.ones(N_SIDE)], 1),    # top     y=1
        np.stack([np.zeros(N_SIDE), t], 1),   # left    x=0
        np.stack([np.ones(N_SIDE), t], 1),    # right   x=1
    ]
).astype(np.float64)  # (120, 2)

# ── jno domain — inject exact grids via context override ─────────────────────
dom = jno.domain.rect(mesh_size=0.05)   # mesh only controls default init points

x, y, _ = dom.variable("interior")
dom.context["interior"] = int_pts[np.newaxis, np.newaxis]  # (1, 1, 900, 2)

xb, yb, _ = dom.variable("boundary")
dom.context["boundary"] = bdy_pts[np.newaxis, np.newaxis]  # (1, 1, 120, 2)

print(f"Interior pts: {int_pts.shape[0]}  Boundary pts: {bdy_pts.shape[0]}")

# ── PDE setup ────────────────────────────────────────────────────────────────
π = jno.np.pi
forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)


def make_model(key):
    """tanh MLP, width=32, depth=1  →  2*32+32 + 32*1+1 = 129 params.

    Weights/biases scaled to std ≈ INIT_SCALE (0.1) to match the reference
    implementation (ngrad/models.py: scale=0.1 in random_layer_params).
    Small init keeps tanh neurons in the near-linear regime, improving the
    numerical range of the energy Gram matrix.
    """
    base = foundax.mlp(
        in_features=2,
        hidden_dims=32,
        num_layers=1,
        activation=jax.nn.tanh,
        key=key,
    )
    # Scale all parameters to std ≈ INIT_SCALE (reference uses N(0, 0.01)).
    scaled = jax.tree_util.tree_map(
        lambda l: l * INIT_SCALE if eqx.is_array(l) else l, base
    )
    return jnn.nn.wrap(scaled)


def evaluate_rel_l2(crux, u_expr):
    u_pred, u_ref = crux.eval([u_expr, u_exact])
    err = jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-30)
    return float(err)


# ── ENGD ──────────────────────────────────────────────────────────────────────
print(f"\n── ENGD (E-NGD), {N_ENGD} iterations ────────────────────────────────────")
net_e = make_model(jax.random.PRNGKey(SEED))
net_e.optimizer(optax.sgd(LR_ENGD))

u_e = net_e(x, y)
lap_e = u_e.laplacian(x, y)     # Δu at interior points
r_e = lap_e + forcing             # residual: Δu + f = 0 at solution
u_e_bc = net_e(xb, yb)           # BC values at boundary

# Energy Gram:  G = (1/N_int) J_int^T J_int + (1/N_bdy) J_bdy^T J_bdy
# which approximates ∫_Ω (Δu_i)(Δu_j) dx + ∫_∂Ω u_i u_j ds  (paper Sec 3)
engd = jno.callbacks.engd(
    gram_terms=[
        (lap_e.grad(net_e), 1.0),    # interior: ∂(Δu)/∂θ  [900×129]
        (u_e_bc.grad(net_e), 1.0),   # boundary: ∂u/∂θ      [120×129]
    ],
    gram_interval=1,
    line_search=True,   # grid search α∈{0.5^0,…,0.5^30} per step (paper §4.1)
)

crux_e = jno.core([r_e.mse, u_e_bc.mse])
crux_e.solve(N_ENGD, callbacks=[engd])
err_e = evaluate_rel_l2(crux_e, u_e)
print(f"  Relative L² error: {err_e:.2e}  (paper median: 2.4e-7)")

# ── Adam (demo: 500 iters; paper uses 200 000) ────────────────────────────────
print(f"\n── Adam, {N_ADAM} iterations ─────────────────────────────────────────────")
net_a = make_model(jax.random.PRNGKey(SEED))
net_a.optimizer(optax.adam(LR_ADAM))

u_a = net_a(x, y)
r_a = u_a.laplacian(x, y) + forcing
u_a_bc = net_a(xb, yb)

crux_a = jno.core([r_a.mse, u_a_bc.mse])
crux_a.solve(N_ADAM)
err_a = evaluate_rel_l2(crux_a, u_a)
print(f"  Relative L² error: {err_a:.2e}  (paper at 200k iters: 1.1e-3)")

# ── GD (demo: 500 iters; paper uses 200 000) ──────────────────────────────────
print(f"\n── GD, {N_ENGD} iterations ───────────────────────────────────────────────")
net_g = make_model(jax.random.PRNGKey(SEED))
net_g.optimizer(optax.sgd(LR_GD))

u_g = net_g(x, y)
r_g = u_g.laplacian(x, y) + forcing
u_g_bc = net_g(xb, yb)

crux_g = jno.core([r_g.mse, u_g_bc.mse])
crux_g.solve(N_ENGD)
err_g = evaluate_rel_l2(crux_g, u_g)
print(f"  Relative L² error: {err_g:.2e}  (paper at 200k iters: 8.2e-3)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
print(f"  GD    ({N_ENGD} epochs, lr={LR_GD}):    rel L² = {err_g:.2e}")
print(f"  Adam  ({N_ADAM} epochs, lr={LR_ADAM}):   rel L² = {err_a:.2e}")
print(f"  ENGD  ({N_ENGD} epochs, lr={LR_ENGD}):   rel L² = {err_e:.2e}")
print()
print(f"  ENGD vs Adam speedup:  {err_a / err_e:.0f}× lower error in same # iters")

results_file = Path(__file__).parent / "engd_poisson2d_results.txt"
with open(results_file, "w") as f:
    f.write(f"engd_poisson2d.py | N_int={int_pts.shape[0]} N_bdy={bdy_pts.shape[0]}\n")
    f.write(f"  GD    ({N_ENGD} epochs) rel_L2={err_g:.6e}\n")
    f.write(f"  Adam  ({N_ADAM} epochs) rel_L2={err_a:.6e}\n")
    f.write(f"  ENGD  ({N_ENGD} epochs) rel_L2={err_e:.6e}\n")

assert err_e < err_a, f"ENGD ({err_e:.2e}) should outperform Adam ({err_a:.2e})"
assert err_e < err_g, f"ENGD ({err_e:.2e}) should outperform GD ({err_g:.2e})"
print("Assertions passed.")
