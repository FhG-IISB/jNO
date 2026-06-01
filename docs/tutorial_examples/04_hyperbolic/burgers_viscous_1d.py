"""04 — 1-D viscous Burgers equation  (manufactured solution + adaptive resampling)

Problem
-------
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f(x,t),   x ∈ [0,1],  t ∈ [0,1]
    u(0,t) = u(1,t) = 0
    u(x,0) = sin(πx)

Manufactured solution
---------------------
    u_exact(x,t) = e^{−t} sin(πx)

Substituting into Burgers gives the source term:
    f = u_t + u u_x − ν u_xx
      = e^{−t} (νπ² − 1) sin(πx)
        + π/2 · e^{−2t} sin(2πx)

The nonlinear term u u_x = e^{−2t} sin(πx)·πcos(πx) = πe^{−2t}/2 · sin(2πx)
creates a higher-frequency component in the forcing.  Because the residual of
this nonlinear term peaks near x = 0.5 (where sin(πx) is maximum), RAD
progressively moves interior collocation points toward that region during
training.

Domain setup
------------
The PDE is solved on the rectangle [0,1] × [0,1] with (x,t) treated as two
spatial coordinates.  This "all-spatial" formulation is the standard approach
in PINN papers and is required here so that adaptive resampling can operate
on the full (x,t) mesh-node pool — time-dependent domain constructors use a
different internal representation that is incompatible with spatial resampling.

Adaptive resampling
-------------------
    RAD (Residual-Adaptive Distribution) replaces a fraction of interior
    collocation points each time it fires, drawing new points from the full
    mesh-node pool (8× larger than the working set) with probability
    proportional to the current PDE residual.  This concentrates points
    where the nonlinear convection residual is largest, without requiring any
    manual spatial grid tuning.

    Reference: Wu et al., "A comprehensive study of non-adaptive and
    residual-based adaptive sampling for physics-informed neural networks",
    Comput. Methods Appl. Mech. Engrg., 403 (2023).
"""

import foundax
import jax
import optax

import jno
from jno import LearningRateSchedule as lrs
from jno import sampler

π = jno.np.pi
ν = 0.05  # viscosity — decrease for sharper gradients
T_end = 1.0

# ── Adaptive resampling strategy ──────────────────────────────────────────────
# RAD fires every 200 epochs (starting at epoch 500, after the network has
# formed a rough solution), replacing 20 % of interior points drawn from the
# full 513-node mesh pool and biased toward high-PDE-residual regions.
strategy = sampler.rad(
    resample_every=200,
    resample_fraction=0.2,
    start_epoch=500,
    k=5,
)

# ── Domain ────────────────────────────────────────────────────────────────────
# 2D rect domain with x ∈ [0,1] (space) and y ∈ [0,1] (time, labelled t below).
# mesh_size=0.05 → 513 interior nodes in the candidate pool.
# sample=(60, None) → 60-point working set (8× pool-to-sample ratio).
domain = 1 * jno.domain(
    constructor=jno.domain.rect(
        mesh_size=0.05,
        x_range=(0.0, 1.0),
        y_range=(0.0, T_end),
    )
)
vars_int = domain.variable("interior", sample=(60, None), resampling_strategy=strategy)
x, t = vars_int[0], vars_int[1]

# IC boundary: t = 0 (bottom face of the rectangle).
vars_bot = domain.variable("bottom", sample=(20, None))
x0, t0 = vars_bot[0], vars_bot[1]

# ── Manufactured solution + source ───────────────────────────────────────────
u_exact = jno.np.exp(-t) * jno.np.sin(π * x)

# f = e^{-t}(νπ² − 1) sin(πx)  +  (π/2) e^{-2t} sin(2πx)
source = jno.np.exp(-t) * (ν * π**2 - 1) * jno.np.sin(π * x) + (π / 2) * jno.np.exp(-2 * t) * jno.np.sin(2 * π * x)

# ── Network  (hard Dirichlet BCs) ────────────────────────────────────────────
# MLP with 2 inputs (x, t); the x*(1-x) factor enforces u(0,t)=u(1,t)=0.
net = jno.nn.wrap(
    foundax.mlp(
        2,
        hidden_dims=64,
        num_layers=4,
        key=jax.random.PRNGKey(3),
    )
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

u = net(x, t) * x * (1 - x)

# ── PDE residual:  u_t + u u_x − ν u_xx − f = 0 ─────────────────────────────
u_t = jno.np.grad(u, t)
u_x = jno.np.grad(u, x)
u_xx = jno.np.grad(u_x, x)
pde = u_t + u * u_x - ν * u_xx - source

# ── Initial condition ─────────────────────────────────────────────────────────
u_0 = net(x0, t0) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
