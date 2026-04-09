"""03 — 2-D heat equation  (parabolic, time-dependent)

Problem
-------
    ∂u/∂t = α ∇²u,   (x,y) ∈ [0,1]²,  t ∈ [0, 0.5]
    u = 0 on ∂Ω  (homogeneous Dirichlet BCs)
    u(x,y,0) = sin(πx) sin(πy)

Analytical solution
-------------------
    u(x,y,t) = exp(−2απ²t) sin(πx) sin(πy)

The x(1−x)y(1−y) factor in the ansatz hard-enforces the Dirichlet BCs on the
unit-square boundary for all times.  The initial condition is a soft constraint
evaluated on the "initial" domain tag.
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
α = 0.1  # thermal diffusivity
T_end = 0.5  # final time
N_t = 4  # number of time slices

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.2),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")
domain.summary()
# ── Analytical solution ───────────────────────────────────────────────────────
u_exact = jno.np.exp(-2 * α * π**2 * t) * jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.np.nn.mlp(
    in_features=3,
    hidden_dims=40,
    num_layers=3,
    key=jax.random.PRNGKey(0),
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))
net.summary()
txy = jno.np.concat([t, x, y])
txy0 = jno.np.concat([t0, x0, y0])

u = net(txy) * x * (1 - x) * y * (1 - y)
u0 = net(txy0) * x0 * (1 - x0) * y0 * (1 - y0)

# ── Constraints ───────────────────────────────────────────────────────────────
pde = jno.np.grad(u, t) - α * jno.np.laplacian(u, [x, y])
ini = u0 - jno.np.sin(π * x0) * jno.np.sin(π * y0)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse], domain).print_shapes()
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"
