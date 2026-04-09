"""03 — 2-D Allen–Cahn equation  (manufactured-solution verification)

Problem (Allen–Cahn with source)
---------------------------------
    ∂u/∂t = ε² ∇²u + u − u³ + f(x,y,t),   (x,y) ∈ [0,1]²,  t ∈ [0,1]

Manufactured solution
---------------------
    u(x,y,t) = e^{−t} sin(πx) sin(πy)

This automatically satisfies homogeneous Dirichlet BCs on ∂[0,1]².
The source term is computed by substitution:

    f = u_t − ε² ∇²u − u + u³
      = e^{−t} sin(πx) sin(πy) (2ε²π² − 2)
        + e^{−3t} sin³(πx) sin³(πy)

Parameters: ε = 0.1  (interface width)
"""

import jax
import jno

import optax
from jno import LearningRateSchedule as lrs

π = jno.np.pi
sin = jno.np.sin
exp = jno.np.exp
eps = 0.1
T_end = 1.0

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.2),
    time=(0, T_end, 4),
)
x, y, t = domain.variable("interior")

# ── Manufactured solution + source ───────────────────────────────────────────
S = sin(π * x) * sin(π * y)
u_exact = exp(-t) * S

coeff = 2 * eps**2 * π**2 - 2
source = exp(-t) * S * coeff + exp(-3 * t) * S**3

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.np.nn.mlp(
    in_features=3,
    hidden_dims=40,
    num_layers=3,
    key=jax.random.PRNGKey(42),
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

txy = jno.np.concat([t, x, y])
u = net(txy) * x * (1 - x) * y * (1 - y)

# ── PDE residual ──────────────────────────────────────────────────────────────
pde = jno.np.grad(u, t) - eps**2 * jno.np.laplacian(u, [x, y]) - u + u**3 - source

# ── Initial condition  (t=0 via 0*t trick) ──────────────────────────────────
u_at_0 = net(jno.np.concat([0 * t, x, y])) * x * (1 - x) * y * (1 - y)
ini = u_at_0 - sin(π * x) * sin(π * y)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ini.mse], domain)

print(f"Allen–Cahn 2-D  (ε={eps})")
history = crux.solve(10)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"
