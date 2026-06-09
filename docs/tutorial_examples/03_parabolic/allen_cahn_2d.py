"""03 — 2-D Allen–Cahn equation (manufactured-solution verification)

Problem
-------
    ∂u/∂t = ε² ∇²u + u − u³ + f(x, y, t)   on [0, 1]²,   t ∈ [0, 1]

Manufactured solution (homogeneous Dirichlet on ∂Ω built into the ansatz)
    u(x, y, t) = e^{−t} sin(πx) sin(πy)

Source by substitution
    f = u_t − ε² ∇²u − u + u³
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
ε = 0.1
T_end = 1.0

# Time-dependent rectangle (jno.domain.rect produces the time-extended sampler;
# PolygonDomain doesn't yet support the ``time=`` axis).
domain = jno.domain.rect(mesh_size=0.05, time=(0, T_end, 4))
x, y, t = domain.variable("interior")

S = jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact = jno.np.exp(-t) * S
source = jno.np.exp(-t) * S * (2 * ε**2 * π**2 - 2) + jno.np.exp(-3 * t) * S**3

# Network with hard Dirichlet BCs in space; t is fed via the trunk input.
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=2,
        n_outputs=1,
        n_layers=3,
        basis_functions=64,
        hidden_dim=40,
        key=jax.random.PRNGKey(42),
    )
)
net.optimizer(
    optax.adam(
        optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=1, decay_steps=500, end_value=1e-5)
    )
)

xy = jno.np.concat([x, y])
# Bind names so partials read like the math:  u.t, u.xx, u.yy, u.xy, ...
u = (net(t, xy) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y, t=t)

pde = u.t - ε**2 * (u.xx + u.yy) - u + u**3 - source

# Initial condition  (t=0 via 0*t trick)
u_at_0 = net(0 * t, xy) * x * (1 - x) * y * (1 - y)
ini = u_at_0 - S

grad_norms = jno.trackers.gradient_norms(interval=500)
crux = jno.core([pde.mse, ini.mse])

print(f"Allen–Cahn 2-D  (ε = {ε})")
crux.solve(5000, callbacks=[grad_norms])

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
if grad_norms.value is not None:
    print(f"Final ∇L norms (pde, ini): {grad_norms.value['norms']}")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
