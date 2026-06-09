"""03 — 1-D heat equation (parabolic, time-dependent)

Problem
-------
    ∂u/∂t = α ∂²u/∂x²   on [0, 1],    u(0, t) = u(1, t) = 0,    u(x, 0) = sin(πx)

Analytical solution: u(x, t) = exp(−απ²t) sin(πx)

Showcases the soft IC + soft BC pattern — three constraints compete in the
loss: PDE residual, initial condition, spatial boundary.
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
α = 0.1
T_end = 0.5

domain = jno.domain.line(mesh_size=0.05, time=(0, T_end, 4))
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")
xb, tb = domain.variable("boundary")

u_exact = jno.np.exp(-α * π**2 * t) * jno.np.sin(π * x)

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=3,
        basis_functions=48,
        hidden_dim=32,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 2000, 0.5, end_value=1e-5)))

u = net(t, x).scalar.bind(x=x, t=t)

pde = u.t - α * u.xx  # PDE residual
ic = net(t0, x0) - jno.np.sin(π * x0)  # initial condition (t=0 slice)
bc = net(tb, xb)  # spatial boundary (u=0)

crux = jno.core([pde.mse, ic.mse, bc.mse], domain)
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 2e-1, f"relative L2 error too large: {rel_l2:.3e}"
