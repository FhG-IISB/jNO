"""04 — 1-D advection-diffusion equation (manufactured solution)"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
c = 1.0
ν = 0.05
T_end = 1.0

domain = jno.domain.line(mesh_size=0.1, time=(0, T_end, 4))
x, t = domain.variable("interior")
x0, t0 = domain.variable("initial")
xb, tb = domain.variable("boundary")

u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source = jno.np.exp(-t) * ((ν * π**2 - 1) * jno.np.sin(π * x) + c * π * jno.np.cos(π * x))

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=1,
        n_outputs=1,
        n_layers=3,
        basis_functions=48,
        hidden_dim=32,
        key=jax.random.PRNGKey(1),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

u = net(t, x).scalar.bind(x=x, t=t)

pde = u.t + c * u.x - ν * u.xx - source
ic = net(t0, x0) - jno.np.sin(π * x0)
bc = net(tb, xb)

crux = jno.core([pde.mse, ic.mse, bc.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 2e-1, f"relative L2 error too large: {rel_l2:.3e}"
