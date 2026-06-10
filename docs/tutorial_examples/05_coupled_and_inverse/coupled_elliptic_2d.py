"""05 — Coupled elliptic PDE system (manufactured solution)"""

import foundax
import jax
import optax
from shapely.geometry import box

import jno

π = jno.np.pi
domain = jno.domain(box(0, 0, 1, 1), mesh_size=0.1)
x, y, _ = domain.variable("interior")

u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
v_exact = jno.np.sin(2 * π * x) * jno.np.sin(π * y)

f = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y) + jno.np.sin(2 * π * x) * jno.np.sin(π * y)
g = 5 * π**2 * jno.np.sin(2 * π * x) * jno.np.sin(π * y) + jno.np.sin(π * x) * jno.np.sin(π * y)

k1, k2 = jax.random.split(jax.random.PRNGKey(0))
u_net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=48, num_layers=4, key=k1))
v_net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=48, num_layers=4, key=k2))
for net in (u_net, v_net):
    net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 50, 5000, 1e-5)))

# 16 * x(1-x)y(1-y) peaks at 1 — keeps the learned field on the right scale.
ansatz = 16.0 * x * (1 - x) * y * (1 - y)
u = (u_net(x, y) * ansatz).scalar.bind(x=x, y=y)
v = (v_net(x, y) * ansatz).scalar.bind(x=x, y=y)

pde_u = -(u.xx + u.yy) + v - f
pde_v = -(v.xx + v.yy) + u - g

crux = jno.core([pde_u.mse, pde_v.mse])
crux.solve(5_000)

_u, _u_exact, _v, _v_exact = crux.eval([u, u_exact, v, v_exact])
rel_l2_u = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
rel_l2_v = float(jax.numpy.linalg.norm(_v - _v_exact) / (jax.numpy.linalg.norm(_v_exact) + 1e-8))
print(f"u rel_L2 = {rel_l2_u:.4e}    v rel_L2 = {rel_l2_v:.4e}")
assert rel_l2_u < 1.5e-1, f"u rel_L2 too large: {rel_l2_u:.3e}"
assert rel_l2_v < 1.5e-1, f"v rel_L2 too large: {rel_l2_v:.3e}"
