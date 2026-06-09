"""01 — 1-D Poisson equation (soft Dirichlet BCs + finite-difference Laplacian)

Problem
-------
    −u''(x) = sin(πx)   on [0, 1],    u(0) = u(1) = 0

Analytical solution: u(x) = sin(πx) / π²

Showcases
---------
* Soft BCs via a separate ``boundary`` tag (loss term in ``jno.core``)
* Finite-difference second derivative (``scheme="finite_difference"``)
* ``.scalar.coords(x=x)`` registers the spatial variable so ``u.xx`` reads
  the same as the math.
"""

import foundax
import jax
import optax

import jno

π = jno.np.pi

domain = jno.domain.line(mesh_size=0.1)
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

u_exact = jno.np.sin(π * x) / π**2

net = jno.nn.wrap(foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

u = net(x).scalar.bind(x=x)
pde = -u.d2(x, scheme="finite_difference") - jno.np.sin(π * x)
bc = net(xb)  # soft BC

crux = jno.core([pde.mse, bc.mse], domain)
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
