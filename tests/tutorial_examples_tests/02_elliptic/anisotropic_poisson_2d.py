"""02 - 2-D anisotropic Poisson equation

Problem
-------
    -(a u_xx + b u_yy) = f(x, y),   (x, y) in [0, 1]^2
    u = 0 on the boundary

Analytical solution
-------------------
    u(x, y) = sin(pi x) sin(pi y)

which gives

    f(x, y) = (a + b) pi^2 sin(pi x) sin(pi y)
"""
import os
import jax
import jno
import foundax
import optax

TEST_MODE = os.getenv("JNO_TUTORIAL_TEST_MODE", "").lower() in {"1", "true", "yes"}

def pick(full, test):
    return test if TEST_MODE else full

pi = jno.np.pi
a = 1.0
b = 3.0

domain = jno.domain(constructor=jno.domain.rect(mesh_size=pick(0.05, 0.3)))
x, y, _ = domain.variable("interior")

u_exact = jno.np.sin(pi * x) * jno.np.sin(pi * y)
forcing = (a + b) * pi**2 * u_exact

net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=pick(32, 16), num_layers=pick(4, 2), key=jax.random.PRNGKey(12)))
net.optimizer(optax.adam(1), lr=jno.schedule.learning_rate.exponential(1e-3, 0.5, 1000, 1e-5))

u = net(x, y) * x * (1 - x) * y * (1 - y)
pde = -(a * u.d2(x) + b * u.d2(y)) - forcing

crux = jno.core([pde.mse], domain)
history = crux.solve(pick(10_000, 1000))

_u, _u_exact = crux.eval([u, u_exact])

rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
if TEST_MODE:
    assert jax.numpy.isfinite(rel_l2), f"non-finite relative L2 error: {rel_l2}"
else:
    assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
