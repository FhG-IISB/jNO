"""01 - 1-D biharmonic equation (beam-like fourth-order problem)

Problem
-------
    u''''(x) = 24,   x in [0, 1]

    u(0) = u(1) = 0,
    u'(0) = u'(1) = 0

Analytical solution
-------------------
    u(x) = x^2 (1 - x)^2

The x^2 (1-x)^2 ansatz hard-enforces the clamped boundary conditions, making
this a compact fourth-order derivative example.
"""

import jax
import jno
import jno.jnp_ops as jnn
import matplotlib.pyplot as plt
import numpy as np
import optax
from jno import LearningRateSchedule as lrs

dire = jno.setup(__file__)
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x = jno.domain.line(mesh_size=0.01)
domain = jno.domain(constructor=x)
x, _ = domain.variable("interior")

u_exact = x**2 * (1 - x) ** 2

net = jno.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(11),
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.6, 8_000, 1e-5))

u = net(x) * x**2 * (1 - x) ** 2
u_xxxx = jnn.grad(jnn.grad(jnn.grad(jnn.grad(u, x), x), x), x)

pde = u_xxxx - 24.0
error = jnn.tracker((u - u_exact).mse, interval=200)

crux = jno.core([pde.mse, error], domain)
history = crux.solve(8000, profile=True)


history.plot(f"{dire}/training_history.png")

pts = np.array(crux.domain_data.context["interior"][0, 0, :, 0])
idx = np.argsort(pts)
xs = pts[idx]
pred = np.array(crux.eval(u)).reshape(xs.shape[0], 1)[:, 0][idx]
true = np.array(crux.eval(u_exact)).reshape(xs.shape[0], 1)[:, 0][idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_title("Biharmonic solution")
ax1.plot(xs, pred, label="PINN")
ax1.plot(xs, true, "--", label="exact")
ax1.set_xlabel("x")
ax1.legend()

ax2.set_title("Pointwise |error|")
ax2.plot(xs, np.abs(pred - true))
ax2.set_xlabel("x")

plt.tight_layout()
plt.savefig(f"{dire}/solution.png", dpi=150)
print(f"Saved to {dire}/")
