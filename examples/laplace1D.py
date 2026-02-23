import jax
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs

π = jnn.pi

# Logging
dire = "./runs/laplace1D"
jno.logger(dire)

# Domain
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, t = domain.variable("interior")

# Analytical
_u = -(1 / (π**2)) * jnn.sin(π * x)

# Neural Network
u_net = jnn.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adamw(1), lr=lrs.exponential(1e-3, 0.3, 1_000, 1e-4))

u = u_net(x) * x * (1 - x)

# Constraints
pde = jnn.grad(jnn.grad(u, x), x) - jnn.sin(π * x)  # 1D Laplace equation
con = jnn.tracker(jnn.mean(u - _u))

# Solve & Save
crux = jno.core([pde.mse, con], domain)
crux.solve(1_000).plot(f"{dire}/training_history.png")
crux.save(f"{dire}/crux.pkl")


pred = crux.eval(u)
true = crux.eval(_u)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(pred[0, :, 0])
plt.plot(true[0, :, 0])

plt.savefig(f"{dire}/solution.png")
