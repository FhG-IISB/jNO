import jax
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs

π = jnn.pi
sin = jnn.sin

# Logging
dire = "./runs/laplace1D"
jno.logger(dire)

# Domain
tst_domain = jno.domain(constructor=jno.domain.line(mesh_size=0.001))
domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.01))
(x,) = domain.variable("interior")

# Analytical
_u = -(1 / (π**2)) * sin(π * x)

# Neural Network
key = jax.random.PRNGKey(0)
u_net = jnn.nn.mlp(1, hidden_dims=64, num_layers=3, key=key)
u = u_net(x) * x * (1 - x)

# Constraints
pde = jnn.laplacian(u, [x]) - sin(π * x)  # 1D Laplace equation
con = jnn.tracker(jnn.mean(u - _u))


# Solve
crux = jno.core([pde.mse, con], domain)

u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1_000, 1e-5))
crux.solve(1_000).plot(f"{dire}/training_history.png")

# Finetune using LoRA
# u_net.lora(rank=1, alpha=1.0)
# u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1_000, 1e-5))
# crux.solve(1_000).plot(f"{dire}/training_history_lora.png")

# Save
crux.save(f"{dire}/crux.pkl")
