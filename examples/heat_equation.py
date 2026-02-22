import jax
import jno
import jno.numpy as jnn
import optax
import jax.numpy as jnp
import equinox as eqx
from jno import LearningRateSchedule as lrs
from jno import WeightSchedule as ws
from jno.architectures.linear import Linear

π = jnn.pi
sin = jnn.sin

# Logging
dire = "./runs/heat_equation"
jno.logger(dire)

# Domain
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, 1, 10),
    compute_mesh_connectivity=False,
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")
C = jnn.constant("C", {"k": 0.1})  # Or load it from most common file types -> Use during computation (same as a constant python value)

# Neural Network — use the built-in MLP factory (requires in_features and key)
key = jax.random.PRNGKey(0)
u_net = jnn.nn.mlp(3, hidden_dims=[64, 64], key=key)
u = u_net(x, y, t) * x * (1 - x) * y * (1 - y)


# Custom equinox module example
class MLP(eqx.Module):
    """Simple fully-connected neural network with scalar output."""

    dense1: Linear
    dense2: Linear
    dense3: Linear

    def __init__(self, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.dense1 = Linear(3, 64, key=k1)
        self.dense2 = Linear(64, 64, key=k2)
        self.dense3 = Linear(64, 1, key=k3)

    def __call__(self, x, y, t):
        h = jnp.concat([x, y, t], axis=-1)
        h = jnp.tanh(self.dense1(h))
        h = jnp.tanh(self.dense2(h))
        u = self.dense3(h)
        return u


# u = jnn.nn.wrap(MLP(key=jax.random.PRNGKey(0)))(x, y, t) * x * (1 - x) * y * (1 - y)

# Constraints
pde = jnn.grad(u(x, y, t), t) - 0.1 * jnn.laplacian(u(x, y, t), [x, y])  # 2D heat equation
ini = u(x0, y0, t0) - sin(π * x0) * sin(π * y0)  # Sinusoidal Initial Condition

# Solve
crux = jno.core([pde.mse, ini.mse], domain)

u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 10_000, 1e-5))
crux.solve(10_000, constraint_weights=ws([1.0, 3.0])).plot(f"{dire}/training_history.png")

# Inference
tst_domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.025),
    time=(0, 1, 10),
    compute_mesh_connectivity=False,
)

# Save
crux.save(f"{dire}/crux.pkl")
