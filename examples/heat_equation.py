import jno
import jno.numpy as jnn
import optax
import jax.numpy as jnp
from flax import nnx
from jaxkan.models.KAN import KAN

π = jnn.pi
sin = jnn.sin

# Logging
dire = "./runs/heat_equation"
jno.logger(dire)

# Domain
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05), time=(0, 1, 10), compute_mesh_connectivity=False)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")
C = jnn.constant("C", {"k": 0.1})  # Or load it from most common file types -> Use during computation (same as a constant python value)

# Neural Network
u = jnn.nn.mlp(hidden_dims=[64, 64])(x, y, t) * x * (1 - x) * y * (1 - y)


class MLP(nnx.Module):
    """Simple fully-connected neural network with scalar output."""

    def __init__(self, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(in_features=3, out_features=64, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=64, out_features=64, rngs=rngs)
        self.dense3 = nnx.Linear(in_features=64, out_features=1, rngs=rngs)

    def __call__(self, x, y, t):
        h = jnp.concat([x, y, t], axis=-1)
        h = jnp.tanh(self.dense1(h))
        h = jnp.tanh(self.dense2(h))
        u = self.dense3(h)
        return u


layer_dims = [3, 12, 12, 1]
req_params = {"D": 5, "flavor": "exact"}


class _KAN(nnx.Module):
    def __init__(self):
        self.KAN = KAN(layer_dims=layer_dims, layer_type="chebyshev", required_parameters=req_params, seed=42)

    def __call__(self, x, y, t):
        h = jnp.concat([x, y, t], axis=-1)
        return self.KAN(h)


u = jnn.nn.wrap(_KAN())(x, y, t) * x * (1 - x) * y * (1 - y)


# u = jnn.nn.wrap(MLP(nnx.Rngs(0)))(x, y, t) * x * (1 - x) * y * (1 - y)

# Constraints
pde = jnn.grad(u(x, y, t), t) - 0.1 * jnn.laplacian(u(x, y, t), [x, y])  # 2D heat equation
ini = u(x0, y0, t0) - sin(π * x0) * sin(π * y0)  # Sinusoidal Initial Condition

# Solve
crux = jno.core([pde, ini], domain)
crux.solve(10_000, optax.adam(1), jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5), jno.schedule.constraint([1.0, 3.0])).plot(f"{dire}/training_history.png")

# Inference
tst_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01), time=(0, 1, 10), compute_mesh_connectivity=False)
crux.plot(operation=u, test_pts=tst_domain).savefig(f"{dire}/u_pred.png", dpi=300)

# Save
crux.save(f"{dire}/crux.pkl")
