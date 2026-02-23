import jax
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs
from jno import WeightSchedule as ws

π = jnn.pi
sin = jnn.sin
dire = "./runs/heat_equation"

jno.logger(dire)

domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05), time=(0, 1, 10), compute_mesh_connectivity=True)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

# DeepONet: branch encodes time (1 sensor), trunk encodes spatial coords (x, y)
u_net = jnn.nn.deeponet(
    n_sensors=1,  # branch input: scalar time
    sensor_channels=1,
    coord_dim=2,  # trunk input: (x, y)
    basis_functions=64,
    hidden_dim=64,
    n_layers=2,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 10_000, 1e-5))

# branch=t, trunk=concat(x,y) → G(t)(x,y) = Σ bᵢ(t) tᵢ(x,y)
u = u_net(t, jnn.concat([x, y])) * x * (1 - x) * y * (1 - y)
u0 = u_net(t0, jnn.concat([x0, y0])) * x0 * (1 - x0) * y0 * (1 - y0)

pde = jnn.grad(u, t) - 0.1 * jnn.laplacian(u, [x, y], "finite_difference")  # 2D heat equation
ini = u0 - sin(π * x0) * sin(π * y0)  # Sinusoidal Initial Condition

crux = jno.core([pde.mse, ini.mse], domain)
crux.solve(10_000, constraint_weights=ws([1.0, 3.0])).plot(f"{dire}/training_history.png")
crux.save(f"{dire}/crux.pkl")
