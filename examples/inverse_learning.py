import jax

jax.config.update("jax_enable_x64", True)  #
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs


π = jnn.pi
sin = jnn.sin

# Logging
dire = "./runs/inverse_problem"
jno.logger(dire)

# Domain
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
(x,) = domain.variable("interior")

# Trainable Parameters
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
a_net = jnn.parameter((1,), key=k1, name="a")
a = a_net()
b_net = jnn.parameter((1,), key=k2, name="b")
b = b_net()
c_net = jnn.parameter((1,), key=k3, name="c")
c = c_net()
_a = 24.8323422495394578345
_b = 1.83472834729837120043
_c = 104.234234275060292911

# Losses
con1 = a * sin(π * x) + _a * sin(π * x)
con2 = b * sin(π * x) - _b * sin(π * x)
con3 = c * sin(π * x) + _c * sin(π * x)

# Solve
crux = jno.core([con1.mse, con2.mse, con3.mse], domain)

for net in [a_net, b_net, c_net]:
    net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 200_000, 1e-5))

crux.solve(200_000).plot(f"{dire}/best_training_history.png")

# Print the parameters
print(crux.models)
