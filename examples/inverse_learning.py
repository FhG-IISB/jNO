import jax

jax.config.update("jax_enable_x64", True)  #
import jno
import jno.numpy as jnn
import optax


π = jnn.pi
sin = jnn.sin

# Logging
dire = "./runs/inverse_problem"
jno.logger(dire)

# Domain
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
(x,) = domain.variable("interior")

# Trainable Parameters
a = jnn.parameter((1), name="a")()
b = jnn.parameter((1), name="b")()
c = jnn.parameter((1), name="c")()
_a = 24.8323422495394578345
_b = 1.83472834729837120043
_c = 104.234234275060292911

# Losses
con1 = a * sin(π * x) + _a * sin(π * x)
con2 = b * sin(π * x) - _b * sin(π * x)
con3 = c * sin(π * x) + _c * sin(π * x)

# Solve
crux = jno.core([con1, con2, con3], domain)
crux.solve(200_000, optax.adam(1.0), jno.schedule.learning_rate.exponential(1e-3, 0.9, 200_000, 1e-5)).plot(f"{dire}/best_training_history.png")

# Print the parameters
print(crux.params)
