import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
from jno import WeightSchedule as ws
import optax
from flax import linen as nn
import jax.numpy as jnp
import nevergrad as ng

dire = "./runs/tuner"
jno.logger(dire)

domain = jno.domain(constructor=jno.domain.disk(mesh_size=0.05))
x, y = domain.variable("interior")
domain.plot(f"{dire}/train_domain.png")

# Architecture space for the model
a_space = jnn.tune.space()
a_space.unique("act", [nn.tanh, nn.selu, nn.gelu, jnp.sin], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")


class MLP(nn.Module):
    arch: jnn.tune.Arch

    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(self.arch("dep")):
            h = self.arch("act")(nn.Dense(self.arch("hid"))(h))
        return nn.Dense(1)(h)


u = jnn.nn.wrap(MLP, space=a_space)(x, y)
u.dont_show()

_u = u * x * (1 - x) * y * (1 - y)
pde = -jnn.laplacian(_u, [x, y]) - 1.0

# Problem
crux = jno.core([pde], domain)

# Training hyperparameter space (separate from architecture)
t_space = jnn.tune.space()
t_space.unique("epochs", [500, 1000])
t_space.unique("optimizer", [optax.adam(1.0), optax.adamw(1.0)])
t_space.unique("learning_rate", [lrs.exponential(1e-3, 0.8, 10000, 1e-5), lrs.exponential(1e-2, 0.9, 5000, 1e-5), lrs.constant(1e-3)])
t_space.unique("weight_schedule", [ws([1.0])])
t_space.unique("batchsize", [1, 1])
crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=10).plot(f"{dire}/best_training_history.png")


# One training sweep, define extra space, optimizer and budget -> returns eval of best run
crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=10).plot(f"{dire}/best_training_history.png")
crux.sweep(space=t_space, optimizer=None, budget=None).plot(f"{dire}/best_training_history.png")

print(f"Best configuration: {crux.best_config}")

# Inference
tst_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
crux.plot(operation=u, test_pts=tst_domain).savefig(f"{dire}/u_pred.png", dpi=300)
crux.save(f"{dire}/crux.pkl")
