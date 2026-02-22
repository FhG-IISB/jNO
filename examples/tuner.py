import jax
import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
from jno import WeightSchedule as ws
import optax
import equinox as eqx
from jno.architectures.linear import Linear
import jax.numpy as jnp
import nevergrad as ng

dire = "./runs/tuner"
jno.logger(dire)

domain = jno.domain(constructor=jno.domain.disk(mesh_size=0.05))
x, y = domain.variable("interior")
domain.plot(f"{dire}/train_domain.png")

# Architecture space for the model
a_space = jnn.tune.space()
a_space.unique("act", [jnp.tanh, jax.nn.selu, jax.nn.gelu, jnp.sin], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")


class MLP(eqx.Module):
    layers: list
    out_layer: Linear
    act: callable

    def __init__(self, arch: jnn.tune.Arch, *, key):
        depth = arch("dep")
        hidden = arch("hid")
        self.act = arch("act")
        keys = jax.random.split(key, depth + 1)

        self.layers = []
        in_dim = 2
        for i in range(depth):
            self.layers.append(Linear(in_dim, hidden, key=keys[i]))
            in_dim = hidden
        self.out_layer = Linear(hidden, 1, key=keys[depth])

    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for layer in self.layers:
            h = self.act(layer(h))
        return self.out_layer(h)


u = jnn.nn.wrap(MLP, space=a_space)(x, y)
u.dont_show()

_u = u * x * (1 - x) * y * (1 - y)
pde = -jnn.laplacian(_u, [x, y]) - 1.0

# Problem
crux = jno.core([pde.mse], domain)

# Training hyperparameter space (separate from architecture)
t_space = jnn.tune.space()
t_space.unique("epochs", [500, 1000])
t_space.unique("optimizer", [optax.adam, optax.adamw])
t_space.unique("learning_rate", [lrs.exponential(1e-3, 0.8, 10000, 1e-5), lrs.exponential(1e-2, 0.9, 5000, 1e-5), lrs.constant(1e-3)])
t_space.unique("weight_schedule", [ws([1.0])])
t_space.unique("batchsize", [1, 1])
crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=4).plot(f"{dire}/best_training_history.png")


# One training sweep, define extra space, optimizer and budget -> returns eval of best run
crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=10).plot(f"{dire}/best_training_history.png")
crux.sweep(space=t_space, optimizer=None, budget=None).plot(f"{dire}/best_training_history.png")

print(f"Best configuration: {crux.best_config}")

# Inference
tst_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
crux.plot(operation=u, test_pts=tst_domain).savefig(f"{dire}/u_pred.png", dpi=300)
crux.save(f"{dire}/crux.pkl")
