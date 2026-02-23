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
x, y, t = domain.variable("interior")
domain.plot(f"{dire}/train_domain.png")

# Architecture space for the model
a_space = jnn.tune.space()
a_space.unique("act", [jnp.tanh, jax.nn.selu, jax.nn.gelu, jnp.sin], category="architecture")
a_space.unique("hid", [4, 8, 16], category="architecture")
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

crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=2).plot(f"{dire}/best_training_history.png")

print(f"Best configuration: {crux.best_config}")
crux.save(f"{dire}/crux.pkl")

# ═══════════════════════════════════════════════════════════════════════
# Per-model tuning with .tune()
# ═══════════════════════════════════════════════════════════════════════
#
# Instead of (or in addition to) a global training space, you can attach
# tunable options directly to each model.  The tuner will search over all
# per-model combinations automatically.
#
# Example: two-model problem where we tune whether the backbone is
# frozen, which LoRA rank to use, and the per-model LR.

print("\n\n=== Per-model .tune() demo ===\n")

domain2 = jno.domain(constructor=jno.domain.disk(mesh_size=0.05))
x2, y2 = domain2.variable("interior")

# Model A — a small "backbone" that we might freeze or LoRA
key2 = jax.random.PRNGKey(42)
backbone = jnn.nn.mlp(2, output_dim=1, hidden_dims=16, num_layers=3, activation=jnp.tanh, key=key2)(x2, y2)
backbone.dont_show()

# Declare per-model tunable options
backbone.tune(
    freeze=[True, False],
    lora=[(4, 1.0), None],  # try LoRA rank 4 or no LoRA
    optimizer=[optax.adam],
    lr=[lrs.constant(1e-3), lrs.constant(1e-4)],
)

_u2 = backbone * x2 * (1 - x2) * y2 * (1 - y2)
pde2 = -jnn.laplacian(_u2, [x2, y2]) - 1.0

crux2 = jno.core([pde2.mse], domain2)

# Global space — only epochs
space2 = jnn.tune.space()
space2.unique("epochs", [200, 500])

stats2 = crux2.sweep(space=space2, optimizer=ng.optimizers.NGOpt, budget=4)
stats2.plot(f"{dire}/per_model_tune.png")
print(f"Best config (per-model): {crux2.best_config}")
