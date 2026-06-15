# Getting Started

This page is the fastest path from installation to a first PDE solve with jNO.

Before you begin, complete setup in [Installation](Installation.md).

## End-to-End Example

The example below solves a **parametric 2-D Poisson equation** with a random diffusion coefficient $k$:

$$-\nabla \cdot (k \, \nabla u) = 1, \quad u\big|_{\partial\Omega} = 0, \quad k \sim \mathcal{U}(0.5, 1.5)$$

It uses a [DeepONet](foundation_models/index.md) to learn the solution operator — mapping any realisation of $k$ to the corresponding field $u$ — and demonstrates the full jNO pipeline: domain setup, operator network, hard boundary enforcement, checkpointing, and inference on a finer test mesh.

```python
import jno
import jax
import optax
import foundax

dir = jno.setup("./runs/test")

# Domain: 500 random realisations of k, mesh spacing 0.05 on [0,2]×[0,1]
dom = 500 * jno.domain.rect(mesh_size=0.05, x_range=(0, 2), y_range=(0, 1))
x, y, _ = dom.variable("interior")
xb, yb, _ = dom.variable("boundary")

random_k = jax.random.uniform(jax.random.PRNGKey(0), shape=(500, 1, 1), minval=0.5, maxval=1.5)
k = dom.variable("k", random_k)

# Neural network: DeepONet maps (k, coords) → u
fx = foundax.deeponet(
    n_sensors=1, coord_dim=2,
    basis_functions=32, hidden_dim=128,
    activation=jax.numpy.tanh,
)
net = jno.nn.wrap(fx)
net.optimizer(optax.adam(
    learning_rate=optax.schedules.cosine_decay_schedule(
        init_value=1e-3, decay_steps=20_000, alpha=1e-5
    )
))

# Hard boundary enforcement via output transformation (u = 0 on ∂Ω automatically)
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.dd(x) + u.dd(y)) + 1.0  # PDE residual

# Checkpoint every 5000 epochs, keep the best 3 by total loss
cb = jno.callbacks.checkpoint(save_interval_epochs=5000, best_fn=lambda m: m["total_loss"])

# Compile → train → save
crux = jno.core(constraints=[pde.mse])
crux.print_shapes()
crux.solve(epochs=20_000, batchsize=32, callbacks=[cb]).plot(f"{dir}/training.png")
jno.save(crux, f"{dir}/model.pkl")

# Inference on a finer mesh with new k values
tst_dom = 16 * jno.domain.rect(mesh_size=0.01, x_range=(0, 2), y_range=(0, 1))
tst_dom.variable("k", jax.random.uniform(jax.random.PRNGKey(1), shape=(16, 1, 1), minval=0.1, maxval=1.9))

pred, x_t, y_t, k_t = crux.eval([u, x, y, k], domain=tst_dom)
print(pred.shape, x_t.shape, y_t.shape, k_t.shape)
```

## Recommended Path

1. Run the example above (or a tutorial from `docs/tutorial_examples/`).
2. Learn domain construction in [Domain and Geometry](Domain-and-Geometry.md).
3. Configure optimization in [Training](training/index.md).
4. Control trainability in [Model Controls](model-controls/index.md).
5. Explore model families in [Foundation Models](foundation_models/index.md).

## Project Setup Helper

`jno.setup()` initializes logging and returns a run directory in one call:

```python
dire = jno.setup("./runs/my_experiment")
```

## Understanding Output

During training, jNO prints progress per epoch:

```text
Epoch  1000/20000| L: 1.2345e-03 | C0: 1.1000e-03
```

- `L` — total weighted loss.
- `C0`, `C1`, … — per-constraint losses.
- `T0`, `T1`, … — tracker values (when trackers are enabled).

