<p align="center">
  <img src="assets/logo.png" alt="jNO logo" width="500"/>
</p>

<p align="center">
    <a href="https://fhg-iisb.github.io/jNO/">
        <img src="https://img.shields.io/badge/docs-GitHub%20Pages-0aa?style=for-the-badge" alt="Dev Docs"/>
    </a>
    <a href="https://github.com/FhG-IISB/jno/actions/workflows/ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/FhG-IISB/jno/ci.yml?branch=main&style=for-the-badge&label=tests" alt="Tests"/>
    </a>
    <a href="https://codecov.io/gh/FhG-IISB/jno">
        <img src="https://img.shields.io/codecov/c/github/FhG-IISB/jno/main?style=for-the-badge&label=coverage" alt="Coverage"/>
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-EPL--2.0-2ea44f?style=for-the-badge" alt="License"/>
    </a>
    <a href="CITATION.cff">
        <img src="https://img.shields.io/badge/cite-CITATION.cff-6b5b95?style=for-the-badge" alt="Citation"/>
    </a>
    <img src="https://img.shields.io/badge/docker-image%20available-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker image available"/>
    <a href="https://arxiv.org/abs/2605.10159">
        <img src="https://img.shields.io/badge/arXiv-2605.10159-b31b1b?style=for-the-badge" alt="arXiv Paper"/>
    </a>
</p>

**jNO** (jax Neural Operators) is a JAX-native library for training neural
operators and physics-informed networks. It unifies data-driven operator
regression, residual-based PINN training, mesh-aware FEM/variational
PINNs, and foundation-model fine-tuning under one symbolic tracing
language — write the PDE once, compile once, train, evaluate, and
checkpoint without rewriting the surrounding code.

> Status: research-level repository under active development. Public API
> is stabilising but may change between minor versions.

## Install

```bash
pip install "jax-neural-operators[fem]"
```

GPU support is included by default (jNO depends on `jax[cuda]`). See
[`docs/Installation.md`](docs/Installation.md) for Pixi, Docker, and
specific-CUDA-version setups.

Foundation models and other neural operator architectures live in a
separate repository ([foundax](https://github.com/FhG-IISB/foundax)),
installed automatically as a dependency so they can also be used on
their own.

## Citation

If you use jNO in academic work, please cite the
[arXiv preprint](https://arxiv.org/abs/2605.10159) (and the specific
version via the `Cite this repository` button on GitHub, which reads
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{armbruster2026jno,
  title   = {jNO: A JAX Library for Neural Operator and Foundation Model Training},
  author  = {Armbruster, Leon and Ramesh, Rathan and Kruse, Georg and Straub, Christopher},
  journal = {arXiv preprint arXiv:2605.10159},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.10159},
  url     = {https://arxiv.org/abs/2605.10159}
}
```

## Example

```python
import jno
import jax
import optax
import foundax

dir = jno.setup("./runs/test")

# Domain
dom = 500 * jno.domain.rect(mesh_size=0.05, x_range=(0, 2), y_range=(0, 1))
x, y, _ = dom.variable("interior")
xb, yb, _ = dom.variable("boundary")

random_k = jax.random.uniform(jax.random.PRNGKey(0), shape=(500, 1, 1), minval=0.5, maxval=1.5)
k = dom.variable("k", random_k)

# Neural Network
fx = foundax.deeponet(n_sensors=1, coord_dim=2, basis_functions=32, hidden_dim=128, activation=jax.numpy.tanh)
net = jno.nn.wrap(fx)
net.optimizer(optax.adam(learning_rate=optax.schedules.cosine_decay_schedule(init_value=1e-3, decay_steps=20_000, alpha=1e-5)))

# Forward pass and hard enforcement of BCs via output transformation
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.dd(x) + u.dd(y)) + 1.0  # PDE Loss

# Checkpointing (saves every 5000 epochs, keeps best 3)
cb = jno.callbacks.checkpoint(save_interval_epochs=5000, best_fn=lambda m: m["total_loss"])

# Create -> Train -> Save
crux = jno.core(constraints=[pde.mse], domain=dom).print_shapes()
crux.solve(epochs=20_000, batchsize=32, callbacks=[cb]).plot(f"{dir}/training.png")
jno.save(crux, f"{dir}/model.pkl")

# Inference via test domain on a finer mesh
tst_dom = 16 * jno.domain.rect(mesh_size=0.01, x_range=(0, 2), y_range=(0, 1))
tst_dom.variable("k", jax.random.uniform(jax.random.PRNGKey(0), shape=(16, 1, 1), minval=0.1, maxval=1.9))

pred, x, y, k = crux.eval([u, x, y, k], domain=tst_dom)
print(pred.shape, x.shape, y.shape, k.shape)
```

