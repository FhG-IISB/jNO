<p align="center">
  <img src="assets/logo.png" alt="jNO logo" width="500"/>
</p>

<p align="center">
    <a href="https://fhg-iisb.github.io/jNO_docs/">
        <img src="https://img.shields.io/badge/docs-GitHub%20Pages-0aa?style=for-the-badge" alt="Docs"/>
    </a>
    <a href="https://fhg-iisb.github.io/jNO_docs/Tutorials/">
        <img src="https://img.shields.io/badge/tutorials-step_by_step-0b8f7a?style=for-the-badge" alt="Tutorials"/>
    </a>
    <a href="https://github.com/FhG-IISB/jno/actions/workflows/python-package.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/FhG-IISB/jno/python-package.yml?branch=main&style=for-the-badge&label=tests" alt="Tests"/>
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-2ea44f?style=for-the-badge" alt="License"/>
    </a>
    <a href="CITATION.cff">
        <img src="https://img.shields.io/badge/cite-CITATION.cff-6b5b95?style=for-the-badge" alt="Citation"/>
    </a>
    <img src="https://img.shields.io/badge/docker-image%20available-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker image available"/>
</p>

Warning: This is a research-level repository. It may contain bugs and is subject to continuous change without notice.

# Install

Quick install from PyPI:

```bash
pip install jax-neural-operators
```

If a Nvidia GPU is avaiable install

```bash
pip instal jax[cuda]
```

For local development (recommended on Linux `aarch64` when `gmsh` wheels are unavailable on PyPI), use `micromamba`:

```bash
micromamba create -n jno python=3.12 pip -y
micromamba activate jno
micromamba install -n jno -c conda-forge gmsh python-gmsh -y
pip install -e .
```



# Minimal DeepONet Neural Operator Example

Create the following file

```python
import jno
import jax
import optax

dir = jno.setup("./runs/test")

# Domain
dom = 500 * jno.domain.rect(mesh_size=0.05, x_range=(0, 2), y_range=(0, 1))
x, y, _ = dom.variable("interior")
xb, yb, _ = dom.variable("boundary")

random_k = jax.random.uniform(jax.random.PRNGKey(0), shape=(500, 1, 1), minval=0.5, maxval=1.5)
k = dom.variable("k", random_k)

# Neural Network
net = jno.nn.deeponet(n_sensors=1, coord_dim=2, basis_functions=32, hidden_dim=128, activation=jax.numpy.tanh)
net.optimizer(optax.adam(learning_rate=optax.schedules.cosine_decay_schedule(init_value=1e-3, decay_steps=20_000, alpha=1e-5)))

# Forward pass and hard enforcement of BCs via output transformation
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.dd(x) + u.dd(y)) + 1.0  # PDE Loss

# Create -> Train -> Save
crux = jno.core(constraints=[pde.mse], domain=dom).print_shapes()
crux.solve(epochs=20_000, batchsize=32).plot(f"{dir}/training.png")
jno.save(crux, f"{dir}/model.pkl")

# Inference via test domain on a finer mesh
tst_dom = 16 * jno.domain.rect(mesh_size=0.01)
tst_dom.variable("k", jax.random.uniform(jax.random.PRNGKey(0), shape=(16, 1, 1), minval=0.1, maxval=1.9))

pred, x, y, k = crux.eval([u, x, y, k], domain=tst_dom)
print(pred.shape, x.shape, y.shape, k.shape)
```

and then run with

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> JNO_SEED=<seed> python <filename>.py
```

### Foundation Models

These models are maintained as separate repositories so they can also be used independently.
If installed, you can access them via:

```python
jno.nn.<model_name>
```

to use a foundation model install one or more of the following repositories (they can also be used as standalone repos without jNO).

<p>
    <a href="https://github.com/FhG-IISB/jax_poseidon">
        <img src="https://img.shields.io/badge/jax_poseidon-1f6feb?style=for-the-badge" alt="jax_poseidon"/>
    </a>
    <a href="https://github.com/FhG-IISB/jax_walrus">
        <img src="https://img.shields.io/badge/jax_walrus-1f6feb?style=for-the-badge" alt="jax_walrus"/>
    </a>
    <a href="https://github.com/FhG-IISB/jax_pdeformer2">
        <img src="https://img.shields.io/badge/jax_pdeformer2-1f6feb?style=for-the-badge" alt="jax_pdeformer2"/>
    </a>
    <a href="https://github.com/FhG-IISB/jax_morph">
        <img src="https://img.shields.io/badge/jax_morph-1f6feb?style=for-the-badge" alt="jax_morph"/>
    </a>
    <a href="https://github.com/FhG-IISB/jax_mpp">
        <img src="https://img.shields.io/badge/jax_mpp-1f6feb?style=for-the-badge" alt="jax_mpp"/>
    </a>
</p>



## Citation

If jNO is used we would appreciate to cite the following paper:

```text
@article{armbruster2026jNO,
  author  = {Armbruster, Leon, ....},
  title   = {{jNO}: A JAX Library for Neural Operator and PDE Foundation Model Training},
  journal = {},
  year    = {},
}
```

