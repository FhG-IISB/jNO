# IREE Deployment

After training, a jNO model can be compiled to an **IREE artifact** — a self-contained binary that runs inference without JAX, NumPy, or any Python ML dependency. The compiled model accepts plain NumPy arrays and returns NumPy arrays.

IREE (Intermediate Representation Execution Environment) supports CPU, CUDA, ROCm, Vulkan, and Metal backends through a single compilation step.

---

## Compiling a trained model

Call `.to_iree()` on any wrapped model after training. The current weights are baked into the artifact as constants.

```python
iree_model = net.to_iree(
    sample_inputs=(jnp.ones((100, 2)),),   # tuple of example inputs matching __call__
)
```

The result is an `IREEModel` that can be called immediately:

```python
import numpy as np

x = np.random.rand(100, 2).astype(np.float32)
output = iree_model(x)   # returns np.ndarray, no JAX required
```

---

## Saving and loading

`IREEModel` is fully serialisable with `jno.save` / `jno.load`:

```python
jno.save(iree_model, "deployed_model.pkl")

# Later — no JAX, no foundax needed
loaded = jno.load("deployed_model.pkl")
output = loaded(x)
```

---

## Target backends

```python
# CPU (default)
iree_model = net.to_iree(sample_inputs, target_backend="llvm-cpu")

# NVIDIA GPU
iree_model = net.to_iree(sample_inputs, target_backend="cuda")

# AMD GPU
iree_model = net.to_iree(sample_inputs, target_backend="rocm")

# Vulkan (cross-platform GPU)
iree_model = net.to_iree(sample_inputs, target_backend="vulkan")
```

The `iree-compile` binary must be available on `PATH`. Install it with:

```bash
pip install iree-compiler iree-runtime
```

---

## Optimization level

```python
iree_model = net.to_iree(
    sample_inputs,
    optimization_level=3,   # 0 = none, 1 = basic, 2 = moderate, 3 = full (default)
)
```

Lower levels compile faster and are useful during development; level 3 gives peak inference throughput.

---

## Compiling a raw JAX function

For cases where you want to compile something other than a wrapped model, use `IREEModel.compile` directly:

```python
from jno.utils.iree import IREEModel
import jax.numpy as jnp

def postprocess(u, v):
    return jnp.sqrt(u**2 + v**2)

compiled = IREEModel.compile(
    postprocess,
    sample_inputs=(jnp.ones((100,)), jnp.ones((100,))),
)

output = compiled(u_np, v_np)
```

---

## Full workflow example

```python
import jno, foundax, jax, optax
import jax.numpy as jnp
import numpy as np

# --- train ---
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y, _ = domain.variable("interior")

net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=64, num_layers=4,
                               key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(1e-3))

u = net(x, y) * x * (1 - x) * y * (1 - y)
pde = jno.np.laplacian(u, [x, y]) + 1.0

crux = jno.core([pde.mse], domain)
crux.solve(5000)

# --- deploy ---
iree_model = net.to_iree(
    sample_inputs=(jnp.ones((1, 2)),),
    target_backend="llvm-cpu",
)
jno.save(iree_model, "poisson_net.pkl")

# --- inference (no JAX needed) ---
model = jno.load("poisson_net.pkl")
pts = np.random.rand(500, 2).astype(np.float32)
predictions = model(pts)
```
