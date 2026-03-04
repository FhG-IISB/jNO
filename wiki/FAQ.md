# FAQ

Frequently asked questions and troubleshooting tips for jNO.

---

## Installation

### Q: I get an error about `libglu1` on Linux

Install the OpenGL utility library:

```bash
sudo apt-get install libglu1
```

### Q: I can't activate the virtual environment on Windows

Run the following in PowerShell to allow script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: How do I enable GPU support?

Install with the CUDA extra:

```bash
uv sync --extra cuda
```

Make sure you have a compatible NVIDIA driver and CUDA toolkit installed.

---

## Training

### Q: Training is slow to start

The first call to `crux.solve()` involves JAX compilation (tracing and XLA compilation). This is a one-time cost per problem configuration. Subsequent calls reuse the compiled code unless the problem structure changes.

**Tips:**
- Avoid calling `solve()` with very few epochs repeatedly — each call triggers recompilation
- Use a single `solve(10_000, ...)` instead of ten `solve(1_000, ...)` calls

### Q: How do I choose the number of epochs?

- **Simple 1D/2D problems:** 5,000–20,000 epochs are usually sufficient
- **Complex coupled systems:** 10,000–100,000+ epochs with multi-phase training
- **Inverse problems:** 50,000–200,000+ epochs may be needed for parameter recovery
- Monitor the loss plot to decide when to stop

### Q: What optimizer should I use?

| Optimizer | Use Case |
|-----------|----------|
| `optax.adam` | General purpose, good default |
| `optax.adamw` | When weight decay is desired |
| `soap(...)` | Second-order optimizer for faster convergence |
| `optax.lbfgs` | Final refinement phase, quasi-Newton |

A common multi-phase strategy:
1. Adam with warmup for initial training
2. SOAP for faster convergence
3. L-BFGS for final polishing

### Q: How do I balance multiple constraints?

Use `WeightSchedule`:

```python
from jno import WeightSchedule as ws

# Fixed weights
crux.solve(epochs, optimizer, lr, ws([1.0, 1.0, 10.0]))

# Adaptive weights based on individual losses
crux.solve(epochs, optimizer, lr,
    ws(lambda e, L: [1.0, 1.0, 10.0 * L[2]]))
```

Higher weights make the solver prioritize those constraints. Start with equal weights and increase weights for poorly converging constraints.

### Q: How do I use float64 precision?

Enable it at the start of your script:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

> **Note:** Float64 is slower on GPUs but can improve numerical stability. It may not be compatible with all optimizers (e.g., SOAP).

---

## Models

### Q: What's the difference between `jnn.nn.mlp()` and `jnn.nn.wrap(MyModule())`?

- `jnn.nn.mlp()` creates a simple fully-connected network with built-in defaults
- `jnn.nn.wrap()` wraps any Flax module for use in jNO — use it for custom architectures

Both return `FlaxModule` wrappers compatible with the training pipeline.

### Q: How do I enforce boundary conditions?

**Hard constraints** (preferred when possible):

```python
# Multiply by functions that vanish at boundaries
u = network(x, y) * x * (1 - x) * y * (1 - y)  # Zero on all edges
```

**Soft constraints:**

```python
xb, yb = domain.variable("boundary")
bc = u(xb, yb) - desired_value
crux = jno.core([pde, bc], domain)
```

### Q: Can I use a pretrained model?

Yes, through Poseidon foundation models or by loading checkpoints:

```python
# Poseidon pretrained models
model = jnn.nn.poseidonT("weights/poseidon_tiny.msgpack")

# Or load a previously saved checkpoint
crux = jno.core.load("checkpoint.pkl")
```

---

## Domains

### Q: How do I create an inference domain with finer resolution?

```python
tst_domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.01),  # Finer mesh
    compute_mesh_connectivity=False,               # Not needed for inference
)
```

### Q: How does domain multiplication work?

```python
domain = 4 * jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
```

This creates 4 "copies" of the domain for batched training. All copies share the same mesh but can have different tensor data (e.g., different PDE coefficients for operator learning).

### Q: Can I import external meshes?

Yes, from Gmsh `.msh` files:

```python
domain = jno.domain('./path/to/mesh.msh')
```

---

## Debugging

### Q: How do I inspect tensor shapes during training?

```python
pde.debug._shape = True   # Print shapes of intermediate tensors
pde.debug._val = True     # Print values
```

> **Warning:** This uses `jax.debug.print` which is extremely slow. Only enable temporarily.

### Q: How do I visualize the computation graph?

```python
crux.visualize_trace(pde).save("trace.dot")
```

Open the `.dot` file at [edotor.net](https://edotor.net/).

### Q: My loss is NaN

Common causes:
- Learning rate too high — try reducing by 10×
- Unstable activation functions — try `tanh` instead of `relu` for PDE problems
- Division by zero in the PDE formulation
- Incompatible float precision settings

---

## Performance

### Q: How do I use multiple GPUs?

```python
crux = jno.core([pde], domain, mesh=(len(jax.devices()), 1))
```

See [Parallelism](Parallelism.md) for detailed configuration.

### Q: How do I reduce memory usage?

- Use smaller `mesh_size` only where needed (coarser for training, finer for inference)
- Use model parallelism: `mesh=(1, n_gpus)`
- Reduce `hidden_channels` or network depth
- Use LoRA for fine-tuning instead of full parameter training

---

## See Also

- [Installation](Installation.md) — setup instructions
- [Training and Solving](Training-and-Solving.md) — training API details
- [Parallelism](Parallelism.md) — multi-GPU configuration
