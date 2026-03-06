# Parallelism

jNO supports multi-device training through JAX's device mesh abstraction, enabling both data parallelism and model parallelism across GPUs and TPUs.

## Device Mesh Configuration

The `mesh` parameter in `jno.core` controls how computation is distributed:

```python
crux = jno.core(
    constraints=[pde, bc],
    domain=domain,
    mesh=(batch_devices, model_devices),
)
```

| Dimension | Name | Purpose |
|-----------|------|---------|
| First | `batch` | Data parallelism — data is split across devices |
| Second | `model` | Model parallelism — parameters are sharded across devices |

The product `batch × model` must equal the total number of available devices.

---

## Common Configurations

### Single Device (Default)

```python
crux = jno.core([pde], domain, mesh=(1, 1))
```

When `mesh=(1, 1)` is specified, jNO automatically expands to `(n_devices, 1)` for pure data parallelism if multiple devices are available.

### Pure Data Parallelism

Each device processes different data samples; parameters are replicated on all devices.

```python
# 4 GPUs — 4× throughput
crux = jno.core([pde], domain, mesh=(4, 1))
```

**Use when:** The model fits on a single GPU and you want maximum training throughput.

### Pure Model Parallelism

Model parameters are sharded across devices; each device holds a portion of the model.

```python
# 2 GPUs — fit 2× larger models
crux = jno.core([pde], domain, mesh=(1, 2))
```

**Use when:** The model is too large to fit on a single GPU.

### Hybrid Parallelism

Combine data and model parallelism:

```python
# 8 GPUs — 4× data parallelism, 2× model parallelism
crux = jno.core([pde], domain, mesh=(4, 2))
```

**Use when:** You have a large model AND large datasets.

---

## Configuration Guide

| GPUs | mesh | Strategy | Best For |
|------|------|----------|----------|
| 1 | `(1, 1)` | No parallelism | Development and small problems |
| 2 | `(2, 1)` | Data parallel | 2× throughput |
| 2 | `(1, 2)` | Model parallel | 2× model capacity |
| 4 | `(4, 1)` | Data parallel | 4× throughput |
| 4 | `(2, 2)` | Hybrid | Balance throughput and capacity |
| 8 | `(8, 1)` | Data parallel | Maximum throughput |
| 8 | `(4, 2)` | Hybrid | Large model + large data |

---

## How It Works

### Data Sharding

Training data (collocation points, tensor tags) is sharded along the batch dimension:

```
Device 0: points[0:N/4], tensor_tags[0:B/4]
Device 1: points[N/4:N/2], tensor_tags[B/4:B/2]
...
```

### Parameter Sharding

With model parallelism (`model > 1`), weight matrices are sharded along their last dimension:

```
Device 0: kernel[:, 0:hidden/2]
Device 1: kernel[:, hidden/2:hidden]
```

Scalar and 1D parameters (biases) are replicated across all devices.

### Automatic Tiling

When the batch size of training data is smaller than the number of data-parallel devices, jNO automatically tiles the data:

```python
# 4 GPUs but only 2 training samples
domain = 2 * jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
crux = jno.core([pde], domain, mesh=(4, 1))
# Data is automatically tiled from (2, N, D) to (4, N, D)
```

---

## Checking Device Setup

```python
import jax

# Check available devices
print(jax.devices())          # List all devices
print(len(jax.devices()))     # Number of devices

# After creating core
crux = jno.core([pde], domain, mesh=(4, 1))
# Log output: "Using 4 device(s): [GpuDevice(id=0), ...]"
# Log output: "Device mesh: Mesh(...) (shape: (4, 1))"
```

---

## Recommendations

1. **Start with data parallelism** — `(n_devices, 1)` is the simplest and most efficient for models that fit on a single GPU
2. **Use model parallelism only when needed** — when OOM errors occur on a single device
3. **Hybrid parallelism** — for the largest workloads where both data throughput and model capacity are bottlenecks
4. **Mesh shape mismatch** — if `batch × model ≠ n_devices`, jNO warns and defaults to `(n_devices, 1)`

---

## See Also

- [Training and Solving](Training-and-Solving.md) — the `core` constructor and training loop
- [Operator Learning](Operator-Learning.md) — batched training across multiple input functions
