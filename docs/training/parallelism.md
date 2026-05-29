# Multi-Device Parallelism

jNO supports data parallelism, model parallelism, and hybrid parallelism via JAX's device mesh.

---

## Device Mesh

```python
# No parallelism (single device, default)
crux = jno.core(constraints, domain, mesh=(1, 1))

# Pure data parallelism: split batches across 4 GPUs
crux = jno.core(constraints, domain, mesh=(4, 1))

# Pure model parallelism: shard model weights across 2 GPUs
crux = jno.core(constraints, domain, mesh=(1, 2))

# Hybrid (2 data × 2 model = 4 GPUs total)
crux = jno.core(constraints, domain, mesh=(2, 2))

# Auto-scale to all available devices
n = len(jax.devices())
crux = jno.core(constraints, domain, mesh=(n, 1))
```

---

## Mesh Shape Rules

- `batch × model` must equal the total number of available devices.
- Data parallelism (`(n, 1)`) maximises throughput when the model fits on a single device.
- Model parallelism (`(1, n)`) allows training models too large for a single device.
