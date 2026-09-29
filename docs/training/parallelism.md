# Multi-Device Parallelism

jNO supports data parallelism, model parallelism, and hybrid parallelism via JAX's device mesh.

---

## Device Mesh

```python
# Default: every visible device, data-parallel -- (1, 1) is expanded to (len(jax.devices()), 1)
crux = jno.core(constraints)

# Pure data parallelism: split batches across 4 GPUs
crux = jno.core(constraints,  mesh=(4, 1))

# Pure model parallelism: shard model weights across 2 GPUs
crux = jno.core(constraints,  mesh=(1, 2))

# Hybrid (2 data × 2 model = 4 GPUs total)
crux = jno.core(constraints,  mesh=(2, 2))
```

`jno.core` has no argument that opts out of the devices JAX can see. To train on one device, make only
that one visible before JAX starts (e.g. `CUDA_VISIBLE_DEVICES=0`, or `JAX_PLATFORMS=cpu`).

---

## Mesh Shape Rules

- `batch × model` must equal the number of devices `jax.devices()` returns; any other shape logs a
  warning and falls back to `(n_devices, 1)`.
- Data parallelism (`(n, 1)`) maximises throughput when the model fits on a single device.
- Model parallelism (`(1, n)`) allows training models too large for a single device.

## How the data is split

Each training array is placed along the mesh's `batch` axis by the axis that holds the work:

- **samples**, when there are at least as many as devices and they divide evenly (operator learning);
- otherwise the **points** of a single sample, when they divide evenly (a PINN) — the loss's mean
  becomes one all-reduce;
- otherwise the array is **replicated** on every device, and the log names the array and why.
