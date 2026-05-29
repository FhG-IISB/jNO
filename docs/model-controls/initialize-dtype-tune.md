# Initialize, Dtype & Tune

## Initialize

`initialize(...)` supports checkpoint paths, pytrees, and callable initializers:

```python
import jax

net.initialize("./weights.eqx")
net.initialize("./runs/checkpoints/2000::1")   # Orbax checkpoint, optional key suffix
net.initialize(other_model.module)
net.initialize(jax.nn.initializers.xavier_uniform(), key=jax.random.PRNGKey(0))
```

`mask(...)` does not provide targeted/partial initialize scoping.

## Dtype

```python
import jax.numpy as jnp

net.dtype(jnp.bfloat16)
```

Casts floating-point parameters before training.

## Tune

`tune(...)` sweeps over combinations of model-control settings:

```python
net.tune(
    freeze=[True, False],
    lora=[(4, 1.0), None],
    optimizer=[optax.adam],
    lr=[lrs(1e-3), lrs(1e-4)],
    dtype=[jnp.float32],
)
```

## Reset

```python
net.reset()
```

Clears all training-time controls: `freeze`, `lora`, `optimizer`, `lr`, `dtype`, `mask`, and init state.
