# Diagnostics

## Logging

At `solve()` time jNO logs:

- parameter-group summary
- overlap/uncovered diagnostics for groups
- zero-match warnings for empty masks/groups
- detailed path samples in the log file (`quiet` logs)

This makes complex model-control chains auditable.

## Paramax Integration

jNO automatically unwraps Paramax wrappers before each forward evaluation (when `paramax` is installed):

```python
import paramax
import jax.numpy as jnp

scale = paramax.Parameterize(jnp.exp, jnp.log(jnp.ones(3)))
print(paramax.unwrap(("abc", 1, scale)))
# ('abc', 1, Array([1., 1., 1.], dtype=float32))
```

If `paramax` is not installed, no unwrapping is attempted.
