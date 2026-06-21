# Mask & Freeze

## Mask

`mask(...)` takes a boolean pytree mask only. There is no `target="..."` argument.

```python
import equinox as eqx
import jax

all_true = jax.tree_util.tree_map(lambda _: True, eqx.filter(net.module, eqx.is_array))
net.mask(all_true).optimizer(optax.adam)
net.mask(all_true).scale(lrs(1e-4))
```

### Regex-style targeting

To target layers by name, build a boolean mask from a regex:

```python
import re
import equinox as eqx
import jax

def regex_mask(module, pattern: str):
    arrays = eqx.filter(module, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(arrays)

    def part(k):
        if hasattr(k, "name"): return str(k.name)
        if hasattr(k, "idx"):  return str(k.idx)
        if hasattr(k, "key"):  return str(k.key)
        return str(k)

    leaves = []
    for path, _ in flat:
        path_str = "/".join(part(p) for p in path)
        leaves.append(bool(re.search(pattern, path_str)))

    return jax.tree_util.tree_unflatten(treedef, leaves)

decoder_mask = regex_mask(net.module, r"decoder")
net.mask(decoder_mask).optimizer(optax.adam)
net.mask(decoder_mask).scale(lrs(3e-4))
```

## Freeze / Unfreeze

```python
net.freeze()                      # freeze entire model
net.mask(decoder_mask).freeze()   # freeze only selected leaves
```

With `mask(...).freeze()`, non-selected leaves remain trainable.
