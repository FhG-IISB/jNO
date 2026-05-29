# Schedules

---

## Learning Rate

The simplest way to set a learning rate is to pass it directly to the optax constructor — no `lr=` argument needed:

```python
net.optimizer(optax.adam(1e-3))
net.optimizer(optax.adamw(5e-4, weight_decay=1e-2))
```

optax chains work the same way — wrap any combination of optax transformations:

```python
net.optimizer(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-3),
    )
)
```

For **dynamic schedules**, pass `lr=` separately. When `lr` is supplied the optimizer should be constructed with a placeholder rate of `1` — jNO scales it internally:

```python
from jno import LearningRateSchedule as lrs

net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

---

## Learning Rate Schedules

`LearningRateSchedule` is a callable `(epoch, individual_losses) → scalar`.

```python
from jno import LearningRateSchedule as lrs

# Constant
lrs.constant(1e-3)
lrs(1e-3)          # shorthand

# Exponential decay: lr(t) = max(lr_end, lr0 * decay_rate^(t/decay_steps))
lrs.exponential(lr0=1e-3, decay_rate=0.9, decay_steps=1000, lr_end=1e-5)

# Cosine decay
lrs.cosine(total_steps=5000, lr0=1e-3, lr_end=1e-6)

# Linear warm-up then cosine decay
lrs.warmup_cosine(total_steps=5000, warmup_steps=500, lr0=1e-3, lr_end=1e-6)

# Piecewise constant
lrs.piecewise_constant(
    boundaries=[1000, 3000],
    values=[1e-3, 5e-4, 1e-4],   # len(boundaries) + 1 values
)

# Custom function (receives current epoch and per-constraint losses)
lrs(lambda epoch, losses: 1e-4 * (0.9 ** (epoch / 500)))
```

All schedules accept `min_lr` and `max_lr` keyword arguments to clamp the output.

---

## Adaptive Loss Weights

Loss weights are **traced placeholders** — call an adaptive balancer with your losses before passing them to `jno.core`, multiply the returned weights into the losses, then pass the weighted losses to the solver.

```python
w_pde, w_bc = jno.fn.adaptive.relobralo([pde, bc])

crux = jno.core([w_pde * pde, w_bc * bc], domain)
```

The weights are recomputed inside the compiled JAX function every step — no Python callback overhead.

### Logging weights

Each weight placeholder exposes a `.tracker()` method that logs its value during training without contributing to the loss:

```python
crux = jno.core([w_pde * pde, w_bc * bc, w_pde.tracker(), w_bc.tracker()], domain)
```

### Available balancers

#### `relobralo` — Relative Loss Balancing via Residual Algorithms

Balances losses relative to their initial values, preventing any single loss from dominating as magnitudes change during training.

```python
w0, w1 = jno.fn.adaptive.relobralo(
    [pde, bc],
    alpha=0.99,          # exponential moving average factor
    tau=0.1,             # temperature for softmax normalisation
    expected_rho=0.999,  # target ratio for balancing
    seed=42,
)
```

#### `softadapt` — SoftAdapt

Weights losses by the softmax of their recent rate of change — losses improving fastest are downweighted.

```python
w0, w1 = jno.fn.adaptive.softadapt(
    [pde, bc],
    beta=0.1,            # sharpness of the softmax
)
```

#### `dwa` — Dynamic Weight Average

Weights by the ratio of each loss's current value to its previous-step value, smoothed by a temperature.

```python
w0, w1 = jno.fn.adaptive.dwa(
    [pde, bc],
    temperature=2.0,
)
```

#### `lbpinns_loss_balancing` — LbPINNs

Learnable log-variance weights updated via an internal Adam step each iteration.

```python
w0, w1 = jno.fn.adaptive.lbpinns_loss_balancing(
    [pde, bc],
    init_s=0.0,    # initial log-variance (per loss, or scalar broadcast)
    lr_s=1e-2,     # learning rate for the log-variance parameters
)
```

#### `rlw` — Random Loss Weighting

Draws weights from a Dirichlet distribution each step, providing stochastic regularisation across losses.

```python
w0, w1 = jno.fn.adaptive.rlw(
    [pde, bc],
    alpha=1.0,   # Dirichlet concentration parameter
    seed=42,
)
```

### Loss preprocessing (`mode`)

All balancers accept a `mode` keyword that normalises the raw loss values before computing weights:

| `mode` | Effect |
|--------|--------|
| `"raw"` (default) | Use loss values as-is |
| `"minmax"` | Scale each loss to [0, 1] over its observed range |
| `"l2"` | Normalise by the L2 norm of the loss vector |

```python
w0, w1 = jno.fn.adaptive.relobralo([pde, bc], mode="minmax")
```
