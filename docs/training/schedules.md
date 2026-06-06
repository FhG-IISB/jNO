# Schedules

---

## Learning Rate Schedules

`LearningRateSchedule` wraps **any** callable `(epoch, individual_losses) → scalar` so it can be passed to `optimizer(..., lr=...)`. Build your own:

```python
from jno import LearningRateSchedule as lrs

# Any (epoch, losses) -> scalar callable is a schedule
lrs(lambda epoch, losses: 1e-4 * (0.9 ** (epoch / 500)))

# Drop the LR when the PDE residual loss plateaus
lrs(lambda epoch, losses: 1e-3 if losses[0] > 1e-2 else 1e-5)
```

### Built-in schedule factories

For common shapes, `lrs` ships these factories:

```python
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
```

All factories — built-in and custom — accept `min_lr` and `max_lr` keyword arguments to clamp the output.

---

## Weight Schedules

`WeightSchedule` scales individual constraint losses:  
`total_loss = Σ wᵢ · constraintᵢ`

```python
from jno import WeightSchedule as ws

# Fixed weights (one per constraint)
ws([1.0, 10.0, 1.0])

# Adaptive weights as a function of (epoch, individual_losses)
ws(lambda e, L: [1.0, 10.0 * L[1], 1.0])  # amplify boundary loss when large

# Using losses from previous step to avoid gradient-through-loss issues
ws(lambda e, L: [1.0, jnp.maximum(1.0, L[1] / (L[0] + 1e-8))])
```

Pass to `solve()`:

```python
crux.solve(5000, constraint_weights=ws([1.0, 10.0]))
```
