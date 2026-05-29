# Optimizer & LR

## Global

```python
net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

## Parameter Groups

Assign different optimizers or learning rates to different parameter groups via masks:

```python
net.optimizer(optax.adamw, lr=lrs(1e-3))                      # global fallback
net.mask(decoder_mask).optimizer(optax.adam, lr=lrs(5e-4))
net.mask(encoder_mask).optimizer(optax.sgd,  lr=lrs(1e-4))
```

`mask(...)` is consumed by the next mutator call. A bare global `optimizer(...)` clears all previously configured groups.

## LR-Only Updates

```python
net.mask(decoder_mask).lr(lrs(1e-5))   # group-specific LR
net.lr(lrs(1e-5))                      # global LR
```

During `solve()`, jNO logs group coverage, overlap, and uncovered-parameter diagnostics.
