# Optimizer & LR

## Global

```python
net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.9, 2000, 1e-5))
```

## Parameter Groups

Assign different optimizers or learning rates to different parameter groups via masks:

```python
net.optimizer(optax.adamw).scale(lrs(1e-3))                      # global fallback
net.mask(decoder_mask).optimizer(optax.adam)
net.mask(decoder_mask).scale(lrs(5e-4))
net.mask(encoder_mask).optimizer(optax.sgd)
net.mask(encoder_mask).scale(lrs(1e-4))
```

`mask(...)` is consumed by the next mutator call. A bare global `optimizer(...)` clears all previously configured groups.

## LR-Only Updates

```python
net.mask(decoder_mask).scale(lrs(1e-5))   # group-specific LR
net.scale(lrs(1e-5))                      # global LR
```

During `solve()`, jNO logs group coverage, overlap, and uncovered-parameter diagnostics.

## `jno.optimizers` — custom second-order optimizers

`jno.optimizers` holds **only** custom optimizers that aren't in optax; for `chain`, learning-rate
schedules and gradient clipping use `optax` directly. Each is an optax `GradientTransformation`, so it
composes with `optax.chain` and drops straight into `.optimizer(...)`.

```python
import optax, jno
k.optimizer(jno.optimizers.ssbroyden())                  # 2nd-order; far fewer steps than Adam on inverse/PINN losses
k.optimizer(jno.optimizers.soap(learning_rate=3e-3))
k.optimizer(optax.chain(                                 # compose with optax directly
    optax.clip_by_global_norm(1.0),
    jno.optimizers.ssbfgs(),
))
```

| Optimizer | What it is | Reference |
|---|---|---|
| `engd()` | Energy Natural Gradient Descent — preconditions with the inverse energy Gram matrix G⁻¹; auto-detects `gram_terms`; compatible with `line_search=True` | Müller & Zeinhofer, *Achieving High Accuracy with PINNs via Energy Natural Gradient Descent*, ICML 2023, arXiv:2302.13163 |
| `ssbroyden()` / `ssbfgs()` | Self-Scaled Broyden / BFGS quasi-Newton with a zoom line search — strong on smooth PINN / inverse losses | Urbán, Stefanou & Pons, *Unveiling the optimization process of PINNs*, J. Comput. Phys. **523** (2025) 113656 |
| `soap()` / `scale_by_soap()` | SOAP — Shampoo with Adam in the preconditioner's eigenbasis | Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* (2024), arXiv:2409.11321 |

**Add your own:** drop an optax-compatible `GradientTransformation` into a new module under
`jno/optimizers/` and re-export it from `jno/optimizers/__init__.py` — it then works everywhere an
optax optimizer does.
