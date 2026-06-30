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
| `ssbroyden()` / `ssbfgs()` | Self-Scaled Broyden / BFGS quasi-Newton with a zoom line search — strong on smooth PINN / inverse losses | Urbán, Stefanou & Pons, *Unveiling the optimization process of PINNs*, J. Comput. Phys. **523** (2025) 113656 |
| `soap()` / `scale_by_soap()` | SOAP — Shampoo with Adam in the preconditioner's eigenbasis | Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* (2024), arXiv:2409.11321 |
| `md()` / `md_decouple()` | Magnitude–Direction Decoupling — a **generic wrapper** that factorizes each weight matrix `W = diag(γ_row) Ŵ diag(γ_col)` into a fixed-norm direction `Ŵ` + learnable per-row/column gains, stepping any optax base optimizer on the direction; removes weight decay / warmup and transfers the LR across width | Hägele, Hernández-Cano, Kosson & Jaggi, *Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors* (2026), arXiv:2606.25971, Algorithm 2 |

### Magnitude–Direction Decoupling (`md`)

`md` wraps **any** optax base optimizer — the base steps the on-sphere direction `Ŵ` and carries the
direction learning rate η_W; a separate Adam steps the magnitude gains at `gain_lr` (η_γ):

```python
import optax, jno
net.optimizer(jno.optimizers.md(optax.adam(1e-3), gain_lr=1e-3))   # recommended (sentinel form)
```

Prefer the `md(...)` **sentinel** form shown above: because the per-step update is the nonlinear
reassembly `W_new − W`, it must not be re-scaled by a learning rate, so jNO automatically forces the
model's outer LR-scale to `1.0` — you do **not** add `.scale(...)`. The bare `md_decouple(base, ...)`
`GradientTransformation` is exported for plain optax loops / power users; there you must keep the
surrounding learning rate at `1.0` yourself.

Only 2-D weight matrices are decoupled; biases, 1-D norm gains and scalars pass straight through the
base optimizer. `gain_axis` selects the gains: `("row", "col")` (default), `("row",)` or `("col",)`.

**Not yet implemented (v1):** `gain_axis="scalar"` (single tied scalar gain, Algorithm 1); conv / >2-D
kernels (they pass through undecoupled); embedding / LM-head per-row unit-norm; the Muon RMS-matching
update-scale factor; gain maps other than softplus; line-search base optimizers
(`ssbroyden`/`ssbfgs`); and driving η_W from the model's `.scale()` schedule.

**Add your own:** drop an optax-compatible `GradientTransformation` into a new module under
`jno/optimizers/` and re-export it from `jno/optimizers/__init__.py` — it then works everywhere an
optax optimizer does.
