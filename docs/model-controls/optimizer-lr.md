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
| `engd()` | Energy Natural Gradient Descent — preconditions with the inverse energy Gram matrix G⁻¹; auto-detects `gram_terms`; compatible with `line_search=True`; `momentum=` / `damping=` give SPRING (Gauss-Newton with momentum, Levenberg-Marquardt damping) | Müller & Zeinhofer, *Achieving High Accuracy with PINNs via Energy Natural Gradient Descent*, ICML 2023, arXiv:2302.13163; SPRING: Goldshlager, Abrahamsen & Lin 2024, arXiv:2401.10190 |
| `ssbroyden()` / `ssbfgs()` | Self-Scaled Broyden / BFGS quasi-Newton with a zoom line search — strong on smooth PINN / inverse losses | Urbán, Stefanou & Pons, *Unveiling the optimization process of PINNs*, J. Comput. Phys. **523** (2025) 113656 |
| `soap()` / `scale_by_soap()` | SOAP — Shampoo with Adam in the preconditioner's eigenbasis | Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* (2024), arXiv:2409.11321 |
| `lm()` | Levenberg-Marquardt on the PINN residuals; the damping adapts to the gain ratio and rejected steps are retried within the epoch, so the loss never rises; dense (smaller of `P x P`, `m x m`) | Levenberg (1944); Marquardt (1963); Madsen, Nielsen & Tingleff, *Methods for Non-Linear Least Squares Problems* (2004), Sec. 3.2 |
| `dngd()` | Dual Natural Gradient Descent — the Gauss-Newton step solved over the `m` residual rows, with geodesic acceleration; `solver="nystrom_pcg"` is matrix-free | Jnini & Vella, *Dual Natural Gradient Descent for Scalable Training of PINNs* (2025), arXiv:2505.21404 |
| `nystrom_ngd()` | Matrix-free natural gradient: Nyström-preconditioned CG on `JᵀJ + μI`, adaptive sketch rank, Armijo steps | Bioli, Marcati & Sangalli, *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*, CMAME (2026), doi:10.1016/j.cma.2026.119456 |
| `dsgnar()` | Doubly-sketched Gauss-Newton; the step length comes from probing for a target gain ratio. **Not yet validated** at the published scale (see its docstring) | Webb, Jerad & Cartis, *An Optimisation Framework for the Well-Conditioned Training of PINNs* (2026), arXiv:2607.02194 |
| `nncg()` | NysNewton-CG — damped Newton by Nyström-preconditioned CG on Hessian-vector products, Armijo steps; a fine-tuning stage after Adam / L-BFGS | Rathore, Lei, Frangella, Lu & Udell, *Challenges in Training PINNs: A Loss Landscape Perspective*, ICML 2024, arXiv:2402.01868 |
| `score()` | SCORE — self-scaled Broyden with a decrement-based step `1/(1+λ)` and a secant pair shifted by the decrement | Si, An, Ma & Yan, *From Non-Convex Self-Concordant Regularization to Scalable Quasi-Newton Training of PINNs* (2026), arXiv:2608.04206 |
| `ss_esoap()` / `scale_by_ss_esoap()` | SS-ESOAP — SOAP with a secant-energy step scale, eigenbasis refreshes when the factors drift, and a downscaled variance state | Wang, Toftrup, Loeschcke, Wang & Anandkumar, *SS-ESOAP: Self-Scaled Adaptive Preconditioning for Physics-Informed Learning* (2026), arXiv:2608.29448 |
| `md()` / `md_decouple()` | Magnitude–Direction Decoupling — a **generic wrapper** that factorizes each weight matrix `W = diag(γ_row) Ŵ diag(γ_col)` into a fixed-norm direction `Ŵ` + learnable per-row/column gains, stepping any optax base optimizer on the direction; removes weight decay / warmup and transfers the LR across width | Hägele, Hernández-Cano, Kosson & Jaggi, *Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors* (2026), arXiv:2606.25971, Algorithm 2 |

### Gauss-Newton optimizers for PINNs (`engd`, `lm`, `dngd`, `nystrom_ngd`, `dsgnar`)

These are **sentinels**, not optax transforms: their step needs the per-point residual of every loss
term, which an optax `update()` never sees. `jno.core.solve()` builds one `(residual.grad(net), 1.0)`
pair per loss term involving the model (or uses `gram_terms=`), runs the step in a callback and
applies it through `optax.sgd(1.0)`. Every one minimises `½‖r̂‖²` with `r̂_k = √(w_k/N_k) r_k`, the
weighted sum of the `.mse` terms, needs float64 for its last digits and `inner_steps=1`.

```python
jax.config.update("jax_enable_x64", True)
net.optimizer(jno.optimizers.lm())          # P up to ~10^4: dense Levenberg-Marquardt
net.optimizer(jno.optimizers.dngd())        # more parameters than residual rows: m x m solve
net.optimizer(jno.optimizers.nystrom_ngd()) # large networks: matrix-free
jno.core([pde.mse, bc.mse]).solve(500)
```

Which to pick: `engd` / `lm` when the parameter count is in the thousands; `dngd` when the network
has more parameters than residual rows (dense up to `m` of a few thousand, `solver="nystrom_pcg"`
beyond); `nystrom_ngd` when neither `J` nor a Gram matrix fits in memory. On the two-dimensional
Poisson problem of the test suite (1020 residual rows) `lm`, `dngd`, `nystrom_ngd` and `engd` reach
rel-L2 below `1e-3` in 200 steps; the optax transforms `nncg`, `score` and `ss_esoap` need the loss
only (`nncg` and `score` read it through `value_fn`, which `jno.core` passes).

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
(`ssbroyden`/`ssbfgs`/`nncg`/`score`); and driving η_W from the model's `.scale()` schedule.

**Add your own:** drop an optax-compatible `GradientTransformation` into a new module under
`jno/optimizers/` and re-export it from `jno/optimizers/__init__.py` — it then works everywhere an
optax optimizer does.
