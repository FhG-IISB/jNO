# FieldView — physics-informed FNO operator

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/11_operator_learning/field_view_physics.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/11-operator-learning/">Back to chapter</a>
</div>

A Fourier Neural Operator learns the solution operator $f \mapsto u$ of the 2-D
Poisson problem

```text
−Δu = f      on Ω = (0,1)²,      u = 0 on ∂Ω
```

mapping a whole forcing field to the whole solution field in one shot. The
operator emits a **grid**, so the coordinates `x`, `y` are *not* inputs to the
network — coordinate-based autodiff of the output is identically zero. PDE
derivatives therefore have to come from **finite differences on the grid**, and
that is exactly what `FieldView` provides:

```python
u = net(f).field.bind(x=x, y=y)   # FD derivatives of the operator's grid output
pde = u.xx + u.yy + f             # −Δu = f  ⇒  u_xx + u_yy + f = 0
```

Because the FD residual is fully differentiable, adding it to the loss makes
`crux.solve` back-propagate through it into the FNO weights — this is real
physics-informed *operator* training, not a forward-only check.

## Why FieldView here

A data-driven operator can fit the solution closely yet still **violate the
PDE** — its grid output has small ripples whose Laplacian is large. `FieldView`
lets you measure (and penalise) that violation directly on the operator's
output:

| Training signal | rel. L2 vs analytic | FD PDE residual |
|---|---|---|
| Data only | ~1–2 % | large (operator ignores physics) |
| **Data + FieldView FD residual** | ~6 % | **~7× smaller** |

The supervised term keeps the operator accurate; the FieldView term pulls its
output back onto the physics. (Numbers above are from a small CPU run; raise
`GRID` / `SAMPLES` / `EPOCHS` for higher fidelity.)

## Walkthrough

### Dataset and operator

```python
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")          # operator input  (forcing)
_u = domain.variable("_u")          # supervised target
x, y, _ = domain.variable("interior")

net = jno.nn.wrap(foundax.fno2d(in_features=1, d_vars=1, d_model=(GRID, GRID), ...))
```

The dataset helpers are shared with `fno_poisson_2d` / `deeponet_poisson_2d` —
the same operator, learned three ways (data, PDE-residual-by-AD, and the
data + FD-residual hybrid shown here).

### Data term + FieldView physics term

```python
pred = net(_f)                          # live operator output (a grid field)
u = pred.field.bind(x=x, y=y)           # FD view: x, y are grid axes, not NN inputs
data = pred - _u                        # supervised term
physics = PHYS_W * (u.xx + u.yy + _f)   # FD Poisson residual (gradient-carrying)

crux = jno.core([data.mse, physics.mse], domain=domain)
crux.solve(epochs=EPOCHS, batchsize=BATCH)
```

!!! note "`.field.bind()` vs `.scalar.bind()`"
    - `.scalar.bind()` → **AD** derivatives through the network graph. Use it
      when coordinates are network inputs (point-wise PINNs / DeepONet —
      see `deeponet_poisson_2d`).
    - `.field.bind()` → **FD** derivatives on a grid output. Use it when the
      operator emits the whole field at once (FNO, U-Net, Poseidon), where
      coordinate-AD would be zero.

### Score the trained operator

```python
p, e = crux.eval([pred, _u])            # prediction vs analytic solution
res_mse = float(np.mean(np.asarray(crux.eval((u.xx + u.yy + _f).mse))))
```

`crux.eval` runs the same compiled graph used in training, so the FD residual is
evaluated on the operator's own prediction — a physics audit of the learned map.

## What to notice

- **FD, not AD.** The operator's coordinates are grid axes, not function
  inputs, so `u.xx`/`u.yy` come from a structured-grid finite-difference stencil
  — the only option for a field-to-field operator.
- **Gradient-carrying.** `net(_f).field.bind(...)` keeps the FD residual
  differentiable, so it can sit *in the loss*, not just in a report.
- **Data anchors accuracy; physics anchors consistency.** A small weight on the
  FieldView term (`PHYS_W`) is enough to cut the PDE residual several-fold while
  keeping the supervised fit.
- **Boundary slices.** `FieldView` also exposes `u.left` / `u.right` / `u.top` /
  `u.bottom` for boundary terms (here the data term carries the Dirichlet
  condition; note the dataset is cell-centred, so the slices sit half a cell
  from the wall).

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/11_operator_learning/field_view_physics.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/11-operator-learning/">Back to 11 Operator Learning</a>
</div>

## Script

```python
--8<-- "tutorial_examples/11_operator_learning/field_view_physics.py"
```
