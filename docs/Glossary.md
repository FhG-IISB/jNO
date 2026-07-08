# Concepts

Short definitions for jNO-specific terminology that appears repeatedly
across the docs. Linked from the [Home page](index.md).

---

### Trace / tracing system

jNO's central abstraction — a single symbolic graph holding domain
points, network calls, derivatives, residuals, supervised losses, FEM
weak forms, and noise terms. `jno.core(...)` JIT-compiles the graph
once into a JAX function reused for both `crux.solve()` (training) and
`crux.eval()` (evaluation), which is why the same expression can serve
as both a residual loss and a quantity of interest. Design details:
[`arXiv:2605.10159`](https://arxiv.org/abs/2605.10159).

### Placeholder

The base class of every symbolic node in the trace. Subclasses include
`Variable` (a coordinate or tensor input), `FunctionCall` (a wrapped
Python function such as `jno.np.sin(x)`), `BinaryOp` (`x + y`, `u * v`),
`ModelCall` (a neural-network forward pass), `Integral` / `IntegralTime`
(`.integrate()` and `.integrate(t)`), `Noise`, and so on.

You rarely instantiate `Placeholder` directly — it is what the
expression-building API produces.

### Constraint

A scalar expression handed to `jno.core`. Canonical form is
`<expression>.mse` (also `.mae`, `.rmse`, …); any reduction to a scalar
qualifies.

```python
pde      = (-u.laplacian(x, y) - forcing).mse
bc       = (u_bnd - 0.0).mse
data_fit = (u_pred - u_obs).mse
crux = jno.core([pde, bc, data_fit])
```

Covers both *physics* (PDE residuals) and *data* (supervised losses).
*Not* used for parameter bounds — those are configured via
`Model.constrain(...)` and called *parameter constraints* for
disambiguation.

### Model controls

The collection of per-parameter knobs set on a network wrapped via
`jno.nn(...)` — `optimizer`, `lr`, `freeze` / `unfreeze`, `mask`,
`lora`, `dtype`, `constrain`, `initialize`. Chain `.mask(M)` before any
of these to scope the next call to a parameter subset. Full reference
and worked examples in [Model Controls](model-controls/index.md).

### Mesh (overloaded)

Two distinct concepts share the word "mesh":

- **Spatial mesh** — the unstructured triangular / tetrahedral / line
  mesh defining the simulation domain (`jno.domain.rect(mesh_size=...)`,
  loaded from `.msh` / `.npz`, or built from a polygon). Collocation
  and integration points come from this mesh.
- **Device mesh** — the JAX `(batch, model)` device topology passed via
  `jno.core(..., mesh=(n_batch, n_model))`. Controls data and model
  parallelism. Unrelated to the spatial mesh.

When ambiguous, the docs say *spatial mesh* vs *device mesh*.

### Variable vs TensorTag

- **`Variable`** — a coordinate-like quantity that lives on a domain
  mesh tag (`domain.variable("interior")` → `x, y, t`). Has a `dim`
  slice into the tag's array.
- **`TensorTag`** — a non-coordinate quantity stored on the domain
  context (e.g., a per-sample diffusion field). Attached and
  referenced through the same call: `domain.variable(name, array)`.

### Tag (domain tag)

A string label on the domain that maps to a point set or tensor
(`"interior"`, `"boundary"`, `"left"`, `"k"`). Tags appear in three
places:

- Mesh pool — points sampled from a region of the spatial mesh.
- Context — the runtime dictionary the compiler reads during evaluation.
- Normals — `domain.normals_by_tag[tag]` holds outward unit normals for
  boundary tags.

### Crux

The object returned by `jno.core(...)`. Holds the compiled step
function, optimiser state, and training history. The variable name
`crux` is a docs convention, not a class — the actual class is
`jno.core.core`.

### Integrate (spatial vs temporal)

- `expr.integrate()` — spatial integral over the auto-detected region
  (boundary vs volume). Reduces to a scalar.
- `expr.integrate(t)` — temporal integral via the trapezoidal rule over
  the time window visible in the current step. Requires
  `min_consecutive >= 2` (or `None`) in `core.solve()`.
- `expr.integrate(x)` (with a spatial Variable) — vectorised integral
  for Fredholm-type kernels.

### `min_consecutive`

A `core.solve()` and `core.eval()` keyword controlling how many
*consecutive* time steps each constraint sees in one forward pass. The
default is `1` (no temporal context). For time-dependent problems with
`.integrate(t)` or temporal residuals, pass `min_consecutive=None`
(full time axis) or `>=2` (windowed). The library logs a hint when it
detects a time-dependent domain still at the default of `1`.

### Adaptive resampling

A family of strategies (`RAD`, `RARD`, `CR3` causal) that redistribute
collocation points toward regions of high residual *during* training.
Attached via `domain.variable("interior",
resampling_strategy=jno.RAD(...))`. See
[Adaptive Resampling](adaptive/resampling.md).

### Constraint weighting / loss balancing

Per-constraint scalar weights applied before summing losses. Static
weights are a list of floats passed to `core(weights=...)`; adaptive
balancers live under `jno.fn.adaptive.*` (ReLoBRaLo, SoftAdapt, etc.).

### Foundation model

A pretrained neural operator whose weights are stored in
[foundax](https://github.com/FhG-IISB/foundax). Examples: PDEformer-2,
generic DeepONet/FNO templates. Fine-tuned inside a jNO trace via
`jno.nn(pretrained_module)` and the standard `model.optimizer` /
`model.lora` controls.
