# Glossary

Short definitions for jNO-specific terminology that appears repeatedly
across the docs. Linked from the [Home page](index.md).

---

### Trace / tracing system

jNO's central abstraction. Domain points, network calls, derivatives,
residuals, supervised losses, FEM weak forms, and noise terms are all
represented as symbolic nodes in a single computation graph (a
"trace"). At `jno.core(constraints, domain)`, the trace is JIT-compiled
once into a JAX function and reused for every training and evaluation
step. This is why the same expression can serve as both a residual
loss during training and as a quantity of interest during `crux.eval()`.

The paper [`arXiv:2605.10159`](https://arxiv.org/abs/2605.10159)
describes the design in detail.

### Placeholder

The base class of every symbolic node in the trace. Subclasses include
`Variable` (a coordinate or tensor input), `FunctionCall` (a wrapped
Python function such as `jno.np.sin(x)`), `BinaryOp` (`x + y`, `u * v`),
`ModelCall` (a neural-network forward pass), `Integral` / `IntegralTime`
(`.integrate()` and `.integrate(t)`), `Noise`, and so on.

You rarely instantiate `Placeholder` directly — it is what the
expression-building API produces.

### Constraint

A single optimisable scalar expression handed to `jno.core`. The
canonical form is `<expression>.mse` (or `.mae`, `.rmse`, ...). Anything
that ends in a reduction to a scalar can be a constraint:

```python
pde      = (-u.laplacian(x, y) - forcing).mse
bc       = (u_bnd - 0.0).mse
data_fit = (u_pred - u_obs).mse
crux = jno.core([pde, bc, data_fit], domain)
```

The term **constraint** is used in jNO for both *physics* (PDE residual)
and *data* (supervised loss). It is *not* used for parameter bounds —
those are configured via `Model.constrain(...)` and called *parameter
constraints* in `docs/Model-Controls.md` for disambiguation.

### Model controls

The collection of per-parameter knobs you set on a network wrapped via
`jno.nn.wrap(...)`:

- `model.freeze()` / `model.unfreeze()` — exclude parameters from training.
- `model.lora(rank, alpha)` — replace dense layers with LoRA low-rank
  adapters.
- `model.dtype(jnp.float32)` — cast parameters/inputs.
- `model.optimizer(optax.adam(1e-3))` and `model.lr(schedule)` — attach a
  per-model optimiser and learning-rate schedule.
- `model.mask(bool_pytree).optimizer(...)` — apply the next mutator to
  a specific parameter subset only.

The full reference lives in
[`docs/model-controls/index.md`](model-controls/index.md).

### Mesh (overloaded)

Two distinct concepts share the word "mesh":

- **PDE / spatial mesh** — the unstructured tetrahedral / triangular /
  line mesh that defines the simulation domain. Created via
  `jno.domain.rect(mesh_size=...)`, loaded from `.msh` / `.npz`, or
  built from a polygon outline. The collocation / integration points
  come from this mesh.
- **Device mesh** — the JAX `(batch, model)` device topology passed as
  `jno.core(constraints, domain, mesh=(n_batch, n_model))`. Controls
  data parallelism (batch axis) and model parallelism (model axis).
  Unrelated to the PDE mesh.

Where the context is ambiguous, the docs use *spatial mesh* vs
*device mesh*.

### Variable vs TensorTag

- **`Variable`** — a coordinate-like quantity that lives on a domain
  mesh tag (`domain.variable("interior")` → `x, y, t`). Has a `dim`
  slice into the tag's array.
- **`TensorTag`** — a non-coordinate quantity stored on the domain
  context (e.g., a per-sample diffusion field). Built via
  `domain.add_tensor_tag(name, array)` and referenced as
  `domain.variable(name)`.

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
`jno.nn.wrap(pretrained_module)` and the standard `model.optimizer` /
`model.lora` controls.
