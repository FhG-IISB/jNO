# Operations

Every quantity in jNO — a coordinate, a network output, a residual, a derivative — is a **traced
expression**, an instance of `Placeholder`. You build a problem by *operating* on these expressions.
This page is the complete menu of what you can do to one, in two families:

- **[Part A](#part-a-operations-on-any-traced-expression)** — operations available on **any** traced
  expression (derivatives, integrals, reductions, units, trackers…).
- **[Part B](#part-b-operations-that-require-trainable-parameters)** — operations that only make sense
  when the expression is backed by **trainable parameters**, i.e. a `Model` returned by `jno.nn(...)`
  (optimizers, LoRA, freezing, Bayesian sampling…).

!!! warning "`.scale(...)` is overloaded — the receiver decides"
    The same method name means two different things depending on what it is called on:

    - On an **expression or a called field** — `x.scale(0.1)`, `net(x).scale(U)` — it declares a
      characteristic **magnitude** for non-dimensionalization (Part A, pairs with `.unit`).
    - On a **model object** — `net.optimizer(optax.adam).scale(lrs.exponential(...))` — it sets the
      **learning-rate schedule** (Part B).

    This is the general rule for the whole page: a handful of controls (`.scale`, `.regularize`,
    `.mask`, `.freeze`, `.lora`, …) require trainable parameters and belong to Part B; everything in
    Part A works on any expression, trainable or not.

---

## Part A — operations on any traced expression

### Differentiation

Every expression carries derivative methods; the differentiation **scheme rides on the call**. The
method forms and the `jno.numpy` (`jnn`) free-function forms are equivalent:

```python
import jno.numpy as jnn

u_x  = u.d(x)                     # ∂u/∂x  (Jacobian)      — same as jnn.grad(u, x)
u_xx = u.d2(x)                    # ∂²u/∂x² (Hessian)      — same as jnn.hessian(u, [x])
u_x  = u.d(x, scheme="finite_difference")                 # the scheme is an argument
lap  = u.laplacian(x, y)          # ∇²u                    — same as jnn.laplacian(u, [x, y])
H    = u.hessian(x, y)            # full Hessian matrix
g    = u.grad(x, y)               # spatial gradient (VectorView)  [∂u/∂x, ∂u/∂y]
```

Aliases: `.diff` = `.d`, `.dd` = `.d2`. Higher-order derivatives chain: `u.d(x).d(x)`.
Vector-calculus helpers live on `jnn`: `jnn.jacobian`, `jnn.divergence`, `jnn.curl_2d`, `jnn.curl_3d`.

**Schemes** (`scheme=` on any derivative). Finite-difference schemes require
`compute_mesh_connectivity=True` on the domain.

| Scheme string | Grad | Lap/Hess | Notes |
|---------------|:----:|:--------:|-------|
| `"automatic_differentiation"` *(default)* | ✅ | ✅ | Exact; any domain |
| `"automatic_differentiation:forward"` / `:reverse` | ✅ | — | `jacfwd` / `jacrev` |
| `"automatic_differentiation:fwd-over-rev"` *(default Hessian)* / `:fwd-over-fwd` / `:rev-over-rev` | — | ✅ | 2nd-order AD variants |
| `"finite_difference"` | ✅ | ✅ | Area-weighted; general unstructured meshes |
| `"finite_difference:lsq"` / `:uniform` / `:inverse_distance"` | ✅ | ✅ | Least-squares / uniform / distance-weighted |
| `"finite_difference:cotangent"` | — | ✅ | Cotangent Laplacian; **2D only** |
| `"spectral"` | ✅ | ✅ | FFT along the grid axes; **uniform grid**, assumes periodicity |
| `"spectral:cosine"` | ✅ | ✅ | Even extension instead — for fields with `u' = 0` at both ends |

Set a project-wide default with `jno.setup(__file__, diff_type="spectral")` — `diff_type` takes
either a whole scheme or an AD sub-mode (`"forward"` / `"reverse"`, its original meaning). A
per-call `scheme=` always overrides it, which is how one term keeps finite differences while the
run is spectral.

#### Spectral differentiation

On a uniform grid the derivative is a multiply in Fourier space, which is **exact** for a
band-limited field rather than merely high order. Measured against the analytic result on a 17×17
grid, same field, same grid:

| | `d u/dx` | `∇²u` |
|---|---|---|
| `"spectral"` | `1.11e-14` | `9.10e-13` |
| `"finite_difference"` | `1.60e-01` | `5.24e+01` |

It also reaches where automatic differentiation **cannot**. An operator fed a *stored* field has no
path from `x` to its output, so `u.laplacian(x, y)` under AD is identically `0.0` — and a physics
residual built on it is silently satisfied by any network at all. Both `"finite_difference"` and
`"spectral"` work on the field's *values* instead and give a real derivative there; spectral is the
accurate one when the field is periodic and band-limited, which is the regime an FNO already
assumes. So the choice for a PINO residual is between those two, and AD is simply wrong:

```python
jno.setup(__file__, diff_type="spectral")

d = jno.Shape.rect(0, 0, 1, 1, size=1/24).domain(structured=True)
x, y, _ = d.variable("interior")
d.variable("_f", forcing)
_f = d.variable("_f")

u   = net(_f).scalar.bind(x=x, y=y)
res = u.xx + u.yy + _f                      # −∇²u = f, from physics alone
crux = jno.core([res.mse])
```

The full Hessian is exactly symmetric (`|H_xy − H_yx| = 0`), because the multiplier `−k_a k_b` is
symmetric in `(a, b)` — the two components are the same computation. A Laplacian takes one forward
transform for all its terms, halving the transforms against separate per-axis second derivatives.

!!! warning "Periodicity is your claim, and the grid must be uniform"

    `"spectral"` assumes the field is periodic along every differentiated axis. On a non-periodic
    field the implied extension has a jump and the derivative rings (Gibbs) — plausible-looking
    numbers that are simply wrong. jNO does **not** check this: the only per-axis periodic flag it
    records is a residue of whether a periodic FDM problem happened to be built earlier in the
    process, so it is not a statement about your geometry.

    `"spectral:cosine"` mirrors the field instead, and is exact when the **odd derivatives vanish at
    both ends** (Neumann-like). That is narrower than "non-periodic": a ramp has `u' ≠ 0` at the
    ends, so its mirrored extension has a kink and still rings — better by ~44×, but still `O(1)`.

    Both need a **uniform** grid: `jno.Shape.rect(...).domain(structured=True)`, or any domain
    carrying `_grid_shape`. Non-uniform spacing and unstructured meshes raise rather than guess.

**Choose the scheme per direction.** `jno.fdm` gets the spectral backend with no wiring, but it is
not a blanket upgrade — a residual usually has different boundary behaviour along different axes.
On `−∇²u = 5π² sin(2πx) sin(πy)`, periodic in `x` and Dirichlet in `y`:

```python
res = -ui.d2(x, scheme="spectral") - ui.d2(y) - f     # exact basis in x, stencil in y
```

| | rel-L2 |
|---|---|
| finite differences in both | `1.96e-02` |
| spectral in both | `2.83e-02` |
| **spectral in `x`, finite differences in `y`** | **`1.14e-03`** |

Spectral applied to the Dirichlet direction gives back more than it gains, because `u′ ≠ 0` at those
ends. Applied only where the geometry is genuinely periodic it is 17× better than the stencil
everywhere. (For a Dirichlet direction the natural basis is a *sine* transform; JAX provides only
DCT-2 and no DST, so that variant does not exist.)

**Spell the Laplacian however reads best.** `u.xx + u.yy`, `u.d2(x) + u.d2(y)` and
`u.laplacian(x, y)` describe the same operator, and `jno.core` compiles all three to the same
single node: a trace pass folds a sum of squared partials over distinct coordinates into one
Laplacian, so the network is evaluated and differentiated once instead of once per coordinate.
On a 2-D+time PINN (513 collocation points, MLP 4×64) that is 308 MFLOP/step for every spelling,
against 390 for the unfused `u.xx + u.yy` and 470 for `u.d2(x) + u.d2(y)`.

The fold is deliberately conservative — it applies only where the two forms are the same
mathematics. Terms keep their own nodes when they repeat a coordinate (`u.xx + u.xx` is
2 ∂²u/∂x²), when they are subtracted rather than added, when they sit over different fields or
different AD modes, when the coordinate is temporal (`u.tt` evaluates through the time path), and
for every `finite_difference` scheme (`:cotangent` returns the whole Laplacian for any requested
dimension, so folding would halve it). FEM weak forms are left untouched — the variational route
lowers them by pattern.

**FEM weak forms — `.bind` then attribute derivatives.** A finite-element trial/test symbol is bound to
its quadrature coordinates once, after which derivatives read as plain attributes:

```python
ui = u.bind(x=xi, y=yi, t=ti)     # bind the symbol to coordinates
ui.x, ui.y, ui.z                  # spatial derivatives ∂u/∂x, ∂u/∂y, ∂u/∂z
ui.t                              # time derivative
```

### Integration

`.integrate()` collapses a field to a scalar by summing over the mesh. The region (**volume** vs
**boundary**) is auto-detected from the `Variable` tags inside the expression — you pass no region
argument. Requires `compute_mesh_connectivity=True`.

```python
vol = u.integrate()               # ∫_Ω u dV   (interior tag → volume weights)
bnd = u_b.integrate()             # ∫_∂Ω u ds  (boundary tag → surface weights)  — jnn.integrate(u) is the alias
```

Flux integrals are written explicitly — request normals and form the dot product yourself:

```python
x_b, y_b, _, nx, ny = dom.variable("boundary", normals=True, split=True)
flux = (u_b.d(x_b) * nx + u_b.d(y_b) * ny).integrate()    # ∮ ∂u/∂n ds
```

An `Integral` is an ordinary scalar node — differentiable and `jax.jit`-compatible, so it drops
straight into a loss (`(u.integrate() - target).square()`) or a tracker.

### Reductions, math & comparisons

Reductions are properties returning a squeezed scalar node; the loss helpers are the ones you reach for
most:

```python
u.mean   u.sum   u.min   u.max   u.std          # reductions
u.mse    # mean(square(x))        u.mae         # mean(abs(x))
u.shape  u.T     u.real  u.imag                 # structural / complex parts
```

Symbolic comparisons return trace nodes: `a.equal(b)`, `a.not_equal(b)`, and the operators
`>`, `<`, `>=`, `<=`. The full elementwise math library (`sin`, `exp`, `sqrt`, `where`, `concat`,
`stack`, `dot`, `matmul`, `norm`, …) lives in `jno.numpy` / `jno.np` — see the
[jno.numpy Reference](operators/numpy-reference.md) for the complete catalog.

### Semantic views & binding

Typed views reinterpret an expression without copying, exposing the right accessors for its role:

```python
u.scalar   u.vector   u.complex   u.matrix   u.voigt   u.field
```

Every view supports `.bind(**named_vars)` (alias `.partials(...)`) to attach the coordinate `Variable`s
a field depends on, so attribute-style derivatives (`.x`, `.t`) work even when those coordinates are not
the network's own inputs.

**Re-binding.** `.bind(...)` may be called on an already-bound view: names merge, and a name given
again wins. That is how you resolve the conflict `u.bind(x=x1) + v.bind(x=x2)` raises — arithmetic
refuses to pick a side, so you say which one explicitly.

```python
u = net(f).scalar.bind(x=x)
u = u.bind(y=y)          # keeps x, adds y
u = u.bind(x=x_other)    # overrides x, keeps y
```

A re-bind returns the **same** view class it was called on, so a `.field` view keeps its
finite-difference derivatives rather than silently reverting to automatic differentiation (which
would be identically zero for an operator network that never takes `x` / `y` as inputs).

### Units & non-dimensionalization

Annotate the **dimension** and characteristic **magnitude** of any leaf, and `jno.units` audits
consistency and extracts the dimensionless groups (Fourier / Péclet numbers) of a residual — then
rewrites it to a well-scaled `O(1)` form.

```python
x = x.unit("m").scale(L)          # dimension + characteristic length
u = net(x, t).unit("K").scale(U)  # dimension + characteristic magnitude of the field
res = u.d(t) - alpha * u.d2(x)

jno.units.check(res)                       # audit dimensional consistency (.warnings is empty if OK)
jno.units.infer(res)                       # the inferred Unit of an expression
report = jno.units.nondimensionalize(res)  # each term's dimensionless group πᵢ = Sᵢ / S_ref
transformed, rescaler = jno.units.rescale(res)   # rewrite to O(1) dimensionless form
rdom = rescaler.rescaled_domain(dom)       # a copy of the domain with coordinates scaled to O(1)
u_phys = rescaler.to_physical(u_hat)       # map a dimensionless solution back:  u = U · û
```

`nondimensionalize` / `rescale` operate on the **additive terms within a single residual**
(`πᵢ = Sᵢ / S_ref`), not on a ratio between two separate losses. Today only coordinates and the network
output are annotatable through the public API; a bare material coefficient has no public `.unit` hook
yet. See the
[Gradient Conflict tutorial](tutorials/07-analysis/gradient-conflict.md) for a worked example.

### Custom functions

Wrap an arbitrary JAX function so it joins the symbolic graph and stays differentiable — for nonlinear
constitutive laws, lookup tables, or anything cleaner as standalone code:

```python
result = jnn.function(lambda x, y: jnp.exp(-x**2) * jnp.sin(y), [x, y])
```

### Trackers, labels & debugging

A **tracker** is logged every `interval` steps but does **not** contribute to the loss:

```python
val_error = jno.np.mean(jno.np.abs(u - u_exact)).tracker(100)   # or jno.np.tracker(expr, interval=100)
crux = jno.core([pde.mse, bc.mse, val_error])
```

Logged values appear in the `statistics` returned by `solve()`. Related metadata methods (all return
`self`, chainable): `.name("label")` tags an expression for logs / W&B, and `.print(what="shape")`
emits a runtime shape/stat/value and passes the value through.

### Gradient control

`.stop_gradient` is identity in the forward pass and zero in the backward pass — freeze part of a graph
or turn an expensive quantity into a constant regulariser:

```python
J_sg    = u.grad(u_net).stop_gradient      # treat the current Jacobian as a constant
ntk_reg = (J_sg @ J_sg.T - target_K).mse
```

### Stochastic terms — `jno.noise`

Noise nodes are ordinary trace expressions that draw a **fresh realisation every training step**,
derived from the solver's step key, so a run is reproducible from the global seed with no key
management. Under `crux.eval()` (no key) they evaluate to zeros, so post-training evaluation stays
deterministic.

```python
u_noisy = net(x) - (u_obs + jno.noise.gaussian(std=0.01))   # (N, 1) — noisy observations
uv      = net(x, y) + jno.noise.gaussian(std=0.01, ndim=2)  # (N, 2) — vector noise
xj      = net(x + jno.noise.uniform(low=-1e-3, high=1e-3))  # jittered input coordinate
```

`gaussian`, `uniform` and `laplace` are **pointwise** — each point draws independently. `grf` is
not: it is a **spatially correlated** Gaussian random field, so it is an input *function* rather
than a perturbation. That is what makes operator learning possible with no dataset at all — a fresh
in-distribution input every step:

```python
f = jno.noise.grf(x, y, length_scale=0.1)        # Matern-3/2 by default
u = net(f).scalar.bind(x=x, y=y)
crux = jno.core([(u.laplacian(x, y) + f).mse])
```

Built by spectral representation (Shinozuka & Deodatis 1991; Rahimi & Recht 2007), with the Matern
spectral density from Rasmussen & Williams §4.2. Knobs: `length_scale`, `variance`, `kernel`
(`"matern"` / `"rbf"`), `nu`, `modes`, `ndim`.

!!! note "An approximate GP sample, and it costs memory"

    Exact only as `modes → ∞`; the covariance error is `O(M^{-1/2})`, so the default `modes=256`
    is ~6%. Exact circulant embedding needs a regular grid, whereas the evaluator sees a flat point
    cloud — hence the spectral method. Cost is `O(B × N × M)` inside the batch vmap, so raise
    `modes` knowingly.

    Like every noise node, one realisation is shared across the timesteps of a single window
    (the step key is not split per timestep), and it does vary across batch samples.

### Parameter Jacobian & the Neural Tangent Kernel

`u.grad(net)` is `.grad` in its **parameter** overload: passed a single `Model` (rather than
coordinates) it returns the Jacobian of the expression w.r.t. the network's trainable parameters,
shape `(B, N, P)` (or `(B, N, D, P)` for vector output). It is an ordinary node — usable as a tracker,
a loss, or evaluated after training.

```python
J   = u.grad(u_net)               # ∂u/∂θ
K   = J[0] @ J[0].T               # (N, N) Neural Tangent Kernel

# Restrict to a parameter subset (cheaper) via a boolean pytree + net.mask(...)
import equinox as eqx, jax
all_false   = jax.tree_util.tree_map(lambda _: False, u_net.module)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)
J_out = u.grad(u_net.mask(output_mask))    # only the output-layer weights
```

See the [Gradient Conflict tutorial](tutorials/07-analysis/gradient-conflict.md) for NTK conditioning
and gradient cosine-similarity diagnostics.

---

## Part B — operations that require trainable parameters

These act on a **`Model`** — the object returned by `jno.nn(module)` (any Equinox module: a `foundax`
model or your own `eqx.Module`). Calling it, `net(x, y)`, yields a `ModelCall`. All controls return
`self` and chain.

```python
import optax, foundax as fx, jno
from jno import LearningRateSchedule as lrs

net = jno.nn(fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3), name="u_net")
```

### Trainable scalar parameters

`jno.np.parameter` creates a trainable array that optimises exactly like a network — the building block
for **inverse problems** (identify unknown PDE coefficients from residuals):

```python
a = jno.np.parameter((1,), key=k1, name="a")
a.optimizer(optax.adam(1e-2))
residual = a * jno.np.sin(π * x) - target
```

### Optimizer & learning rate

```python
net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.8, 5000, 1e-5))   # optimizer + LR schedule
```

Per-group control via `.mask(...)` (consumed by the next mutator; a bare `optimizer(...)` clears all
groups):

```python
net.optimizer(optax.adamw).scale(lrs(1e-3))          # global fallback
net.mask(decoder_mask).optimizer(optax.adam).scale(lrs(5e-4))
```

`jno.optimizers` adds custom second-order optimizers not in optax (`engd`, `ssbroyden`, `ssbfgs`,
`soap`, `md`) — each an optax `GradientTransformation`, composable with `optax.chain`. See
[Optimizer & LR](model-controls/optimizer-lr.md) and [Schedules](adaptive/schedules.md).

### Regularization

`.regularize` is called on a **field** (a called network or a FEM nodal parameter) and returns a
pointwise penalty term — FEM-exact where possible, else the autodiff form:

```python
reg = net(x, y).regularize("h1seminorm", x, y)       # smooth / tv / nonneg / bounded / l2(FEM)
crux = jno.core([pde.mse, reg])
```

### Parameter selection & freezing

```python
net.freeze()            net.unfreeze()               # exclude / re-include from training
net.mask(param_mask)                                 # one-shot boolean-pytree scope for the next control
net.constrain(jax.nn.softplus)                       # reparameterize params before every forward pass
```

### Fine-tuning (LoRA)

```python
net.lora(rank=4, alpha=1.0)                           # low-rank adapters; per-target via specs=
```

Details and per-target configuration: [LoRA](model-controls/lora.md).

### Bayesian & variational inference

Turn a point estimate into a posterior — MCMC (`.bayesian`) or a variational ELBO fit (`.vi`, mutually
exclusive):

```python
net.bayesian(blackjax.nuts, warmup=500, keep=1000)               # posterior via MCMC
net.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-3))        # variational approximation
net.posterior_samples        net.posterior_diagnostics   # draws + per-step diagnostics (or None)
```

See [Bayesian Sampling](training/bayesian.md).

### Precision & initialization

```python
net.dtype(jnp.float64)                                # set params + compute dtype
net.initialize("weights.eqx")                         # load pretrained weights (path / pytree / initializer)
```

### Diagnostics, sweeps & deployment

```python
net.summary()          net.dont_show()               # print / suppress the model-control summary
net.tune(optimizer=[optax.adam, optax.sgd], lr=[1e-3, 1e-4])   # declare per-model HP-sweep options
net.reset()                                           # reset training config to defaults
net.to_iree(sample_inputs)                            # compile to an IREEModel for deployment
```

Weights are persisted with the free functions `jno.save` / `jno.load` (not model methods). See
[Hyperparameter Tuning](Hyperparameter-Tuning.md) and [IREE Deployment](model-controls/iree.md).
