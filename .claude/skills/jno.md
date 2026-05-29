# jNO skill

You are a coding assistant specialized in the **jNO** library (`jax-neural-operators`), a research-grade JAX framework for physics-informed neural operators and PDE solving.

## Package overview

jNO couples a **tracing layer** (lazy symbolic graph built from `Placeholder` ops) with a **compiler + evaluator** that JIT-compiles the graph into JAX. Networks come from the companion package **foundax** (MLP, FNO, DeepONet, Poseidon, etc.) or any custom `equinox.Module`. The high-level entry point is `jno.core`.

## Core five-step workflow

```python
import jno, jax, optax, foundax

# 1. Directory + logging
dir = jno.setup("./runs/my_experiment")

# 2. Domain
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y, _ = domain.variable("interior")    # collocation points → Placeholder
xb, yb, _ = domain.variable("boundary") # boundary points

# 3. Network from foundax, wrapped for tracing
fx = foundax.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=4,
                 key=jax.random.PRNGKey(0))
net = jno.nn.wrap(fx)
net.optimizer(optax.adam(1e-3))

# 4. Symbolic expressions (lazy — no actual computation yet)
u = net(x, y)
pde = u.dd(x) + u.dd(y) + source_term   # Laplacian via automatic differentiation

# 5. Solve
crux = jno.core(constraints=[pde.mse, bc.mse], domain=domain)
crux.solve(epochs=10_000, batchsize=32)
```

## Key API reference

### Domain
| Call | Meaning |
|---|---|
| `jno.domain(constructor=jno.domain.rect(mesh_size, x_range, y_range))` | 2-D rectangle |
| `jno.domain(constructor=jno.domain.rect(mesh_size, x_range))` | 1-D interval |
| `jno.domain(constructor=jno.domain.polygon(points, mesh_size))` | Arbitrary 2-D polygon |
| `jno.domain.poseidon(nx=128, ny=128)` | Structured 2-D grid for Poseidon-style workflows |
| `dom.variable("interior")` | Returns `(coord1, coord2, ...)` Placeholders for interior |
| `dom.variable("boundary")` | Same for boundary |
| `dom.variable("boundary", normals=True)` | Returns `(*coords, nx, ny)` with outward normals |
| `dom.variable("k", array)` | Register a named input variable (e.g. a parameter field) |

### Differential operators (`jno.numpy` / `jnn`)
| Call | Meaning |
|---|---|
| `jnn.grad(u, x)` | ∂u/∂x (automatic differentiation by default) |
| `jnn.grad(u, x, scheme="finite_difference")` | FD instead of AD |
| `jnn.laplacian(u, [x, y])` | ∇²u (alias: `jnn.laplace`) |
| `jnn.jacobian(u, [x, y])` | Full Jacobian [∂u/∂x, ∂u/∂y] |
| `jnn.hessian(u, [x, y])` | Full Hessian matrix |
| `jnn.divergence([Fx, Fy], [x, y])` | ∇·F |
| `jnn.curl_2d(Fx, Fy, x, y)` | 2D curl scalar |
| `jnn.curl_3d(Fx, Fy, Fz, x, y, z)` | 3D curl vector |

Placeholder methods (`u.d(x)`, `u.dd(x)`, `u.laplacian(x, y)`) are also available.

### Integral operators
```python
# Volume integral ∫_Ω expr dV
vol = expr.integrate()           # or jnn.integrate(expr)

# Boundary integral ∫_∂Ω expr ds
bnd = expr_on_boundary.integrate()

# Flux integral ∫_∂Ω F·n ds
xb, yb, _, nx, ny = dom.variable("boundary", normals=True)
flux = (u_b.d(xb) * nx + u_b.d(yb) * ny).integrate()
```

Integration weights (nodal volumes / arc lengths) are precomputed at domain creation. Region is auto-detected from variable tags.

### Loss / reduction on Placeholders
| Property | Meaning |
|---|---|
| `expr.mse` | Mean-squared error (most common loss) |
| `expr.mae` | Mean-absolute error |
| `expr.mean` | Mean |
| `expr.sum` | Sum |
| `expr.std`, `expr.max`, `expr.min` | Statistics |

### `jno.core`
```python
crux = jno.core(
    constraints=[pde.mse, bc.mse],  # list of scalar loss Placeholders
    domain=dom,
    mesh=(1, 1),                     # (batch_devices, model_devices) for sharding
    rng_seed=42,
)
crux.print_tree()                    # debug: print symbolic computation tree
crux.print_shapes()                  # debug: print inferred tensor shapes
crux.solve(epochs=N, batchsize=B, callbacks=[...])
crux.eval(u)                         # evaluate expression on training domain
crux.eval(u, domain=test_dom)        # evaluate on a different domain
crux.predict(points=pts, operation=u, context=ctx)  # arbitrary point arrays
```

### Running training
```python
stats = crux.solve(
    epochs=5000,
    batchsize=128,              # None = full batch
    checkpoint_gradients=False, # True → gradient checkpointing (saves memory, ~30% slower)
    offload_data=False,         # True → keep dataset on CPU, stream mini-batches
    constraint_weights=ws([1.0, 10.0]),  # optional WeightSchedule
    profile=True,               # capture Perfetto traces to <log_dir>/traces/
)
stats.plot("history.png")
stats.epoch; stats.total_loss; stats.losses; stats.weights
stats.training_time; stats.trainable_params; stats.total_params
```

Multi-phase training: call `solve()` multiple times (Adam → L-BFGS → SOAP). Solver resumes from where it left off.

### Saving / loading
```python
jno.save(crux, f"{dir}/model.pkl")
crux2 = jno.load(f"{dir}/model.pkl")

# After loading, re-attach optimizer
crux.set_optimizer(optax.adam, lr=lrs(1e-4))
crux.solve(1000)
```

### Checkpoints
```python
for i, ckpt in enumerate(crux.checkpoints):
    print(i, ckpt["step"], ckpt["time"])
crux.models = crux.checkpoints[-1]["models"]
```

## Model controls (`jno.nn.wrap`)

`jno.nn.wrap` accepts any equinox module. All controls return `self` and are chainable.

### Optimizer and LR
```python
from jno import LearningRateSchedule as lrs

# Simple (pass optax optimizer directly)
net.optimizer(optax.adam(1e-3))

# With dynamic schedule (optax constructed with lr=1)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(5000, 500, 1e-3, 1e-4))
net.optimizer(optax.adam(1), lr=lrs.cosine(5000, 1e-3, 1e-6))
net.optimizer(optax.adam(1), lr=lrs.piecewise_constant([1000,3000], [1e-3,5e-4,1e-4]))
net.optimizer(optax.adam(1), lr=lrs(lambda epoch, losses: 1e-4 * 0.9**(epoch/500)))

# LR-only update
net.lr(lrs(1e-5))

# Parameter groups via mask
net.mask(decoder_mask).optimizer(optax.adam, lr=lrs(5e-4))
net.mask(encoder_mask).optimizer(optax.sgd,  lr=lrs(1e-4))
```

### Freeze / unfreeze / mask
```python
net.freeze()                      # freeze entire model
net.unfreeze()
net.mask(decoder_mask).freeze()   # freeze only mask-selected leaves
```

Build a boolean mask from a regex:
```python
import re, equinox as eqx, jax

def regex_mask(module, pattern):
    arrays = eqx.filter(module, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(arrays)
    def part(k):
        if hasattr(k, "name"): return str(k.name)
        if hasattr(k, "idx"):  return str(k.idx)
        if hasattr(k, "key"):  return str(k.key)
        return str(k)
    leaves = [bool(re.search(pattern, "/".join(part(p) for p in path))) for path, _ in flat]
    return jax.tree_util.tree_unflatten(treedef, leaves)

decoder_mask = regex_mask(net.module, r"decoder")
```

### LoRA
```python
net.lora(rank=8, alpha=16)                    # all layers
net.lora(rank=8, target="encoder")            # regex on pytree path
net.mask(encoder_mask).lora(rank=8)           # from boolean mask

# Per-target specs
net.lora(specs=[
    {"target": "encoder", "rank": 4,  "alpha": 1.0},
    {"target": "decoder", "rank": 16, "alpha": 4.0},
])

# Combine with freeze (freeze all; only adapters train)
net.freeze().lora(rank=8, alpha=16)
```

**LoRA Zoo (Linear):**
| Class | Trainable params | Key idea |
|-------|-----------------|----------|
| `LoRALinear` | `r·(in+out)` | Standard LoRA |
| `rsLoRALinear` | `r·(in+out)` | Scale = α/√r — stable across ranks |
| `DoRALinear` | `r·(in+out)+out` | Magnitude + direction decomposition |
| `PiSSALinear` | `r·(in+out)` | SVD principal-component init |
| `LoRAFALinear` | `r·out` | Frozen A; halves adapter params |
| `LoRAXSLinear` | `r²` | Frozen A,B; trainable r×r core |
| `VeRALinear` | `out+r` | Seed-based frozen A,B; only b,d trained |
| `MiLoRALinear` | `r·(in+out)` | Adapts minor SVD components |
| `IA3Linear` | `out` | Per-output scale vector; no rank param |
| `LoKrLinear` | depends | Kronecker product adapter |
| `OFTLinear` | `n_blocks·r²` | Orthogonal fine-tuning |

Matching `*Conv` variants exist for conv layers. Import from `jno.lora`.

### Initialize, dtype, tune, reset
```python
net.initialize("./weights.eqx")
net.initialize("./runs/checkpoints/2000::1")   # Orbax checkpoint
net.initialize(other_model.module)
net.initialize(jax.nn.initializers.xavier_uniform(), key=jax.random.PRNGKey(0))

net.dtype(jnp.bfloat16)   # cast floating-point params before training

net.tune(
    freeze=[True, False],
    lora=[(4, 1.0), None],
    optimizer=[optax.adam],
    lr=[lrs(1e-3), lrs(1e-4)],
)

net.reset()   # clear all controls
```

### IREE deployment
```python
iree_model = net.to_iree(
    sample_inputs=(jnp.ones((100, 2)),),
    target_backend="llvm-cpu",   # or "cuda", "rocm", "vulkan"
    optimization_level=3,        # 0-3, default 3
)
output = iree_model(np.random.rand(100, 2).astype(np.float32))  # returns np.ndarray

jno.save(iree_model, "deployed.pkl")
loaded = jno.load("deployed.pkl")  # no JAX needed

# Compile a raw JAX function
from jno.utils.iree import IREEModel
compiled = IREEModel.compile(my_fn, sample_inputs=(jnp.ones((100,)),))
```

## Adaptive loss weights

Weights are traced placeholders — recomputed inside the compiled JAX function every step:

```python
w_pde, w_bc = jno.fn.adaptive.relobralo([pde.mse, bc.mse])
crux = jno.core([w_pde * pde.mse, w_bc * bc.mse, w_pde.tracker(), w_bc.tracker()], domain)
```

Available balancers: `relobralo`, `softadapt`, `dwa`, `lbpinns_loss_balancing`, `rlw`.
All accept `mode=` (`"raw"`, `"minmax"`, `"l2"`).

## Weight schedules

```python
from jno import WeightSchedule as ws

ws([1.0, 10.0])                              # fixed per-constraint weights
ws(lambda e, L: [1.0, 10.0 * L[1]])         # adaptive function
crux.solve(5000, constraint_weights=ws([1.0, 10.0]))
```

## Callbacks

```python
# Early stopping
cb = jno.callbacks.early_stopping(
    patience=1000, min_delta=1e-6, mode="min",   # or "max", "rel"
    metric_fn=lambda **kw: float(kw["individual_losses"][1]),  # custom metric
    baseline=1e-3,
)

# Checkpointing
cb = jno.callbacks.checkpoint(
    directory="runs/ckpts",
    save_interval_epochs=500, max_to_keep=3,
    best_fn=lambda m: m["total_loss"],
    async_checkpointing=True,   # default
)
state = cb.restore()            # latest
state = cb.restore(step=2000)   # specific step

crux.solve(50_000, callbacks=[cb_ckpt, cb_early])
```

## Adaptive resampling

```python
from jno import sampler

x, y = domain.variable(
    "interior",
    resampling_strategy=sampler.rad(resample_every=100, resample_fraction=0.1, start_epoch=1000),
)
```

| Strategy | Cost | Best For |
|----------|------|----------|
| `sampler.random` | Very low | Preventing grid overfitting |
| `sampler.rad` | Low | Localized high-error regions (residual clusters) |
| `sampler.rard` | Low | Smooth residual fields (importance sampling) |
| `sampler.ha` | Low | General use, avoids over-concentration |
| `sampler.cr3` | Medium | Time-dependent PDEs (causal resampling) |
| `sampler.pinnfluence` | High | High-accuracy / small-data regimes |

Always delay with `start_epoch > 0`. Use `resample_fraction` 0.1–0.3.

## Trackers

Logged each step but do not contribute to the loss:

```python
from jno.numpy import tracker

val_error = tracker(jno.np.mean(jno.np.abs(u - u_exact)), interval=100)
crux = jno.core([pde.mse, bc.mse, val_error], domain)
```

## Custom functions and trainable parameters

```python
import jno.numpy as jnn

# Wrap arbitrary JAX function into the graph
result = jnn.function(my_fn, [x, y])

# Trainable scalar/array (for inverse problems)
a = jno.np.parameter((1,), key=k1, name="a")
a.optimizer(optax.adam(1e-2))
residual = a * jnn.sin(π * x) - target
crux = jno.core([residual.mse], domain)
_a = crux.eval([a])
```

## Parameter Jacobian / gradient analysis

```python
J = u.grad(u_net)             # (B, N, P) — Jacobian w.r.t. trainable params
J_norm = tracker(jno.np.mean(J ** 2), interval=50)

# NTK condition number
J_flat = J[0]; K = J_flat @ J_flat.T
K_cond = tracker(jno.np.max(K) / (jno.np.min(K) + 1e-8), interval=100)

# Gradient cosine similarity between losses
g_pde = jno.np.mean(pde_expr.grad(u_net)[0], axis=0)
g_bc  = jno.np.mean(bc_expr.grad(u_net)[0],  axis=0)
cos_sim = tracker(jno.np.dot(g_pde, g_bc) / (jno.np.norm(g_pde) * jno.np.norm(g_bc) + 1e-8))

# Use stop_gradient when Jacobian is used as a loss (avoids 2nd-order AD cost)
J_sg = u.grad(u_net).stop_gradient()
ntk_reg = (J_sg @ J_sg.T - target_K).mse
```

## Multi-device parallelism

```python
crux = jno.core(constraints, domain, mesh=(1, 1))   # single device
crux = jno.core(constraints, domain, mesh=(4, 1))   # 4-GPU data parallel
crux = jno.core(constraints, domain, mesh=(1, 2))   # model parallel over 2 GPUs
crux = jno.core(constraints, domain, mesh=(2, 2))   # hybrid (4 GPUs)

n = len(jax.devices())
crux = jno.core(constraints, domain, mesh=(n, 1))   # all available
```

## FEM and variational PINNs

```python
domain.init_fem(
    element_type="TRI3",     # "TRI3", "TRI6", "QUAD4", …
    quad_degree=2,
    bcs=[
        jno.dirichlet(["left", "right", "top", "bottom"]),
        jno.neumann("right"),
    ],
    fem_solver=True,
    vec=1,                   # 2 for vector-valued unknowns
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss", split=True)

weak = jnn.grad(u, xg) * jnn.grad(phi, xg) + jnn.grad(u, yg) * jnn.grad(phi, yg) - 1.0 * phi

# Linear FEM solve
A, b = weak.assemble(domain, target="fem_system")
u_h  = jnp.linalg.solve(A, b)

# Nonlinear Newton loop
R = weak.assemble(domain, target="fem_residual")
u_h = jnp.zeros(R.size)
for _ in range(20):
    J, rhs = R.linearize(u_h)
    u_h = u_h + jnp.linalg.solve(J, rhs)

# VPINN: trial function is a neural network
u_nn = net(xg, yg) * xg * (1-xg) * yg * (1-yg)
weak_vpinn = jnn.grad(u_nn, xg)*jnn.grad(phi,xg) + jnn.grad(u_nn, yg)*jnn.grad(phi,yg) - 1.0*phi
crux = jno.core([weak_vpinn.mse], domain)
```

Boundary conditions:
```python
jno.dirichlet("left")                          # zero
jno.dirichlet(["left", "right"], 0.0)
jno.dirichlet("top", lambda x, y: jnp.sin(x)) # spatially varying
jno.dirichlet("wall", (0.0, 1.0))             # vector: u_x=0, u_y=1
```

## Foundation models (foundax)

```python
import foundax as fx

# Core architectures
fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)
fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
fx.unet2d(in_channels=1, out_channels=1, depth=4)
fx.deeponet(n_sensors=100, sensor_channels=1, coord_dim=2, basis_functions=128, hidden_dim=256, n_layers=4)

# Foundation model wrappers (namespace style — preferred)
fx.poseidon.T()    # also .B(), .L()
fx.morph.Ti()      # also .S(), .M(), .L()
fx.mpp.B()
fx.walrus.base()
fx.dpot.S()
fx.prose.fd_1to1()
```

All are equinox modules — wrap with `jno.nn.wrap(...)` to add jNO controls.

## `jno.numpy` utilities

```python
import jno.numpy as jnn

# Math
jnn.sin, jnn.cos, jnn.exp, jnn.log, jnn.sqrt, jnn.abs, jnn.power, jnn.tanh, ...
jnn.pi, jnn.e, jnn.inf, jnn.nan

# Reductions (support axis=, keepdims=)
jnn.sum, jnn.mean, jnn.std, jnn.var, jnn.min, jnn.max, jnn.norm, jnn.prod

# Array ops
jnn.concat([x,y], axis=-1); jnn.stack; jnn.reshape; jnn.squeeze; jnn.expand_dims

# Conditional
jnn.where(cond, x, y); jnn.maximum(x, y); jnn.minimum(x, y)

# Linear algebra
jnn.dot(x, y); jnn.matmul(x, y); x @ A

# Constants from file/dict
C = jnn.constant("C", {"k": 1.5, "rho": 2700})
C = jnn.constant("C", "params.json")   # JSON, YAML, TOML, npz

# View factor (radiation BC)
xb, yb, tb, nx, ny, VF = domain.variable("boundary", normals=True, view_factor=True)
VF_op = jnn.view_factor(VF)
q_inc = VF_op @ q_emitted
x = VF_op.solve(rhs, alpha)   # (I - αF)x = rhs
```

## Output transformation (hard BC enforcement)

```python
u = net(x, y) * x * (1 - x) * y * (1 - y)  # zero on all four walls of [0,1]²
```

## Common debugging scenarios

### Shape errors / `print_shapes`
Call `crux.print_shapes()` immediately after `jno.core(...)` — fastest way to catch dimension mismatches.

### NaN from step 1
1. Division by zero in PDE (e.g. `1/u` when u≈0 at init).
2. Learning rate too high — try `1e-4` or a warmup schedule.
3. Use `pde.debug._shape = True` / `pde.debug._mean = True` to inspect intermediate values.

### Gradient not flowing
- Verify network is inside `jno.nn.wrap(...)`.
- Constraint list must use `.mse` (scalar), not a raw Placeholder array.

### FD vs AD
Use `scheme="finite_difference"` on irregular meshes where AD would require repeated `jax.grad` calls. Requires `compute_mesh_connectivity=True` in the domain.

### Profiling
```python
crux.solve(5000, profile=True)   # writes Perfetto traces to <log_dir>/traces/
```
Open traces at [ui.perfetto.dev](https://ui.perfetto.dev). The first step (JIT compilation) is skipped; next 50 steady-state steps are captured.

## Dev / contributor workflow

```bash
pixi run fmt && pixi run lint   # format + lint
pixi run test                   # run tests
pixi run ci-fmt && pixi run ci-lint && pixi run ci-test  # CI-equivalent
```

All dev dependencies live in the `dev` pixi environment. Use `pixi add` — not bare `pip install`.

## Style conventions
- Line length: 124 (ruff enforced).
- Module-level lambdas allowed (`E731` ignored).
- `jno.np` and `jno.numpy` are aliases for the internal `jnp_ops` module — use them instead of bare `jax.numpy` in jNO source files.
- Re-exports in `__init__.py` are intentional; `F401` suppressed there.
