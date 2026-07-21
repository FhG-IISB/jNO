# API Reference

This page is auto-generated from in-source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). When something here
looks wrong, fix the docstring — not this file.

For conceptual prose (what these objects *are* and *why* they exist),
see the [Getting Started](Getting-Started.md) guide and the
[Concepts](concepts.md) page.

---

## `jno.core`

The top-level solver. Wraps a list of constraint expressions and a
domain, compiles them once, then exposes `solve()` / `eval()`.

::: jno.core.core
    options:
      members:
        - __init__
        - compile
        - solve
        - eval
        - sweep
        - print_shapes

---

## Domain

`jno.domain` is the entry point for spatial geometry, mesh management,
sampling, and tensor tags.

::: jno.domain.domain
    options:
      members:
        - __init__
        - variable
        - sample
        - summary
        - line
        - rect
        - polygon
        - export
        - plot_mesh

### `jno.domain.csg`

::: jno.domain.polygon_domain.PolygonDomain
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_root_full_path: false

---

## Neural-network controls

`jno.nn` lifts a plain Equinox / foundax module into a jNO
`Model` so it can participate in the trace and accept per-model
optimisers, masks, LoRA, freezing, and so on.

::: jno.architectures.models.nn
    options:
      members:
        - wrap

::: jno.trace.Model
    options:
      members:
        - optimizer
        - lr
        - freeze
        - unfreeze
        - mask
        - lora
        - dtype
        - constrain
        - initialize
        - tune

---

## Symbolic math (`jno.np`)

A NumPy-compatible namespace that returns traced placeholders instead
of concrete arrays. Use it inside any expression that you intend to
feed into `jno.core(...)`.

::: jno.jnp_ops
    options:
      members:
        - concat
        - sin
        - cos
        - exp
        - log
        - sqrt
        - abs
        - grad
        - pi

---

## Function helpers and loss balancers (`jno.fn`)

`jno.fn` provides PDE-named helpers (`heat`, `wave`, `burgers_1d`, ...),
loss reductions (`mse`, `mae`, `rmse`, `huber`, `log_cosh`, ...), and
the adaptive loss balancers under `jno.fn.adaptive.*`.

::: jno.fn
    options:
      heading_level: 3
      members:
        - _module_call

---

## Training history

`solve()` returns a `statistics` object. The most common operations:

::: jno.utils.statistics.statistics
    options:
      members:
        - total_loss
        - total_loss_history
        - plot
        - save
        - load

---

## Differential and integral operators

These provide the residuals you put inside constraints
(`u.laplacian(x, y)`, `u.d(x)`, `(grad_u * n).integrate()`).

### Scheme strings

Every differential operator (`.d`, `.diff`, `.d2`, `.dd`, `.laplacian`,
`.hessian`) accepts a `scheme=` kwarg that selects the backend:

| Scheme | Backend |
| --- | --- |
| `"automatic_differentiation"` *(default)* | global default — see `jno.setup(diff_type=..., hessian_type=...)` |
| `"automatic_differentiation:forward"` | first-order via `jax.jacfwd` |
| `"automatic_differentiation:reverse"` | first-order via `jax.jacrev` |
| `"automatic_differentiation:fwd-over-rev"` | second-order `jacfwd(jacrev(f))` *(= historical `jax.hessian`)* |
| `"automatic_differentiation:fwd-over-fwd"` | second-order `jacfwd(jacfwd(f))` |
| `"automatic_differentiation:rev-over-rev"` | second-order `jacrev(jacrev(f))` |
| `"automatic_differentiation:rev-over-fwd"` | second-order `jacrev(jacfwd(f))` |
| `"finite_difference"` | central-difference stencils on mesh (with `:lsq` / `:uniform` / `:inverse_distance` / `:cotangent` sub-schemes) |

Forward-mode is typically cheaper when the input dim (≤ 3 spatial dims for
PINNs) is ≤ the output dim; reverse-mode is cheaper for scalar losses with
many inputs. Set the project-wide default once via `.jno.toml`:

```toml
[jno]
diff_type    = "forward"        # default for first-order operators
hessian_type = "fwd-over-rev"   # default for second-order operators
```

or per script via `jno.setup(__file__, diff_type="forward")`. Per-call
`scheme=` always overrides the default.

::: jno.differential_operators.DifferentialOperators

::: jno.integration_operators.IntegrationOperators

::: jno.utils.ad_mode

---

## Solvers and preconditioners

`jno.solve` and `jno.precond` are the slots that `fem.solve(linear=…, nonlinear=…,
precond=…, time=…)` composes (see the [FEM guide](fem.md)). The families:

| Kind | `jno.solve` |
| --- | --- |
| **Linear — direct** | `lu` (sparse LU), `dense` |
| **Linear — iterative (Krylov)** | `cg`, `bicgstab`, `gmres`, `fgmres`, `minres`; `lstsq` (LSQR, least-squares); `chebyshev` (polynomial) |
| **Linear — multigrid** | `amg` (GPU AMG / NVIDIA AmgX via jaxamg) |
| **Nonlinear** | `newton`, `picard` |
| **Eigenproblem** | `eigs` (generalized `Kx = λMx`) |
| **Matrix functions** (stochastic Lanczos, matrix-free) | `logdet`, `trace`, `applyfun` (`f(A)·v`), `diagonal` |
| **Time integration** | `theta` (θ-method), `exponential` (exponential integrator) |

**Preconditioners** (`jno.precond`, for the iterative solvers): `jacobi`, `chebyshev`,
`amg`, `ams` (H(curl) auxiliary-space Maxwell), `form` (weak-form auxiliary operator),
`inner` (any solver as `M⁻¹`), `block_diag` / `triangular` (block / Schur), and `cached`.

::: jno.solve

::: jno.precond

---

## Tracing primitives

Most users never instantiate these directly — they are what the
expression-building API returns. Documented here for reference and for
authors of custom operators.

::: jno.trace.Variable

::: jno.trace.Integral

::: jno.trace.Noise
