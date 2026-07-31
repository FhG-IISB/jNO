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
| **Eigenproblem** | `eigs` (generalized `Kx = λMx`) — dense reduction, or preconditioned LOBPCG with `precond=` |
| **Singular values** | `svd` (partial SVD of a **rectangular**, matrix-free operator — POD bases, inverse-problem ill-posedness) |
| **Matrix functions** (stochastic Lanczos, matrix-free) | `logdet`, `trace`, `applyfun` (`f(A)·v`), `diagonal` |
| **Time integration** | `theta` (θ-method), `exponential` (exponential integrator), `adaptive` (step-doubling adaptive step size) |

### Eigenproblems at scale

`jno.solve.eigs` / `FEM.eigs` have two paths, chosen by the arguments. With none of the iterative
arguments the pencil is reduced **densely** — exact, and right when you want the whole low spectrum of
a small problem, but it materializes the operator (`O(N²)` memory). Passing `precond=` selects
**preconditioned LOBPCG** (Knyazev, *SIAM J. Sci. Comput.* **23**(2), 517–541, 2001), which only applies
`K`/`M` as matvecs and so runs where the dense reduction cannot:

`K` is the **source-less** `jno.fem` whose bilinear form is the stiffness; `mass=` takes the mass form
as a plain term list, which `eigs` assembles onto the same space for you:

```python
u, v = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)

K = jno.fem([ui.x * vi.x + ui.y * vi.y])                       # stiffness — no source term
lam, X = K.eigs(mass=[ui * vi], k=6)                           # dense;  λ = ω² on a Neumann box
lam, X = K.eigs(mass=[ui * vi], k=6, precond=jno.precond.amg())  # LOBPCG, never densified
```

`jno.solve.eigs(...)` is the lower-level form of the same thing and takes two **assembled operators**
rather than a fem and a term list — `jno.solve.eigs(k=6)(K.operator[0], M)`.

The Rayleigh–Ritz runs in the **M-inner product**, so an ordinary FEM form's consistent (non-lumped)
mass matrix is handled directly, and `XᵀMX = I` holds on both paths. Eigenvalues are differentiable on
both — for **simple** eigenvalues; a degenerate cluster makes `∂λ/∂θ` ill-defined either way (use the
trace of the cluster). LOBPCG freezes the converged eigenvector and differentiates the Rayleigh
quotient, which gives that derivative exactly without differentiating through the sweeps, but its
**eigenvectors** carry no gradient where the dense path's do.

`tol`/`maxiter` tune the iteration and are **rejected** without `precond=`, so a tolerance can never be
silently ignored by the dense path. Do not set `tol` near machine precision: on an ill-conditioned
pencil the residual floors well above it (≈`4.4e-8` on a singular all-Neumann Laplacian with
`cond(K) ≈ 2e16`), and a tolerance below that floor burns the budget and **NaN-poisons** the result —
which is the deliberate contract for an exhausted budget, never a quietly under-converged spectrum.

### Singular values — `jno.solve.svd`

`eigs` solves the *symmetric* pencil `Kx = λMx`. The two questions that are **not** eigenproblems need
the SVD of a possibly rectangular map, via Golub–Kahan bidiagonalization (Golub & Kahan,
*J. SIAM Numer. Anal. Ser. B* **2**(2), 1965):

```python
U, s, Vt = jno.solve.svd(snapshots, k=6)      # POD basis from a (n_time, n_dofs) trajectory
U, s, Vt = jno.solve.svd(jacobian_op, k=20)   # ill-posedness of a parameter-to-observable map
```

* **POD / reduced-order models** — the singular vectors are the energy-optimal basis and `s` says how
  many modes the trajectory actually needs.
* **Ill-posedness** — the singular spectrum of the parameter-to-observable map says which parameter
  modes are recoverable at all; those below the noise floor are not, whatever the optimizer does.

`A` is touched only through its matvec, so it can be the JVP of a differentiable FEM solve rather than
an assembled matrix, and `s` differentiates back to whatever that matvec closes over.

`depth` (bidiagonalization steps, default `2k+10`) **must exceed `k`** — the Ritz values converge from
below, so at `depth == k` only the largest singular value is meaningful (measured 95 % error on the
rest, against ~1e-15 at `depth = 2k`). Convergence is fast on the *decaying* spectra that make POD and
ill-posedness analysis worth doing, and slow on clustered ones (~3 % error at `depth = 4k` on a tight
cluster) — inspect `s` for a plateau if the spectrum may be flat.

**Preconditioners** (`jno.precond`, for the iterative solvers): `jacobi`, `chebyshev`,
`nystrom` (randomized low-rank — the rung between `jacobi` and multigrid),
`amg` (algebraic multigrid), `gmg` (geometric multigrid — a structured-grid V-cycle),
`ams` (H(curl) auxiliary-space Maxwell), `form` (weak-form auxiliary operator),
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
