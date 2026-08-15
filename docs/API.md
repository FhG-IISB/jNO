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
        - tag
        - by_region
        - by_tag
        - attached
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
| **Eigenproblem** | `eigs` (generalized `Kx = λMx`) — dense reduction, preconditioned LOBPCG with `precond=`, or interior modes nearest a shift with `sigma=` |
| **Singular values** | `svd` (partial SVD of a **rectangular**, matrix-free operator — POD bases, inverse-problem ill-posedness) |
| **Matrix functions** (stochastic Lanczos, matrix-free) | `logdet`, `trace`, `applyfun` (`f(A)·v`), `diagonal` |
| **Time integration** | `theta` (θ-method), `exponential` (exponential integrator), `adaptive` (step-doubling adaptive step size) |

### Eigenproblems at scale

`jno.solve.eigs` / `FEM.eigs` have three paths, chosen by the arguments. With none of the iterative
arguments the pencil is reduced **densely** — exact, and right when you want the whole low spectrum of
a small problem, but it materializes the operator (`O(N²)` memory). Passing `precond=` selects
**preconditioned LOBPCG** (Knyazev, *SIAM J. Sci. Comput.* **23**(2), 517–541, 2001), which only applies
`K`/`M` as matvecs and so runs where the dense reduction cannot. Passing `sigma=` targets the `k`
eigenvalues **nearest the shift** — interior modes (a cavity resonance inside a band, a Brillouin-zone
point away from the band edge), which no extremal-end iteration can reach — by shift-invert block
subspace iteration (Ericsson & Ruhe, *Math. Comp.* **35**, 1980; Bathe & Wilson, *IJNME* **6**, 1973):
`θ = 1/(λ−σ)` makes the near-σ modes dominant with enormous transformed gaps, so the transformation is
its own preconditioner and `precond=` is rejected there. The inner solves against `K − σM` default to a
host sparse LU **factorized once** (every sweep is then triangular substitutions); `linear=` swaps in a
different inner solver when a factorization is too big. Constrained pencils (Dirichlet pins, periodic
ties) compose: the reduced `K − σM` is assembled sparsely through the same triplet remap the periodic
solve reduction uses.

`K` is the **source-less** `jno.fem` whose bilinear form is the stiffness; `mass=` takes the mass form
as a plain term list, which `eigs` assembles onto the same space for you:

```python
u, v = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)

K = jno.fem([ui.x * vi.x + ui.y * vi.y])                       # stiffness — no source term
lam, X = K.eigs(mass=[ui * vi], k=6)                           # dense;  λ = ω² on a Neumann box
lam, X = K.eigs(mass=[ui * vi], k=6, precond=jno.precond.amg())  # LOBPCG, never densified
lam, X = K.eigs(mass=[ui * vi], k=4, sigma=60.0)               # the 4 modes nearest λ = 60
```

`jno.solve.eigs(...)` is the lower-level form of the same thing and takes two **assembled operators**
rather than a fem and a term list — `jno.solve.eigs(k=6)(K.operator[0], M)`.

**Sweeps warm-start.** `X0=` seeds the LOBPCG block with eigenvector guesses in the full DOF space
(restricted through any constraint elimination for you) — the classic sweep accelerator: a
parameter/frequency/k-point sweep passes each point the previous point's eigenvectors, so the
iteration only tracks the drift instead of re-finding the subspace from random. Fewer columns than
`k` are padded with the seeded random block. `X0=` is rejected on the dense path (it would be
silently ignored) and not yet wired into `sigma=` (the transformation converges from random in a
handful of sweeps anyway):

```python
lam, X = K.eigs(mass=mass, k=6, precond=jno.precond.jacobi())            # first sweep point
lam2, X2 = K2.eigs(mass=mass, k=6, precond=jno.precond.jacobi(), X0=X)   # next point: warm
```

The Rayleigh–Ritz runs in the **M-inner product**, so an ordinary FEM form's consistent (non-lumped)
mass matrix is handled directly, and `XᵀMX = I` holds on both paths. Eigenvalues are differentiable on
both — for **simple** eigenvalues; a degenerate cluster makes `∂λ/∂θ` ill-defined either way (use the
trace of the cluster). LOBPCG freezes the converged eigenvector and differentiates the Rayleigh
quotient, which gives that derivative exactly without differentiating through the sweeps, but its
**eigenvectors** carry no gradient where the dense path's do.

`tol`/`maxiter` tune the iterative paths and are **rejected** without `precond=` or `sigma=`, so a
tolerance can never be silently ignored by the dense path. Do not set `tol` near machine precision: on
an ill-conditioned pencil the residual floors well above it (≈`4.4e-8` on a singular all-Neumann
Laplacian with `cond(K) ≈ 2e16`), and a tolerance below that floor burns the budget and
**NaN-poisons** the result — which is the deliberate contract for an exhausted budget, never a quietly
under-converged spectrum. The shift-invert gate measures the **original pencil's** residual of the `k`
returned pairs (a θ-space gate would flatter it), and a shift landing exactly ON an eigenvalue makes
`K − σM` singular — the garbage its factorization produces fails the same gate; perturb σ off the
eigenvalue.

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

### What runs compiled

A slot-composed solve runs as **one compiled program** where it can, rather than calling the Krylov
iteration from eager Python and paying dispatch on every step. How much that is worth depends on the
device: the eager cost is host-bound and so barely varies between machines, while the compiled cost
is device-bound — the faster the GPU, the larger the ratio. On an RTX 3070 at 13759 DOFs,
`bicgstab + jacobi` 114.1 ms → 18.1 (6.3x), `cg + jacobi` 97.5 → 14.5, `minres + jacobi` 115.2 →
20.2, `gmres + jacobi` 398.4 → 183.2, `fgmres + jacobi` 536.3 → 300.8 (1.8x — jNO's own restart loop
does more real arithmetic, so less of its time was dispatch). On CPU, 1.8–4.2x; on a faster GPU,
`bicgstab + jacobi` reached 16.6x. Same answers throughout. Nothing to switch on; write the
slots as usual, and
write them inline if you like (`fem.solve(linear=jno.solve.cg(), precond=jno.precond.jacobi())`) —
equivalently configured specs share one compilation, so a solve in a loop compiles once.

These combinations stay **eager**, and are correct but not accelerated:

| Slot | Why |
| --- | --- |
| `solve.chebyshev`, `precond.chebyshev` | measures spectrum bounds, then branches on what it measured |
| `precond.amg` **unbuilt**, `precond.ams`, `precond.form` | assembles an auxiliary operator host-side (scipy / pyamg) |
| `precond.jaxamg`, `solve.amg` | AmgX builds its hierarchy from the matrix values; unverified under a tracer |
| `solve.lu`, `solve.dense`, `solve.amg` | one direct call — no per-iteration dispatch to remove |
| a bare callable in either slot | jNO knows nothing about it, so it makes no assumption |
| a multi-device (`shard=`) solve | already compiles itself, with the operator partitioned |

### First-run compilation cost

Compiling is not free the first time. Building a 13.8k-DOF 2-D Poisson problem issues ~209 XLA
compilations totalling ~3.7 s — assembly evaluates reference-element expressions whose every distinct
operation and shape is its own small program. The cost is paid **once per distinct mesh shape**, and
it is cached within a process: rebuilding the same problem costs ~320 ms, and rebuilding with freshly
constructed term objects costs the same (the cache is keyed on shapes, not on object identity). A
*different* mesh pays it again, so a remeshing loop (`fem.adapt`, a refinement study) pays it per
iteration.

Across **separate processes** — running a script twice, a test suite, CI — nothing is reused by
default. JAX can persist compilations to disk, which takes the same build from 4757 ms to 1143 ms
(measured, 214 entries, 932 KB):

```python
import jax
jax.config.update("jax_compilation_cache_dir", "~/.cache/jax")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)  # REQUIRED here
```

The second line is not optional for this workload: the default threshold is 1.0 s and every one of
these compilations is far below it, so with the cache directory alone nothing is ever written. jNO
does not set either of these for you — a library should not start writing to a user's disk uninvited.

### When AMG is worth it

`jno.precond.amg` has the best asymptotics on offer: Jacobi's iteration count grows as `√n` (79 → 166
→ 288 at n = 3k → 12k → 47k on a 2-D Poisson), AMG's is `O(1)`. **Build the hierarchy once and reuse
it** — an unbuilt spec re-runs pyamg's host-side setup on every solve, and stays off the compiled path
because that setup cannot be traced:

```python
M = jno.precond.amg().build(fem.operator[0])       # or jno.precond.amg().cached()
u = fem.solve(linear=jno.solve.cg(tol=1e-10), precond=M)
```

Per solve, against `cg + jacobi` at the same tolerance — the advantage grows with the problem, which
is the `√n`-vs-`O(1)` law showing up directly:

| DOFs | cg + jacobi | cg + amg (built) | | setup | break-even |
| --- | --- | --- | --- | --- | --- |
| 3,013 | 6.4 ms | 4.4 ms | 1.5x | 135 ms | 66 solves |
| 18,289 | 13.6 ms | 6.6 ms | 2.1x | 317 ms | 45 solves |
| 46,677 | 31.5 ms | 10.0 ms | 3.2x | 303 ms | 14 solves |
| 95,061 | 80.3 ms | 16.4 ms | **4.9x** | 419 ms | **7 solves** |

So AMG is for **repeated solves against the same operator** — a transient run, a parameter sweep, a
Newton loop — where the setup amortises. For a single one-shot solve below ~100k DOFs, Jacobi still
wins on wall clock: you would pay 419 ms of setup to save 64 ms. This is why AMG is not the default;
the right choice depends on how many times you solve, which only you know.

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
