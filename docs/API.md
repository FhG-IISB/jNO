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
        - attach
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
| `"spectral"` | FFT along the grid axes — exact for band-limited periodic fields; **uniform grid only** |
| `"spectral:cosine"` | even (mirror) extension — for fields with vanishing odd derivatives at both ends |

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

`diff_type` accepts a **whole scheme** as well as an AD sub-mode, so
`jno.setup(__file__, diff_type="spectral")` makes every unqualified derivative in the run spectral
— see [Operations → Spectral differentiation](operations.md#spectral-differentiation) for the
accuracy numbers and the periodicity caveat.

::: jno.differential_operators.DifferentialOperators

::: jno.integration_operators.IntegrationOperators

::: jno.utils.ad_mode

---

## Solvers and preconditioners

`jno.solve` and `jno.precond` are the slots that `fem.solve(linear=…, nonlinear=…,
precond=…, time=…)` composes (see the [FEM guide](fem/index.md)). The families:

| Kind | `jno.solve` |
| --- | --- |
| **Linear — direct** | `lu` (sparse LU), `dense` |
| **Linear — iterative (Krylov)** | `cg`, `bicgstab`, `gmres`, `fgmres`, `minres`, `cocg` (complex-symmetric); `lstsq` (LSQR, least-squares); `chebyshev` (polynomial) |
| **Linear — multigrid** | `amg` (GPU AMG / NVIDIA AmgX via jaxamg) |
| **Nonlinear** | `newton`, `picard` |
| **Eigenproblem** | `eigs` (generalized `Kx = λMx`) — dense reduction, preconditioned LOBPCG with `precond=`, or interior modes nearest a shift with `sigma=` |
| **Singular values** | `svd` (partial SVD of a **rectangular**, matrix-free operator — POD bases, inverse-problem ill-posedness) |
| **Matrix functions** (stochastic Lanczos, matrix-free) | `logdet`, `trace`, `applyfun` (`f(A)·v`), `diagonal` |
| **Time integration** | `theta` (θ-method), `exponential` (exponential integrator), `adaptive` (step-doubling adaptive step size) |

### Matrix functions — what `Ax = b` cannot express

`logdet`, `trace`, `applyfun` and `diagonal` touch the operator only through its matvec, so they scale
where a factorization cannot, and they are differentiable. They answer questions a linear solve
cannot: the Bayesian **log-evidence** of a FEM precision, an **effective-degrees-of-freedom** count, a
per-DOF **uncertainty map**, and one exact **exponential-integrator step**.

```python
A, _ = jno.fem([ui * vi + ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0]).operator

jno.solve.logdet(A, samples=64)                          # log det A          — log-evidence
jno.solve.trace(A, fun=lambda z: 1 / z, samples=64)      # tr(A⁻¹)            — effective DOFs
jno.solve.diagonal(A, fun=lambda z: 1 / z, samples=256)  # diag(A⁻¹) as a FIELD, plottable on the mesh
jno.solve.applyfun(A, u0, fun=lambda z: jnp.exp(-dt * z))  # exp(-dt·A)·u₀    — one exact step
```

!!! measured "Accuracy on a 198-DOF FEM precision operator (cond 45)"
    | quantity | estimate | exact | rel |
    |---|---|---|---|
    | `logdet` (samples=64) | 168.99 | 167.22 | 1.1e-02 |
    | `trace(1/z)` (samples=64) | 119.65 | 118.20 | 1.2e-02 |
    | `diagonal(1/z)` (samples=256) | — | — | 1.1e-01 (L2 over the field) |
    | `applyfun` exp step | — | — | **1.1e-15** |

    The first three are **stochastic** — Hutchinson probes plus Lanczos quadrature — so a percent or
    so is the expected accuracy, not a defect: variance falls with `samples`, bias with `order`.
    `applyfun` is **deterministic** (a Krylov approximation, no probes) and essentially exact.

    Differentiating works through the estimator: `d(log det cA)/dc` came back **152.198** against the
    closed form `n/c = 152.308`.

!!! danger "`order` must stay below the Krylov dimension — and a pinned FEM operator's is small"
    Lanczos can only build a subspace as large as the number of **distinct** eigenvalues the probe
    sees. A jNO FEM operator has far fewer than it has rows: every Dirichlet-pinned DOF is an identity
    row, so eigenvalue 1.0 carries the pinned count as its multiplicity.

    Measured on 2-D Poisson at mesh 0.25 — n=30, 16 pinned rows, only **15 distinct** eigenvalues —
    the default `order=25` overran that and `logdet` returned `NaN`. It now raises instead, naming the
    cause. The operator above is fine (198 rows, 151 distinct); a coarse mesh with a large boundary
    fraction is not. Lower `order`, or apply the estimator to the free-DOF operator.

    `applyfun` is **not** affected: its `order` is an upper bound, and it stops at the order that has
    actually converged (see below). The stochastic three have no such ladder, so `order` is a real
    request there.

!!! measured "`applyfun` picks its own order"
    `order` is an upper **bound** on the Krylov dimension, not an exact request — the same meaning
    `maxiter` has for every iterative solver here. Running past the dimension a problem supports used
    to degrade catastrophically and silently, because the Lanczos sub-diagonal does not collapse to
    zero as a textbook "happy breakdown" would; it **explodes** (0.27 → 2.2 → … → 184.7), the
    residual being pure round-off whose normalisation gives basis vectors of noise. On the 30-DOF
    operator above, `exp(A)·1` against a true 49.02:

    | `order` | before | now |
    |---|---|---|
    | 15 | 3.35e-15 | 3.26e-15 |
    | 20 | 1.44e-10 | **3.26e-15** |
    | 25 | **5.11e+35** | **3.17e-15** |
    | 29 | **1.85e+74** | **3.17e-15** |

    Note order 20 — this is not only a fix for the catastrophic end, it is *more accurate* wherever
    round-off has begun to contaminate the basis. The rule is the standard a-posteriori one (Saad,
    *SIAM J. Numer. Anal.* **29**(1), 1992, §4): accept the first order whose approximation agrees
    with its predecessor. Every nested approximation comes from the **same** decomposition, so it
    costs small dense eigendecompositions and **no extra matvecs** — measured overhead 0.17 ms at
    n=513 and 0.41 ms at n=8355.

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

### Large systems on a GPU

Past a few hundred thousand DOFs, what stops a GPU solve is rarely the matrix. Three things bite
first, and only the third is about size at all.

**Assembly runs on the host, automatically.** jNO's element loop allocates temporaries far larger
than the matrix it produces, so assembling on the device runs out of memory at roughly a third of the
size the finished operator would happily occupy. `jno.fem(...)` therefore builds on the CPU whenever
a GPU is the default backend, and the solve moves the operator across:

```python
fem = jno.fem([...])                     # assembled on the host, whatever the default backend
u = fem.solve(linear=jno.solve.fgmres(), precond=M)   # solved on the GPU
```

Nothing is given up for this. `jax.device_put` across backends is traceable and `jax.grad`
differentiates through it, so topology optimisation, neural coefficients and trainable coordinates
all still work across the split. Measured on a 9,970-DOF Poisson, assembly now costs **8 bytes** of
device memory — flat from n=242 to n=3,797 — against 104 MB before. Budget roughly **6 GB of host RAM
per million DOFs**; host memory, not the card, becomes the binding constraint.

It is not slower for a one-shot build, either — the host wins on every first call, because it also
compiles faster:

| | assemble on host | on device |
| --- | --- | --- |
| Poisson, 43k DOFs | **4.02 s** | 4.98 s |
| mixed N1E x Lagrange, 43k DOFs | **9.28 s** | 11.00 s |

The device does win on *repeated* assembly of a heavy form once compilation has amortised (5.60 s on
the host against 3.69 s on the device, mixed N1E at 43k DOFs), which matters in a Newton or
optimisation loop that reassembles many times. There is no jNO argument for this — `jax.default_device`
is JAX's own way of saying where work goes, and an explicit one wins:

```python
with jax.default_device(jax.devices("gpu")[0]):
    fem = jno.fem([...])                 # assemble on the device instead
```

**Set the async allocator.** XLA's default BFC allocator fragments and will refuse an allocation
about a gigabyte below the card's actual free memory — on a GPU shared with a desktop this is the
difference between running and not:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
```

A 2.70 GB operator that BFC rejected ran without complaint under `cuda_async`.

**Sizing.** A BCOO operator costs **16 bytes per nonzero** in `float64` (an 8-byte value plus two
`int32` indices), and peak device memory during a solve is about **1.3x** that, the extra being SpMV
scratch. Measured on an RTX 3070 (7.81 GiB usable, ~3 GB held by a desktop), on the fused real
`2n` block of a complex mixed N1E x Lagrange system at ~50 nonzeros per row:

| DOFs | nonzeros | operator | peak | SpMV | effective |
| --- | --- | --- | --- | --- | --- |
| 1,215,312 | 60.3 M | 0.90 GB | 1.24 GB | 7.9 ms | 114 GB/s |
| 2,216,184 | 113.3 M | 1.69 GB | 2.22 GB | 15.0 ms | 113 GB/s |

Bandwidth holds flat across sizes, so a Krylov solve costs about *iterations x SpMV*: budget from the
iteration count, not from the DOF count.

**`float32` is not a lever.** The same 60.3 M-nonzero operator takes **8.05 ms in `float64` and
9.09 ms in `float32`** — half the value bytes, and slightly slower. Indices stay `int32` either way
and the gather of `x` dominates, so halving precision buys about a quarter of the memory and none of
the time. Reach for a coarser mesh or a better preconditioner instead.

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

---

## Numerical-method front doors

Each takes a **term list** and returns an object carrying the assembled problem. The narrative
guides are [FEM](fem/index.md), [FDM](fdm.md) and [RCWA](rcwa.md); this is the signature-level
reference.

### `jno.fem`

::: jno._fem.fem

### The `FEM` object

::: jno._fem.FEM
    options:
      members:
        - solve
        - eigs
        - eval
        - residual
        - jacobian
        - operator
        - offsets
        - blocks
        - block_index
        - region_dofs
        - points
        - field_points
        - dofs
        - stats
        - is_linear
        - is_transient
        - is_complex

### `jno.fdm`

::: jno.fdm.fdm

### `jno.rcwa`

::: jno.rcwa.rcwa

::: jno.rcwa.Rcwa
    options:
      members:
        - solve
        - spec

::: jno.rcwa.RcwaSpec

#### The RCWA solution

Returned by `rcwa(...).solve()`. Every readout is a differentiable JAX array, so a transmission or
per-order objective can be optimised straight through the modal solve.

::: jno.rcwa._Sol
    options:
      heading_level: 5
      members:
        - efficiency
        - order
        - jones
        - field
        - field3d
        - aerial

::: jno.rcwa.RcwaError

---

## Geometry

::: jno.geometry.shape.Shape

::: jno.geometry.path.Path

---

## Optimizers (`jno.optimizers`)

Optax-compatible transformations. Anything optax exposes works too — these are the additions jNO
needs for PDE-constrained and topology-optimisation work.

::: jno.optimizers

---

## Bayesian inference (`jno.bayesian`)

Backs `model.bayesian(...)` / `model.vi(...)`; see [Bayesian Sampling](training/bayesian.md).

::: jno.bayesian
    options:
      members:
        - rhat
        - ess
        - priors
        - default_gaussian_prior
        - laplace
        - pathfinder
        - LaplaceInitializer
        - PathfinderInitializer
        - SVGDInitializer

---

## Noise nodes (`jno.noise`)

::: jno.noise._NoiseNamespace
    options:
      members:
        - gaussian
        - uniform
        - laplace
        - grf

---

## Units & non-dimensionalization (`jno.units`)

Annotate the dimension and characteristic magnitude of a leaf with `.unit(...)` / `.scale(...)`, then
audit consistency, extract the dimensionless groups (Fourier, Péclet, …) of a residual, and rewrite
it to a well-scaled `O(1)` form. Worked usage: [Operations → Units &
non-dimensionalization](operations.md#units-non-dimensionalization).

::: jno.trace.units
    options:
      members:
        - check
        - infer
        - nondimensionalize
        - rescale
        - Unit
        - Rescaler
        - NondimReport
        - UnitLogger

---

## Adaptive resampling (`jno.sampler`)

Residual-adaptive collocation strategies — see [Adaptive Resampling](adaptive/resampling.md).

::: jno.utils.adaptive.resampling
    options:
      members:
        - sampler
        - ResamplingStrategy
        - RAD
        - RARD
        - CR3
        - R3
        - PINNFluence
        - HA
        - RandomResampling

---

## Parameter-efficient fine-tuning (`jno.lora`)

Attached with `.lora(...)` on a wrapped model — see [LoRA](model-controls/lora.md).

::: jno.lora

---

## Training trackers (`jno.trackers`)

Diagnostics attachable as callbacks — see [Explainability](training/explainability.md).

::: jno.trackers

---

## Deployment (`jno.iree`)

::: jno.utils.iree.IREEModel
