# `jno.info` — what is this, and is it what I meant?

Every jNO object knows a great deal about itself. Almost none of it used to be reachable without
knowing the attribute name to ask for, and what *was* reachable lived behind three different
spellings on three different classes. `jno.info` is the one front door.

```python
print(jno.info())                    # the environment: x64, device, memory, versions
print(jno.info(d))                   # a domain: mesh, quality, tags, attachments
print(jno.info(fem))                 # a form: mode, blocks, every term as classified
print(jno.info(fem, deep=True))      # + operator symmetry and empty rows
print(jno.info(jno.solve.newton(direct=True)))   # what a spec actually does
```

It takes the assembled things — a `domain`, a `jno.fem` form, a `jno.fdm` or `jno.rcwa` solver, a
`jno.core`, any `jno.solve` / `jno.precond` spec — and the **smaller parts they are built from**:

```python
print(jno.info(sh))                  # a shape, before .domain(): regions, bounds, the CSG tree
print(jno.info(pde))                 # an expression: which region it samples, its derivatives
print(jno.info(net))                 # a network: architecture, parameters, dtype
print(jno.info(d.variable("lid")))   # a variable: tag, points, extent, normals
print(jno.info(sol, context=fem))    # a result, split by the form's own field blocks
```

The small parts matter more than the assembled ones: the assembled object is where you find out
something was wrong, and the small ones are where it went wrong. `Info.as_dict()` returns the same content as data, so a test can
assert on structure rather than on formatted text.

Three rules it keeps:

* **Cheap by default.** Anything costing an assembly sits behind `deep=True` — and `deep` itself
  stays **sparse**: symmetry and the empty-row count are computed from the operator's indices, not
  from `A.todense()`, which at the 90,814 dofs of an ordinary 3-D solve would be 66 GB.
* **Bounded output.** Every section caps at `jno.info.MAX_ROWS` (24) and says how many rows it
  dropped. A device mesh can carry hundreds of tags; silent truncation would be worse than either.
* **Never force a lazy result.** A transient `solve()` returns a trace node on purpose; `info`
  reports that it is one rather than evaluating it.

## The environment — `jno.info()`

```
─── environment ──────────────────────────────────────────────
  build
    float64 (x64)  **OFF** — jNO assembles in float64; set
                   jax.config.update('jax_enable_x64', True) before the first array
    default dtype  float64 (numpy) / float32 (jax)
    backend        gpu
  devices
    [0] cuda:0  NVIDIA GeForce RTX 3070 (gpu)   ·   0 B / 5.7 GB used (0 %)
  versions
    jax 0.10.2 · jaxlib 0.10.2 · numpy 2.5.0
```

The first row is the one that silently ruins an answer. jNO assembles in **float64**; with `x64`
off, the stiffness matrix is built in float32 and every convergence rate in these docs stops
applying. It is a process-wide flag that must be set before the first array, so discovering it late
is discovering it too late.

The second is "am I actually on the GPU", which is not answerable from any other jNO output and
changes wall-clock by an order of magnitude.

## A domain

```
  mesh
    built     yes
    points    84
    triangle  136  ·  h 0.138–0.195 · worst aspect 1.24
    extent    [0, 2] × [0, 2]
  tags
    lid       8 points   [0, 2] × [2, 2]
  attached (d.<prop>)
    d.k  lo=5.0, hi=1.0   [volume]
    d.h  lid=25.0         [surface]
```

`worst aspect` is the Shewchuk length/inradius ratio, normalised so a **regular** simplex is `1.0`
— the same definition [`domain.cell_aspect`](fem/geometry.md) uses, so the report and an adaptivity
criterion cannot disagree. A cell with non-positive measure is reported separately as `INVERTED n`;
that is a correctness signal, not a quality complaint.

Tag extents are the check that `lambda x, y: y > 1 - 1e-9` caught the edge you meant. The
`attached` section is what `d.<prop>` will resolve to — see [`domain.attach`](fem/geometry.md).

## A form

A Taylor–Hood Stokes cavity (P2 velocity `u`, P1 pressure `p`, pinned):

```
  form
    mode           linear
    dofs           350
    saddle blocks  p
  terms (as classified)
    [0]  volume
    [1]  volume
    [2]  dirichlet@boundary[x]
    [3]  dirichlet@boundary[y]
    [4]  dirichlet@_gauge_pin_p
  field blocks
    u  dofs 0:306  (306) · value_shape (2,) · P2
    p  dofs 306:350  (44) · P1
  operator
    nnz               4,237  ·  fill 3.46e-02  ·  ~12.1/row
    dense equivalent  957.0 kB
    dtype             float64
    load ‖b‖          5.519
```

**`terms (as classified)` is the important one.** It names every term by how jNO recognised it, so a
boundary condition that landed nowhere is visible here rather than as a wrong answer later.

**Field-block order is first appearance — including inside a term.** Blocks are numbered in the
order each trial field first appears in the term list. Writing the momentum equation above as
`-p*trace(grad v) + inner(grad u, grad v)` instead makes `p` block 0 (dofs 0:44) and `u` block 1.
`offsets` indexes this order, so read it here before slicing a solution vector.

`load ‖b‖ = 0` means the right-hand side is identically zero: the solve will return zeros with a
*perfect* residual. A missing source term looks exactly like a converged solve without this line.

With `deep=True` the operator section also reports `max|A - Aᵀ|` and the count of empty rows — an
empty row is a singular system, and knowing it before the solver says "may be singular/ill-posed"
saves guessing which condition is missing.

### After a solve

Once the form has been solved, two more sections come from [`fem.stats`](solvers.md#diagnostics-what-the-solver-actually-did):
`last solve` (wall time, the slots, the nonlinear verdict, and `FAILED` with the error if the solve
raised) and, for a march, `march`:

```
  last solve
    wall       1.13 s  (first solve of this form: includes tracing/compilation)
    linear     default
    nonlinear  continuation/newton · residual 1.039e-09 / bound 1.000e-08  ✓
  march
    kind           continuation
    steps          5 over k ∈ [0, 2]
    time per step  median 75.5 ms · slowest step 1 (k=0) 840 ms
                   step 1 took 840 ms and includes tracing/compilation
    convergence    all 5 steps converged · tightest step 5 (k=2): residual at 10.4% of its bound
```

A load-path or transient march is one compiled `lax.scan`, so it reports a mean time per step and
says so, rather than per-step times it cannot see. A transient solve returns a deferred node: its
`march` section says *not run yet* until the node is evaluated with `.fn()`.

## The smaller parts

### An expression — `jno.info(pde)`

```
  reads
    regions    interior
    networks   MLP
    weak form  trial yes · test yes   ← a jno.fem term
  structure
    d/dt order        0
    derivative nodes  2
    tree size         11 nodes
```

**`regions` is the one to read.** A PDE residual accidentally bound to `boundary` instead of
`interior` produces a plausible, wrong answer and is invisible everywhere else. `deep=True` adds the
node tree.

Works on a bare IR node and on the `ScalarView` / `VectorView` wrappers a weak form is written in —
including a bound-but-underived term, whose coordinates live on the view rather than in the tree.

### A shape — `jno.info(sh)`

```
  geometry
    bounds     [0, 2] × [0, 2]
    meshed     no — call .domain() (jno.info on the domain then reports quality)
  regions
    lo  k=5.0
    hi  k=1.0
  CSG tree
    regions
      lo: leaf
        Rect
```

Checks what your booleans and `.attach` calls actually produced, without paying gmsh for it. An
unmeshed shape reports **no** mesh facts — `info` never builds something in order to have more to
say.

### A result — `jno.info(sol, context=fem)`

```
  array
    shape          (350,)
    dtype          float64
    range          [-9.64035, 10.2115]
    norm           22.7794
    rel. residual  1.403e-09   ‖Au − b‖ / ‖b‖ against the form's operator
  by field block
    u  306 dofs · [-0.246683, 1]
    p  44 dofs · [-9.64035, 10.2115]
```

`context=` splits the vector by the form's **own** `offsets` — first-appearance order, as above —
and labels each block with its trial field's name. It also computes the relative residual against
the form's operator. That costs a sparse matvec, which is why it is here, on request, and not in
the per-solve log line. An adaptive transient returns a trajectory instead, and that
reports frames, the time span, and the per-frame dof range.

## A Bayesian model — `jno.info(a)`

```
─── model · a ────────────────────────────────────────────────
    architecture  _Parameter      parameters  1
  inference
    method     bayesian · nuts
    warmup     100    keep  200    thin  1
    prior      default gaussian
    kernel step_size            0.01
    kernel inverse_mass_matrix  ArrayImpl shape (1,)
  posterior
    draws            shape (1, 200, 1)  (1 chain(s) x 200 draws)
    mean / sd        3.23926 / 1.04949
    R-hat (max)      0.9963   ✓ < 1.01
    ESS (min)        42.2   ← < 100 effective draws
    divergences      0 of 200   ✓ none
    acceptance_rate  mean 0.9064
```

**R-hat and ESS are the Bayesian answer to "did it converge"** — the analogue of the relative
residual on a deterministic solve. A chain that has not mixed is not a posterior, and nothing else
in the report would say so. Divergences matter separately: a divergence invalidates the draws
around it regardless of what R-hat says.

Note that `jno.np.parameter(...)` returns a `ModelCall`; `jno.info` unwraps a bare one to the model
behind it, so asking about a parameter gives you its inference settings rather than a one-node
expression. And a sampler is reported as a **sampler** — `training backend: a (MCMC sampler)` —
not as an optimizer.

## Domain decomposition — `jno.info(d)`, `jno.info(jno.core([a, b]))`, `jno.info(jno.dd.couple(...))`

A domain's `tags` section lists every `domain.region(...)` (with its bounds) and every
auto-created `interface_A_B` tag (with its node count). The reversed spelling `interface_B_A` is
shown as an alias. A region added after the mesh was built and sampled with no count is **one point,
redrawn every step** (Monte-Carlo mode), and its row says so: pass `sample=(n, None)` for a fixed set.

A `jno.core` whose items are subdomain solves reports them as solves, not as losses. The coupling
method comes from the same geometric test the driver uses (overlapping regions → Schwarz, a
partition → Dirichlet–Neumann on the line):

```
  subdomain solves
    [0]  fdm · on region A · owns 80 of 142 nodes · [0, 0.6] × [0, 1]
    [1]  fdm · on region B · owns 87 of 142 nodes · [0.4, 1] × [0, 1]
  coupling
    method                overlap-Schwarz  (the regions overlap)
    interface conditions  none declared — value continuity (and flux, across a line) is inferred
  training
    training backend  not needed — .solve() couples the subdomain solves (jno.dd)
```

Each subdomain solve spans the whole mesh with its complement pinned, so the useful count is how many
nodes each one **owns**. `jno.dd.couple(...)` reports the same, and after `.solve()` its last run:

```
  last solve
    iterations    8 of max 60
    overlap jump  4.840e-07 vs tol 1.0e-06  ✓
```

## An `rcwa` problem, and its energy check

```
─── rcwa (problem, unsolved) ─────────────────────────────────
  setup
    orders         9
    formulation    JONES_DIRECT_FOURIER
    period         (0.6, 0.6)      wavelength  1.0
    source face    bottom
  layers
    [0]  semi-infinite ambient  ·  eps 1
    [1]  thickness 0.3048  ·  eps 4
    [2]  semi-infinite ambient  ·  eps 1
```

and after `.solve()`:

```
  result
    efficiency T / R  0.815064 / 0.184936
    T + R             1.000000   ✓ energy conserved
```

**`T + R` is the oracle.** For a lossless stack it is exactly 1, so a value below it means either a
genuinely absorbing stack or too few retained `orders` — which is the one number that says whether
the truncation was enough. A patterned layer reports its permittivity *range* and grid shape rather
than the grid itself.

## When it does not know the object

`jno.info` never returns a blank report and never refuses a jNO object it has not met:

* **No handler** → a generic report: the class, the recognisable attributes, and every field the
  object carries with its type and shape.
* **A handler that finds nothing** → the same generic report, prefixed with
  *"the `<x>` handler found nothing — its attributes have probably moved"*.
* **A handler that raises** → the generic report, naming the exception.
* **A genuinely foreign object** (a `dict`, a string) → a `TypeError` listing what *is* handled.

The middle two exist because of how every handler in this file first failed. Each was written by
reading attribute names out of the source, and each was wrong — `core.models` is a dict keyed by
model id, the `rcwa` attributes live on `.spec`, a network inside an expression is a `ModelCall`
wrapping the `Model`. The shared symptom was a report that came back **empty and looked like a
finding**. A moved attribute is now visible rather than silent — and the generic dump is what
tells you where the attribute went.

## Reading the result as data

`Info.as_dict()` gives the same content as data. A section whose rows have unique keys is a
`dict`; a section whose rows share a key (a rendered tree, a note — every row keyed `""`) is a
**list**, because collapsing it to a dict kept only the last row and a four-node CSG tree reported
as one.

## Registering another type

```python
jno.info.REGISTRY["MyThing"] = lambda obj, deep: jno.info.Info("my thing", [("", [("k", 1)])])
```

Keyed by class name, so a module that lives elsewhere can register itself at import time.
