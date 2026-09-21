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

Two rules it keeps:

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

```
  form
    mode           linear
    dofs           374
    saddle blocks  p
  terms (as classified)
    [0] volume   [1] volume   [2] dirichlet@left   [3] dirichlet@_gauge_pin_p
  field blocks
    field 1  dofs 0:326  (326) · value_shape (2,) · P2
    field 3  dofs 326:374  (48) · P1
  operator
    nnz               8,979  ·  fill 6.42e-02  ·  ~24.0/row
    dense equivalent  1.1 MB
    dtype             float64
    load ‖b‖          0   ← ALL ZERO
```

**`terms (as classified)` is the important one.** It names every term by how jNO recognised it, so a
boundary condition that landed nowhere is visible here rather than as a wrong answer later.

**Field-block order is the assembler's, not yours.** Above, the pressure is block *3* even though
the momentum term was written first — `offsets` indexes this order, so read it here before slicing a
solution vector.

`load ‖b‖ = 0` means the right-hand side is identically zero: the solve will return zeros with a
*perfect* residual. A missing source term looks exactly like a converged solve without this line.

With `deep=True` the operator section also reports `max|A - Aᵀ|` and the count of empty rows — an
empty row is a singular system, and knowing it before the solver says "may be singular/ill-posed"
saves guessing which condition is missing.

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
    shape  (224,)   range [-6.90236, 1.19927]   norm 36.2271
  by field block
    field 1  194 dofs · [-6.90236, 1.19927]
    field 3  30 dofs · [-1.05379, 0.524342]
```

`context=` splits the vector by the form's **own** `offsets`, which are the assembler's order and
not the order you wrote the terms in. An adaptive transient returns a trajectory instead, and that
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
