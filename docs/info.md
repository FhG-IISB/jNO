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

It takes a `domain`, a `jno.fem` form, a `jno.fdm` or `jno.rcwa` solver, a `jno.core`, or any
`jno.solve` / `jno.precond` spec. `Info.as_dict()` returns the same content as data, so a test can
assert on structure rather than on formatted text.

Two rules it keeps:

* **Cheap by default.** Anything costing an assembly or a dense factorisation sits behind
  `deep=True`. A default `info` is reads and arithmetic on what is already built.
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

## Registering another type

```python
jno.info.REGISTRY["MyThing"] = lambda obj, deep: jno.info.Info("my thing", [("", [("k", 1)])])
```

Keyed by class name, so a module that lives elsewhere can register itself at import time.
