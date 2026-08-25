# One-dimensional problems

A line domain is a first-class FEM domain in jNO, not a degenerate case: systems, arbitrary order,
coupled and mixed-order fields, parameters, neural coefficients, complex forms, variational PINNs,
adaptivity and the classical beam element all run on it. It is also the cheapest dimension to
prototype in.

!!! note "What a line domain does *not* support"
    The vector and triangle-only non-nodal families — `"RT"`, `"N1curl"`, `"Argyris"`, `"Morley"` —
    have no 1-D counterpart and raise a clear error on a line. Mesh motion via a geometry term is not
    wired either: a 1-D "moving boundary" is a single endpoint.

## Systems on a line

A 1-D line domain takes a **vector unknown** (`value_shape=(n,)`), so a 1-D *system* — a two-species
model, a Timoshenko pair, a bar with several dofs per node — is one field with node-major dofs and
per-component essential conditions (`u(region)[i] - g`).

## Order and vertex superconvergence

**Any order `k ≥ 1`** is available. Degree `k` adds `k-1` interior dofs per element, laid out after
all vertices, so read `fem.points` for the coordinates the solution actually lives on.

Orders above 2 are tabulated by basix on the reference interval through the same builder the 2-D/3-D
path uses, so a P*k* line and a P*k* triangle agree on what degree `k` means.

At the **vertices** 1-D Lagrange is superconvergent at O(h^2k), and each P*k* reproduces a degree-`k`
solution exactly. Measured on `-u'' + u = f`:

| element | interior rate | vertex rate |
|---|---|---|
| P1 | O(h²) | **O(h²)** |
| P2 | O(h³) | **O(h⁴)** |
| P3 | O(h⁴) | **O(h⁶)** |

## Coupled and mixed-order fields

A coupled 1-D system carries a **per-field order**, so a mixed-order pair (the 1-D Taylor–Hood shape)
assembles. The blocks are then unequal and the coupling blocks rectangular — read `fem.field_points`
for each field's dof coordinates.

!!! note "Algebraic fields make the block a DAE"
    In a coupled *transient* system a field may be **algebraic** (no `u_t`): its mass rows are zero,
    so the implicit step solves `A p = c` on those rows. That is how a constraint / closure field — a
    pressure, a saturation, an equilibrium concentration — is written.

## Parameters and neural coefficients

A `jno.np.parameter` coefficient (scalar or nodal field) works on a 1-D form, so a differentiable
inverse problem runs through `crux.solve` — as does a **neural** (`jno.nn(...)`) coefficient, so a
learned `k(x)` can be trained from 1-D data.

| 1-D form | parametric? | note |
|---|---|---|
| steady linear | ✅ | |
| steady nonlinear | ✅ | Newton on `R(·, θ)`; implicit differentiation gives `∂u/∂θ` |
| transient | ✅ | recovering a diffusivity from a time series works — **but see below** |
| coupled steady (linear & nonlinear) | ✅ | block kernels publish the same `volume_vars` / neural-table keys, so the shared evaluator reads them regardless of field layout |
| coupled transient | ❌ | fails loud |

!!! warning "Two things a parameter must not sit on"
    Both are assembled **once**, so a parameter there would be silently frozen at its placeholder.
    Both fail loud rather than doing that.

    - The **transient mass** — put the parameter on the stiffness / residual instead.
    - A **coupled transient block**.

## Variational PINNs

a network trial (`u = net(x)` inside the weak form, test-projected onto the FE
test space) works on a line, so the cheapest dimension for prototyping a variational PINN is
available. The native `fem_context` it projects onto now builds on an interval: `lagrange_interval`
is the 1D sibling of `lagrange_triangle`/`lagrange_tet` from the same basix builder, and an
interval's facets are its two endpoints (outward normals ∓1). Still single-field only.

## Adaptivity

`fem.solve(adapt=jno.solve.remesh(...))` works on a line, steady and transient.

Measured on a boundary layer `-eps u'' + u' = 0`, adaptive refinement from **11 dofs** reaches
**5.7e-4**, where uniform refinement at **81 dofs** is still at **1.2e-2**.

??? note "Why a line needs no mmg"
    mmg has no 1-D mode and needs none: an interval mesh is a sorted vertex list, so honouring a size
    field is **subdivision** rather than remeshing — exact where mmg is approximate, and with no
    optional dependency. mmg's `hgrad` gradation rule is imposed by two monotone sweeps.

    Solution transfer is the same code as 2-D/3-D: an interval is a 1-simplex, so its barycentric
    weights are the two linear hat values.

!!! note "Not wired in 1-D"
    Mesh motion via a geometry term — a 1-D "moving boundary" is a single endpoint.

## The beam element — `space="Hermite"`

the C¹ cubic Hermite, i.e. the classical
Euler-Bernoulli beam element and the 1D counterpart of Argyris/Morley on triangles. Two dofs per
vertex, `(w, dw/dx)`, laid out `2*node` / `2*node + 1`; sharing the slope dof across elements is what
makes the space C¹, so the fourth-order operator has a well-defined weak form:

```python
u, v = d.fem_symbols(space="Hermite")
ui, vi = u.bind(x=xi), v.bind(x=xi)
lap = jno.np.laplacian
fem = jno.fem([EI * lap(ui, [xi]) * lap(vi, [xi]) - q * vi,   # EI w'''' = q
               u(xl) - 0.0, u.dn(xl) - 0.0])                  # clamped at the left end
```

The classical supports are just *which* of a node's two dofs are pinned:

| Support | Terms |
|---|---|
| simply supported | `u(region)` |
| clamped | `u(region)` **and** `u.dn(region)` |
| guided | `u.dn(region)` alone |
| free | neither |

The slope condition rides the same `u.dn` essential-rotation channel the 2-D C¹ plate families use,
so a beam and a plate are clamped by the same notation.

!!! tip "Nodally exact for a uniform load"
    To machine precision: a cantilever gives `qL⁴/8` at the tip and `qL³/6` for the tip slope;
    simply supported gives `5qL⁴/384` at mid-span; clamped-clamped `qL⁴/384`.

## Complex forms

A `1j` anywhere in a 1-D weak form routes through the same real-equivalent Re/Im split the 2-D/3-D
and non-nodal paths use, so 1-D Helmholtz-type problems — complex coefficient, complex source, or
both — solve and return a complex `u`.

A runtime parameter inside the complex coefficient keeps **both legs** parametric, so the complex
**inverse** works in 1-D too.

!!! warning "Scope: steady, linear, single field"
    Each of these raises:

    - complex **transient**
    - complex **nonlinear**
    - a complex **coupled** 1-D system
    - a **complex essential value** — the two legs share one Dirichlet row set, which can impose
      `Re u = g` with `Im u = 0`, but not a prescribed `Im u`
