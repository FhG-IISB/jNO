# Domain decomposition

Solve **part** of a domain with one method and the rest with another, coupled at the interface. The
motivation is rarely parallelism in jNO — it is that different regions want different discretisations:
a boundary layer that needs FEM next to a bulk that a finite-difference stencil handles fine, a
subdomain where you only have a trained network, or a material interface across which the coefficient
jumps.

The coupling is a **fixed point**: each subdomain solves its own problem with the neighbour's data as
boundary data, and the exchange repeats until the two agree. jNO drives that loop for you, and — the
part that matters for inverse problems — differentiates *through* the converged fixed point rather than
through the sweeps.

!!! abstract "The whole page in one snippet"
    ```python
    d.region("A", boxA)                     # name the regions
    d.region("B", boxB)

    a = jno.fdm([-aa.d2(xa) - aa.d2(ya) - fa, u(xb, yb) - 0.0])   # each subdomain's own problem,
    b = jno.fem([ub.x * vb.x + ub.y * vb.y - fb * vb, ...])       # authored exactly as it would be alone

    sol = jno.core([a, b]).solve()          # jno.core couples them
    ```
    No driver object, no interface bookkeeping: a subdomain problem is an ordinary `jno.fdm` /
    `jno.fem`, and the region comes from the coordinates its PDE was written on.

## The public entry — `jno.core([...])`

Write each subdomain as the problem it is, on coordinates taken from a **named region**. `jno.core`
recognises the subdomain solves, infers each region from those coordinates, and couples them:

```python
d = jno.domain(boxA.union(boxB), mesh_size=0.06)
d.region("A", boxA)
d.region("B", boxB)

xa, ya, _ = d.variable("A", split=True)          # region A's nodes
xb2, yb2, _ = d.variable("B", split=True)
xb, yb, _ = d.variable("boundary", split=True)   # the OUTER boundary, shared

u = d.unknown()
aa, ab = u.bind(x=xa, y=ya), u.bind(x=xb2, y=yb2)

a = jno.fdm([-aa.d2(xa) - aa.d2(ya) - fa, u(xb, yb) - 0.0])   # PDE on A + outer BC
b = jno.fdm([-ab.d2(xb2) - ab.d2(yb2) - fb, u(xb, yb) - 0.0])  # PDE on B + outer BC

a.region, b.region        # 'A', 'B' — inferred from the PDE coordinates, not declared
sol = jno.core([a, b]).solve()
```

!!! measured "It reproduces the monolithic solve"
    Two overlapping halves of the unit square (`x ≤ 0.6` and `x ≥ 0.4`, mesh 0.06, 416 nodes),
    `−Δu = f` with the manufactured `u* = sin(πx)sin(πy)`:

    | | |
    |---|---|
    | coupled vs **monolithic single-mesh solve** | **1.84e-08** |
    | overlap jump at convergence | 3.59e-08 |
    | Schwarz iterations | 11 |
    | monolithic vs the exact `u*` | 1.55e-02 (the discretisation error, unchanged) |

    That first row is the claim worth testing: the decomposition must not change the answer, only how
    it was computed. The discretisation error is what it always was.

## The explicit driver — `jno.dd.couple`

`jno.core` is the front door; `jno.dd.couple` is the same machinery with the regions passed by hand,
which you want when the region is not a named tag or you want the convergence report:

```python
from jno.dd import couple

sA, sB = jno.Shape.rect(0.0, 0.0, 0.6, 1.0), jno.Shape.rect(0.4, 0.0, 1.0, 1.0)
sol, info = couple([(a, sA), (b, sB)]).solve(tol=1e-7, max_iter=60, return_info=True)

info    # {'mode': 'overlap-Schwarz', 'iterations': 11, 'overlap_jump': 3.59e-08,
        #  'interfaces': {'count': 0, 'flux': 0, 'value': 0}}
```

The region is a `jno.Shape`, resolved to a node subset by the analytic, shapely-free
[`Shape.contains`](Domain-and-Geometry.md) — 2-D **and** 3-D.

## Two coupling modes, chosen by the geometry

jNO reads the mode off the regions rather than asking:

| regions | mode | exchange |
|---|---|---|
| **overlap** | `overlap-Schwarz` | each subdomain pins its complement to the neighbour's field |
| **share a line** (partitioning tags) | `line-DN` | Dirichlet–Neumann: value one way, flux the other |

`info["mode"]` reports which one ran.

## Heterogeneous by construction

Nothing requires both sides to be the same method — that is the point. Each subdomain is whatever
solves it best:

```python
femL = jno.fem([uif.x * vif.x + uif.y * vif.y - f(xL, yL) * vif, uf(xb, yb) - 0.0])   # FEM on the left
fdmR = jno.fdm([-uiR.d2(xR) - uiR.d2(yR) - f(xR, yR), u(xb, yb) - 0.0])               # FDM on the right

sol = jno.core([femL, fdmR]).solve()
```

!!! warning "A PINN region is demonstrated, not driven"
    Coupling an exact solve to a **trained network** on the neighbouring region works — the FEM reads
    the net's interface values as its Dirichlet data, and the net is warm-started each round against
    its own PDE plus the FEM's interface values — but it is an explicit alternating loop in
    `tests/test_domain_decomposition.py`, **not** something `couple([...])` drives for you today.

    And it converges to the **network's** accuracy floor, not the solver's: a few percent, with the
    overlap jump going *noisy* once it gets there, because a half-trained network injects noise into
    the exchange. The gate that test asserts is that the coupling drives the jump down by a large
    factor and the combined field is a valid few-percent solution — not the 1e-08 monolithic
    equivalence the solver-to-solver cases reach.

## Declaring the interface yourself

Naming two regions and **then meshing** auto-creates a first-class `interface_<A>_<B>` tag — every
mesh node on the line where they meet — alongside `boundary` / `interior`. The order matters: the tag
is built during `build_mesh`, so declare the regions first. `interface_R_L` is an alias for the same
nodes, so you need not remember which way round you named them.

You can then write the coupling **in the same jNO syntax as everything else** and put it in the
`jno.core` list:

```python
d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
d.region("L", regL)
d.region("R", regR)
d.build_mesh(mesh_size=0.05)          # the interface tag appears here

xif, yif, _ = d.variable("interface_L_R", split=True)
nrm = d.variable("interface_L_R", normals=True)

uL_if, uR_if = uf.bind(x=xif, y=yif), u.bind(x=xif, y=yif)
value_cond = uL_if - uR_if              # value continuity  (reads like a periodic tie)
flux_cond  = uL_if.d(nrm) - uR_if.d(nrm)  # flux continuity

sol = jno.core([femL, fdmR, value_cond, flux_cond]).solve()
```

A residual that references an `interface_*` tag is recognised and **classified** — one carrying a
normal derivative is a flux condition, one without is a value condition — so `info["interfaces"]`
comes back as `{'count': 2, 'flux': 1, 'value': 1}`. Declaring them routes the coupling from what you
wrote instead of only from the geometry.

## Differentiable through the fixed point

This is what makes an **inverse** domain-decomposition problem ordinary. When a subdomain carries a
trainable `jno.np.parameter`, `.solve()` returns a differentiable trace node exactly as `fem.solve()`
does, and the gradient flows through the *converged* Schwarz fixed point via `jax.lax.custom_root` —
never through the unrolled sweeps:

```python
kL = jno.np.parameter((1,), name="kL")          # conductivity to recover, in the FEM region
kx = jnn.where(xi < 0.5, kL, kR)
femA = jno.fem([kx * (uif.x * vif.x + uif.y * vif.y) - fsrc * vif, uf(xb, yb) - 0.0])
fdmB = jno.fdm([-kR * (uiB.d2(xi) + uiB.d2(yi)) - fsrc, u(xb, yb) - 0.0])

node = couple([(femA, sA), (fdmB, sB)]).solve(tol=1e-9, max_iter=300)
jno.core([(node - u_obs).mse]).solve(epochs)     # recovers kL THROUGH the coupling
```

!!! note "Why the coefficient case is the one that is tested"
    A parameter in the **source** would move the right-hand side only; one in the **coefficient** is
    re-assembled into `A(θ)` on every solve, which is exactly what a source-only gradient silently
    misses. `tests/test_domain_decomposition.py` asserts the coefficient gradient against finite
    differences (to 1%) for both the overlap and the line-DN coupling.

## The low-level handles

The driver is built out of two pieces you can also use directly:

| handle | what it is |
|---|---|
| `fem.pinned_solver(node_ids)` | a reusable `f(values) -> field` solving with `node_ids` pinned to `values`. The matrix is **prefactored once** and each exchange only re-solves against a new right-hand side |
| `fdm.pinned_solver(node_ids)` / `fdm.solve_pinned(node_ids, values)` | the same for a strong-form subdomain; the reusable form is built once and JIT-compiled, so the Schwarz loop does not recompile per iteration |
| `fem.region` / `fem.region_geometry` | the region a subdomain problem owns — the name, and its geometry, which is how `jno.core` recognises a subdomain solve |

Reach for them when you want a Schwarz variant jNO does not drive — a different exchange order,
multiplicative rather than additive sweeps, or a coupling to something outside jNO entirely.

## Limits

- The subdomains live on **one shared mesh**. Independent per-subdomain meshes are the
  [tied-interface](fem/geometry.md) route instead.
- Overlap coupling converges linearly in the **overlap width**: a thin overlap needs more iterations.
  `info["iterations"]` tells you what it took. The forward sweeps are **multiplicative** (each
  subdomain sees the neighbour's *just-updated* field, which is the faster of the two); the residual
  the implicit differentiation is built on is the **additive** map, which has the same fixed point
  and is the one that transposes cleanly.
- The differentiable path needs the fixed point to *converge*: a solve stopped by `max_iter` has no
  fixed point to differentiate, so tighten `tol` before trusting a gradient.
