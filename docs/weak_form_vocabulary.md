# Weak-form vocabulary

Everything you write inside `jno.fem([...])` is a **symbolic term** built from a small set of
primitives, composed by ordinary arithmetic. This page is a reference for *what you can already
write* — the declarative building blocks — so you reach for an escape hatch only when you truly
need one. A term's **region** is carried by the symbols you bind into it (no string kwargs): bind a
field to an interior/boundary tag and the term assembles there.

```python
d = jno.domain(box(0, 0, 1, 1), mesh_size=0.05)
u, v = d.fem_symbols()                       # trial, test
xi, yi = d.variable("interior", split=True)  # coordinates carry the region
ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])   # ∇u·∇v = f, Dirichlet
```

## Fields, coordinates, derivatives

| You want | Write |
|---|---|
| trial/test fields (scalar/vector, P1/P2, RT/Nédélec) | `u, v = d.fem_symbols(value_shape=(2,), order=2, space="RT", names=("u","v"))` |
| coordinates + region | `xi, yi, ti = d.variable("interior", split=True)`; bind via `u.bind(x=xi, y=yi, t=ti)` or `u(xi, yi)` |
| partial derivatives (any order) | `ui.x`, `ui.y`, `ui.t`, `ui.xy`, `ui.xx` |
| grad / div / curl / laplacian / hessian | `ui.grad(xi, yi)`, `ui.div(xi, yi)`, `ui.curl(xi, yi)`, `ui.laplacian(xi, yi)`, `ui.hessian(xi, yi)` |

## Algebra & nonlinear physics

Closed-form nonlinear physics is already symbolic — `+ - * / **` and the `jno.np` library compose
straight into a term:

```python
D = k * (1e-3 + ui.x**2 + ui.y**2) ** 0.3          # nonlinear diffusivity D(|∇u|)
react = ui * ui * vi                                 # u² reaction (nonlinear)
fem = jno.fem([ui.t*vi + D*(ui.x*vi.x + ui.y*vi.y) + react - f*vi, ...])
```

| Group | Primitives (`jno.np.*`) |
|---|---|
| elementary | `exp log sin cos tan sqrt cbrt abs sign square power` (and `**`) |
| **conditional / piecewise** | `where(cond, a, b)`, `maximum(a, b)`, `minimum(a, b)`, comparisons `u > 0` |
| vector / tensor | `inner(a, b, n_contract=)`, `dot cross outer trace sym`; vector `.norm() .dot() .cross()` |
| matrix / Voigt / complex | `MatrixView` (`.det .inv .eigvals .sym`), `VoigtView` (`.von_mises .deviatoric .invariants`), `ComplexView` (`.real .imag .conj`) |

## Integrals (local **and** non-local)

```python
energy = (ui.x**2 + ui.y**2).integrate()            # ∫_Ω |∇u|² dΩ  (scalar)
heat   = (u * v).integrate(ti)                       # ∫ u·v dt      (time window)

# non-local / Fredholm kernel: ∫ K(x,y) u(y) dy, returned per collocation point
x, _ = d.variable("interior"); y, _ = d.variable("interior")
Ku = (kernel(x, y) * u_of(y)).integrate(var=x)
```

`expr.integrate()` is a domain (or boundary) integral; `expr.integrate(var=x)` is the **non-local**
kernel form (one value per collocation point — Fredholm-type operators); `expr.integrate(t)` is a
time-window integral.

## Geometry symbols

| Symbol | Meaning | Use |
|---|---|---|
| `d.variable(tag, normals=True, split=True)` → `nx, ny` | boundary outward normal | flux / Robin terms `nx*ui.x + ny*ui.y` |
| `d.cell_size` | element size `h` (\|detJ\|^(1/dim)) at quad points | **SUPG/GLS stabilization** `τ = h/(2·|β|)` |
| `d.enclosure(tags)` | view-factor matrix + measures | grey-body radiation (`.view_factor`, `.field()`, `.load()`) |

```python
# SUPG-stabilized advection–diffusion, fully declarative:
h    = d.cell_size
beta = jno.np.vector(1.0, 0.5)
tau  = h / (2 * beta.norm())
adv  = beta[0]*ui.x + beta[1]*ui.y
fem  = jno.fem([adv*vi + nu*(ui.x*vi.x + ui.y*vi.y) - f*vi
                + tau * adv * (beta[0]*vi.x + beta[1]*vi.y), u(xb, yb) - 0.0])
```

## Coefficients & the escape hatch

| You want | Write |
|---|---|
| learnable PDE coefficient (inverse design) | `k = jno.np.parameter((1,), name="k")` |
| fixed coefficient / field / table | `jno.np.constant(...)`, `d.variable(tag, sample=array)` |
| **arbitrary JAX** not covered above | `jno.fn(lambda a, b: ..., [ui, vi])` → a traced node usable in any term |

`jno.fn` is the escape hatch: any differentiable JAX function of traced arguments becomes a term.
Reach for it only when the math isn't expressible from the primitives above — which, with
conditionals, non-local integrals, the geometry symbols, and the full tensor calculus, is rare.

## Introspection

`fem.term_kinds` (provisional) classifies each PDE term — `is_local` (pointwise reaction/mass) vs.
global (neighbour-coupling diffusion/advection), its temporal order, trial/test gradient channel,
and linearity — the basis for operator-splitting routing. See [`fem.md`](fem.md).
