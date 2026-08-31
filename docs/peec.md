# Partial-Element Equivalent Circuits (PEEC)

`jno.peec` solves for the **current distribution in a conductor network** — resistance, loop
inductance, current sharing between parallel paths, and the ohmic loss that feeds a thermal model. It
is the method of choice for a busbar, a power-module layout, a planar transformer winding or a bond-wire
loop: geometry that is *mostly metal in mostly air*, where meshing the surrounding space would be the
whole cost.

Unlike [`jno.fem`](fem/index.md), a partial-element method **never meshes the air**. A conductor becomes
filaments carrying its centreline and cross-section, and the operator is Ruehli's Neumann double
integral over their pairs (Ruehli, *Inductance calculations in a complex integrated circuit
environment*, IBM J. Res. Dev. **16**(5), 470 (1972); *Equivalent circuit models for three-dimensional
multiconductor systems*, IEEE Trans. Microwave Theory Tech. **22**(3), 216 (1974)). Kirchhoff's laws
over those partial elements are the whole system.

!!! info "No optional extra"
    `jno.peec` is part of the core install. It needs no backend beyond JAX and SciPy.

## The front door — geometry, material, and what is impressed on the terminals

The input is a **constraint list**, exactly as [`jno.fem`](fem/index.md) takes one. The conductors and
their conductivity come from the geometry; the list says only what is impressed on the terminals.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jno

SIG = 5.8e7                                          # copper, S/m
L, W, T = 0.040, 0.004, 0.3e-3                       # a power-module trace, ONE cell thick
trace = (jno.Shape.box(0, 0, 0, L, W, T, size=(1e-3, 1e-3, T))
         .attach(sigma=SIG).name("trace"))

d = trace.domain()
d.tag("A", lambda x, y, z: x < 1.1e-3)               # the two pads, by position
d.tag("B", lambda x, y, z: x > L - 1.1e-3)

i, v = d.peec_symbols()
at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]

sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e5).solve()
print(f"R = {float(sol.R) * 1e6:.2f} uOhm    L = {float(sol.L) * 1e9:.2f} nH")
```

`d.peec_symbols()` hands back the two circuit quantities — a terminal **current** `i` and a nodal
**potential** `v` — and `at(...)` splats a tagged region into the coordinates a port is written on.

### Four constraint forms, and only four

| form | meaning |
|---|---|
| `v(A) - v(B) - g` | a **source**: `g` volts impressed across the port |
| `v(A) - g` | a **fixed potential**: `A` held at `g` |
| `i(A) - g` | a **fixed current**: `g` amps injected at `A` |
| `v(A) - v(B) - Z*i(A)` | a **two-terminal device**: Ohm's law, `Z` ohms between the pads |

The first three are the whole vocabulary of a port. The fourth is what puts a **component** in the
network without meshing it as metal:

```python
# a MOSFET's on-resistance, as a circuit element rather than as geometry
constraints = [
    v(*at("DCP")) - v(*at("DCN")) - 1.0,
    v(*at("MD")) - v(*at("MS")) - 5e-3 * i(*at("MD")),
]
```

!!! tip "A device parameter belongs in the circuit, not the geometry"
    Voxelise a 0.18 mm die on a 0.285 mm grid and its resistance follows the grid — which is both
    wrong and the reason the grid had to be fine. As a device it is exact and costs nothing.

## Conductors — lines and solids

A conductor is one of two things.

=== "A line — `jno.Shape.line`"

    A tube swept along a polyline: a bond wire, a via, a round conductor. Its filaments follow the
    centreline, and its cross-section is the tube radius.

    ```python
    wire = jno.Shape.line(
        [(0, 0, 0), (5e-3, 0, 2e-3), (10e-3, 0, 0)],    # the arc of a bond wire
        r=1.9e-4, size=1e-3,
    ).attach(sigma=5.8e7).name("wire")
    ```

=== "A solid — any closed-form shape"

    Voxelised onto a lattice: nodes at cell centres, elements the bars joining adjacent centres. Any
    shape with a closed-form `contains` works, not just a box — including CSG.

    ```python
    plate = jno.Shape.box(0, 0, 0, 0.02, 0.006, 1e-3, size=(1e-3, 1e-3, 1e-3))
    plate = plate.attach(sigma=5.8e7).name("plate")
    ```

Solids on the same lattice **share it**, and a line landing on a solid is **welded** where the metal
touches — so a bond wire arcing onto a trace carries current across the joint with no extra
declaration.

!!! warning "A lattice conducts centre-to-centre"
    Nodes sit at cell centres, so the conducting span of an `n`-cell run is `extent × (n-1)/n`, not
    `extent`. This is the standard convention (pypeec uses it too) and it converges as `1/n`, but on a
    coarse grid it is a visible offset rather than a rounding error.

## Conductivity is a design variable as often as it is a constant

`.attach(sigma=...)` takes three spellings, and a gradient flows back through all three:

```python
.attach(sigma=5.8e7)                              # a material
.attach(sigma=lambda x, y, z: SIG * rho(x, y))    # a FIELD: the density is the design
.attach(sigma=SIG * rho)                          # one value per element
```

A callable is evaluated at each element — cell centres for a solid's lattice, midpoints for a wire —
and its arity is positional, exactly as an attached FEM coefficient's is, so `lambda x, y` is a planar
field. That is the usual one: a trace is thin, and its material varies across the board rather than
through the 0.57 mm of it.

!!! tip "Prefer the callable to the vector"
    A callable says nothing about the pitch, so it survives a change of `size=`. A per-element vector
    cannot — and it must be the right length, which for a lattice is **one value per cell, not per
    bar**. A 40 × 4 × 1 cell trace has 160 cells and 276 bars; the vector is 160 long. A wrong length
    raises and names both numbers.

A field is what makes a **density (SIMP) topology optimisation** expressible here. What it does *not*
do is move the geometry: the lattice is fixed, so a cell whose conductivity goes to zero is still a
cell, still joined by bars, and still counted as metal by the thickness runs behind the skin term.
That is the ordinary fixed-mesh treatment, and it is why a converged density has to be **read back out
as a shape** rather than assumed to be one.

A conductivity may also be **traced**, which is what closes an electro-thermal loop — copper is about
31 % more resistive at 100 °C than at 20 °C.

## Frequency and the skin effect

Pass `freq=` a scalar or an array. Each filament carries **one** current, and its self term is the DC
geometric mean distance, so the skin effect **within** a filament is not resolved by subdivision — it
enters as an analytic **surface impedance** on elements that span the whole thickness.

That gives a hard rule, and `jno.peec` enforces it at `build()` rather than returning a wrong number:

!!! danger "One cell through the thickness, or thirty times finer"
    An element may take the surface impedance only when it **is** the whole thickness — there it is
    exact at any frequency. Otherwise the cells must resolve the skin depth (at least two per `δ`).
    Anything in between falls back to the DC resistance, and is **refused**:

    ```
    jno.peec: 712 of 712 elements sit in a conductor that is 2 elements thick where each is
    1 mm, against a skin depth of 0.06609 mm at 1e+06 Hz -- 30.3 skin depths through it.
    Neither model applies there: ... Use ONE cell through the thickness -- the surface
    impedance is exact there at any frequency -- or at least 30x finer so two cells fit in a
    skin depth.
    ```

    A power-module trace is 0.3 mm of copper and `δ` at 100 kHz is 0.21 mm, so **one cell through the
    thickness is the right model** for almost every layout problem — and it is also the cheap one.

## Freeze the geometry once — `build()`

`solve()` redoes all of the host work — which cells are metal, which nodes a pad owns, which filaments
weld to which — on every call. That differentiates, but it does not jit. A design loop wants that pass
done **once**:

```python
emag = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e5).build()   # host, once

@jax.jit
def loss(rho):
    return emag.solve(sigma={"trace": SIG * rho}).joule
```

What comes back solves in **pure JAX**, so it jits and composes into `jno.core` through `jno.fn`
exactly as `fem.solve()` does — which is what puts a PEEC objective behind `.optimizer()`,
`jno.optimizers.mma` and `jno.le`. Measured on a 1,540-bar plate: **42×** on the value and **23×** on
the gradient, because the host pass, not the solve, was the cost.

!!! warning "A built network freezes its GEOMETRY"
    A design variable may change a **conductivity**, never a **shape**. Build again for a new shape.
    The exception is the wire path — see below.

## Reading the solution

| readout | what it is |
|---|---|
| `sol.Z` / `sol.R` | terminal impedance of the source port, ohm |
| `sol.L` | loop inductance from the **field energy**, `Iᴴ Lp I / \|I_port\|²`, henry |
| `sol.joule` | ohmic dissipation at the solved excitation, `Σ R_k \|I_k\|²`, watt |
| `sol.dissipation()` | `{region: W/m³}` — per-conductor loss, shaped for `d.by_region` |
| `sol.current(t)` | net current injected at terminal `t`, amp |
| `sol.i` | the filament currents themselves |
| `sol.partial` | the dense `(n, n)` partial inductances — **formed on demand**, never at solve time |

`L` comes from the energy rather than `Im(Z)/ω` so that it is defined at DC too, and so that it stays
the loop inductance the currents actually produce: at a frequency where they redistribute, that is a
different (smaller) number than the DC one, which is the effect worth seeing.

### Feeding a thermal solve

`dissipation()` is shaped for `jno.domain.by_region`, which is how a per-region quantity enters a weak
form — and it is **jittable**, so an electro-thermal objective reaches `jno.core`:

```python
q = d.by_region(emag.solve().dissipation(), default=0.0)
heat = kappa * dot(grad(T, coords), grad(s, coords)) - q * s
```

## Solver controls

All three are optional and all three live on `solve()`.

```python
sol = built.solve(restart=48, matrix_free=None, devices={"MD": 6.1e-3})
```

`restart`
:   GMRES restart depth on the matrix-free path. The default of 16 is where the curve flattens for a
    few thousand elements, and **not** where it flattens for twenty thousand: on a real module at a
    0.7 mm pitch (21,980 bars) 16 leaves a 9.3e-06 residual where 1e-6 is wanted, while 48 converges —
    and finishes *sooner*, 176 s against the 290 s the shallower one spends failing. Raise it when the
    solve refuses.

`matrix_free`
:   `None` decides by structure: a network containing a bar lattice is applied by **FFT**, anything
    else forms the dense operator. `False` forces the dense path — exact, but `O(N²)` memory.

`devices`
:   `{terminal: Z}` overriding a device's impedance at solve time. A device value that **depends on
    the solved state** cannot be a constant in the constraint list; a SiC die's on-resistance rises
    about 0.5 %/K, so an electro-thermal fixed point has to re-impress it every pass.

!!! failure "A solve that does not converge raises"
    It never returns the unconverged iterate. The message names both levers — `solve(restart=48)` and
    `solve(matrix_free=False)`.

## Geometry as a design variable

Two things about the geometry *can* move under a gradient without rebuilding.

### A wire's route and gauge

`line_filaments(points=..., radii=...)` recomputes a wire's geometry in JAX from vertices that may be
traced, while the **structure** — how many filaments a segment is cut into, which endpoints are the
same node — stays as the reference geometry decided it. The lattice cannot do this: its occupancy is a
discrete in/out test per cell and the FFT needs a regular grid, so its nodes cannot move at all. Wires
can, and on a power module they are where the loop inductance lives.

Gauge matters for a reason that is not "thicker is better" (it is, trivially): given a **fixed total
cross-section**, which wires should carry it? A wire carrying no power current is spending copper for
nothing.

### A pad's position — weighted terminals

`solve(weights=...)` makes a terminal a prescribed current **distribution** over its nodes instead of a
short across them — one weight per node of its support.

This is what makes a terminal's **position** a design variable. Unweighted, a pad is an equipotential
node *set*, and which nodes are in the set is a step function of where the pad is: sliding a die a
quarter of a millimetre changes the answer by nothing at all, and then by 8 % when a node crosses the
boundary. Weighted, the support is a frozen superset covering the travel and the weights are smooth in
the position, so **the gradient exists** — the same structure-frozen, values-traced split as `sigma`.

!!! warning "Keep the support tight"
    The support is a frozen superset covering the whole travel, and its size drives the conditioning
    of the constraint block. A die that needs ±5 mm of travel should not be given a ±20 mm support:
    measured, widening supports to 120–168 nodes pushed the required `restart` from 16 to 160 and a
    single 11,000-bar solve to about twelve minutes.

## Validation

Every layer is checked against an oracle that does not go through the layer below it.

| what | oracle |
|---|---|
| self inductance of a straight round wire | `L = (μ₀l/2π)[ln(2l/a) − 3/4]`, to 2 % |
| a circular loop | the closed-form loop formula; convergence in quadrature order **and** segment count |
| DC resistance of a lattice | `span / (σ·W·T)` exactly, to machine precision |
| series / parallel / crossover networks | closed-form `ΣR + jωΣLp` and the DC conductance ratio |
| the FFT matrix-free apply | the **dense** operator built by `pair_matrix`, on every case |
| every gradient | a central difference, to ~1e-6 relative |
| the whole solve, end to end | **pypeec 5.8.0**, an independent PEEC code, on the same 20,480-cell grid: `R` agrees to 0.28 %, `L` to 0.08 % |

## Cost — what to expect

Measured on one CPU core-set, a copper bar discretised as a plain lattice. `build()` is the host pass;
*first* includes the XLA compile; *warm* is a repeat solve on the built network.

| bars | `build()` | first solve | warm, DC | warm, 10 kHz |
|---|---|---|---|---|
| 58 | 0.30 s | 1.4 s | 0.026 s | — |
| 6,688 | 0.44 s | 7.2 s | 0.21 s | — |
| 23,688 | 0.44 s | 17.9 s | 0.76 s | — |
| 57,472 | 0.52 s | 61 s | 2.15 s | **76 s** |

Three things to read off it.

- **`build()` is cheap and flat.** The host pass is well under a second even at 57k bars. Building once
  and solving many times is the intended shape, and it is why `build()` exists.
- **The first solve is dominated by compilation**, not arithmetic — 61 s against a 2.15 s warm solve at
  57k bars. For a one-shot answer that is the number you feel; in a design loop it is paid once.
- **AC on a plain lattice is the weak spot.** At 10 kHz the same 57k-bar network takes 76 s where DC
  takes 2.15 s — a factor of 35 on an identical operator size.

!!! warning "Why AC on a lattice is slow, and what does not fix it"
    A **welded** network is preconditioned by a sparse LU of the whole block system with the near field
    of `Z`; a **plain lattice** is preconditioned by a Schur complement of `diag(Z)` only. Each suits
    its own case and neither suits both — the near-field preconditioner does not converge on a lattice
    at all. On a lattice at AC, `diag(Z)` discards the inductive coupling that is the entire difficulty,
    and the Krylov iteration count is what you pay.

    `restart=` is **not** a lever here: measured at 57,472 bars and 10 kHz, restart 8 / 16 / 32 give
    76.3 s / 75.9 s / 79.7 s and bit-identical answers. Deepening the subspace does not help when the
    preconditioner is the bottleneck. A stronger lattice preconditioner is the open work.

## Limits, up front

- A conductor is a `Shape.line` tube **or** a closed-form solid. A CSG *plan* has no centreline and no
  cross-section, so it cannot be a line — build the conductor from `Shape.line`, or discretise it as a
  solid.
- Each filament carries **one** current: the skin effect within a filament is not represented. See
  [Frequency and the skin effect](#frequency-and-the-skin-effect).
- A **built** network freezes its geometry. A conductivity may be traced; a lattice shape may not.
- A network of wires alone has no lattice structure, so it forms the **dense** operator — that is the
  small-network path.
- There is no magnetic material and no dielectric: this is a conductor solver, not a full-wave one.
