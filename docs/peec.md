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

### Composing a layout — `+` is not union

Two different operators, and mixing them up is the first thing that bites:

| | |
|---|---|
| `a \| b`, `a - b`, `a & b` | **CSG**, within one conductor — an L-shaped trace, a hole through a plane |
| `a.name(..) + b.name(..)` | **region composition** — separate conductors, each with its own `sigma` |

A whole layout is therefore CSG for each part, `+` between them, and the cell pitch goes on the
**Shape** (`size=`), not on `.domain()`:

```python
mm, P = 1e-3, 0.5e-3
lower = (jno.Shape.box(0, 0, 0, 30*mm, 6*mm, 0.5*mm, size=(P, P, P))
         - jno.Shape.cylinder(12*mm, 3*mm, -1*mm, 0, 0, 3*mm, r=1.2*mm)).name("lower")
via   = jno.Shape.box(26*mm, 2*mm, 0.5*mm, 29*mm, 4*mm, 1.5*mm, size=(P, P, P)).name("via")
upper = (jno.Shape.box(20*mm, 2*mm, 1.5*mm, 40*mm, 4*mm, 2.0*mm, size=(P, P, P))
         | jno.Shape.box(36*mm, 2*mm, 1.5*mm, 40*mm, 10*mm, 2.0*mm)).name("upper")

d = (lower.attach(sigma=CU) + via.attach(sigma=CU) + upper.attach(sigma=AL)).domain()
d.tag("IN",  lambda x, y, z: x < 0.6*mm)
d.tag("OUT", lambda x, y, z: (y > 9.4*mm) & (z > 1.4*mm))     # a pad on the OTHER layer
```

Two layers, a via between them, a cut-out, an L-shape and two materials — the via conducts because
the geometry says the metal touches, not because anything declared a connection.

!!! warning "One lattice, one pitch"
    Every solid voxelises onto the **same** grid, so the pitch has to resolve the thinnest feature in
    the whole layout. A 35 µm foil beside a 3 mm busbar is expensive for that reason, not because
    either is hard on its own. Wires (`Shape.line`) are exempt — they carry their own analytic
    cross-section and weld to whatever they land on.

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
.attach(sigma=(sx, sy, sz))                       # ANISOTROPIC — and each of the three may
                                                  # itself be any of the above
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

!!! info "Anisotropy is diagonal, and a wire projects it"
    The bars are axis-aligned, so an off-diagonal conductivity has nowhere to live in this
    discretisation — `(sx, sy, sz)` is the whole vocabulary. A `Shape.line` is one-dimensional, so
    what reaches it is the component along its own tangent, `t · σ · t`, which is the physically
    right answer rather than a refusal: a transverse conductivity cannot drive current along a
    filament. A `(3,)` array on a conductor with exactly three elements is ambiguous and raises.

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
| `sol.field(points)` | magnetic flux density at `(n, 3)` positions, tesla — see below |
| `sol.export_vtk(path)` | the solved currents as line cells, for ParaView |
| `sol.partial` | the dense `(n, n)` partial inductances — **formed on demand**, never at solve time |

`L` comes from the energy rather than `Im(Z)/ω` so that it is defined at DC too, and so that it stays
the loop inductance the currents actually produce: at a frequency where they redistribute, that is a
different (smaller) number than the DC one, which is the effect worth seeing.

### The field off the metal

A partial-element method never meshes the air, so the field away from the conductors is not a solved
unknown — it is a Biot-Savart sum over the currents that **were** solved for:

```python
B = sol.field(probe_points)          # (n, 3) tesla, free space
```

It is a readout, not a second problem: no boundary condition, and differentiable in the currents
**and** in the points — which is what an EMI objective over a keep-out volume needs.

!!! warning "Free space, and off the metal"
    There is no magnetic material in this solver, so a nearby core would change the answer and is not
    represented. A probe inside a conductor's own cross-section is **refused**: the kernel is singular
    there, and the field inside the metal is not what this computes. (A point on the *axis* of a
    straight filament is not the dangerous case — the field vanishes there by symmetry.)

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

## A conductor with no terminal

A body no port touches — a ground plane under a trace layer, a shield, a floating heatsink — carries
only the currents the rest of the network induces in it. Its potential is undetermined (add any
constant and every current is unchanged), so `jno.peec` **pins one of its nodes automatically** and
says so:

```
peec: 1 conductor piece(s) carry no terminal, so their potential floats. One node of each has
been pinned as a reference. This removes a singular direction and changes no current, impedance
or loss -- the balance it replaces is implied by the others.
```

It changes nothing measurable: on an isolated body `1ᵀA = 0`, so the current balance it replaces is
already implied by the others. **Just include the plane** — no ground constraint of your own.

!!! tip "Ground it yourself only to say where zero is"
    If you do add `v(GND) - 0.0`, tag exactly **one** node. A multi-node terminal ties its nodes
    equipotential, which is a real short across that part of the conductor rather than a reference —
    worth 0.5 % on a module's loop inductance, measured.

## Stacked layers — one cell each

A uniform lattice has one pitch per axis, so **one cell per conductor through the thickness** needs
every layer to be `dz` thick *and* to start on a grid line. A real DBC stack (0.37 mm bottom metal,
0.60 mm ceramic, 0.57 mm traces) is tiled by no single `dz`.

Take `dz` from the layer whose thickness matters most and place the others an integer number of cells
away:

```python
T   = (3.65 - 3.08) * mm            # the trace copper -> dz
z1  = 3.08 * mm - T                 # the plane's top: one empty cell below the traces
plane = jno.Shape.box(x0, y0, z1 - T, x1, y1, z1, size=(P, P, T))
```

That gives plane / gap / traces — three cells, one per conductor, so the surface impedance is exact
at any frequency. The cost is a stated approximation to the geometry: the plane is modelled at the
trace thickness and its separation snaps to a whole cell.

!!! warning "It is worth the trouble"
    On a real half-bridge module at 1 MHz, omitting the ground plane gave **28.3 nH** against the
    module's own reference of 21.7 nH — **+31 %**. With the plane included on a snapped grid:
    **20.9 nH, −3.6 %**. A return plane is not a detail.

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
| the `O(N log N)` inductance | the `O(N²)` pair sum it replaced, to 1e-11 relative |
| a rectangular bar's partial inductance | Grover's rectangular-bar formula, `L = (μ₀l/2π)[ln(2l/(w+t)) + ½ + 0.2235(w+t)/l]` |
| a near-neighbour mutual between two cells | a volume Monte-Carlo of the Neumann double integral — wrong in a completely different way from a Gauss rule |

!!! warning "A lattice element is a cube, and the quadrature costs what that implies"
    Sub-points sample an element's **volume**: `quad` along its axis, `quad_t` across each transverse
    direction, so `quad · quad_t²` = **12** per element at the defaults. Sampling only along the axis
    leaves the cross-section a point, which over-counts every near-neighbour mutual — measured against
    the volume integral, **+1.2 %** when an element is 8× longer than it is thick, **+15.3 %** at a
    cube, **+48.8 %** at 8× shorter. That is a wrong *trend*, not just a wrong number: it survives a
    convergence study, because refining a lattice adds exactly the mutuals that are over-counted.

    The price is that the count rose from 3 to 12, and the **dense and welded** paths pay it
    quadratically — a welded module that solved in 27 s can now take minutes. The FFT lattice path is
    unaffected in `N`; only its one-off kernel build grows. Lower `quad_t` if a model is all wires,
    where a filament is thin by construction and gains nothing from transverse sampling.

## Cost — what to expect

Measured on one CPU core-set, a copper bar discretised as a plain lattice. `build()` is the host pass;
*first* includes the XLA compile; *warm* is a repeat solve on the built network.

| bars | `build()` | warm solve (`R`), CPU | `L` |
|---|---|---|---|
| 712 | 0.22 s | 0.04 s | 0.02 s |
| 6,688 | 0.19 s | 0.16 s | 0.02 s |
| 23,688 | 0.21 s | 0.46 s | 0.03 s |
| 57,472 | 0.52 s | 1.16 s | 0.04 s |
| 113,800 | 0.40 s | 2.03 s | 0.08 s |

Three things to read off it.

- **`build()` is cheap and flat.** The host pass is well under a second even at 57k bars. Building once
  and solving many times is the intended shape, and it is why `build()` exists.
- **The first solve is dominated by compilation**, not arithmetic — a few seconds against a warm solve
  of well under one. For a one-shot answer that is the number you feel; in a design loop it is paid once.
- **The solve is linear in the bars**, and the frequency does not change its cost.

!!! tip "The GPU is worth using, and was not always"
    The solve is matrix-free and the partial-inductance apply is an FFT, so it belongs on a GPU: at
    23,688 bars that apply is **3.2× quicker** there. It did not show up in the solve until the
    constraint block stopped being assembled one jax op per node — thousands of eager dispatches that
    cost the same whichever device they dispatch to, and hid the operator they were wrapped around.

    | 23,688 bars, warm solve | CPU | GPU |
    |---|---|---|
    | before | 0.781 s | 0.801 s |
    | after | 0.641 s | **0.317 s** |

    Note the CPU column moved too: this was never a GPU-specific problem, only a GPU-visible one.

    The second thing in the way was the preconditioner's sparse LU, re-run on every `solve()` — at
    40,000 nodes that was 1.5 s, **53 % of the whole solve**, all of it on the host. It is now cached
    on content, which is what makes the GPU pull ahead at size:

    | 113,800 bars (40,000 cells) | jNO CPU | jNO GPU |
    |---|---|---|
    | DC | 2.03 s | **0.92 s** |
    | 10 kHz | 2.12 s | **0.95 s** |

    The cache is keyed on the conductivity and the frequency, so a repeated solve and a gradient's
    adjoint pass are free — and a design loop that moves the conductivity every iteration is not.
    That miss is deliberate: reusing a factorisation built for a *different* conductivity is a claim
    about preconditioning, not a caching decision, and this does not make it.

!!! tip "`L` is a quadratic form, evaluated through the operator"
    `sol.L` used to walk every pair — `O(N²)` behind a solve that is linear — and cost **76 s at 57,472
    bars against a 2.3 s solve**, thirty times the answer it was reporting on. It now contracts through
    the same block-Toeplitz apply the matrix-free solve is built on, which is `O(N log N)`:

    | bars | 6,688 | 23,688 | 57,472 |
    |---|---|---|---|
    | pair sum | 1.43 s | 16.2 s | 76.5 s |
    | through the apply | 0.02 s | 0.03 s | 0.04 s |

    to the same value in every case. A network with no lattice — a polyline's filaments are not
    Toeplitz — keeps the pair sum, which is the honest path when there is no structure to exploit.

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
