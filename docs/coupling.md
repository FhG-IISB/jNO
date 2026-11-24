# Coupling problems on their own meshes

`jno.core` couples `jno.fem`, `jno.fdm` and `jno.fvm` problems, each on its own mesh, through **interface
conditions written as for one problem**: a value that agrees on both sides, a flux that leaves one side and
enters the other, a load one side puts on the other. Each problem keeps its own discretisation, mesh, time step
and solver. The coupling solves them in turn until the interface conditions hold (a partitioned scheme, as
preCICE does).

```python
gas  = jno.fvm([... the gas ..., a slip wall on "liner", inflow, outflow, initial])
wall = jno.fem([... conduction and thermo-elasticity ..., outer convection, ...])
qg, Ts = q.bind(x=xg, y=rg), T.bind(x=xs, y=rs)          # each side on its own boundary of the interface
sol = jno.core([gas, wall,
    Tg - Ts,                                             # value: the temperatures agree
    kg * Tg.d(ng) + ks * Ts.d(ns),                       # flux: what leaves the gas enters the wall
    sigma_s @ ns + pg * ns,                              # load: the gas pressure on the wall
]).solve(coupling=jno.solve.coupling(window=1e-3, accel="iqn-ils"))
```

`sol` holds each problem's solution, in the order given; for a transient coupling, its states at the window
ends. The coupled problem is kept as `core.coupled` (`.iterations`, `.history`, `.assign`, `.window_iterations`).
The whole of such an example, with all three kinds of condition, is the tutorial
[Hot gas in a thick-walled vessel](tutorials/15-coupling/hot-gas-vessel.md).

!!! measured "Checked against closed forms"
    | case | result |
    |---|---|
    | two slabs in series, non-matching meshes (fvm and fem, fem and fvm, fdm and fvm) | the series resistance to 1e-10 in 3 sweeps |
    | three slabs in series (fvm, fem, fdm), the middle one floating | exact to 1e-7 |
    | two solids brought into contact (fem k 1, fvm k 4) | the contact temperature (Carslaw & Jaeger) to 1.2 %, the erf profiles to 2 % at h ≈ 0.02, dt 1e-3 |
    | a capacity-free slab on a heating wall (quasi-steady partner) | the equivalent Robin problem to the O(h²) of the transfer |
    | a steady conducting gas at rest (started with `x0=`) on a steady fem wall | the gas alone with the wall as its Robin condition, to 1e-8 |
    | a thick cylinder under an fvm gas at rest and a radial temperature (plane strain) | the radial displacement (Timoshenko & Goodier) to 0.2 % |
    | the axisymmetric meridian of a Lamé cylinder under an axisymmetric gas | to 1 % |
    | two insulated bars brought into contact, the fvm bar on ten explicit steps per window | their heat conserved to 2.4 % of the heat moved, 0.8 % at twenty steps (the end-of-window flux: 24 %) |

## Who takes which condition

Every interface condition goes to **one** problem as its own boundary condition. The other problem's part of it
becomes data: that problem evaluates it at its interface nodes and it crosses the interface. Which problem takes
a condition follows from its structure:

- **A value and a flux condition between the same two problems are a Dirichlet–Neumann pair.** The value goes to
  the side with the smaller interface stiffness, the flux to the other (Giles 1997). The stiffness is probed once:
  how much the side's flux changes for a unit rise of its interface value, its other interfaces held. For a
  transient problem it is probed over one window from rest, which gives an effusivity rather than `k/L`. Then the
  iteration contracts by the ratio of the two stiffnesses: gas takes the temperature and metal the flux.
  `dirichlet=problem` overrides the choice.
- **`alpha=` makes the pair Robin–Robin**: each side takes `flux + alpha * value`, and `alpha="auto"` sets it to
  `sqrt(S_a S_b)` (Quarteroni & Valli 1999). It is robust when neither side is much stiffer.
- **A steady problem with no value condition of its own that would take only fluxes** would be singular (a
  floating subdomain). One of its pairs becomes Robin–Robin, which pins it.
- **A condition that differentiates only one side is a load on that side** (`sigma_s @ ns + pg * ns` loads the
  wall). If the other side hears nothing back, the coupling is one-way and a single sweep settles each window.

A problem owns its unknowns and its own coordinates and normals, so in `pg * ns` the gas owns `pg` and the wall
owns `ns`.

## How data cross the interface

The two meshes need not match. The data are nodal fields on each side's interface:

- **values** (a temperature, a displacement) by **interpolation**: every node of the receiver reads the sender's
  piecewise-linear field at the closest point of its facets. Constants and linear fields pass exactly.
- **fluxes and loads** (a heat flux, a traction) by the **L2 (mortar) projection**, `M_tt⁻¹ M_ts`. It keeps the
  total `∫ q dμ` to round-off, and a linear flux crosses a shared straight interface unchanged.

A side reads its own quantities from its discretisation: a value is evaluated at its interface nodes, a flux comes
from its own discrete flux (the fvm boundary-face rule, the fdm boundary stencil, the fem consistent reaction).

`measure="axisymmetric"` makes the interface measure `2π r ds`, with `r` the second coordinate (an `(x, r)`
meridian); `"axisymmetric:0"` takes the first coordinate as `r`. Write your forms with the same `2π r`. A fem
side's interface load is then written with it too.

## Transient coupling

With transient problems in the list, the coupling advances **window by window**:

- Each transient problem marches the window `[t, t + window]` with its own steps and the interface data held over
  it.
- A steady problem is solved again in every sweep: it is quasi-steady, as a gas next to a slowly heating wall is.
  A steady gas has no state at zero, so give it one to start from, `jno.solve.coupling(x0={gas: state})` (a
  state as its `solve(x0=)` takes, e.g. at rest).
- The interface data start where the problems do: each at what its sender gives from its initial state or `x0=`,
  or, when the sender has neither, at the value that holds the condition at its receiver's start (a gas starting
  at rest reads its wall at its own temperature, not at 0).
- A flux sent by a problem that took several steps in the window is its **mean over them** (the trapezoid rule
  for an explicit or IMEX march, the step ends for backward Euler), so what the receiver takes in over the window
  is what the sender gave out. Give a fast-changing sender a step finer than the window.
- `implicit=True` (the default) iterates each window until the data stop changing. `implicit=False` takes one
  sweep per window, which is cheaper and first order in the window.

`window=` defaults to the largest step of the transient problems and must be a whole number of each one's step.
`time=` gives the schemes, one for all or `{problem: scheme}`; a problem that does not read `t` marches every
window as the same compiled program.

| `jno.solve.coupling(...)` | |
|---|---|
| `scheme=` | `"gauss-seidel"` (each problem reads what the ones before it just produced) or `"jacobi"` |
| `tol=`, `max_iter=` | the relative change of the interface data at which a window stops, the most sweeps |
| `accel=` | `"aitken"` (Küttler & Wall 2008), `"iqn-ils"` (Degroote, Bathe & Vierendeels 2009), `None`, or a fixed relaxation |
| `reuse=` | the number of earlier windows whose IQN-ILS columns are kept |
| `dirichlet=`, `alpha=` | see above |
| `measure=` | `None`, `"axisymmetric"`, `"axisymmetric:0"` |
| `implicit=`, `window=`, `time=` | see above |
| `x0=` | `{problem: state}`: where steady `jno.fvm` / `jno.fdm` problems start (first sweep, stiffness probe) |
| `compile=` | a window of one sweep (one-way data, or `implicit=False`) as one compiled program, traced once (`core.coupled.compiles`); default `True` |

A stall below `tol` (the participants' own solves are not that accurate) is named as such.

## Solvers that come with it

- **`jno.solve.ptc()`**, pseudo-transient continuation (Kelley & Keyes 1998), is a steady solve for problems that
  Newton cannot start. Its pseudo-time step is a local CFL number and grows as the residual falls. On steady
  viscous Burgers from rest it reaches the state the time march reaches, where Newton lands elsewhere.
- **`jno.solve.imex()`**, IMEX Runge–Kutta (Ascher, Ruuth & Spiteri 1997), takes a `jno.fvm` problem's fluxes
  explicitly and its cell-local volume terms (reactions, sources, an algebraic closure) implicitly, with a Newton
  on the `nf × nf` block of each cell. That is the chemistry route: species as extra unknowns, Arrhenius rates as
  sources. A 0-D ignition matches scipy's Radau to 1 % in the delay at a step two thousand times past the explicit
  limit, and a premixed flame runs at its travelling-wave speed to 2 %.

## A gas that takes a wall temperature

An inviscid gas has no conduction and cannot take a temperature on its wall; with conduction in its energy
equation it can. `jno.fvm` takes one condition per unknown and face, and a condition names one unknown, so the
temperature is an unknown of its own with the algebraic row `T - p/rho` (solved in every cell by
`jno.solve.imex`), beside the vector of conserved variables that takes the slip condition. The conduction goes
into the energy row, `jnn.stack([0, 0, 0, k*(T.xx + T.yy)], axis=-1)`, or as the divergence of a stacked flux,
`jnn.stack([0, 0, 0, -k*T.x], axis=-1).d(x, scheme="finite_difference") + ...`: either is a face flux of that
row, conservative. See the [tutorial](tutorials/15-coupling/hot-gas-vessel.md).

## Limitations

- Gradients do not reach a parameter through the coupled fixed point yet.
- `jno.solve.ptc` loses positivity on a steady supersonic Euler problem; march such a gas in time.
- A `u.tt` problem cannot take part in a transient coupling, and a transient `jno.fem` side cannot send a flux (it
  takes the flux side of a pair).
- A vector load on a `jno.fem` side must be a coefficient times the boundary normal (`p * n`).
- A vertex-centred fvm side sends a Dirichlet interface's flux from its least-squares gradient (exact for linear
  fields, not conservative).
- The window mean of an explicit sender's flux is the trapezoid rule over its steps, not its own stage
  quadrature: right after a jump (two bodies brought into contact) its first step is off.
- A gas wall moved by its wall (two-way fluid–structure interaction), turbulence models and grain burnback are
  outside it.

## References

- Giles (1997), *Int. J. Numer. Meth. Fluids* 25, 421–436.
- Quarteroni & Valli (1999), *Domain Decomposition Methods for Partial Differential Equations*, OUP.
- Küttler & Wall (2008), *Comput. Mech.* 43, 61–72.
- Degroote, Bathe & Vierendeels (2009), *Comput. Struct.* 87, 793–801.
- Bungartz et al. (2016), *Comput. Fluids* 141, 250–258.
- de Boer, van Zuijlen & Bijl (2007), *Comput. Methods Appl. Mech. Eng.* 196, 1515–1525.
- Farhat, Lesoinne & Le Tallec (1998), *Comput. Methods Appl. Mech. Eng.* 157, 95–114.
- Kelley & Keyes (1998), *SIAM J. Numer. Anal.* 35, 508–523.
- Mulder & van Leer (1985), *J. Comput. Phys.* 59, 232–246.
- Ascher, Ruuth & Spiteri (1997), *Appl. Numer. Math.* 25, 151–167.
- Pareschi & Russo (2005), *J. Sci. Comput.* 25, 129–155.
- Carslaw & Jaeger (1959), *Conduction of Heat in Solids*, OUP.
- Timoshenko & Goodier (1970), *Theory of Elasticity*, McGraw-Hill.
- Williams (1985), *Combustion Theory*, Benjamin/Cummings.
