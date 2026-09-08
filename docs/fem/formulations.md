# Vector, coupled, and nonlinear formulations

## Vector, coupled, and higher-order problems

* **Vector / elasticity** — `u, phi = d.fem_symbols(value_shape=(2,))`; write the elasticity bilinear
  form `λ (∇·u)(∇·φ) + 2μ ε(u):ε(φ)` with `jno.np.symgrad` and `jno.np.inner(..., n_contract=2)` —
  **or component-wise**: `u[i]` and `u[i].x` / `u[i].d(var)` are first-class on a vector Lagrange field,
  and the two spellings mix freely in one term (the shape conventions are shared). A boundary traction on
  one component is `-t * phi_b[i]`.
* **Finite-strain (hyperelastic) mechanics** — the component spelling is what makes it expressible:
  build `F = I + ∇u` from `u[i].d(x_j)`, then `det F`, `F⁻ᵀ` and `log` are ordinary term algebra, and a
  form nonlinear in `∇u` routes to the matrix-free Newton automatically (use
  `nonlinear=jno.solve.newton(line_search=True)` for large steps). Compressible Neo-Hookean
  `P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ` is verified in `tests/test_fem_vector_components.py`: it matches the
  coupled-scalar spelling to machine precision and linear elasticity in the small-load limit. Ramp a hard
  load with a warm-started **`sequence` axis**: `space.sequence("load", ramp, keep="last")` then
  `crux.sweep(space)` — measured on the cantilever: cold *default* Newton fails at `load = 0.1` while
  the four-step warm-started ramp reaches it.
* **Coupled / mixed (Stokes)** — call `fem_symbols(...)` once per field and add one momentum and
  one continuity term; an inf-sup-stable Taylor–Hood pair is `order=2` velocity + `order=1`
  pressure. Pure-Dirichlet velocity leaves the pressure defined only up to a constant; gauge-fix
  that null space by adding `p.pin()` to the constraint list (`p.pin(value)` sets the gauge).

  > **Which gauge, and when it matters.** `p.pin(value)` fixes one vertex's *discrete* value to a
  > *continuous* one. That is fine whenever only the pressure **gradient** is used, and wrong as soon
  > as the **level** is read, because the constant it leaves behind does not shrink with the mesh.
  > Measured on a manufactured 3-D Stokes solution (P2/P1 tets, direct solver), the pressure `L2`
  > error under refinement is `3.57e-2 → 1.08e-2 → 1.34e-2 → 5.56e-3` — it *rises* at `h = 0.22`, and
  > the observed order is `6.16 / -0.89 / 4.38`, i.e. no order at all. `p.pin(mean=True)` gauges to
  > `∫p dx = 0` instead and the same problem gives `8.13e-3 → 4.50e-3 → 2.66e-3 → 1.88e-3`, order
  > `3.05 / 2.17 / 1.75` against the theoretical `O(h²)`. The velocity is identical either way — the
  > field was always right up to that constant (Bochev & Lehoucq, *SIAM Review* 47(1), 2005, §3).
  > A **natural (do-nothing) outflow** fixes the level on its own, so a channel with an outflow wants
  > no pin at all. The normalisation applies wherever a solution is returned — steady vector,
  > transient trajectory, or a lazy solve node — and is plain arithmetic, so it survives `jit`/`grad`.
* **1D and 3D** — a 1D interval or a 3D `cube`/extruded `gmsh` volume use the identical API with
  one fewer / one more coordinate (`ui.z`, `u(xb, yb, zb) - g`).

### Reading a multifield solution back

A coupled solve returns **one flat vector**, so the `fem` object carries the handles that say which
part is which. They are the same handles the block preconditioners key on:

```python
st = jno.fem([mu * inner(gu, gv, n_contract=2) - pp * trace(gv),   # momentum
              -qq * trace(gu),                                     # continuity
              u(xb, yb)[0] - u_profile(yb), u(xb, yb)[1] - 0.0, p.pin()])

st.dofs           # 850          — total, both fields
st.blocks         # [slice(0, 746), slice(746, 850)]   — per-field slices into the flat vector
st.offsets        # [0, 746, 850]                       — the same thing as boundaries
st.block_index(u) # 0            — resolve a SYMBOL to its block, never hardcode the position
st.block_index(p) # 1
st.field_points   # [(373, 2), (104, 2)]  — each field's own node coordinates (P2 velocity, P1 pressure)
st.is_complex     # False        — also is_linear, is_transient

sol = st.solve()
velocity = sol[st.blocks[st.block_index(u)]]
```

!!! warning "Resolve the block by symbol, not by position"
    Field order follows the order the terms were written, which is not always the order you think —
    on a phase-field system the degradation factor is written first. `block_index(sym)` is the only
    spelling that cannot silently pick the wrong field.

### What the fluid path is verified to do — and what it is not

Scope first, since it is not obvious from the API: jNO's FEM fluid path is **laminar incompressible**
flow. There is no RANS model (no k–ε, no k–ω SST, no wall functions), no compressible or Euler path,
no free surface / VOF / level set, and no fluid–structure interaction. Nothing about `jno.fem` stops
you writing those terms; nothing in the library implements or verifies them.

The one qualification is **algebraic (zero-equation) LES**, below: a subgrid eddy viscosity is a
formula of the resolved velocity gradient, so it is written in the term list like any other
coefficient. Those formulas are checked against a textbook oracle in 2-D and 3-D, and run inside a
Newton solve in the [LES subgrid tutorial](../tutorials/08-fem-and-varpinns/les-subgrid-3d.md) — where
Vreman and WALE vanish in simple shear (1.1e-6 and 2.0e-19 of `nu`) while Smagorinsky reports 8.0e-2.
They are still *not* a validated LES capability: that needs a turbulent benchmark against DNS
(wall-resolved channel, decaying isotropic turbulence), which this library has not run.

#### Subgrid eddy viscosity — a formula, not an API

> Worked end to end, on two flows and with the failure modes measured, in the
> [LES subgrid tutorial](../tutorials/08-fem-and-varpinns/les-subgrid-3d.md).

An algebraic LES model adds `ν_t(∇u)` to the molecular viscosity. It needs no new API: the filter
width is `d.cell_size` (or `d.cell_metric` if you want it direction-aware), and the model is
arithmetic on `grad(u)`. Vreman (*Phys. Fluids* **16** (2004) 3670, eq. 5), written through
invariants so it reads the same in 2-D and 3-D:

```python
g     = grad(u, ax)
tr_b  = delta**2 * inner(g, g, n_contract=2)                       # tr(beta),  beta = delta^2 g gᵀ
tr_b2 = delta**4 * einsum("...ik,...jk,...jl,...il->...", g, g, g, g)   # tr(beta^2)
B     = where(0.5 * (tr_b**2 - tr_b2) > 0, 0.5 * (tr_b**2 - tr_b2), 0.0)
nu_t  = jno.lag(0.07 * sqrt(B / (inner(g, g, n_contract=2) + 1e-30) + 1e-30))

mom = (MU / RHO + nu_t) * inner(g, grad(v, ax), n_contract=2) + ...   # into the viscous term
```

!!! danger "Three things that will bite, all measured"
    * **`ν_t` must be lagged.** It is a square root, so its slope at `u = 0` is infinite and Newton
      diverges from a rest state outright. `jno.lag` freezes it within each linearisation — the same
      treatment a Carman–Kozeny drag needs.
    * **Clamp the invariant.** `B_β` and WALE's `S_d:S_d` are non-negative in exact arithmetic but are
      computed as a *difference of nearly equal numbers*. Measured in pure shear: `B_β` lands at
      1.4e-20 where it should be 0, from a relative cancellation of 2.7e-16. Unclamped, one round-off
      excursion below zero is not a small error — `sqrt` and `**1.5` return **NaN**.
    * **Smagorinsky does not vanish in laminar shear**, and that is a modelling defect, not a detail:
      `|S|` is non-zero in any shear, so it invents eddy viscosity throughout a laminar boundary layer
      and needs Van Driest damping. Vreman and WALE (Nicoud & Ducros, *Flow Turb. Combust.* **62**
      (1999) 183) vanish identically there. `tests/test_fem_les.py` pins exactly that difference.

Within that scope, measured rather than asserted:

| | verified by |
|---|---|
| 2-D steady Stokes, Taylor–Hood P2/P1 | exact fields recovered to ~1e-13 with a direct solver |
| 2-D steady Navier–Stokes | Kovasznay (closed form) in the convergence matrix: velocity `O(h³)`, pressure `O(h²)` |
| 2-D transient Navier–Stokes | lid-driven cavity at Re = 200, backward Euler + Newton |
| **3-D Stokes, Taylor–Hood P2/P1 tets** | fitted order 3.12 velocity / 2.29 pressure against theory 3 / 2 |
| **3-D Navier–Stokes** (convective term) | fitted order 3.13 / 2.37 at `ν = 0.05`, cell Péclet ≈ 4 |
| coupled (Boussinesq) | its own convergence row, three fields |
| **an external benchmark** | DFG 2D-1 cylinder at Re = 20 — `c_D` to **0.02 %** of the published value |
| natural (do-nothing) outflow | carried by that same benchmark |
| forces on a body | reaction-based drag/lift via `fem.eval` + `region_dofs` |

Two ceilings worth knowing before you plan a run:

* **Direct-solver fill-in in 3-D.** Measured on one GPU, a 3-D Stokes solve is trivial to ~10k DOF
  (0.60 s at 9.1k) and then turns over sharply — 4.02 s at 18.5k, i.e. roughly `O(N^2.7)`. That puts
  the practical ceiling for `lu()` around 30–60k DOF; past it use the block/Schur preconditioners in
  `jno.precond` (verified in 3-D), or `lu(backend="pardiso"/"cudss")`.
* **Stabilisation is a formula, not a feature.** SUPG / PSPG / grad-div are terms you write in the
  term list, and the two pieces they need are there: `jno.np.laplacian` accepts a **vector** field
  (the momentum strong residual carries `nu*lap(u)`), and `dom.cell_metric` gives the element metric
  `G = J^-T J^-1` that a direction-aware `tau` is built on (`dom.cell_size` is isotropic and cannot
  see a stretched cell). Verified: SUPG cuts the upstream oscillation of a Peclet-1000 transport
  problem by 3300x, and PSPG makes **equal-order P1/P1** flow converge (Kovasznay, observed rates
  1.78 velocity / 1.70 pressure, a 4.3x better pressure than unstabilised) — see the
  [stabilised-flow tutorial](../tutorials/08-fem-and-varpinns/navier-stokes-stabilised-2d.md).
  What is **not** settled: the grad-div/LSIC coefficient made both errors worse at Re = 20 and is
  left uncalibrated, and no unstabilised-P2/P1 Reynolds ceiling has been measured.
* **Higher-order Lagrange** — `order=k` gives degree-`k` elements (P2, P3, P4, … on triangles and tets);
  read the solution at `fem.points`. The geometry stays affine-P1 (straight-sided), so on a *curved*
  boundary the geometric error caps the observed order regardless of `k` — measure high-order convergence
  on straight-sided/polygonal domains.

---

## Elasto-plasticity — a trace formula, not a module

Plasticity is not a module in jNO; it is a **formula** in the term list (the FEM contract). The J2 (von
Mises) radial return contracts against the test strain to a *scalar* per Gauss point — the same trick the
elastic form uses (`lam*trace*trace + 2*mu*inner`, never an identity), via `dev(A):B = A:B - tr(A)tr(B)/3`
and `||dev(A)||^2 = A:A - tr(A)^2/3`. So the whole return map is six lines of `jno.np`, behind your aliases:

```python
sym, grad, trace, inner, sqrt, maximum = jno.np.sym, jno.np.grad, jno.np.trace, jno.np.inner, jno.np.sqrt, jno.np.maximum
lam, mu = lame(E, nu); K = lam + 2*mu/3; rt = 1.5**0.5
eps = lambda w: sym(grad(w, [x, y, z]))
eu, ev = eps(u), eps(phi)
tru, trv = trace(eu), trace(ev)
ddev = sqrt(maximum(inner(eu, eu, 2) - tru*tru/3, 0) + 1e-30)   # ||dev eps(u)||, safe von-Mises norm at 0
dg   = maximum(rt*2*mu*ddev - sy, 0) / (3*mu + H)               # plastic multiplier
dev_ev = inner(eu, ev, 2) - tru*trv/3                          # dev eps(u) : eps(phi)
mech = K*tru*trv + 2*mu*dev_ev - 2*mu*rt*dg*dev_ev/ddev        # = the integrand of  sigma(eps(u)) : eps(phi)
sol  = jno.fem([mech, u(*bc) - 0.0]).solve(nonlinear=jno.solve.newton())
```

`jno.fem` sees the nonlinear form and routes to Newton; the element Jacobian is the consistent
elastoplastic tangent for free (AD of the formula). The solve is differentiable — thread `sy` (or `H`,
`E`) as a `jno.np.parameter` to recover it from an observed deformation (material-identification inverse
problem). This is Hencky deformation theory (virgin every solve): exact for monotonic proportional loading.

### Flow theory — the path-dependent march

(path-dependent; unloading leaves a permanent set) is the *identical* formula reading the
previous step's per-quadrature-point state with the step-history index `.i(k)`: `ee = eps(u) - ep.i(-1)`
and `sy -> sy + H*al.i(-1)`, with `ep, al` declared like any field via `fem_symbols`. How each state
*advances* is a **named update term** in the same list — `state.evolves(<formula>)`, an update, not an
equation (and not an operator: `==` is reserved for identity, `<` for comparison). The load is written as
a function of the pseudo-time coordinate `tau`, the domain carries a `tau=` load grid, and `fem.solve()`
**marches** the path with **nothing passed** — triggered by `.i(k)` exactly as `u.t` triggers transient:

```python
d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.1).domain(tau=(0.0, 1.0, 40))   # pseudo-time load path
x, y, z, tau = d.variable("interior", split=True)            # τ is a coordinate, like t
dev = lambda A: A - trace(A) / 3 * I3                         # I3 = jno.np.identity(3)
nrm = lambda A: sqrt(maximum(inner(A, A, 2), 0) + 1e-30)      # safe Frobenius norm
ee  = eps(u) - ep.i(-1); D = dev(ee); dd = nrm(D)             # elastic predictor about the previous state
dg  = maximum(rt * 2 * mu * dd - (sy + H * al.i(-1)), 0) / (3 * mu + H)
n   = D / dd
sig = K * trace(ee) * I3 + 2 * mu * D - 2 * mu * rt * dg * n  # returned stress
P   = peak * (1 - jno.np.abs(2 * tau - 1))                    # load ramps 0 → peak → 0 with τ
traj = jno.fem([
    inner(sig, eps(phi), 2) - P * inner(zhat, phi, 1),       # equilibrium              (test phi)
    ep.evolves(ep.i(-1) + rt * dg * n),                      # plastic strain advances  (a named update)
    al.evolves(al.i(-1) + dg),                               # hardening advances
    u(*bc) - 0.0,                                            # clamp
]).solve()                                                   # (n_steps, n_dofs) load-path trajectory
```

??? note "How the build infers the history depth"
    `.i(-k)` **reads** history, `.evolves` **writes** it. The build infers the keep-depth from the
    most-negative index and threads a zeroed per-quadrature-point buffer through the march's `lax.scan` carry
    (one compiled residual, reused every step; frozen-constant in the tangent → the consistent return-map
    tangent). The whole march rides `custom_root`, so it stays differentiable end-to-end: thread `sy` as a
    `jno.np.parameter` and `∂(unloaded state)/∂sy` flows through the entire load path (a material-
    identification inverse). A **primary-unknown** history (`u.i(-1)`/`u.i(-2)`, e.g. a BDF2 time scheme) is
    auto-buffered from the solved `u` — no `.evolves`; an **internal** state read at `.i(-1)` with no
    `.evolves` on a `tau=` domain is a build error (never a silently frozen buffer = deformation theory).

!!! warning "Scope"
    small-strain, isotropic, linear-hardening; 3-D (2-D is plane strain). Kinematic / nonlinear
    hardening and contact are separate (not built).

**A state can be shared by a coupled system.** The march is not single-field: history buffers are indexed
by *cell*, never by field, and the readout gathers every field's cell DOFs at once — so a state written
by one field and read by another is the same march. That is the phase-field / gradient-damage shape,
where an irreversible history `H = max_τ ψ⁺(u)` couples a displacement field to a damage field:

```python
psi = 0.5 * inner(grad(u, X), grad(u, X), 1)                 # driving force, from the u field
deg = (1 - dm)**2 + eta                                      # degradation, from the dm field
jno.fem([
    deg * inner(grad(u, X), grad(phi, X), 1) - load(tau)*phi,        # equilibrium      (test phi)
    (gc/l)*dm*q + gc*l*inner(grad(dm, X), grad(q, X), 1)
        - 2*(1 - dm)*Hs.i(-1)*q,                                     # damage evolution (test q)
    Hs.evolves(maximum(Hs.i(-1), psi)),                              # irreversibility  (a named update)
    u(*bc) - 0.0,
]).solve()                                                           # nothing passed
```

The running `maximum` is what makes it irreversible: at zero load the damage is retained rather than
healing. Note the block order is *first appearance in the term walk* — here `dm` precedes `u`, because the
degradation factor is written first — so resolve a block with `fem.block_index(dm)`, never a hardcoded
index.

A form that is **linear in every unknown** but reads `.i(k)` marches too — the AT1 damage equation with a
fully lagged driving force is exactly that shape. Each load step is a different linear system whose
coefficients the buffers set, so it routes through the same residual operator as the nonlinear march
(Newton converges in one step on a linear residual). You pay roughly one extra linear solve per step
versus a pure linear assembly, in exchange for one march path rather than two.

The coupled march is differentiable in a material parameter, exactly as the single-field one is: thread a
`jno.np.parameter` into the form and `∂trajectory/∂θ` flows through the whole scan. The same holds for a
coupled *steady* **nonlinear** form. What still refuses a runtime parameter is a coupled form that is
**linear and carries no history**, because that one assembles as a matrix/rhs pair and the coupled linear
assembly has no parametric route; anything on the residual path re-evaluates at the runtime args and is
field-agnostic.

Not carried, each rejected with a clear error: a real `u.t` transient (drive time through `tau` instead),
a complex form, 1D, non-nodal (Argyris/Morley/edge) elements, VPINN, and periodic ties.

!!! danger "A step that did not converge is refused, not carried forward"
    The march runs its per-step Newton
    inside a single `lax.scan`, and the driver's own convergence check needs a *concrete* residual — so
    inside the scan it disables itself, exactly where the signal matters most. A load path compounds the
    loss: a non-converged step becomes the next step's initial state *and* its history buffers, so one
    silent failure contaminates everything after it, and the trajectory still comes back finite and
    entirely plausible. Measured on a 3-D Yeoh phase-field march whose undamped Newton overshot into an
    inverted element (`J = det F ≤ 0`, so `J**(-2/3)` is NaN, which is absorbing): with the grip *pinned*
    to 0.4 the returned displacement read 0.70, with no error raised.

??? note "How a failed step is detected out of the scan"
    So the per-step residual is carried out of the scan and tested where it is concrete, against the
    driver's **own** `rtol`/`atol` — the net can only catch what the driver would have caught eagerly, and
    never second-guesses a solve configured loosely. Under `bounds` it scores the **min-map**, not the bare
    residual: on an active constraint that residual is non-zero by construction (it *is* the multiplier),
    and scoring against it would read a correct answer as a divergence. The check costs two residual
    evaluations per step — measured at **2.4%** (30.30 s → 31.03 s) of an 8-step, 576-DOF Yeoh march. It is
    a no-op under `jax.grad` of a runtime-parametric march, where the norms are themselves traced; there,
    as everywhere else in jNO, the solver's iteration cap is all there is.

The error names the fixes in order of what usually works: globalize the per-step solve
(`jno.solve.newton(line_search=True)` / `staggered(line_search=True)`, or `damping<1`), take smaller
steps (a finer `domain(tau=(...))` grid, or the adaptive path below), or raise `max_steps`. In the Yeoh
case above, `line_search=True` alone recovers the exact answer — and note P1 solves the same form
undamped: a higher-order element's full Newton step produces larger gradients at its extra quadrature
points, so P2 is the more exposed one.

### Adaptive load stepping

A uniform load grid is
wrong in both directions at once: it wastes steps while nothing happens and takes too-large ones through
the event. On a path-dependent march that second failure is not merely coarse — a step can converge
perfectly and skip the entire transition, leaving a valid sequence of equilibria with no resolved event
between them, which is a *different* answer, not a coarser one.

```python
sol = fem.solve(tau=jno.solve.adaptive(limit=0.05))            # bound every DOF's per-step change
sol = fem.solve(tau=jno.solve.adaptive(limit=[(dm, 0.05)]))    # per field — the usual case
```

The criterion is deliberately not the transient's. A rate-independent load path has **no local
truncation error to estimate** — each step is an equilibrium, not an approximation to a trajectory — so
the `rtol`/`atol` step-doubling estimate that sizes `time=` measures nothing here. `limit` bounds how
much the solution may change in one step; a step is rejected (and cut by `shrink`) when the solve fails
to converge *or* the change exceeds `limit`, and a comfortable step grows by `grow`.

??? note "Mechanism — pilot, freeze, replay"
    Mechanism: **pilot → freeze → replay**. March eagerly with rejection to discover the schedule, freeze
    it, replay it as a fixed-length differentiable scan. Rejection is exactly why the pilot must be
    separate: the transient marcher accepts every attempt on purpose, because a discarded state makes the
    per-step adjoint run at zero cotangent and returns a NaN gradient. The replay has nothing to reject.
    The schedule is piecewise constant in the parameters, so the gradient over a frozen one is the true
    derivative almost everywhere — the same contract `adapt=` makes for a frozen mesh sequence.

The trajectory is resampled back onto the domain's declared `tau=` grid (as the transient resamples onto
`save_ts`), so the returned shape does not depend on the steps taken and the resampling error is bounded
by `limit` itself. `fem.tau_schedule` reports what the pilot chose.

!!! warning "A parametric form refuses to pilot, by design"
    The pilot needs concrete values to accept or reject a
    step and a differentiable solve hands it tracers; piloting at the parameters' *stored* values would
    silently adapt to whatever they happen to be — 0.0 for a fresh `jno.np.parameter`, i.e. a load path that
    never happened. Discover the schedule forward, then replay it:

```python
fem.solve(tau=jno.solve.adaptive(limit=0.05))   # forward, at the values you want
fem.solve(tau=fem.tau_schedule)                 # differentiable replay of that schedule
```

`tau=<array>` accepts any strictly increasing grid spanning the declared path, so it doubles as the
"non-uniform grid I chose myself" spelling. Not composable with a per-load-step field
(`freeze_path(frames)`), whose frames are indexed by the declared step count.

Note what adaptivity does **not** fix. If the step is cut to the floor and the change still exceeds
`limit`, that is an **unstable branch**, not a step that is merely too big: under load control a
snap-back has no nearby equilibrium, so no refinement finds one. The error says so and points at
[`fem.solve(tau=jno.solve.arclength(...))`](#arc-length-passing-a-limit-point), which advances along the
equilibrium path rather than along the load and so can turn around.

!!! warning "Arc-length exists; it does not yet compose with `staggered`"
    It is refused with `nonlinear=jno.solve.staggered(...)`, because freezing the load factor while
    sweeping a block *is* load control — which has no equilibrium past the fold. Since a phase-field
    energy needs alternate minimization (see below), this particular study is still without an
    instrument. The refusal says so rather than converging to something plausible.

Keep `dm` in range with [`dm.bounds(0, 1)`](boundary-conditions.md#inequalities-uboundslo-hi), which composes with the march.
Note that the bound does **not** replace the floor `eta` on the degradation: at `dm = 1` exactly,
`(1-dm)²` makes the displacement block singular, so the floor is a well-posedness requirement in its own
right. And a monolithic Newton is not expected to converge on this energy at all — drive it with
[`jno.solve.staggered([u, dm])`](../solvers.md).

### Finite strain

Tensor constants broadcast correctly (`jno.np.identity(n)` carries
a leading batch axis), so `F = I + ∇u`, `E = ½(FᵀF − I)`, `S = λ tr(E) I + 2μ E` and the internal virtual
work `∫ (F S):∇δu` are written directly — St. Venant-Kirchhoff in five lines, no module:

```python
grad, trace, inner, einsum, I = jno.np.grad, jno.np.trace, jno.np.inner, jno.np.einsum, jno.np.identity(3)
H = lambda w: grad(w, [x, y, z])
F = I + H(u);  E = 0.5*(einsum("...ki,...kj->...ij", F, F) - I);  S = lam*trace(E)*I + 2*mu*E
mech = inner(einsum("...ij,...jk->...ik", F, S), H(phi), 2)      # ∫ (F S):∇δu
```

`jno.fem` routes the nonlinear form to Newton (exact 20%-stretch patch test; reduces to linear elasticity
as strain → 0). Combine with the plastic return map for finite-strain plasticity — both are formulas.

### Hyperelasticity — the energy is the input

For anything past St. Venant-
Kirchhoff, hand-deriving the stress is where the algebra errors live. `diff` differentiates a **scalar
expression with respect to another expression** (the constitutive counterpart of `grad`, which
differentiates w.r.t. a coordinate), so you write the energy from the paper and get the 1st
Piola-Kirchhoff stress:

```python
det, trace, einsum, jac, I = jno.np.det, jno.np.trace, jno.np.einsum, jno.np.jacobian, jno.np.identity(3)
F   = I + jac(u, X)                                     # bind it ONCE, then reuse
I1b = det(F)**(-2/3) * trace(einsum("...ki,...kj->...ij", F, F))    # first isochoric invariant
psi = C10*(I1b - 3) + C20*(I1b - 3)**2 + C30*(I1b - 3)**3           # Yeoh, 3rd order
mech = inner(jno.np.diff(psi, F), jac(phi, X), 2)       # P = ∂psi/∂F, then ∫ P:∇δu
```

Measured: on a Yeoh solid this reproduces the hand-derived `S = 2 ∂psi/∂C`, `P = F S` residual
**bit-for-bit**, and the Neo-Hookean `P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ` to 1e-11. The consistent tangent
`∂P/∂F` comes out of the assembler's own element differentiation — you never write it.

Two scope limits, both fail-loud rather than silent:

* **It is pointwise.** The derivative is taken independently at each quadrature point, which is what a
  constitutive law is; an `Integral` inside the target is refused (differentiate the integrand, then
  integrate).
* **`wrt` is matched by identity.** Bind `F` to a variable and pass that same object. A rebuilt copy
  (`diff(psi, I + jac(u, X))` written inline) is a different node, and rather than differentiate to a
  silent zero it raises.

Any energy-derived law works the same way — Mooney-Rivlin, Ogden, Gent, anisotropic tissue models — as
does a chemical potential `mu = diff(f, c)` or an electro/magnetostrictive coupling.

---

### Volumetric locking — `cellwise`, and the one-line B-bar

Everything above has a failure mode that does not announce itself. As `nu -> 0.5` a standard
displacement element goes rigid — **volumetric locking** — and since J2 plastic flow is *isochoric*,
that is plasticity's default regime, not an edge case. The answer is simply too stiff; nothing warns.

`jno.np.cellwise(expr)` is the quadrature-weighted mean of an expression over each cell,
`(∫_K expr)/(∫_K 1)`, broadcast back to its quadrature points — the L2 projection onto the piecewise
constants. B-bar is that projection applied to the volumetric strain, and nothing else. Written as a
scalar contraction (the way the elastic and plastic forms above are written, `dev(A):dev(B) =
A:B - tr(A)tr(B)/dim`, never an identity tensor):

```python
eu, ev   = eps(ui), eps(vi)
tru, trv = trace(eu), trace(ev)
cu, cv   = cellwise(tru), cellwise(trv)          # <- the whole change
dev_uv   = inner(eu, ev, 2) - tru * trv / dim    # dev(eu) : dev(ev)
mech     = lam * cu * cv + 2 * mu * (dev_uv + cu * cv / dim)     # sigma(ebar) : ebar(phi)
```

Delete the two `cellwise` calls and you have the standard form back, so **every J2 return map already
written against an `eps` alias becomes locking-free by rebinding that one alias**.

!!! measured "Refinement does not cure locking — that is what makes it locking"
    Plane-strain Q1 cantilever under a body load, `nu = 0.4999`, max `|u_y|`:

    | cells | standard | B-bar |
    |---|---|---|
    | 16x4 | 2.68e-02 | 3.03e-01 |
    | 64x16 | 6.24e-02 | 3.01e-01 |
    | 128x32 | 1.30e-01 | 3.02e-01 |

    B-bar is converged on the coarsest mesh. The standard element is 11.3x too stiff there and still
    **2.3x too stiff at 128x32**, having spent 64x the cells to get halfway. At `nu = 0.3`, where there
    is nothing to cure, the two agree to 5.6%.

F-bar is the finite-strain counterpart — project the determinant, and put the projection inside the
variable you differentiate so `diff` still sees a pointwise expression:

```python
F    = I + jac(u, X)
Fbar = (cellwise(det(F)) / det(F)) ** (1 / dim) * F
mech = inner(diff(psi(Fbar), Fbar), jac(phi, X), 2)
```

Three scope limits, all fail-loud except the first, which is a property of the element rather than a
restriction:

* **It does nothing on P1 triangles/tets.** Their strain is already constant over the cell, so the
  projection is the identity. B-bar is for quadrilateral/hexahedral cells and higher-order simplices;
  for P1, use Q1/hex cells, raise the order, or write a mixed u-p formulation.
* **Native-Lagrange volume terms only.** 1-D, non-nodal (Argyris/Morley/RT/N1E), surface terms and
  VPINN/collocation residuals raise — averaging over those points would not be a per-cell mean.
* **Not inside a `diff` target.** `diff` evaluates as `grad(sum(...))`, which is the per-point
  derivative only because the quadrature axis is a batch axis; a projection couples the points and the
  result would silently be the cell-summed derivative. Write `cellwise(diff(psi, F))`, or use the F-bar
  ordering above, where the projection lives inside `wrt` and never reaches the differentiated
  expression. The error says both.


### Arc-length — passing a limit point

Everything above drives the path by the **load**. That works until the load-deflection curve turns
over: past a limit point there is no equilibrium at a higher load, so no step size finds one, and the
adaptive stepper says exactly that rather than grinding to its floor.

`fem.solve(tau=jno.solve.arclength(...))` makes the load factor an **unknown** and constrains the
*increment* instead (Crisfield, *Computers & Structures* 13 (1981) 55-62; *Non-linear FEA of Solids and
Structures* Vol. 1 §9.3.2):

$$\Delta u \cdot \Delta u + \psi^2 \Delta\lambda^2 = \Delta s^2$$

Nothing in the term list changes — the load is still a formula in `tau`. What changes is that `tau` is
solved for. This is cheap in jNO because **`tau` is not a DOF**: it reaches the residual as a plain
scalar argument, so the border rides a residual wrapper and `op.size` never moves.

```python
sol = fem.solve(tau=jno.solve.arclength(ds=1e-3))
load = fem.tau_schedule          # the load factors reached — NON-monotone across a snap-back
```

!!! measured "It finds the fold, and goes round it"
    Bratu's problem `-Δu = λ e^u`, whose fold is analytic at `λ_c = 3.5138307…`. Arc-length peaks at
    **3.51871** (0.14% high, and the error halves under mesh refinement), then the load factor comes
    back **down** to 2.11 while `‖u‖∞` keeps climbing 1.18 → 2.80. The same problem under load control
    over the same span does not return a path at all — it fails, which is the honest outcome.

Reading the knobs:

* `ds` — the arc per step. Left `None` it is calibrated from the declared span, with the consequence
  that on a **linear** problem arc-length reproduces the declared uniform grid exactly. With the
  default `psi=0` it is a pure displacement measure, i.e. the same quantity `adaptive(limit=...)`
  bounds.
* `psi` — the weight on the load term; `0` is Crisfield's cylindrical constraint. Note the deviation:
  Crisfield weights this by the reference load vector `qᵀq`, which jNO does not have (the load is an
  arbitrary formula in `tau`). `psi` is a plain weight, and nothing is substituted for `qᵀq`.
* `domain(tau=(start, end, n))` is **reinterpreted**: `end` sets the direction and the default arc, not
  a target. So `n` buys resolution, not reach — to follow the path further, widen `end`.

Refused by name rather than approximated: `nonlinear=jno.solve.staggered(...)` (freezing the load
factor is load control), `newton(direct=True)` (the bordered tangent is not assembled — the system goes
to a matrix-free Newton-Krylov whose `jax.linearize` builds the border for free), a runtime-parametric
form, and `freeze_path(frames)`. The trajectory is differentiable; `fem.tau_schedule` is concrete
observability.

---

## The trial may be a network — VPINN and Deep Ritz

Everything above solves for FE coefficients. The same term list also accepts a **neural trial**: write
the network where the unknown would go and `jno.fem` detects it, test-projects the weak form onto the
FE basis, and returns a trainable residual instead of an operator. Nothing else about the authoring
changes — same domain, same symbols, same `jno.fem([...])`.

```python
net    = jno.nn(foundax.mlp(2, hidden_dims=32, num_layers=3, activation=jax.nn.tanh, key=key))
ansatz = xi * (1 - xi) * yi * (1 - yi)          # hard-BC ansatz: vanishes on the boundary
u_net  = net(xi, yi) * ansatz                   # the trial IS the network
vi     = phi.bind(x=xi, y=yi)                   # the test function is still the FE basis

pde = jno.fem([grad(u_net, xi) * grad(vi, xi) + grad(u_net, yi) * grad(vi, yi) - f * vi,
               u(xb, yb) - 0.0])                # declares WHICH test functions vanish
jno.core([pde.mse], domain=dom).solve(2500)     # train the weights
```

This is the Petrov–Galerkin variational PINN of Kharazmi, Zhang & Karniadakis (*hp-VPINNs*, **CMAME**
374 (2021) 113547). The Dirichlet term is not optional decoration: it tells `jno.fem` which test
functions vanish on the boundary, so their irreducible `∂u/∂n` flux is masked out of the loss.
Without it the loss minimum is *not* the PDE solution.

!!! measured "What the network trial actually reaches, against analytic solutions"
    | problem | rel L2 |
    |---|---|
    | Poisson, hard-BC ansatz | 4.2e-05 |
    | **3-D** Poisson on the cube | 4.6e-04 |
    | **3-D** Neumann flux face | 8.4e-04 |
    | Neumann flux (`u = x`) | 6.8e-04 |
    | cubic nonlinearity (`+ u³`) | 9.8e-05 |
    | vector Poisson, `u* = (a, 2a)` | 1.3e-04 / 2.8e-04 |
    | Deep Ritz (energy, Gauss quadrature) | 8.8e-04 |
    | network trial **+ inverse parameter** | `k`: 1.00 → 2.901 (truth 3.00), field 3.4e-02 |

    The last row is the combination worth knowing: a network trial and a `jno.np.parameter` train in
    the same loss, so the field and an unknown coefficient are recovered together. Note it needs a
    **data** term — the residual alone is degenerate, since `k·a(u,v) = (f,v)` with `u` free is
    satisfied by any `k` with `u` rescaled.

### A source on a vector field — one DSL, either trial

A value-channel coefficient is a scalar per quadrature point, or one vector per point — **quadrature
first, value axis trailing**. A constant carries no quadrature axis at all, so it broadcasts. All of
these lower **identically**, and are accepted by the FEM trial and the network trial alike:

```python
g * vi[0] + (2 * g) * vi[1]                          # per component
jno.np.inner(jnp.array([1.0, 2.0]), vi, 1)           # a CONSTANT vector
jno.np.inner(g * jnp.array([1.0, 2.0]), vi, 1)
jno.np.inner(jno.np.stack([g, 2 * g], axis=-1), vi, 1)
```

!!! warning "`stack` defaults to `axis=0`, which is the wrong axis here"
    `jno.np.stack([f0, f1])` builds a **component-first** `(vec, Nq)` array, and the value axis is
    trailing throughout the assemblers. Both lowerings refuse it by name and point at `axis=-1`,
    rather than transposing it silently — a silent transpose is a guess at intent, and it would make
    one lowering accept a weak form the other rejects.

Two conventions worth stating once, since they are the only places the layout is not inferable:

* a bare `(k,)` coefficient with `k` equal to the quadrature-point count is read as a per-point
  **scalar**, that being the ordinary case — so a constant vector of exactly that length needs an
  explicit quadrature axis, `(1 + 0*x) * jnp.array(...)`;
* the value axis is **trailing**, everywhere, for both trials.

### A coupled system, as one vector field

Two separate `fem_symbols` calls raise. That is a real boundary — the lowering wraps a single primary
unknown — but it is rarely the end of the road, because a coupled system whose fields share a test
space is *the same system* as one vector field. Inter-field coupling becomes a cross-component term:

```python
u, phi = d.fem_symbols(value_shape=(2,))         # u = (a, b), one field
net    = jno.nn(foundax.mlp(2, output_dim=2, ...))   # one network, two outputs
u_net  = net(xi, yi) * ansatz
vi     = phi.bind(x=xi, y=yi)

#  -Δa = fa + b ,  -Δb = fb
jno.fem([inner(jac(u_net, X), jac(vi, X), 2)
         - fa * vi[0] - fb * vi[1]      # per-component sources
         - u_net[1] * vi[0],            # the coupling: b enters a's equation
         u(xb, yb) - (0.0, 0.0)])
```

!!! measured "The rewrite is exact; the training is the limit"
    Solving that *identical form* with an FEM trial recovers the manufactured `(s, 2s)` to **9.3e-04 /
    9.2e-04**, so the vector rewrite of the coupled system is correct. The network trial on the same
    form reaches **6.8e-02** on the coupled component and **5.0e-05** on the uncoupled one, and does
    not improve with more steps — the two component residuals compete under an equal-weight loss.
    That is loss balancing, a standard PINN concern, not a formulation error; weight the terms if you
    need the coupled component tighter.

Fields that genuinely need **different** test spaces — a Taylor–Hood velocity/pressure pair — have no
route yet, since the lowering builds one test context.

### Deep Ritz — and the quadrature that makes it honest

For an energy-minimising formulation there are no test functions at all: write the functional and
minimise it.

```python
energy = (0.5 * (ux**2 + uy**2) - f * uu).integrate(quadrature="gauss")
jno.core([energy], domain=dom).solve(4000)
```

E & Yu, *Commun. Math. Stat.* **6**(1) (2018). Use `quadrature="gauss"`: the default nodal rule samples
the energy only at mesh vertices, and a network expressive enough to develop structure *between* them
drives the discrete energy below the true minimum — a variational crime in which the reported loss
keeps falling while the solution degrades.

!!! note "One DSL, either trial — swept operator by operator"
    A weak form should mean the same thing whichever trial it carries, so the operators were checked
    side by side: the same form built once with `d.fem_symbols()` and once with a network, comparing
    what each lowering accepts. `grad·grad`, `inner(grad, grad)`, reaction, `u³`, `exp(u)`,
    `laplacian(u)·v`, a coefficient `k(x)`, a Neumann boundary term, `inner(jac, jac, 2)`,
    `symgrad : symgrad`, `div(u)·div(v)` and component terms (`u[0]·v[0]`, `u[1]·v[0]`) all lower on
    both.

    Two of those did not, until this sweep found them. A **constant scalar source** — `- 1.0 * phi`,
    the plainest one there is — raised on the network trial, because a constant carries no quadrature
    axis and the value channel demanded one per point; every shipped VPINN happens to write a
    *coordinate* source, which does. And **`div`**, which has no node of its own (it is
    `trace(jacobian(phi, X))`, the way a book writes it), was not recognised by the channel extractor,
    so grad-div stabilisation and an incompressibility penalty assembled on the FEM path only.

    `div(v)` is `I : grad(v)`, so it lowers to the grad channel with the identity as its coefficient.
    Checked end to end: a VPINN trained on a grad-div form (`γ = 5`) matches the FEM solve of the
    identical form to **1.3e-02**, and on the FEM trial `trace(jac(w))` equals the longhand
    `∂w₀/∂x + ∂w₁/∂y` exactly.

    One asymmetry is inherent rather than a gap: a bound field view offers `ui.x`, while a network
    expression is not a field view and takes `jno.np.grad(u_net, xi)`. That is the trial object
    differing, not the weak-form language.

### Boundary conditions, and complex forms

| | network trial |
|---|---|
| **Dirichlet, homogeneous** | declaration `u(region) - 0.0`; the network satisfies it through its **ansatz** |
| **Dirichlet, inhomogeneous** | **refused** — put `g` in the ansatz (`g + ansatz * net`), see below |
| **Neumann flux** | a boundary term; its coefficient must carry a coordinate from that region |
| **Robin** `a·u·v − g·v` | a boundary term with the network evaluated *on* the face |
| **Mixed** (Dirichlet + flux on different regions) | works |
| **Vector**, all components or one (a roller) | works |
| **Complex coefficient** (a `1j` in the form) | works — the residual stays complex |
| **`fem_symbols(complex=True)`** | refused: a `ComplexPair` is two coupled real fields, i.e. multi-field. Use one field with `value_shape=(2,)` and a 2-output network for (Re, Im) |

!!! warning "An essential condition DECLARES; it does not impose"
    For a network trial `u(region) - g` says *which test functions vanish* — the value never reaches
    the residual. Measured with one fixed network: `- 0.0`, `- 0.5`, `- 7.0` and `- sin(πx)` all give a
    **bit-identical** residual. A non-zero value therefore looked like a boundary condition and did
    nothing, so it is now refused by name. Put it where the network can satisfy it exactly:

    ```python
    u_net = g + ansatz * net(xi, yi)     # ansatz vanishes on the region
    jno.fem([..., u(region) - 0.0])      # the declaration stays homogeneous
    ```

!!! warning "A complex form's `.mse` is a complex loss"
    `pde.mse` on a complex weak form comes back `complex128`, and minimising a complex number is not
    defined. Build a real objective explicitly — `(r.real**2 + r.imag**2).mean` — the same caveat that
    applies to a complex FEM inverse problem.

!!! warning "Scope — refused by name"
    * **Steady only.** The lowering test-projects onto a *spatial* FE basis, so a form carrying the
      time coordinate is refused. It used to build and evaluate, and the number meant nothing:
      measured on a heat form over `domain(time=(0, 0.1, 5))`, the spatial quadrature carried 120
      points against the time coordinate's 5, the declared grid never reached the residual (5 steps
      and 17 gave a bit-identical value), and the initial condition was discarded entirely
      (`u(initial) - 0` and `u(initial) - 7` also bit-identical). Use an FE trial for a transient weak
      form, or drive a time-dependent network as a **collocation PINN** through `jno.core`, where the
      residual and the initial condition are both explicit losses.
    * **A boundary coefficient must carry a coordinate.** A bound test function keeps its binding on
      the *view*, not in the expression tree, so once the weak form is flattened `-1.0 * v_right` and
      `-1.0 * v_interior` are indistinguishable and both read as volume. Write
      `(g + 0.0 * xr) * v_r` against that region's own coordinates; a bare constant is refused rather
      than integrated over the volume, which trains happily and is wrong (measured 3.9e-01 against
      6.8e-04 for the same problem). The FEM trial classifies the raw constraint, where the binding
      survives, and needs no such spelling.
    * **Single field.** The lowering wraps one primary unknown, so two separate fields raise —
      but a coupled system whose fields share a test space **is** one vector field, and that works.
      See below.
    * **No periodic ties.** A tie is an algebraic reduction of FE trial DOFs, and a network trial has
      none. Impose periodicity inside the network instead (a periodic input embedding).
