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

### What the fluid path is verified to do — and what it is not

Scope first, since it is not obvious from the API: jNO's FEM fluid path is **laminar incompressible**
flow and nothing else. There is no turbulence model (no RANS, no LES), no compressible or Euler path,
no free surface / VOF / level set, and no fluid–structure interaction. Nothing about `jno.fem` stops
you writing those terms; nothing in the library implements or verifies them.

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
* **No stabilisation.** There is no SUPG/GLS/grad-div term in the library, so convection-dominated
  flow is unaddressed. The cavity tutorial sits at Re = 200; the practical ceiling for unstabilised
  P2/P1 is somewhere in the low hundreds and has not been measured. `dom.cell_size` gives you the
  element size `h` if you want to write a stabilised form yourself.
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
displacement/arc-length control, which is a different instrument and is not built.

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
