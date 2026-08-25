# Solvers & preconditioners

`jno.solve` and `jno.precond` are **slots**: you hand one to `fem.solve(...)` or
`fdm.solve(...)` and it replaces the default without changing a single term of the
weak form. The same slot objects work for both front doors.

Between "accept the default" and "write a full `solve_fn`" sits the **slot API**: the solver
factorises into four orthogonal slots, each a configured **callable** (never a string) from the
`jno.solve` / `jno.precond` namespaces — or your own with the same contract. Every `None` keeps
today's default; `solve_fn=` stays the total override (passing both is an error).

```python
u = fem.solve(
    x0        = u_guess,                 # warm start (previous solve, coarse solve, a surrogate…)
    nonlinear = jno.solve.newton(),      # linearization driver (nonlinear problems)
    linear    = jno.solve.gmres(),       # inner linear solve: lu / dense / cg / bicgstab / gmres
    precond   = jno.precond.jacobi(),    # v -> M⁻¹v spec, materialized against the assembled A
)
```

Everything shipped is **pure JAX** — `jit`/`vmap`-native, differentiable (Krylov on
`lax.custom_linear_solve`, Newton on `lax.custom_root`), receives the assembler's **BCOO** operator
directly (no densification), and composes with the periodic reduction and every parametric/inverse path.
Pick by structure:

| structure | solver | notes |
|---|---|---|
| SPD (Poisson, elasticity, mass) | `cg` | cheapest per iteration |
| non-symmetric (advection, SUPG) | `bicgstab` / `gmres` | `bicgstab` with `jacobi()` is the default |
| **iterative preconditioner** (inner Krylov, block/Schur) | `fgmres` | flexible right preconditioning — Saad, *SISSC* 14(2), 1993 |
| symmetric **indefinite** (Stokes/Biot saddle, biharmonic) | `minres` | monotone residual, `O(1)` memory — Paige & Saunders, *SINUM* 12(4), 1975 |
| SPD, batched/GPU-heavy | `chebyshev` | inner-product free (no reductions) — Golub & Varga 1961 |
| indefinite, single solve | `lu` | sparse-direct; **no vmap rule** — use a Krylov solver inside batched solves |
| cuSolver refuses it, or is slow | `lu(backend="host")` | factors on the HOST (SuperLU) and drives it from the device; same answer and same gradients (wrapped in `custom_linear_solve`, transpose via SuperLU `trans="T"`). Measured **faster** where cuSolver also works — Stokes 21,839 DOFs 0.27 s vs 1.67 s, H(curl) 17,072 complex DOFs 13.3 s vs 36.4 s — and it runs meshes cuSolver rejects (Stokes 26,908, H(curl) 26,154, both of which fail on GPU). Affordable because a direct solve factorises **once**: the operator crosses PCIe once, not per iteration. Read the win as *cuSolver's sparse LU is weak*, not *GPUs lose* — see the row below |
| **shift-invert eigs, or a constant-operator transient** | `lu(backend="cudss")` | NVIDIA cuDSS — the **fastest** direct backend wherever it runs, because it separates the symbolic *plan* from the numeric *factorization* and jNO caches on the **sparsity**, so the plan survives a change of values. Against `backend="host"` on an RTX 3070 (fp64 at 1/64 rate — the *unfavourable* card): Stokes saddle factorization **3.4 ms vs 79.9 ms**, lap3d 50³ **576 ms vs 64,856 ms**, and **64.7× per Newton step** at n=64,000. Also factors the Stokes saddle cuSolver calls *singular*, with smaller residuals. Needs the optional stack (`nvmath-python`, `cudss`, `cupy`); raises a clear `ImportError` otherwise. Fill-in still governs 3-D (69×→218× nnz growth at lap3d 20³–40³), so it moves the ceiling and makes **device memory** the binding constraint — it is not a substitute for a preconditioner  Two things happen automatically: a **block right-hand side** `(n, k)` is solved in one call (measured **2.7× at k=4 to 5.4× at k=16** over the same factorization solved column by column — this is what the shift-invert eigensolver's subspace iteration needs every sweep), and an exactly symmetric operator is factored as **LDLᵀ instead of general LU** (1.41× faster, **1.38× less peak device memory** on lap3d 40³). Symmetry is tested to within a few ulps rather than bitwise — an assembled FEM tangent for a symmetric form is symmetric only up to *assembly round-off* (a vector element block contracts components in a different order for `(a,i),(b,j)` than for `(b,j),(a,i)`, measured **0.25 ulps**), so a bitwise rule sent every vector and coupled problem down the general-LU branch. Within the gate the two triangles are **averaged**, a correction bounded by the asymmetry it removes and therefore no larger than the round-off already in the matrix; outside it nothing is touched. The margin is not a knife-edge: the weakest *genuine* asymmetry that can be constructed — an advection coefficient of 1e-12, meaningless beside the Laplacian — is already **191 ulps**. Measured end-to-end on a 3-D vector tangent, general → symmetric is **1.07–1.13×** at 3.4k–11.8k DOFs (growing with size) and **1.10–1.62×** on lap3d 20³–50³. The rest of the rule stands: symmetry is still tested and SPD is never inferred — a matrix symmetric only to ~1e-15 would otherwise be silently factored as `(A+Aᵀ)/2`, and a wrong SPD guess returns NaN. A **replaced pivot** is first given iterative refinement against the true matrix (Wilkinson 1963 — the remedy cuDSS's own docs prescribe), so a zero-pivot-but-nonsingular operator or a marginally conditioned saddle is *solved* (refined to ~1e-10 per solve, one SpMV per step) rather than refused; a genuinely singular operator — one whose refinement residual will not contract — still **raises**: cuDSS signals it through neither an exception nor a NaN, returning a finite plausible vector instead |
| **a Newton loop** (or no GPU / a factorization too big for device memory) | `lu(backend="pardiso")` | Intel MKL PARDISO, multithreaded CPU. **The fastest factorization of the four**, and like cuDSS it splits symbolic analysis from numeric factorization, so a Newton step reuses the analysis. On lap3d 50³ (n=125,000) against single-threaded SuperLU's 65,212 ms: factorization **298 ms**, Newton re-factorization **296 ms — 220×**, where cuDSS reaches 115×. Its adjoint is cheaper too — `Aᵀx = b` comes from the *same* factorization rather than a second one. An exactly symmetric operator uses LDLᵀ (1.9× on lap3d 50³, 13× on a saddle), which needs the upper triangle with an **explicit diagonal** — without that a saddle's constraint rows come back empty and PARDISO rejects the matrix. Like cuDSS it returns finite garbage on a singular operator, so jNO checks the perturbed-pivot count and **raises**. `pip install jax-numerical-operators[pardiso]`, x86-64 |
| small systems / coarse blocks | `dense` | LAPACK, vmap-native |

> **jNO tells you when this applies.** `jno.fem` detects a saddle system structurally at build time —
> a field whose own test function never meets its own trial function contributes no diagonal block —
> and `fem.solve()` warns, naming the field, when it is about to use the matrix-free default on one.
> It warns rather than refuses: a 2-D saddle of moderate size does solve acceptably that way. Passing
> `linear=` or `precond=` silences it, since that is the deliberate choice it asks for. The detection
> is structural, so it holds in every mode with no tangent to assemble, and it fires under
> `jit`/`vmap`/`grad` — which matters, because the runtime residual guard needs a concrete residual
> and steps aside on a tracer, leaving a transformed solve otherwise unguarded.

> **Choosing between `cudss` and `pardiso`: pick by the phase your problem repeats.** A Newton loop re-*factorizes* every iteration, so PARDISO wins (220× vs 115× over SuperLU). A shift-invert eigensolve or a constant-operator transient re-*solves* against one factorization, and there cuDSS is ~11× faster per solve (3.5 ms vs 40 ms at lap3d 50³) and takes a whole block of right-hand sides at once. There is deliberately no `auto`: which wins depends on hardware jNO cannot inspect. Install both with `pip install jax-numerical-operators[fem]`.

**Preconditioner specs** (declarative — materialized against the assembled operator at solve time; a
preconditioner never changes the converged solution, only the speed, so specs need no gradient path):

* `jno.precond.jacobi()` — diagonal.
* `jno.precond.chebyshev(degree=…)` — fixed-degree Chebyshev **polynomial** preconditioner: matvecs and
  AXPYs only, the GPU-era substitute for Gauss-Seidel/ILU smoothing, and a fixed *linear* map so it
  legally preconditions `cg`/`minres`. Spectrum bounds come from `lmin`/`lmax` when you pass them, else
  **both** ends are measured by Lanczos (Lanczos 1950, §II — the extreme Ritz values), at the same cost
  as a power iteration on the top end alone. This matters because the polynomial is a contraction only
  *inside* the interval it is fitted to: the `lmin = lmax/30` guess, whenever the true ratio is smaller,
  leaves the lowest modes outside that interval where the polynomial amplifies them. Without the
  optional `matfree` package that guess is the fallback.
* `jno.precond.nystrom(rank=…)` — **randomized Nyström** low-rank preconditioner (Frangella, Tropp &
  Udell, *SIAM J. Matrix Anal. Appl.* 44(2), 2023, Alg. 2.1 and §3), the rung between `jacobi` and a
  multilevel method. Sketches `A` against a random `n × rank` matrix — exactly `rank` matvecs, no
  factorization, no triangular sweep — and deflates the captured top of the spectrum. That is what
  `jacobi` cannot do: a diagonal rescales, it cannot *separate* a few large outlying eigenvalues, and on
  such a spectrum Jacobi can be worse than no preconditioner at all (measured on a rank-15-outlier SPD
  system: 124 CG iterations for `jacobi`, 98 unpreconditioned, **46** for `nystrom(rank=20)`). **SPD
  only** — the sketch takes a Cholesky, so an indefinite operator gives NaN rather than a quiet wrong
  answer.
* `jno.precond.inner(solver)` — any `jno.solve` solver as the `M⁻¹` application (an inexact block/system
  solve). Iterative inner ⇒ flexible outer (`fgmres`).
* `jno.precond.form([...terms], inner=…)` — **preconditioners as weak forms**: assemble an auxiliary
  operator from ordinary traced terms and invert it as `M⁻¹` (weighted mass matrices, shifted-Laplacian
  Helmholtz twins, low-order proxies — written in the PDE's language). Pass a **callable**
  `form(lambda sol: [...terms...])` for a **solution-dependent** auxiliary — a `(1/μ(u))`-weighted
  Schur mass whose weight is computed from the current solution. It is re-assembled once per outer
  solve from that solve's entry iterate (warm start / previous march step) — the **Picard-lagged
  preconditioner** (Elman, Silvester & Wathen 2014, §9.2): the coefficient trails the solution by one
  outer solve, which changes convergence *speed* only, never the answer. Eager by necessity — every
  Newton loop is a `lax.while_loop`, so the per-step iterate is a tracer no host assembly can see; a
  fully traced solve that never supplies a concrete iterate gets a loud `NotImplementedError`, not a
  garbage preconditioner.
* `jno.precond.block_diag((field, spec), …)` / `jno.precond.triangular((field, spec), …)` — per-field
  composition over `fem.blocks`. `triangular` is the standard saddle-point shape: last block solved
  first, substituted back through the assembled off-diagonal matvecs.
* `jno.precond.saddle(mass_weight=…, laplace_weight=…)` — that standard shape as **one call**, for the common case.
  It finds the constraint block *structurally* (the field with no diagonal entry — the same detection
  behind the saddle-system warning), puts `amg()` on the momentum block and a weighted pressure
  **mass** matrix on the constraint block, and assembles that mass on the domain's own P1 space, so
  no symbols are passed and it reads identically in 2-D and 3-D. `mass_weight` is explicit and never
  inferred (`1/μ` for Stokes): digging `μ` out of an arbitrary weak form is fragile, and a variable
  viscosity would silently take the wrong weight — which costs iterations without ever failing. A
  wrong weight changes convergence *speed* only, never the answer. Pair with `jno.solve.fgmres`;
  `minres` does not apply (block-upper-triangular is nonsymmetric). Needs `pyamg`, and refuses by
  name on a non-saddle system or a constraint field that is not P1 — for anything outside that,
  compose `triangular` yourself. **The mass matrix alone is the pure-Stokes approximation:** a strong
  reaction term (Brinkman / Darcy drag, or the `1/dt` mass of a small implicit step) makes the Schur
  complement stop looking like a mass matrix, and the mesh-robustness degrades. `laplace_weight`
  switches to the **Cahouet–Chabard** approximation, a pressure mass *plus* a pressure Laplacian,
  `S⁻¹ ≈ μ·M_p⁻¹ + α·L_p⁻¹` (Cahouet & Chabard, *IJNMF* **8**, 869–895, 1988, §3). Both weights are
  the **reciprocal of the coefficient the term stands for**, so `mass_weight=1/μ` and
  `laplace_weight=1/α`. Measured through this spec on a 2-D Brinkman channel (`μ=1`, `mesh_size=0.12`,
  preconditioned GMRES to 1e-8, `restart=200`): `α = 0 / 1e2 / 1e3 / 1e4` → **83 / 135 / 503 / 2312**
  with the mass alone, **83 / 60 / 76 / 74** with Cahouet–Chabard. The point is the *flatness*, not the
  31× at `α=1e4` — the count stops tracking `α`. The same collapse appears with the momentum block
  inverted exactly instead of by AMG (31 / 63 / 134 / 189 → 31 / 26 / 25 / 24), which is how you can
  tell it is the Schur approximation doing the work and not the multigrid. At `α=0` the two are the
  same preconditioner — that column is one number measured twice — so leave `laplace_weight` at its
  `None` default on pure Stokes and the Laplacian is never assembled. The pressure Laplacian is pure-Neumann
  and therefore **singular** — and a sparse LU of it factors happily and then applies nonsense, so the
  auxiliary carries its own gauge pin and the applier projects the constant out on both sides. It is
  derived for a *constant* `α`; a spatially varying drag (topology optimisation's `α(ρ)`) takes a
  representative scalar and degrades gracefully rather than failing.
* `jno.precond.amg(cycles=…)` — **hybrid algebraic multigrid**: setup once on the host via the *optional*
  `pyamg` (Vaněk, Mandel & Brezina, *Computing* 56, 1996; PyAMG — Bell et al., *JOSS* 8(87), 2023),
  applied as a pure-JAX V-cycle with Chebyshev smoothing (Adams et al., *JCP* 188, 2003). The apply is
  `jit`/`vmap`-native and exactly linear, so it preconditions `cg`/`minres` too. Mesh-independent
  convergence ⇒ *the* choice for large elliptic blocks. Inside traced/parametric solves, pre-build
  eagerly: `spec = jno.precond.amg(); spec.build(fem.A)`. Without pyamg, `amg` raises a clear install hint.
* `jno.precond.jaxamg(symmetric=…)` — GPU AMG via NVIDIA **AmgX** as the `M⁻¹` application (setup and
  apply both on the device). `symmetric=False` builds a second hierarchy on `Aᵀ` so the adjoint
  (reverse-mode) solve of a non-symmetric operator is preconditioned too. Measured caveat: each AmgX
  application carries ~11 ms of fixed handle overhead, so it pays only where the iteration savings
  exceed it — for most problems prefer `precond.amg()` (whose V-cycle compiles into the solve) or
  `linear=jno.solve.amg()` (ONE AmgX crossing per solve, with warm structure-keyed re-setup measured
  ~10x cheaper on repeats). `benchmarks/amg_scaling.py` holds the numbers.
* `.cached(refresh=…)` on any spec — reuse an expensive setup across solves: `False` frozen, `True`
  rebuild on shape/sparsity change, an **int k** to rebuild every k-th materialization (the cadence
  for a march whose operator values drift step by step), or a `ctx -> key` callable.

The flagship pattern — Taylor–Hood **Stokes** by FGMRES with an inexact velocity block solve and the
viscosity-weighted pressure-mass Schur approximation (Elman, Silvester & Wathen, 2014, §9.2):

```python
sol = fem.solve(
    linear  = jno.solve.fgmres(tol=1e-10, restart=40),
    precond = jno.precond.triangular(
        (u, jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=60))),   # Â⁻¹: inexact CG
        (p, jno.precond.form([(1.0/mu) * pp * qq], inner=jno.solve.dense())),  # Ŝ ≈ μ⁻¹ M_p
    ),
)
```

…and the same recipe when the defaults suit — multigrid on the momentum block, exact pressure mass:

```python
sol = fem.solve(linear=jno.solve.fgmres(tol=1e-10, restart=150),
                precond=jno.precond.saddle(mass_weight=1.0/mu))
```

Give FGMRES enough `restart`: the default `restart=30` **stagnates** on a 3-D Taylor–Hood block
preconditioner and does so quietly — it still returns, just slower and less accurate. Measured at
4302 dofs, `tol=1e-10`: `restart=30` → 2.30 s for 3.3e-6; `restart=150` → 0.49 s for 3.3e-9. On
3-D Stokes this is what makes the recipe overtake a direct factorisation — measured crossover at
~15k dofs, and 3.4–3.9× faster (23.2 s → 5.9–6.9 s over repeat runs) at 41k, where LU's fill-in also
costs more memory. `benchmarks/saddle_scaling.py` runs that sweep.

On a **reaction-dominated** system — Brinkman/Darcy drag, or a small implicit time step — add the
Laplacian leg, and the iteration count stops tracking the reaction coefficient:

```python
sol = fem.solve(linear=jno.solve.fgmres(tol=1e-10, restart=150),
                precond=jno.precond.saddle(mass_weight=1.0/mu, laplace_weight=1.0/alpha))
```

**Picard / lagged coefficients — `jno.lag`.** When a solution-dependent coefficient's Newton tangent
destroys the linearized system's structure (the classic case: a shear-thinning viscosity `μ_eff(u)` in
non-Newtonian Stokes flow, whose full-Newton velocity block defeats AMG/block preconditioners), freeze
it with `jno.lag(...)` and drive with `jno.solve.picard()`:

```python
fem = jno.fem([2 * jno.lag(mu_eff) * inner(eps(ui), eps(vi)) - pi * div(vi), ...])
sol = fem.solve(nonlinear=jno.solve.picard(damping=0.7), linear=jno.solve.fgmres(),
                precond=jno.precond.triangular(...))
```

`lag` is `stop_gradient` on the traced expression, so the residual's linearization *is* the Picard
operator — each outer step re-solves the lagged system (linear convergence, but every inner system keeps
its symmetry/definiteness); the converged solution is identical to full Newton's. Without any `lag`
marker, `picard(damping=…)` is exactly damped Newton (`jno.solve.newton(damping=…)`). Caveat for inverse
problems: implicit differentiation then also uses the lagged Jacobian — drop `lag` when exact parameter
gradients matter more than per-step solvability.

**What did the solver do? — `fem.stats`.** After any `fem.solve()`, `fem.stats` reports what happened
without changing the solve's return: `mode`, `dofs`, `wall_s` (dispatch time — JAX is async; block on
the result for compute time), the `linear`/`precond` slot reprs, `nonlinear` (driver, final residual
norm against its bound, converged flag, and the step count where the driver runs its forward loop
eagerly — `newton_direct` reports steps; the drivers whose loop lives inside `custom_root` report
`None`), and `amgx_cache` occupancy when jaxamg served the solve. Populated on eager paths; a solve
wrapped whole in `jit`/`vmap`/`grad` records the slots but no residuals — the same concrete-only
self-disabling as the convergence guards.

```python
sol = fem.solve(nonlinear=jno.solve.newton(direct=True), linear=jno.solve.lu(backend="host"))
fem.stats
# {'mode': 'nonlinear', 'dofs': 44, 'wall_s': 0.31, 'linear': 'jno.solve.lu-host(...)',
#  'precond': None, 'nonlinear': {'driver': 'newton_direct', 'residual': 1.6e-07,
#                                 'bound': 1.7e-06, 'steps': 3, 'converged': True}}
```

**Alternate minimization — `jno.solve.staggered([u, d])`.** Some coupled energies are **non-convex in
the fields jointly but convex in each separately**. A monolithic Newton then has no descent guarantee
and simply diverges; solving one field at a time turns the problem into a sequence of convex solves:

```python
sol = fem.solve(nonlinear=jno.solve.staggered([u, dm]))   # sweep u, then dm, until both converge
```

Variational phase-field fracture is the canonical case — `(1-d)²|∇u|²` is quartic in the pair, while
`u` alone is a linear elasticity problem and `d` alone a linear elliptic one. Measured on the coupled
damage form in `tests/test_fem_staggered.py`: `newton()` leaves with a residual around **1e+25** (it
raises), and the staggered sweep converges to a genuine root of the *coupled* system. Fixed-stress Biot
poroelasticity and thermo-mechanical staggering have the same shape.

Algorithm: Bourdin, Francfort & Marigo, *Numerical experiments in revisited brittle fracture*, JMPS
**48** (2000), §3 — as the staggered operator split with a history field, Miehe, Welschinger & Hofacker,
IJNME **83** (2010).

**`direct=True` factorizes each field's diagonal block** rather than solving it matrix-free, and pairs
with a `linear=` slot:

```python
fem.solve(nonlinear=jno.solve.staggered([u, dm], direct=True), linear=jno.solve.lu(backend="pardiso"))
```

It exists because the matrix-free sub-solve **cannot be preconditioned**. A `precond=` spec materializes
against an assembled operator, and a staggered sub-problem is a restriction *closure*
(`x -> R(u with block set to x)[block]`) with no matrix, so an ill-conditioned block is solved by
**unpreconditioned BiCGStab** — near-incompressible elasticity (ν → 0.5) being the usual victim. That
was also why a direct `linear=` slot used to be refused against `staggered` outright: there was nothing
to factorize.

The extraction avoids a data-dependent `nnz` (which would break the static shapes a traced Newton
needs): instead of slicing `J[b][:, b]` out, the *complement's* rows and columns are zeroed and given a
unit diagonal, so the block is solved as `[[J_bb, 0], [0, I]]` against `[-r_b, 0]`. The padding is pure
diagonal — no fill-in — so the factorization cost stays the block's. With `bounds`, the matrix handed to
the factorization is the **min-map's** semismooth Jacobian (identity rows on the active set), which
`jax.linearize` derives for free on the matrix-free path but an assembled tangent does not.

Measured on a 3-D Yeoh phase-field march (576 DOFs, 8 load steps, CPU): **42.80 s → 10.78 s, 4.0×**, to
the same answer. Two honest caveats: the **full** tangent is assembled to use one block of it (a
sparsity-caching backend — `pardiso`/`cudss` — then pays only the numeric re-factorization per sweep),
and on a well-conditioned problem the matrix-free default is cheaper. This is not a free upgrade and it
is not the default. The load-path march threads the tangent too, which it previously never did: before
this, a direct driver inside a `tau=` path silently fell back to the matrix-free inner solve.

**`line_search=` globalizes each sub-solve's Newton steps**, and it is now on by default:

| value | what it does |
|---|---|
| `"backtrack"` *(default)* | residual-norm Armijo, halving from `damping`. What this used to do under `line_search=True`. |
| `True` | **exact** line search — bisect for the root of the directional derivative `R(x+λd)·d`, i.e. the minimizer of the energy along the Newton direction |
| `False` | no line search; take `damping` |

The default moved from `False` to `"backtrack"` because no line search is a genuine footgun on
finite-strain forms: measured here, a 3-D Yeoh P2 march produced **NaN on load step 1** without one
(the undamped step inverts an element, `det F ≤ 0`, and `J^(-2/3)` is NaN) and solved cleanly with one.

`True` implements Heinzmann, Vicentini, Carrara et al., *Iterative convergence in phase-field brittle
fracture computations: exact line search is all you need*, Computational Mechanics (2026),
[arXiv:2511.23064](https://arxiv.org/abs/2511.23064), §3 Algorithm 2 — the same algorithm they
contributed to PETSc as `SNESLineSearchBisection`. Their Props. 1–2 and Remark 4 chain into a
convergence guarantee for the whole alternate-minimization scheme, provided each sub-problem is
strictly convex and coercive (jNO's `bounds` min-map is semismooth, so that proof does not cover it —
the same gap they note for reduced-space active sets).

**It is not the default, because we have not measured it beating backtracking.** On the problems
tested here — a 2-D SENT plate with and without a volumetric-deviatoric split, at several load levels —
both need the same 21–22 staggered sweeps, and the exact search costs ~15% more wall time from the
extra residual evaluations. Note the paper's own failure cases arise only where the *mechanical*
sub-problem is non-linear (their §: the residual "reduces to an affine form in the absence of an energy
decomposition"), at critical load steps reached along a path — a regime not reproduced here. Reach for
`True` when a sub-solve stalls; the theory is on its side even where our measurements are neutral.

**`over_relax=ω` accelerates the sweep itself.** Alternate minimization *is* a nonlinear block
Gauss–Seidel iteration, so over-relaxation accelerates it exactly as it does the linear one — go `ω`
times as far along each sub-step's own update direction, per block:

```python
fem.solve(nonlinear=jno.solve.staggered([u, dm], over_relax=1.4))
```

Algorithm: Farrell & Maurini, *Linear and nonlinear solvers for variational phase-field models of
brittle fracture*, IJNME **109** (2017) 648–667, **Algorithm 2 (ORAM), §2.1**. `ω = 1` is plain
alternate minimization. Kahan's classical bound gives `ω ∈ (0, 2)` as necessary for SOR to converge, and
anything outside raises.

**Whether it pays is problem-dependent, and there is no way to know in advance.** The paper is explicit —
they "rely on the naïve strategy of numerical experimentation on coarser problems" and defer automatic
selection to future work. Their own results split cleanly: Table I (a propagating crack, where AM
converges slowly) gains **58–73% fewer iterations**; Table II (where AM already converges fast) gets
**0% — over-relaxation hinders it**, going 37 → 111 → 185 → 326 → 747 iterations as ω runs 1.0 → 1.8.
On a 2-D small-strain phase-field problem that drives damage to 1, ω = 1.4 was **1.95× faster**
(3096 → 1587 ms, warm median of 3). So it defaults to 1, and a short ω sweep on a coarse version of the
problem is the only way to know — the paper's own two tables disagree with each other.

**ω cannot diverge.** The extrapolation is not taken on trust: the step retreats by bisection on
`[1, ω]` until the trial point has a **finite** residual and is **feasible**, and `ω = 1` is the
sub-solve's own converged answer, already evaluated and therefore admissible by construction. So the
worst case is that the sweep degrades to plain alternate minimization. This matters most on
finite-strain forms, where an unguarded step past a converged answer inverts an element (`det F ≤ 0`,
so `J^(-2/3)` is NaN) — measured on the 3-D Yeoh SENT march, which NaN'd on load step 1 at *every* ω
down to 1.1 before the guard and runs to completion with it.

The test is finiteness and feasibility, **not** descent. Over-relaxation is deliberately not a descent
method, so demanding a residual decrease would reject almost every ω > 1 and silently collapse the
feature back to ω = 1 while appearing to keep it. Feasibility is the paper's own rule (§2.1 backs ω off
by bisection on `[1, ω]` for the bound constraint); finiteness is the generalization.

Over-relaxation also acts on the **free** DOFs only. Farrell & Maurini's `ũ` lives in the constrained
space `C_ū`, where a prescribed DOF has `δ = 0`; jNO imposes essential conditions as residual rows, so
without the mask the sub-solve's exact hit on the prescribed value gets extrapolated past — measured on
one row with `g = 2`, ω = 1.7 gave 3.40 → 1.02 → 2.69, an oscillation decaying only as `|1−ω|ᵏ`, worst
on a *ramped* condition where `g` moves every load step.

Cost when ω ≠ 1: one extra full residual evaluation per block per sweep.

Under `bounds`, over-relaxation steps *past* the sub-solve's answer — which is feasible by construction,
while the extrapolation need not be — so the driver takes the box projector from the `bounds` wrapper and
clips. That deviates from the paper, which backs the scalar `ω` off to the largest feasible value;
clipping is componentwise, also feasible, and keeps more of the step.

**The trade is the convergence rate, and it is not small.** Alternate minimization converges *linearly*
where Newton is quadratic, so it can need hundreds of sweeps near a propagating crack — hence the
`max_sweeps=200` default. It buys robustness, not speed: where Newton converges, Newton is the better
choice (Farrell & Maurini, CMAME **312**, 2017, compare the two directly). Sweeping is Gauss-Seidel, so
the **order matters**, and every field block must be listed — an unlisted field's equations would never
be solved, which is rejected rather than skipped. Each field is solved alone; sweeping a *group* of
fields together (a Stokes velocity/pressure pair inside one sweep) is not wired.

Differentiable in the ordinary way: at convergence the full residual is zero, so the sweep is just a way
of *finding* that root and `lax.custom_root` supplies the gradient from the full Jacobian — the
alternating structure is absent from the derivative by construction.

**Sparse-direct Newton — `jno.solve.newton(direct=True)`.** The default Newton solves each linear step
**matrix-free** (BiCGStab on the JVP), which stalls on an **indefinite / ill-conditioned** tangent with no
good preconditioner — a Taylor–Hood velocity/pressure saddle, a stiff Carman–Kozeny phase-change drag in a
melt pool. `direct=True` instead **assembles and factorizes** the tangent each step with a sparse LU (the
transient stepper factorizes the backward-Euler step tangent `M/dt + ∂R/∂u`; the steady path `∂R/∂u`).
It composes wherever the assembler provides that tangent — `fem.solve(nonlinear=jno.solve.newton(direct=True))`
on a native nonlinear problem, **steady or the transient march** — and stays differentiable: implicit
differentiation uses a *direct, transposable* tangent solve on the tangent assembled at the root (the adjoint
solves `Jᵀ` directly too, not with a stalling Krylov). `damping` / `line_search` apply unchanged. It needs the
assembled tangent, so it does **not** apply to the matrix-free-only paths (a coupled-residual wrapper,
complex) — those fail loud.

The tangent it factorizes is always the **element-scattered sparse** one, in 1-D as in 2-D/3-D. 1-D used
to build its nonlinear tangent with a global `jax.jacfwd` instead, on the assumption that only the
matrix-free default would ever ask for it; `direct=True` does ask, from inside the Newton loop, where a
dense tangent cannot be sparsified at all (`BCOO.fromdense` needs a concrete `nse`). Besides working, the
scattered tangent is `O(nnz)` rather than `O(N²)` — which is what 1-D node counts want.

**A direct `linear=` slot selects it.** `lu`, `dense` and `amg` all need an assembled matrix, so pairing one
with the *matrix-free* Newton has nothing to factorize. `fem.solve(linear=jno.solve.lu(backend="host"))` on a
nonlinear or transient problem therefore routes to the direct Newton, and that slot is the solver that runs on
the assembled tangent (and on `Jᵀ` in the adjoint); `precond=` materializes against the same assembled
operator. Which factorization you pick is not cosmetic here: on a 26-step Rayleigh–Bénard march (three fields,
saddle, nonlinear) the default matrix-free Jacobi-BiCGStab takes 20.1 s, `linear=jno.solve.lu()` 7.6 s and
`linear=jno.solve.lu(backend="host")` **3.1 s**, all to the same 2.8e-07 per-step Newton residual. An *explicit*
matrix-free `nonlinear=` alongside a direct `linear=` is contradictory and raises rather than picking one
silently.

> Limits, since they are not obvious. This routing only fires when the *linear* slot is direct — a
> `precond=` that needs an assembled matrix (`jacobi`, an unbuilt `amg`) raises where it is
> materialized. `picard()` has no assembled-tangent form: its linearization is the lagged
> *matrix-free* JVP, so `nonlinear=picard(), linear=lu()` raises — moving to `newton(direct=True)`
> there changes the algorithm, not just the solver. And the direct Newton needs the assembler to
> supply a tangent, so the matrix-free-only routes (a coupled-residual wrapper) fail loud either way.

**User extension** is duck-typed — a linear solver is any `fn(A, b, *, M=None, x0=None) -> x` with `A` a
`jno.solve.LinearOperator` (`.mv`, `.T`, `.diag()`, `.bcoo`, `.dense()`); a preconditioner is any
`ctx -> (v -> M⁻¹v)`:

```python
def my_precond(ctx):                      # ctx.A, ctx.diag(), ctx.fem
    inv = 1.0 / ctx.diag()
    return lambda v: inv * v
u = fem.solve(linear=jno.solve.cg(), precond=my_precond)
```

Calling a solver **directly** (outside `fem.solve`) takes the preconditioner as `M=`, which is the
*application* `v -> M⁻¹v`. A `jno.precond.*` spec is accepted there too and materialized against `A` on
the way in — `jno.solve.cg()(A, b, M=jno.precond.jacobi())`. A **bare callable** is always the applier,
never a `ctx -> applier` factory: as a `precond=` slot it would be the factory, nothing about a callable
tells the two apart, and guessing wrong would apply a preconditioner you did not ask for. Specs that
need eager preparation (`form`, which assembles an auxiliary operator) require `fem.solve(precond=...)`
— a direct call has no owning FEM.

If your callable is pure JAX it inherits `jit`/`vmap`/AD automatically. On the matrix-free **nonlinear**
path the `precond` spec is materialized *per Newton/Picard linearization* against the JVP operator — so
`form`, `inner(...)`, `chebyshev`, a pre-built `amg`, and their `block_diag`/`triangular` compositions
all work; only specs that need the assembled matrix (`jacobi`, an unbuilt `amg`) raise.

**Transient problems.** The slots configure the *per-step* solves of the default theta-method integrator:
`linear`/`precond` see the step operator `M + θ·dt·A` — when it is time-independent the step matrix is
formed **once** and the preconditioner materialized **once before the time loop** — and `nonlinear` drives
each implicit step of a nonlinear block. Second-order-in-time (`u_tt`) flows through the same augmented
block. Each step warm-starts from the previous state (so `x0=` is rejected).

> **`lu(backend="host")` factorizes a constant step operator once, not once per step.** The step matrix is
> formed once (above), and the host factorization is cached on the operator's *content*, so a march
> that solves against the same matrix every step pays one factorization for the whole trajectory — and
> the transpose solve reuses it, so the adjoint pass adds none. On a 51-step heat march that is worth
> 1.55× of whole-solve wall clock at 8,355 DOFs and **2.9× at 23,934**. The gain grows with mesh size,
> because factorization cost grows faster than the per-step solve; at 513 DOFs it is 1.05×, where the
> factorization is sub-millisecond.
>
> What it does **not** help: a *nonlinear* march, or any Newton loop. There the tangent's values
> change every iteration, so every call legitimately misses and pays a content hash (~1–2% of a
> factorization) for nothing. Reusing only the *symbolic* analysis — which genuinely is constant when
> the sparsity pattern is fixed — is not something SuperLU exposes through scipy. It is also
> host-path-only: `lu()` on GPU goes through cuSolver's `spsolve`, a single fused call with no
> factorization object to keep.

> **Adjoint memory: the march is gradient-checkpointed.** Reverse-mode through the default integrator
> rematerializes each step in the backward pass instead of storing every step's solver internals for
> the whole trajectory. Measured at 8,355 DOFs × 399 steps: peak memory **968 → 112 MB (8.6×)** for a
> gradient cost of **+60%** (2.98 → 4.76 s), gradient identical to 10 digits. The trade is
> deliberate: a differentiable march OOMs long before it is time-walled on consumer cards. A pure
> forward solve is unaffected — checkpointing is the identity outside differentiation.

**Complex problems** are assembled as one real `2n` system over the stacked `[Re; Im]` state — the
real-equivalent block `[[A_r, -A_i], [A_i, A_r]]` — at assembly rather than at solve time. A complex
transient is therefore an ordinary transient block (the slots configure its per-step solve as above, and
`theta` / `adaptive` / `exponential` all apply), and a complex steady problem is an ordinary linear
system: the `linear` / `precond` slots and `x0=` work on it, with a **complex** `x0` mapped into the
block layout for you. Its default solver stays **sparse-direct**, not the matrix-free BiCGStab real
elliptic systems get — the real-equivalent block is indefinite for Helmholtz/PML, where Jacobi-BiCGStab
does not converge.

One exception keeps a dedicated path: a **complex-native** preconditioner (`ams`) solves `A_r + i·A_i`
directly rather than the block, so the Re/Im legs are retained for it.

A **Bloch** (quasi-periodic) tie fuses like everything else. Its complex prolongation `P` cannot
reduce the Re/Im legs independently, but on the fused `[Re; Im]` state the same tie is the *real*
prolongation `B(P) = [[P_r, -P_i], [P_i, P_r]]`, and the ordinary real congruence `B(P)ᵀ A B(P)`
equals the Hermitian reduction `P^H A_c P` the Bloch space requires. Consequences: `solve_fn=`, the
`linear`/`precond` slots and `x0=` all apply to a Bloch problem (each used to be silently discarded by
a dedicated block routine); a Bloch tie composes with a **complex transient** (the quasi-periodic
plane-wave march, previously a dtype crash); and a **real** weak form with a Bloch tie is promoted to
the complex path automatically — the phase makes the field complex anyway, and the real path's
bilinear `Pᵀ A P` is not a Galerkin projection for a complex `P` (measured 8.1 rel-L2 off the
Hermitian answer on a manufactured mode, with the tie itself satisfied exactly).

A **coupled (multi-field) complex steady system** — coupled Helmholtz-type equations — takes the same
Re/Im split through the coupled assembler: one fused real `2n` block over `[Re_all; Im_all]`, with
`fem.offsets` still listing the per-field blocks of the recombined complex solution. Scope: steady and
linear (a complex *nonlinear* coupled form and a complex coupled *transient* refuse, as everywhere).

**Essential values on a complex form must be real.** The two legs share one Dirichlet row set, which
imposes `Re u = g` with `Im u = 0` — right for a real `g`, and the usual case (the complexity lives in
the operator and the source). A *complex* `g` is not expressible there: pinning `Im u = g_i` would need
the imaginary leg's rows zeroed rather than set to identity, and the symmetric elimination's
known-column lift is cross-leg (the real equation needs `A_r[:,j] g_r - A_i[:,j] g_i`, which no per-leg
elimination produces). It raises a clear error. Carry the complex part in the operator or the source.

`adapt=` composes with a complex **transient** too: the stacked `[Re; Im]` halves transfer across
each remesh as a doubled field layout, the **modulus** `|u|` drives the remesh metric (refining on
`Re` alone would miss a rotating phase), and the saved frames come back complex.

Not yet supported (clear errors): a Bloch tie on a **real** transient march (the phase forces a
complex field — make the problem complex, or use a plain tie) or on a **nonlinear** form (complex
nonlinear forms are not wired).
