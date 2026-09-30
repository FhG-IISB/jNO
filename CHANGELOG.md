# Changelog

## [0.4.0] — 2026-09-29

jNO 0.4.0 is mostly a correctness release. It fixes more than thirty bugs that returned plausible but wrong results, and many former silent failures now raise a named error. It also adds scalable preconditioners, new time integrators, moving-mesh adaptivity with checkpointing, mesh-free geometry sampling, `jno.info` diagnostics and multi-device execution. Read the breaking changes before upgrading: several APIs were renamed or retired and some defaults changed.

### Breaking changes and migration

**Renamed and removed**
- `jno.Shape` is now `jno.shape` (#131). `Shape` remains an alias. `from jno.geometry import shape` now gives the class, not the submodule.
- `d.by_region("steel", k=16.0)` and `d.by_tag(...)` are retired. Use `d.attach("steel", k=16.0)` and read it back as `d.k`. A bare `d.attach(k=1.0)` sets the default for unnamed volume regions (#142).
- `d.summary()` is retired: use `print(jno.info(d))`. `crux.print_tree(path)` and `crux.print_shapes()` are retired: use `print(jno.info(crux, deep=True))` (#142).
- The root exports `jno.dirichlet` / `jno.neumann` are removed. Write the condition as a term; `domain.dirichlet(...)` is unchanged (#132).
- `vec=` is removed from `jno.fem`. Drop the argument; the inferred value is used (#123).
- The grid-built multigrid V-cycle is removed. `jno.precond.gmg()` keeps its name and now builds from the operator being solved (#145).

**Changed defaults, behaviour and shapes**
- The default nonlinear Newton uses the assembled tangent with an iterative inner solve (#149). The inner solve checks its answer and falls back to Jacobi-GMRES (#150). `direct=False` restores the matrix-free Newton.
- A transient march checks every step, and the linear-solve residual check now fires under asynchronous (GPU) dispatch and on the transpose solve used by gradients (#123, #130). Solves that returned a non-converged iterate now raise. An eager `LinearSolver` call blocks on its result.
- `jno.np.transpose` defaults to the tensor transpose of the trailing two axes. An explicit `axes=` is unchanged (#130).
- `jno.callbacks.engd` defaults to `line_search=True`, matching `jno.optimizers.engd` (#129).
- A second `solve()` on the same `jno.core` continues the optimizer state instead of restarting optax schedules. Build a new `jno.core` to restart (#115).
- `shape(...).domain()` is mesh-free until something mesh-derived is read; `.mesh` is built lazily. `variable(tag, sample=(n, None))` draws `n` fresh points instead of clipping to mesh nodes. A shape with `size=` keeps its old meaning (#121).
- Geometry sampling is seeded per domain: a run repeats exactly, and successive draws still differ (#134).
- A march with a geometry term infers runtime connectivity, so a reconnection does not recompile. P2/P3 fields, contact gaps and surface readouts fall back to baked connectivity; an explicit `d.dynamic_topology()` refuses them (#143).
- On a multi-device machine, PINN training and FEM solves and marches use every visible device automatically (#149). `fem.solve(shard=False)` keeps a solve on one device; to train on one device, make only that one visible (`CUDA_VISIBLE_DEVICES`).
- FEM assembly runs on the host by default on a GPU backend, and the finished operator moves to the device once; an explicit `jax.default_device` overrides it (#126).
- A structurally singular system is reported at `solve()`, not at build (#129). Non-convergence errors count steps from 1, and after a failed solve `fem.stats` describes that solve, with `error` (#142).
- On a vector network, `net(x)[..., i]` and `net(x).vector[i]` are component `i` on every trial and path, with shape `(N, 1)`, as are `.real`/`.imag` of a per-point scalar. A bare `[i]` indexes points; FE symbols keep `u[i]` as a component (#150).
- Context tensors `(B, n, k)` and `(B, T, k)` are stored as `(B, 1, n, k)` and `(B, T, 1, k)`. Ambiguous shapes, and lazy sources in these layouts, raise (#150).

**New refusals** (in addition to the silent wrong answers below that now raise)
- `jno.domain(shape, mesh_size=...)` raises; it never applied the size. Use `shape.sized(h)` (#129).
- `domain.variable(tag)` without a count, on a mesh-free domain without `size=`, raises and names both ways out (#121).
- `grad(u, X)[i]` on FE symbols raises at build (use `[..., i]`), as do `fem.eigs` with a trainable, a VPINN on a batched domain, and a time variable evaluated without a time (#150).
- A linear or transient form with a moving slip surface raises at build (#146).

### Fixed: silent wrong answers
- Calling `solve()` in chunks restarted every optax schedule on each call (#115).
- `crux.eval` returned the parameter value from when `jno.core` was built, ignoring later edits to the module; it now raises and names the working spellings (#117).
- The slip condition `n·u = 0` on a nodal field was parsed, then dropped from the solve (#118).
- Slip constraints kept build-time normals when `.trainable()` coordinates moved the surface (#146).
- The matrix-free default under-solved saddle-point systems; `solve()` now warns, naming the field (#119).
- `jno.np.parameter` leaked float32, so under x64 `jno.fdm` round-tripped every residual evaluation through single precision (#123).
- The transpose (adjoint) linear solve behind every gradient, including the default `fem.solve()`, was never checked for convergence (#123).
- `applyfun` degraded silently past the Krylov dimension; `order` is now an upper bound, and `logdet`/`trace`/`diagonal` raise instead of returning NaN (#123).
- A volume term on a named cell region was misplaced; it is now refused (#126).
- Adaptivity indicators assumed field 0 starts at DOF 0 and mixed the components of an interleaved vector field (#127).
- A multi-component `jno.np.parameter` used only its first entry (#128).
- A Neumann flux could fall through a duplicated term classifier into the volume channel and be dropped (#128).
- A transient VPINN ignored the time grid and initial condition, and an inhomogeneous VPINN Dirichlet value did nothing; both now raise (#128).
- `dom.cell_size` off the native path returned `h = 1.0` (#130).
- `jno.np.transpose` also transposed the quadrature axis of a tensor field (#130).
- AMG built a NaN/zero coarse grid on an indefinite operator; it now refuses (#130).
- A transient march returned capped or diverged Newton steps as a trajectory (#130).
- On GPU, a failed linear-solve residual check was swallowed into a log line (#130).
- A curved tied interface fell back to collocation coupling, which fails the patch test, instead of mortar (#133).
- The `every="step"` driver on a periodic transient block sliced the reduced state with full-space offsets; it is now refused (#139).
- A scalar factor on a weak term deleted its stiffness (#140).
- A march that never solved returned its input; it now raises. A mixed-order group the splitter cannot reach is refused (#140).
- On the rebuild path, `Domain.sample` wrote to an alias, so constraints read stale points (#143).
- `jno.fdm`: `u.tt`, flux conditions on structured and coupled grids, unknown region tags, coupled-field order, periodic ties leaking into later problems, and parametric/tracer leaks now work or raise (#145).
- The nonlinear θ-step tangent missed θ, so Crank–Nicolson gradients were wrong (#149).
- Vector-network components meant different things on different trials, so vector PINN residuals were wrong; the coupled VPINN example in the docs coupled point 1 (#150).
- An initial condition mentioning `t0` read `t0` as `x` (FEM nodal, non-nodal, 1-D and FDM) (#150).
- VPINN with P2 test functions divided vertex rows by ∫φ ≈ 0 (#150).
- A trainable `jno.nn` coefficient inside `jno.fdm` stayed at its stored weights (#150).
- Per-node data `(B, n, k)` on a batched domain reached only node 0; per-step data `(B, T, k)` gave every step step 0's value or the whole series (#150).
- The adaptivity criterion divided by the signed ∫φ, ≈ 0 at P2 vertices; from P2 on it is a consistent-mass L2 projection (#150).
- A network given its optimizer after an eager `jno.fdm` solve was never trained; this now raises (#150).

### Added

**FEM solvers and time integration**
- `jno.solve.bdf2()` (#130), also for a state-dependent mass (#139); `jno.solve.sdirk(order=2|3)` and linearly implicit `jno.solve.rosenbrock("ros34pw2"|"ros2")` (#149).
- `jno.solve.continuation`, including a direct Newton on a reduced system (#118).
- `jno.solve.newton(direct=True, reuse=True)`, a lagged-Jacobian Newton; `fem.stats["nonlinear"]["factorizations"]` counts factorizations (#146). It differs from `lu(reuse=...)`, which keeps a factorization cache.
- Anderson acceleration via `picard(anderson=m)` and `staggered(anderson=m)`, and `vmap` batching rules for the LU backends (#149); `jno.solve.cocg` for complex-symmetric systems (#126).
- `fem.solve(k=value)` solves a parametric problem at a value without rebuilding it (#118, #150); `fem.solve(contact=...)` re-searches contact pairs from the deformed configuration (#129).
- `jno.derived(fn, inputs=[...], on=u)`: a nonlocal quantity that enters a weak form as a nodal value, with a lagged sparse tangent (#139).
- `p.pin(mean=True)` re-levels the pressure to zero mean (#119); `dom.cell_metric` and second derivatives of vector fields, for SUPG/PSPG/grad-div (#130).
- Complex non-nodal FEM: the complex steady form assembles once; non-nodal spaces use two-pass sparse assembly (#126).

**Preconditioners**
- `jno.precond.saddle(mass_weight=, laplace_weight=, schur=)`: block-triangular Stokes preconditioner, with Cahouet–Chabard for reaction terms; pair it with `fgmres` (#120, #122, #130).
- `jno.precond.lsc()`, `jno.precond.pcd()`, and `+` / `@` on preconditioner specs (#130).
- `jno.precond.schwarz(parts, overlap, coarse, restricted, nullspace)`: overlapping Schwarz on a METIS partition with a coarse level, for `jno.fem` and `jno.fdm` (#149).
- `jno.precond.fsai(power)` for SPD systems, and `float32=` on every `jno.precond` constructor (#149).
- `jno.precond.real_equivalent(inner)`, `hypre(kind="ams"|"boomeramg")` via PETSc, a host-only `ilu()`, `inner(..., precond=)` and `form(..., inner=)` (#126).
- AMG can precondition a march, built once before the scan; `cached(spec, refresh=k)` rebuilds it every `k` steps (#140).

**Adaptivity and moving meshes**
- p-adaptivity: `space="cover"` elements and `fem.solve(adapt=jno.solve.enrich(criterion=...))` (#127).
- `relocate(objective=)` takes a weak form; `cell_aspect()`; an inequality mesh condition adds nodes when moving them stops helping; moving meshes accept per-step solver slots (#118).
- ALE mesh velocity, `remesh(alpha=...)` reconnection so bodies can merge, and free-surface capillary traction (#139).
- Runtime connectivity (`Domain.dynamic_topology()`), conservative L2 state transfer across a remesh, and a per-node `alpha` length scale (#143).
- `relocate()`, `relocate(escalate=tol)` and `remesh(anisotropic=True)` on a moving mesh; Monge–Ampère on a disconnected mesh (#143).
- `fem.solve(checkpoint=jno.solve.checkpoint(path))` with `keep="last"` and `resume=True`, for `adapt=` marches; `JNO_MARCH_MEMDEBUG=1` and `JNO_RELOCATE_STRICT=1` (#143).
- Named volume regions from a mesh file, carried through local refinement and metric remeshing (#126).

**FDM** (#145)
- Vector and tensor unknowns; coupled, time-dependent and second-order-in-time systems, including algebraic fields.
- `jno.fd(...)` stencils (any order, unstructured local fits, upwinding), `linear=` / `precond=` slots on `jno.fdm`, and affine value conditions.
- Multigrid built from the operator (Galerkin coarse operators, block smoother) behind `jno.precond.gmg()`, the structured linear path and the Newton inner solve.

**NN + FEM / training**
- `fem.eval(F, u)` of a test-free `F` returns ∫F dΩ, or ∮F ds on a tag; `F.integrate(fem)` makes it a differentiable objective for `jno.core` (#144).
- `fem.residual` covers linear forms and accepts a field from any source (#128).
- VPINN in 3-D, with `div`, constant scalar sources and an inverse parameter; a frozen field from another form resolves by space (#128).

**Geometry**
- `shape` domains sample analytically in 1-D, 2-D and 3-D, boundaries, tags and `revolve` included, without gmsh (#121).
- `shape.sdf(points)` / `shape.sdf(x, y)`: signed distance, exact for rect, box, disk, sphere, cylinder and polygon, usable for hard boundary conditions (#137).
- Graded mesh size `size=f(x, y, z)` (#130).

**Diagnostics: `jno.info`** (#142)
- `jno.info(obj)` describes any jNO object; `deep=True` for detail; `jno.info.REGISTRY` for other types.
- `fem.solve()` logs what was built, which solver ran and what came back; new `fem.mode` and `fem.stats` keys `march`, `error`, `solve_index`.

**Multi-device** (#149, including #148)
- PINN collocation points split across devices; each device used to evaluate all of them.
- Linear transient marches split `M` and `A`; nonlinear solves and marches split the element loop, under every time scheme; `schwarz` places one block of subdomains per device under `shard=`.
- An assembled tangent is built on one device. Marches with solver slots, a parametric operator, or inside a trace stay on one device unless `shard=` is given.

### Fixed
- `crux.solve(inner_steps=)` could not run; `lu(backend=)` on a dense operator now logs that the backend is ignored (#123).
- Now accepted instead of raising: `sum(f[k] * v[k] for k in ...)` as a source (#119), a frozen field used only in a boundary term (#139), a surface readout on a volume-only form (#144).
- AMG's Chebyshev smoother no longer trusts a bound that can amplify the residual; it falls back to Gershgorin (#140).
- Preconditioner auxiliary forms assemble sparse, not dense (#130). The factorization cache holds a block preconditioner, `lu(reuse=False)` stops a Newton march leaking a factorization per step, and re-naming a geometry region by tag warns (#140).
- A march rebuild releases its compiled program; two rebuild-path leaks are fixed (#143).
- Pressure gauge after a mesh change, h-adaptivity with several trial functions, surface objectives across a remesh, runtime-parameter kwargs in the moving-mesh driver, differentiating a parametric solve; a perturbed cuDSS pivot is refined, not refused (#118).
- Tied-interface, mortar and tag-ownership fixes; `domain.tag(..., region=)` owns one body's facets and normals (#129).
- Crashes: `fem.solve(continuation=..., x0=...)` on a reduced system (#146); reverse mode through a march whose Newton uses a solver slot (#149); a PINN on a domain that already carries a `jno.fem` problem, and a nonlinear transient on a non-nodal space (#150).
- Resampling on a CPU+GPU build put arrays on two devices (#135); explainability gradient-tracker cosines use full matmul precision (#150).
- AMS on a mixed operator reports it is unbuilt, and the `real_equivalent` inner no longer freezes against the complex operator (#126).
- Unsupported VPINN cases (multi-field, periodic ties, a component-first `stack`, a coordinate-free boundary coefficient) refuse by name instead of failing on an internal tag (#128).
- The JIT warm-up compiles without walking the parameters (6b9f8f0d). A parameter-only fit is not refused as ambiguous between two solve meshes (db171ac3). `sym(grad(u))` stays linear (e57524dc). Triangle-facet membership slack is above float32 eps (8eb9886d).
- An error message and docstrings named the non-existent `init_fem()`; they now name `init_fem_native` (#132, #136).

### Packaging, CI and docs
- `meshio` and `scipy` are direct dependencies (c1c8c069). Optional `pymetis` comes with the new `[metis]` extra, pulled in by `[fem]` and a new `[fdm]` extra (#149). `benchmarks/` is no longer in the repository (163e4a3c).
- `publish-pypi` refuses a tag that does not match the `pyproject` version; `docker-release` runs the suite one process per file (06c89dc7).
- Tests: `jax_enable_x64` is scoped per test, with leak guards (#116); `test_fdm.py` and `test_rcwa.py` join their CI groups (#117); every test file passes on GPU (#135); RCWA tests run in float64 (5e6eba7c).
- New cross-path tests: FEM, VPINN, FDM and PINN on one problem, and trainables through every solve type (#150). The drop-oscillation test no longer pins an over-damping that was an assembly defect (6b7a6ae9).
- Documentation rework: `mkdocs build --strict` clean, the FEM guide split into pages, new domain-decomposition and troubleshooting pages, tutorials embed their verified scripts, a test resolves every `jno.*` reference in `docs/`, and the README gains inverse and coupled Rayleigh–Bénard examples (#123).
- New docs: `docs/info.md` (#142), a VPINN section (#128), moving-mesh sections in `docs/fem/geometry.md` (#143), scope limits in `docs/fem/limitations.md` (#127, #143, #144, #146). Docs and docstrings match the code (5e199722); `jno.fd` parameters are annotated (deb373f2).
- New tutorials: 3-D Stokes and Navier–Stokes and the DFG 2D-1 cylinder (#119); stabilised flow, vortex shedding, high-Re cavity, LES and a laser melt pool (#130); droplets (#139).
