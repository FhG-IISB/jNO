# Differentiable solve & inverse problems

## Coefficient fields — known (`.freeze()`) vs trainable

A coefficient in the weak form (a conductivity `k`, an emissivity, a source weight) can be a plain
constant, a **coordinate function** `jno.fn(lambda x, y: ...)`, or a `jno.np.parameter` — written
straight into the math like any other value:

```python
k = jno.np.parameter(phi).initialize(lambda x, y: 1.0 + 4.0 * x).freeze()   # KNOWN coefficient
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
u_h = jnp.linalg.solve(fem.A, fem.b)                                        # non-parametric forward solve
```

A `jno.np.parameter` is a **trainable** unknown by default — it makes the system runtime-parametric
(resolved through `crux`, below). Marking it **`.freeze()`** declares it a *known* coefficient:
`jno.fem` evaluates its `.initialize` value at the quadrature points — exactly like `jno.fn` — so the
system assembles non-parametrically (`fem.A` / `fem.b`, no `crux`) **and works in every form**
(steady-linear, nonlinear, transient, coupled). The frozen value is a **scalar** (`.initialize(3.0)`)
or a **coordinate function** (`.initialize(lambda x, y: ...)`, scalar- or vector-valued); a raw
per-node array, a JAX initializer, or no value all fail loud. Leave the parameter **un-frozen** to
make it an inverse unknown — the next section. A vector-valued coefficient is best written **per
component** with scalar functions (a single function returning a tuple hits a kernel limit shared with
`jno.fn`).

---

`fem.solve()` is the **differentiable forward solve as a trace node** — the entry point for
inverse problems. Put a `jno.np.parameter` in the weak form, compare `fem.solve()` to data, and
train the parameter through `crux.solve`. The gradient flows through the solve back to the
parameter (see also [Inverse problems](../inverse-problems.md)).

```python
import jax, optax
k = jno.np.parameter((1,), name="k")                      # unknown scalar
k.dtype(jnp.float64); k.initialize(jax.nn.initializers.constant(2.0)); k.optimizer(optax.adam(5e-2))
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs_domain)
crux.solve(200)                                           # recovers k
recovered = crux.eval([k])                                # the array (do not index [0])
```

`fem.solve(solve_fn)` lets you choose the solver, but every problem ships with a differentiable
default (no external dependency): the linear default is a sparse-direct factorisation
(`sparse_lu_solve`, JAX `spsolve`), with a Jacobi-preconditioned matrix-free BiCGStab as the iterative
alternative; the nonlinear default is a matrix-free Newton-Krylov, and the transient default
backward-Euler over those. All are implicit-diff, so `crux.solve` recovers parameters through them.

### Solving *at* a value — `fem.solve(k=...)`

A parameter is not always an unknown to recover; often it is a knob you sweep. Give it a value and the
solve happens **now**, returning the array rather than a node for `crux`:

```python
cap = jno.np.parameter((1,), name="cap")
fem = jno.fem([...])                                # built ONCE
x = None
for v in np.geomspace(G / 2000.0, G, 8):            # a continuation, as a plain Python loop
    x = fem.solve(cap=v, x0=x, nonlinear=jno.solve.newton(direct=True))
```

The alternative — rebuilding `jno.fem` per value — re-meshes, re-assembles and re-compiles the whole
problem to change one number. Measured on an 8-value sweep: **6.20 s rebuilding, 0.55 s this way**.
The solve is staged **once** (6 tracings for 8 values, not 48) because the value arrives as a runtime
argument and jit keys on shape, not on value; `x0=` warm-starts across the sweep without re-staging.

It composes with everything the ordinary solve does, including `newton(direct=True)` on a **reduced**
(slip / periodic) system — the case `fem.solve(continuation=...)` still cannot serve, since that driver
hands its solver no assembled tangent. `fem.stats` reports the verdict as usual: the jit hides the
driver's own convergence check, so the judgement is remade outside it, on the *reduced* residual where
there is one.

Give **every** parameter a value or none: a partly-supplied problem is refused by name, and with none
supplied `fem.solve()` is the trace node it always was, for `crux` to resolve.

---

### Multiple devices — `fem.solve(shard=...)`

Sharding is **automatic**: on a machine with more than one visible device, the assembled operator's
nonzero axis is partitioned across all of them, each device scatter-adds its slice, and one
`all-reduce` combines the partials. The operator shards; the vectors stay replicated.

```python
u = fem.solve()                 # automatic: every visible device
u = fem.solve(shard=False)      # opt out — single device (1 means the same)
u = fem.solve(shard=2)          # pin a device count; over-requesting fails loud
u = fem.solve(shard=jax.devices()[:4])   # pin exactly these
```

It is on by default because the change is **answer-preserving** — same operator, same solvers, only
the reduction order moves (~1e-14) — and because the realistic alternative on a multi-GPU box is not
a tuned single-device run, it is idle silicon. On a single-device host it resolves to the untouched
single-device path, so the default carries no risk there.

The reason no solver needed changing is that the operator is ~100× the vector, so replicating the
vectors costs nothing and removes the entire distributed-FEM apparatus: no mesh partitioning, no halo
exchange, no ghost DOFs, no DOF renumbering. Every Krylov step is either a matvec (sharded,
`all-reduce` inside) or a vector operation on replicated data (identical on every device, no
communication).

**What shards:** the default steady-linear solve, and the slot-composed solve
(`linear=` any Krylov solver, with `precond=None` or `jno.precond.jacobi()`).

**Parametric / differentiate-through solves shard too, but only on an explicit `shard=`:**

```python
u = fem.solve(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi(), shard=4)
```

`device_put` cannot place a tracer, so this route uses `lax.with_sharding_constraint` from inside the
trace. Gradients flow through it unchanged. It is opt-in rather than automatic for a safety reason,
not out of caution: inside a trace jNO is a guest in someone else's computation, and a sharding
constraint has to agree with the device commitments of every other value in that `jit`. Under `crux`
it does not — the optimiser's parameters arrive committed to a single device while the constraint
spans all of them, and JAX rejects the mix. That conflict cannot be detected in advance
(`get_abstract_mesh()` is empty there) and cannot be caught locally, because it surfaces when the
*outer* `jit` compiles, long after the solve was traced. There is no fallback to write, so automatic
placement leaves traced operators alone; an explicit `shard=` is a request you can diagnose.

> Two traps on this route were invisible in the answers and only showed up in the compiled HLO, which
> is why the tests assert on collectives rather than on values. Padding the triplet axis to a multiple
> of the device count makes XLA `all-gather` the **whole operator** onto every device to feed the
> concatenate; constraining an *uneven* axis makes it gather the index array instead (the same 8 bytes
> per triplet as the data). Both produced correct answers and correct gradients with the memory saving
> entirely gone. jNO therefore shards the divisible prefix and leaves the sub-device-count tail
> replicated — at most 3 triplets on a 4-device run.

**What does not** — each falls back silently to the single-device path rather than raising:

| | why |
|---|---|
| sparse-direct branches (periodic, 1-D, fused-complex) | route to `spsolve` — single-device, no batching rule |
| `linear=jno.solve.lu()` | `spsolve` is single-device with no batching rule — a genuine wall, not a wiring gap. Distributing it means a distributed sparse-direct solver (SuperLU_DIST class), which is not a placement change |
| `linear=jno.solve.dense()` | not wired. Dense LU with partial pivoting shards poorly, but the `N²` matrix itself would split — the win here would be capacity, not speed |
| `precond=amg()` / `ams()` | the hierarchy is built host-side through scipy/pyamg; distributing the V-cycle is a distributed-AMG project, not a placement change |
| `precond=chebyshev()` / `form()` | not wired yet, and **not** a hard limit — Chebyshev is matvec-only by construction (spectral bounds by power iteration), so it composes with the sharded matvec directly; `form`'s auxiliary operator is just another assembled BCOO |
| other `precond=` | the applier closes over the assembled operator, so a full copy would be replicated anyway. Jacobi is the exception: it needs only the diagonal, computed from the *sharded* triplets |
| parametric / differentiate-through solves | **opt-in only** — needs an explicit `shard=`, see below |
| transient | not wired yet. A sharding constraint inside the `lax.scan` body already produces the right collectives with the operator still closed over; threading it in as a jit argument additionally makes the per-device footprint provable (measured: exactly `nnz/N` per device) |

No speedup figure is quoted here because none has been measured — the development machine has one
GPU. What *is* verified, on simulated devices, is correctness, the even split, that XLA emits
`all-reduce` and **zero** `all-gather` (no device ever reconstitutes the matrix), and that the
fallbacks decline rather than silently gathering.

### Reduced-order solves — `fem.solve(basis=U)`

A periodic prolongation and a **reduced-order basis** are the same object: a tall `(n_dofs, k)` map `U`
defining `UᵀAU`, `Uᵀb`, and the lift `u = U x`. So `basis=` reuses the reduction the periodic ties
already drive, and the answer comes back in the **full** space — nothing downstream changes.

Solve the family a few times, keep the recurring shapes, then every later solve costs `k` unknowns:

```python
snapshots = jnp.stack([build(p).solve() for p in sweep])    # (n_snapshots, n_dofs)
U, s, Vt  = jno.solve.svd(snapshots.T, k=10)                # columns of U are the spatial modes
u = build(p_new).solve(basis=U)                             # 10 unknowns; full field returned
```

Mind the orientation: for a `(n_snapshots, n_dofs)` snapshot matrix the spatial modes are `Vt.T`, and
for its transpose they are `U`. Passing the wrong one is refused by shape, with the fix in the message.

This is the **only** path here that returns an approximation, so it is measured rather than trusted: the
relative residual of the full system at the lifted solution is computed each call (one matvec), kept on
`fem.basis_residual`, and a basis that does not span the solution **raises** instead of returning a
plausible wrong field. Deliberately coarse work (a rank sweep) can raise `fem.BASIS_RESIDUAL_LIMIT`.

The basis is per-call, must be orthonormal (a non-orthonormal one would need `(UᵀU)⁻¹` when restricting
a state, and would be silently wrong), and composes with `linear=` / `precond=`, which see the reduced
operator. `∂u/∂U` flows under `jax.grad`, so the subspace itself can be **learned** — note an orthonormal
basis lives on the Stiefel manifold, so put the orthonormalisation inside the differentiated function
(`net -> QR -> basis`) rather than projecting the step afterwards.

**Transient too** — and that is what a ROM is really for, since the cost avoided is a whole time
integration rather than one solve. The block is reduced once at solve time (`PᵀMP`, `PᵀAP`, restricted
`state0`) and the marcher steps in the reduced space, returning the trajectory at full width:

```python
snaps = np.concatenate([np.asarray(build(p).solve().fn()) for p in sweep])   # (n_sweep*n_t, n_dofs)
U = np.linalg.svd(snaps.T, full_matrices=False)[0][:, :8]
traj = build(p_new).solve(basis=U).fn()                                     # 8 unknowns per step
```

A transient solve is certified differently: the steady residual has no analogue, so what is measured is
the **projection error of the initial state**, `‖u0 − U Uᵀ u0‖/‖u0‖`. If the span cannot represent where
the trajectory starts, the march is wrong from step 0. It is a floor, not a bound — it says nothing
about whether the span keeps up *later* (measured on a nonlinear case it came in below the true
trajectory error), and the docstring is explicit about that.

Scope, each refused with its own reason: **second-order-in-time** (`u_tt` marches the augmented `[u; v]`
state, so a field basis needs `blkdiag(U, U)` and the row convention is unsettled), **complex** (solves
through an internal real-equivalent 2n layout), a **periodic tie** (composing two prolongations has no
decided convention yet), and a `jno.np.parameter` basis (a trace node, not an array — `jax.grad` over a
concrete basis is the supported differentiable path). A reduced **nonlinear** solve works, but is a
memory win, not a speed one: the full-order residual is still evaluated per Newton step (no
hyper-reduction).

### Field parameters `k(x)` + regularization

`jno.np.parameter(phi)` is a **nodal field** on the trial space — a trainable value per node. Field
inversion is ill-posed, so add a smoothness/structure prior with `k.regularize(...)` (`"h1seminorm"`,
`"l2"`/`"tikhonov"`, `"tv"`, `"nonneg"`, `"bounded"`):

```python
k = jno.np.parameter(phi, name="k")                       # P1 field, one DOF per node
crux = jno.core([(fem.solve() - u_obs).mse, 1e-3 * k.regularize("h1seminorm").mean], domain=obs)
```

### Neural coefficients — `jno.nn(net)` inside the weak form

A network called inside a weak form is a trainable **coefficient** on an assembled FE system —
mesh-independent (remeshing never touches the weights), smooth by architecture, and trained through the
same differentiable `fem.solve()` as any parameter:

```python
net = jno.nn(foundax.mlp(2, hidden_dims=16, num_layers=2,
                              activation=jax.nn.tanh, key=key))
net.dtype(jnp.float64)                                       # match the f64 assembly
net.optimizer(optax.adam(1e-2))

# k(x) = 1 + net(x, y): the offset keeps A(θ) nonsingular at the (near-zero) net init
fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs).solve(600)   # trains the weights
```

The kernel re-evaluates the network at the quadrature points during every re-assembly, so the
coefficient is *not* interpolated on the mesh — it composes with scalar/nodal parameters, per-region
masks, vector trials, and surface (Robin/Neumann) terms. `net.freeze()` makes it a **known** network
coefficient. This is the unsupervised coefficient-recovery setting of NN-EUCLID (Flaschel, Kumar &
De Lorenzis, *J. Mech. Phys. Solids* 165, 2022) and Tartakovsky et al. (*Water Resour. Res.* 56, 2020).

**Learned constitutive laws — `net(u)`, `net(∇u)`.** A network may also take the *solution* (or its
derivatives) as input — then it is a material law, not a spatial map, and the form becomes nonlinear in
`u` (routed to the matrix-free Newton path automatically). Observe `u`, learn the hidden law
unsupervised through the residual:

```python
net = jno.nn(foundax.mlp(1, hidden_dims=16, num_layers=2,
                              activation=jax.nn.tanh, key=key)).dtype(jnp.float64)
# hidden truth k(u) = 1 + 0.5 u²; learn it from a single observed field
fem = jno.fem([(1.0 + net(ui)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs).solve(600)
```

`net(ui.x, ui.y)` (a `k(∇u)` law) and mixed inputs (`net(xi, yi, ui)`) work the same way; a net whose
arguments carry the unknown makes the form nonlinear. **Transient forms** recover a diffusivity or
constitutive law from a `u(t)` trajectory; a coordinate `net(x)` on the **mass** (`u_t`) term is
supported (an unknown density `ρ(x)·u_t`), but a *solution-dependent* `net(u)` on the mass is rejected (a
nonlinear mass the semidiscrete form cannot express). Net coefficients also compose with **complex**
steady forms and **coupled (multi-field)** forms, and with the scalar C¹ families
(`"Argyris"`/`"Morley"`/`"Hermite"`) and a *scalar coordinate* `net(x)` on the vector edge families
(`"RT"`/`"N1E"`); the non-nodal path assembles a *dense* operator, so wants an explicit dense `solve_fn`.

**Unknown boundary / initial conditions.** A network as an *essential value* is a trainable *profile*
(it enters the lift, not the operator): a *Dirichlet* value `u(∂Ω) - net(xb, yb)` (an unknown boundary
profile) or an *initial condition* `u(initial) - net(xi, yi)` (an unknown starting state, recovered from
a trajectory). Both supported for a **bare** `net(x)`, native Lagrange single-field — the Dirichlet on
steady / nonlinear / linear-transient / nonlinear-transient forms, the IC on a linear-transient form; a
compound value, or a net IC on a nonlinear transient, fails loud.

*Current scope:* steady/transient/steady-complex on the native 2D/3D Lagrange assembler (single or
coupled multi-field), steady scalar C¹ non-nodal, a bare `net(x)` steady/nonlinear/linear-transient/
nonlinear-transient Dirichlet value, and a bare `net(x)` linear-transient initial condition. Not yet
(each fails loud): a compound net essential value, a net IC on a *nonlinear* transient, a net Dirichlet
with a state-dependent mass, `net(u)` on the mass term, k(u) in complex forms, the complex transient, a
net Dirichlet combined with a time-varying `g(x,t)` Dirichlet, `net(u)` on the vector edge families, and
1D domains.

### Transient inverse

For a transient form, `fem.solve()` returns the **trajectory** `u(save_ts)` (default: backward
Euler over the assembled `dt`, sampled at the domain time grid), differentiable in the
parameters — so a rate constant is recovered from a time series:

```python
fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(*ci) - u0])
crux = jno.core([(fem.solve() - u_traj).mse], domain=obs).solve(200)   # recovers alpha
```

`fem.solve(my_integrator, save_ts=...)` swaps the integrator: `my_integrator(block, args, save_ts) ->
trajectory`. Build your own (e.g. diffrax) from the block's `block.M` / `block.A` / `block.state0` — form
`u_dot = M⁻¹(c − A u)`; the implicit backward-Euler default is preferred for Dirichlet problems.

---
