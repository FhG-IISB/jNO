# Limits, build time, and worked examples

## Known limitations

Almost every boundary below is an explicit, fail-loud `NotImplementedError`, raised when you
**assemble a weak form** or solve a **transient problem through the time route**. The residual-PINN
path is unaffected.

!!! danger "Two limits are silent — every other one raises"
    The assembler cannot distinguish these from a legitimate modelling choice, so nothing stops you.

    - **Affine geometry on a curved boundary** — the domain is approximated to O(h²), which caps
      every element order above it. The solve is simply suboptimal.
    - **The `2πr` measure on an axisymmetric *vector* form** — exact for scalars, wrong for vectors.

    Both are detailed below, and stated again at the point where you make the choice.

| Area | The limit | How it fails |
|---|---|---|
| Transient mass | must be parameter-free — put affine trainable parameters on the stiffness / residual, not on `u_t * phi` | raises |
| Second order in time | nodal Lagrange only, 1-D/2-D/3-D, scalar or vector; the **temporal** side must stay linear | raises |
| Reduced-order `basis=` | steady + first-order transient only | raises |
| Runtime Dirichlet parameters | steady linear, steady nonlinear, linear transient | raises |
| Affine parameter lowering | one trainable scalar per additive term, not nested | raises |
| Enclosure radiation | 2-D / axisymmetric, needs a direct solve; you write the radiosity yourself | manual composition |
| Plasticity | small-strain, isotropic, linear-hardening, whole-domain | raises |
| Element order on RT / N1E / P0 / Hermite / Argyris / Morley | each family has one intrinsic order | raises |
| `eigs` on a non-symmetric pencil | eigenvalues differentiate, **eigenvectors do not** | NaN, not a silent zero |
| **Curved-boundary geometry** | affine (straight-edge) at every order | **silent** |
| **Axisymmetric vector forms** | the `2πr` measure is wrong for vectors | **silent** |

### The detail

??? note "Second order in time (`u_tt`) — what composes and what refuses"
    A **nonlinear spatial** operator (sine-Gordon, cubic Klein–Gordon, large-deformation
    elastodynamics) *is* supported — Newton on the augmented `[u; v]` block. The **temporal** side
    must stay linear: a state-dependent mass or damping `c(u)·u_tt` is refused, since `M2`/`C` are
    extracted by differentiating at `u=0` and would otherwise be frozen there.

    A **coupled** 2-D/3-D system flows through the *same* assembler as a single field, so damping,
    the nonlinear path and a driven boundary `g(x,t)` all compose with coupling. A coupled form
    still refuses:

    - a field with **no** `u_tt` term — its velocity rows would be singular;
    - **runtime parameters** — the parametric coupled steady assembly underneath is not wired;
    - periodic ties.

    Time-varying Dirichlet is refused on nonlinear forms. A **complex coefficient** on any `u_tt`
    form is refused by name — it used to be silently cast to real. Write the problem first-order in
    time instead; the complex transient is supported.

    A coupled 1-D system carries `u_tt` on narrower terms (linear, undamped): the augmented state is
    `[u_all; v_all]`, so `fem.offsets` lists the displacement blocks then the velocity blocks.

??? note "Reduced-order solves — what `basis=` covers"
    Steady and **first-order transient**, linear and nonlinear. Second-order-in-time (`u_tt`),
    complex, and periodic-tied problems each refuse with their own reason. Nonlinear reduces, but
    without hyper-reduction that is a memory win, not a speed one.

??? note "Runtime Dirichlet parameters — where the gradient flows"
    A trainable `jno.np.parameter` may sit in an essential value (`u(top) - g`, or scaling a
    coordinate profile `u(top) - g*sin(pi*x)`), and the boundary value is recovered from data like
    any other parameter — `∂b/∂g` flows through the symmetric elimination (linear), and `∂/∂g`
    through the solve's / each step's `custom_root` (nonlinear / transient).

    Refused loudly: a value that is **both** parametric and t/τ-dependent (`u(top) - g*tau`). Train
    the amplitude through a Neumann / body term instead. A FIELD-sized optimizer-less parameter stays
    the nodal data-field value (a neighbour's field in a DD solve), gathered per node.

??? note "Plasticity — what runs today"
    Deformation theory (monotonic / proportional) and the path-dependent flow-theory **`tau=`
    load-path march** both run. The march assembles on the real, steady native-Lagrange path,
    **single-field or coupled** — not transient / complex / 1-D / non-nodal / periodic, each rejected
    with a clear error.

    The internal-state readout runs on every cell; sub-region-restricted plasticity is not wired.
    Kinematic / nonlinear (Voce) hardening and contact are separate formulas / machinery, not built.

??? measured "Curved boundaries: P3 buys nothing over P2"
    There is no isoparametric mapping, so on a curved boundary the domain itself is approximated to
    O(h²) and that error caps every element order above it. Measured on the unit disk
    (`-Δu = 4`, `u|∂Ω = 0`, exact `u* = 1 − r²`), L2 rates under refinement:

    | order | expected | measured | error at `h = 0.05` |
    |---|---|---|---|
    | P1 | 2 | **2.00** | 1.14e-03 (1 536 dofs) |
    | P2 | 3 | **2.02** | 7.46e-04 (6 015 dofs) |
    | P3 | 4 | **2.01** | 7.40e-04 (13 438 dofs) |

    P3 buys *nothing* over P2 — 13 438 dofs for the same answer — and both are barely ahead of P1.
    The cap comes from imposing the boundary condition at straight-edge nodes that sit on the chord,
    O(h²) inside the true arc.

    On a **polygonal** domain the advertised rates hold exactly (the suite measures P2/P3 there).
    Until an isoparametric mapping exists, prefer `h`-refinement (or the adaptive loop) over
    `order ≥ 2` near curved boundaries.

??? note "Element order on a non-nodal family — refused, not applied"
    RT / N1E / P0 / Hermite / Argyris / Morley each have one intrinsic order. `space="N1E", order=2`
    used to return the same lowest-order space silently; it now raises. The mesh is the only accuracy
    knob on an H(curl)/H(div) problem — see [*Mesh resolution for wave
    problems*](elements.md#mesh-resolution-for-wave-problems) for what a given points-per-wavelength
    buys, measured.

??? warning "`eigs` routing, differentiability, and when `linear=` pays"
    `jno.solve.eigs` / `FEM.eigs` route on the operator's **actual symmetry**. A symmetric pencil
    takes the symmetric reductions (real spectrum, differentiable). A genuinely non-self-adjoint
    operator goes to **ARPACK/Arnoldi** (Lehoucq & Sorensen 1996) and returns the **complex**
    spectrum it actually has — the case that matters for stability problems (resistive MHD growth
    rates, drift waves, anything with a mean flow), where the sign of the growth rate *is* the
    physics. Neither path ever returns the spectrum of `½(K+Kᵀ)` as though it were the answer.

    The routing probe is a randomized bilinear test and is concrete-only, so **under `jit` the
    symmetric path is assumed**.

    **Differentiability of the non-symmetric path.** The eigenvalues *are* differentiable in reverse
    mode — `dλ = wᴴ(dA − λ dB)v / (wᴴBv)` for a simple eigenvalue (Wilkinson 1965 ch. 2), with the
    left eigenvector obtained by inverse iteration on `(A − λᵢB)ᴴ`, verified against finite
    differences to 1e-09. The **eigenvectors are not**, and differentiating through them yields
    **NaN** rather than the silent zero a plain callback would give. A **defective** eigenvalue has
    no derivative at all (its perturbation series runs in `√ε`) and is detected via the eigenvalue
    condition number, giving NaN instead of the enormous finite number the formula would produce.
    Because that guard is a `custom_vjp`, **forward mode (`jax.jvp`/`jacfwd`) is unavailable** — use
    `jax.grad` / `jacrev`.

    **`linear=` is opt-in because it is not always a win.** It accepts
    `jno.solve.lu(backend="pardiso"/"cudss"/"host")` to drive ARPACK's shift-invert factorization —
    those kernels are plain numpy functions, so ARPACK can call them from host code — but refuses
    `backend="device"` (a JAX primitive), the Krylov solvers, and `precond=`, rather than ignoring
    them. ARPACK applies the inverse ~50–70 times, so it trades one fast factorization against
    per-application overhead. Measured with PARDISO:

    | pencil size | speed-up |
    |---|---|
    | n = 3,000 | **0.72×** (slower) |
    | n = 20,000 | **10.05×** (21.6 s vs 217 s) |

    It needs `k < n-1`; smaller pencils take an exact dense `scipy.linalg.eig`. Order with
    `which="LR"` / `"SR"` (real part — the growth rate) or target an interior region with `sigma=`.

??? warning "Axisymmetric vector forms are your responsibility"
    The `2πr` measure is exact for scalars and wrong for vectors — elasticity hoop strain, and for
    vector Maxwell the cylindrical curl's own `1/r` terms plus the meridional/azimuthal decoupling.
    jNO ships no axisymmetric H(curl)/H(div) element, and multiplying by `r` is arithmetic the
    assembler cannot distinguish from a legitimate radial coefficient, so **nothing raises**.

    Use a full 3-D mesh for vector Maxwell, or write the cylindrical operator out yourself. See
    [*Axisymmetric (bodies of revolution)*](geometry.md#axisymmetric-bodies-of-revolution).

---

## Build time: what to expect, and the one knob

`jno.fem([...])` **fully assembles** the operator — it returns concrete matrix values, which is why
`fem.solve()` is then only the linear solve (measured **7 ms** on 3-D Poisson at 27,833 nodes).

Some libraries instead defer assembly into their solve. That makes their "build" look faster and
their solve slower, so compare **build + solve** — and remember jNO assembles *once*, while a
per-solve assembler pays again on every Newton iteration.

Most of a cold build is **XLA compilation**, and the cost is fixed per problem *structure* rather
than per DOF: a 15× larger mesh still compiles about the same number of programs. Two caches attack
it, **both on by default**.

### Across processes — the persistent XLA cache

Enabled at `import jno`, stored in `~/.cache/jno/xla`.

!!! measured "3-D Poisson, 27,833 nodes"
    | | first build | repeat build |
    |---|---|---|
    | no cache | 4.75 s | 2.48 s |
    | persistent cache (default) | **2.22 s** | **1.51 s** |

The very first run on a machine is *slower* — populating the cache costs more than not having one.
It pays back from the second process onward, which is jNO's normal life: sweeps, optimisation loops,
test suites, re-running a script after an edit.

!!! note "Opting out"
    Any one of: `JNO_COMPILE_CACHE=0`, `jno.setup(__file__, compile_cache=False)`, or per project in
    `.jno.toml`:

    ```toml
    [jno]
    compile_cache = false
    ```

### Within a process — identical-problem reuse

Rebuilding an *identical* problem (same mesh content, same terms) reuses the already-compiled
assembly kernels outright, keyed on content rather than object identity. A rebuild then costs meshing
plus host prep and **no XLA work at all** — measured 1.94 s → 0.28 s on 3-D Poisson at 29k nodes.

Anything that changes the operator — a different mesh, a different coefficient — recompiles exactly
the kernels that bake it. Structure a tokenizer cannot key by value simply never caches (a safe
miss); the coverage is measurable, not guessed (`jno.utils.solver.fem_utils._ELEM_MAP_STATS`).

---

## Worked examples

The [FEM tutorials](../tutorials/08-fem-and-varpinns/poisson-2d-fem.md) cover every pattern in this
guide:

| | |
|---|---|
| **Elliptic** | Poisson · mixed Dirichlet/Robin reaction–diffusion · mixed-BC Helmholtz |
| **Nonlinear** | Allen–Cahn interface |
| **Solid & fluid** | linear-elastic cantilever · Stokes channel flow |
| **Transient** | heat · two **second-order-in-time** wave examples (vibrating membrane, vector-elastodynamics cantilever) |
| **Inverse** | a hidden diffusivity field · a transient rate |
| **Non-nodal** | H(div) mixed Poisson · H(curl) Maxwell / eddy currents |
| **Neural** | a **variational PINN** — a neural trial in the same weak form |
