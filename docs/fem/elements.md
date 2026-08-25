# Element families

The family is one argument on `fem_symbols` — everything downstream (assembly, BCs,
`fem.solve()`) is unchanged:

```python
d = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain()

u, v = d.fem_symbols()                          # P1 Lagrange, the default
u, v = d.fem_symbols(order=2)                   # P2; order=3, … for higher
u, v = d.fem_symbols(value_shape=(2,))          # a vector unknown (elasticity, flow)

u, v = d.fem_symbols(space="RT")                # H(div) Raviart–Thomas
u, v = d.fem_symbols(space="N1E")               # H(curl) Nédélec edge ("N1curl" also works)
u, v = d.fem_symbols(space="Morley")            # C¹ biharmonic, 6 DOF — plates
u, v = d.fem_symbols(space="Argyris")           # C¹ conforming quintic
u, v = d.fem_symbols(space="Hermite")           # C⁰ value + gradient
```

Each non-nodal family has **one intrinsic order** set by the element definition, so
`order=` is a nodal-Lagrange knob only — passing it alongside `space="N1E"` raises rather
than silently handing back the same space (see below).


## Non-nodal element families: H(div) and H(curl)

> **⚠️ Experimental.** Validated on 2-D triangular meshes at lowest order (RT₀ / N1E₀); the API may
> still change. Steady-linear, steady-nonlinear (Newton), and transient `M u̇ + A u = c` (including the
> mixed/saddle DAE, e.g. transient Darcy) all work, as does the differentiable `fem.solve()` inverse
> (a **scalar** or spatially-varying **P1 field** `k(x)` in a *volume* term, steady and transient). Every
> edge/cell (RT/N1E/P0) operator — the steady `A`, and the transient **mass `M` and spatial `A`** (also when
> re-assembled per step for a parametric march) — is assembled **sparsely, one element at a time** (a `BCOO`,
> mirroring the native Lagrange assembler), never a dense global `jacfwd`; the mixed-DAE initial state is
> projected by a matrix-free CG on the (SPD) field mass block. So a 3-D N1E **eddy-current / time-domain
> Maxwell** transient scales instead of hitting the `O(n_dof²)` dense-assembly wall past ~10⁴ edges.
>
> **3-D Nédélec (H(curl)) is supported** on a **tetrahedral** mesh: the first-kind `"N1E"` element
> assembles the H(curl) **mass and curl-curl** forms (`inner(u, v) + inner(u.vector.curl(x,y,z),
> v.vector.curl(x,y,z))`) — the correct edge discretisation for **Maxwell / eddy currents** (nodal
> Lagrange gives spurious modes). The covariant push-forward `Φ_phys = J^{-T} Φ_ref` is dimension-agnostic
> and the curl is taken from the physical gradient. The essential **PEC wall** `n×E = 0` is supported,
> written `u.vector.cross(d.variable(region, normals=True))` (facet-based; it pins every boundary-face edge
> DOF of the region). On a tet mesh **only N1E is wired** — RT / P0 / Hermite / Argyris / Morley remain
> 2-D-triangle only and raise. On a **line** mesh `"Hermite"` selects the 1D cubic beam element
> (see above); the other families raise.
>
> **Each of these families has ONE intrinsic order** — RT₀ / N1E₀ lowest, P0 constant, Morley quadratic,
> Hermite cubic, Argyris quintic — set by the element definition, not chosen. `order=` is a nodal-Lagrange
> knob and **is refused** here rather than ignored: `space="N1E", order=2` used to hand back the same
> lowest-order space silently (an identical operator), which is the worst failure shape for a wave problem
> — you pay for accuracy, get first-order convergence, and only find out from a convergence study that
> stalls. Refine the mesh instead (see *Mesh resolution for wave problems* below), or use a nodal Lagrange
> field, where `order=` does apply.
>
> **Not yet:** the rest of the zoo in 3-D (RT / C¹ / plate are 2-D only); the *inhomogeneous* `n×E = g` on
> N1E; higher order; other families (BDM, second-kind Nédélec, Bell); quad / non-triangular meshes; a
> runtime parameter **or trainable neural coefficient** in a **host-assembled** RT-pressure / plate
> boundary term (the N1E tangential-trace impedance / incident **surface** BC *is* differentiable in a
> boundary-term parameter *and* a learned `net(x)` coefficient); the
> constraint-consistent algebraic initial state at `t0` in the saddle-DAE transient (only the reported `t0`
> algebraic value is affected).

Beyond nodal Lagrange (P1/P2), `jno.fem` assembles **edge-DOF** families on 2-D triangles — for
problems whose natural space is *not* H¹. Pick one with the `space=` knob on `fem_symbols`:

| `space` | Space | DOF | Use |
|---------|-------|-----|-----|
| `"Lagrange"` (default) | H¹ | nodal value | standard PDEs |
| `"RT"` | **H(div)** Raviart–Thomas | edge normal flux `∫ₑ u·n` | mixed Poisson, Darcy, conservation |
| `"N1E"` | **H(curl)** Nédélec (1st kind) | edge tangential `∫ₑ u·t` | Maxwell, eddy currents |
| `"P0"` | L² (piecewise constant) | one per cell | the pressure / multiplier of a mixed pair |
| `"Hermite"` | C⁰ cubic, **vertex value + ∇ DOFs** | `u`, `∂u/∂x`, `∂u/∂y` at vertices (+ centroid) | smooth/gradient-aware fields; the foundation for C¹ elements |
| `"Argyris"` | **C¹** quintic (TUBA-6) | value + `∇u` + `D²u` at vertices, `∂u/∂n` at edge midpoints | conforming **biharmonic** / plate / Cahn–Hilliard |
| `"Morley"` | **non-conforming** quadratic (6 DOF) | value at vertices, `∂u/∂n` at edge midpoints | **cheap biharmonic** / plate — scales to fine meshes |

> **Hermite** is the first element with a per-cell **DOF-mixing** transform `M(cell)` (global derivative
> DOFs are the physical gradient `∇u` at the vertices). It is **C⁰** (not C¹), so it is *not* a conforming
> biharmonic element — it de-risks the `M(cell)` machinery the **C¹ Argyris** element reuses. A
> value-Dirichlet `u(region) - g` pins boundary-vertex value DOFs (derivatives free); composes with the
> steady / transient / nonlinear paths (`tests/test_fem_hermite.py`).

> **These families assemble sparsely.** Their *linear* operator is built element-by-element like every
> other family, so peak memory grows as `n^1.0` and is set by the operator rather than by an
> intermediate. Argyris reaches **22,511 DOFs** on an 8 GB card; Morley and Hermite go further still.
>
> The solve stays **sparse-direct** for these families. Going sparse would otherwise have handed 4th-order
> biharmonic operators to the Jacobi-preconditioned BiCGStab that serves real elliptic systems, where it
> does not converge — a solver change disguised as a storage change. Their **second-order-in-time** path
> is still dense, as it is for every non-nodal family.

> **Argyris** is the **C¹-conforming** quintic triangle (21 DOF) — the element for **4th-order PDEs**.
> Across a shared edge both `u` and `∂u/∂n` are continuous, so `∫Δu·Δv` is a *convergent* biharmonic
> discretisation (the conformity caveat above does not apply). The reference dual basis is mapped to each
> physical cell by the affine-equivalence DOF-transform `M(cell)` (Kirby, *SMAI J. Comput. Math.* 4, 2018;
> element: Argyris–Fried–Scharpf 1968); its **globally-oriented edge-normal DOF** is what makes it C¹ on an
> *unstructured* mesh. Essential BCs are the two plate traces — `u(region) - g` (deflection) and
> `u.dn(region) - h` (rotation `∂u/∂n`); the boundary curvature `∂²u/∂n²` is always left **free**. These are
> wired for **axis-aligned boundary edges** only; a non-axis-aligned edge is **rejected with a clear error**
> (use Morley there). Composes with steady / transient / nonlinear `fem.solve()` and **inverse** (scalar or
> P1-field `k(x)`, steady and transient — `tests/test_fem_argyris.py`, `tests/test_fem_inverse.py`).

> **Morley** is the **cheapest** biharmonic element (6 DOF, quadratic). It is **non-conforming** — yet
> passes the patch test and converges (energy `O(h)`, L² `O(h²)`). It reuses Argyris's `M(cell)` transform
> and globally-oriented edge-normal DOF but with ~3.5× fewer DOF, so it is **much cheaper per node**
> and scales to finer meshes. **Modelling subtlety:** because it is non-conforming,
> the biharmonic form must be the **full-Hessian inner product** `inner(hessian(u), hessian(v))` (`∫D²u:D²v`),
> *not* `∫Δu·Δv` — the Laplacian form is singular for Morley (`xy` has `Δu = 0` but `D²u ≠ 0`, a spurious
> kernel). Its two plate traces `u(region) - g` and `u.dn(region) - h` work on **any** boundary orientation.
> **Periodic ties** compose too — `u(top) - u(bottom)` ties the vertex-value *and* the edge-normal-derivative
> DOFs across a matched (conforming) boundary pair, so a y-periodic biharmonic solve recovers a manufactured
> `sin(πx)sin(2πy)` at the optimal L² rate (`tests/test_fem_morley.py::test_morley_periodic_biharmonic_convergence`);
> the reduction requires **conforming** periodic boundaries (matching vertices/edges) and is Morley-only for now —
> the other C¹ families (Argyris, Hermite) raise a clear `NotImplementedError`
> (Morley, *Aeronautical Quarterly* **19**, 1968; `tests/test_fem_morley.py`).

> **Plate boundary conditions.** For a 4th-order (plate/biharmonic) field the boundary trace has two
> essential parts to pin independently — the **deflection** `u(region) - g` and the **rotation**
> `u.dn(region) - h` — plus two conjugate **natural** parts (bending moment `M_n`, effective shear `V_n`)
> that emerge on any trace you *don't* pin. The classical BCs compose from these:
>
> | BC | Physics | How to write it (on a region) |
> |----|---------|-------------------------------|
> | **Clamped** | `w=0`, `∂w/∂n=0` | `u(reg)-g`, `u.dn(reg)-0` |
> | **Simply-supported** | `w=0`, `M_n=0` | `u(reg)-g` |
> | **Guided / sliding** | `∂w/∂n=0`, `V_n=0` | `u.dn(reg)-h` |
> | **Free** | `M_n=0`, `V_n=0` | *(write neither — natural)* |
>
> A free edge gets the natural `M_n=V_n=0` from the ν-weighted plate energy
> `(1-ν)·inner(hessian(u),hessian(v)) + ν·laplacian(u)·laplacian(v)` (validated against Timoshenko's
> square-plate coefficients). A *nonzero* prescribed edge moment is the boundary load `M_n * phi.dn(region)`
> (`∮_region M_n ∂φ/∂n ds`), built from each cell's geometry so it applies on **any** edge orientation
> (though on Argyris the *essential* pin is still axis-aligned-only). The conjugate **shear** load `V_n` is
> not yet wired.

```python
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")   # H(div) flux
p, q = d.fem_symbols(names=("p", "q"), space="P0")                     # piecewise-constant scalar
```

**Vector operators** (on a bound vector view): `u.div(x, y)` is the divergence and `u.curl(x, y)` the
2-D scalar curl `∂uy/∂x − ∂ux/∂y`; in 3-D `u.curl(x, y, z)` returns the **curl vector** (used for the
N1E tet curl-curl form). After binding, the no-arg `u.div()` / `u.curl()` reuse the bound coordinates.

**Essential (edge-trace) BCs** — the outward normal is `d.variable(region, normals=True, split=True)`:

| Family | Trace | Term |
|--------|-------|------|
| RT  | normal flux `u·n = g` | `u(b)[0]*nx + u(b)[1]*ny - g` |
| N1E | tangential `u×n = g`  | `u(b)[0]*ny - u(b)[1]*nx - g` |

For the RT mixed-Poisson saddle, a Dirichlet condition on the scalar `p` is *natural* — add the weak
term `p_D * (v[0]*nx + v[1]*ny)`, no essential constraint on the flux. A BC may target a sub-region
(a `Shape.rect` edge tag or any `d.tag(...)` boundary subset; sub-region normals are computed from the
geometry).

Tutorials: `mixed_poisson_rt_2d.py` (H(div)), `maxwell_nedelec_2d.py` (H(curl): magnetostatics + eddy
current), and `maxwell_nedelec_3d.py` (the **3-D PEC cube cavity resonator** — recovers `k²=π²(l²+m²+n²)`,
spurious-free, converging to `2π²` from below).

---

## Quadrilateral and hexahedral (tensor-product) cells

jNO meshes simplices by default — triangles in 2-D, tetrahedra in 3-D. Tensor-product cells are
available two ways, and which one you can use is decided by what a mesher can actually produce:

```python
# 2-D quadrilaterals on ARBITRARY geometry, via gmsh recombination
d = jno.Shape.disk(0, 0, 1, size=0.1).quad().domain()
u, v = d.fem_symbols(order=2)          # Q2 / Q3 work exactly as P2 / P3 do

# structured grids, no mesher involved — a rectangle of quads, a box of hexes
d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=40).quad().domain()
d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).structured(n=16).quad().domain()
```

Nothing else in the term list changes — the weak form, the boundary conditions and the solve are
written exactly as they are on a simplex mesh.

**Why bother.** Linear triangles and tetrahedra are stiff: they lock in near-incompressible
elasticity and are poor in bending, which is why mechanics codes are built on hexahedra. A
quadrilateral grid is also the mesh the topology-optimisation literature is written on, and for
layered geometry (a thin-film stack, a wound magnetic component, a PCB) a structured hex mesh is both
better conditioned and far cheaper per node than tetrahedra.

**Why 3-D is structured-only.** gmsh cannot hexahedral-mesh general geometry. Measured here,
`Recombine3DAll` on a plain box returns **944 tetrahedra and zero hexahedra**; hexes come only from
sweeping/extruding or transfinite meshing. `Shape.quad()` therefore refuses a 3-D shape by name
rather than quietly handing back tetrahedra. 2-D recombination has no such limit — a disk
recombines to pure quadrilaterals just as a rectangle does.

**Accuracy.** Q1 recovers O(h²), measured on `-Δu = 2π²sin(πx)sin(πy)`:

| mesh | rates per halving | note |
|---|---|---|
| structured quads, Q1 | 1.87, 1.92, 1.96 | same error as the triangulation of the same grid, from half the cells |
| structured hexes, Q1 | 1.79, 1.86 | same error as the tetrahedralisation, from a sixth the cells |
| recombined quads (`Shape.quad()`), Q1 | 1.83, 1.91 | ~5× the L2 error of triangles at equal node count on this smooth problem |
| structured quads, **Q2** | 3.94, 3.96 | ~8× *lower* error than P2 triangles at the same node count |
| structured quads, **Q3** | 3.87, 3.95 | ~5× lower than P3 triangles at the same node count |
| structured hexes, **Q2** | 4.27, 4.29 | ~10× lower than P2 tetrahedra at the same node count |

**Order is the lever, not cell shape.** At Q1 the two cell types are indistinguishable; raising the
order is what moves the rate. A Q{k} quad mesh and a P{k} triangulation of the same grid have
*exactly* the same node count, so those comparisons are DOF-for-DOF rather than favourable. (The
rates are nodal RMS on structured meshes, where nodal superconvergence is expected — read them as a
comparison between cells, not as the theoretical O(h^{k+1}).)

That last row is worth reading honestly: **recombination is not a free accuracy win**. It leaves
some poorly-shaped cells, and on smooth scalar Poisson the triangulation is more accurate per node.
The tensor-product advantage is in bending and near-incompressibility, not here.

The geometry map is formed **per quadrature point**, because a bilinear quad or trilinear hex has a
Jacobian that varies within the cell even when the cell looks straight-sided — the same machinery
`Shape.curved()` introduced. The quadrature degree is raised by 2 for the same reason it is on curved
cells: the map makes the integrand rational, so no rule is exact.

**Scope — what is not supported yet**, each refusing by name rather than approximating:

**Measured coverage.** Every row below was run on a tensor-product mesh *and* on the simplex mesh of
the same grid, so an API mistake could not be mistaken for a limitation. Working on both quads and
hexes: steady linear, Neumann / surface terms, nonlinear (Newton), **vector fields (elasticity)**,
coupled multifield including **Taylor-Hood Q2/Q1**, transient marches, runtime parameters differentiated with
`jax.grad`, `u.bounds`, `by_region`, `by_tag`, eigensolves, `fem.eval` readout, and the direct
solver slots. **Whole-domain periodic BCs work on both**, in every direction — matched faces collapse
onto one DOF, so the periodicity holds exactly rather than to a tolerance.

| not supported on quad/hex | why |
|---|---|
| **integrated (mortar)** ties across a hexahedral facet | the mortar rows clip triangles and have no quadrilateral analogue. The **collocated** coupling does support it, and is what a hex tie uses — see below |
| **h-adaptive re*meshing* on HEXES** (`adapt=`) | not for want of plumbing: no general all-hex mesher exists (gmsh's `Recombine3DAll` on a plain box returns tetrahedra and no hexahedra), so there is nothing to remesh *to*. Hexes adapt by **local refinement with hanging nodes** instead — `refine_domain`, see below |
| **r-adaptivity** (`relocate=`) on quad/hex | the monitor, the cell measures and the validity check are all barycentric; a bilinear cell's validity is the sign of the sampled `det J`, not one determinant |
| 4th-order forms (plates, phase-field) | the physical-Hessian push-forward assumes an affine cell — the same refusal curved simplices already carry |
| non-nodal families (N1E, RT, Argyris, Morley) | Argyris and Morley are *defined* on triangles; the quad analogues (RTCF/NCE, Bogner–Fox–Schmit) are different elements |
| `Shape.quad().curved()` | a curved quadrilateral is a 9-node block the emitter does not produce |

The **recovery error estimator** runs on both cell families. On a simplex the P1 gradient is constant
per cell, so inverting the edge matrix is the answer; on a bilinear cell the gradient varies, and the
sample is taken at the cell **centroid** — the superconvergent (Barlow) point of a Q1 gradient, where
it is `O(h²)` accurate against `O(h)` elsewhere, which is the property Zienkiewicz–Zhu recovery needs
of its samples. Measured as the effectivity index `eta / true error` on `-Δu = f`:

| n | quads | triangles |
|---|---|---|
| 8 | 1.301 | 1.018 |
| 16 | 1.147 | 1.012 |
| 32 | 1.069 | 1.006 |

Both asymptotically exact. The indicator integrates the gradient gap over the cell with the element's
own quadrature rather than sampling it at the centre: on a quad *both* the recovered field and `∇u_h`
are superconvergent at the centre, so a centroid rule compares two good approximations and misses the
error, and its effectivity **decayed** (0.81 → 0.53 → 0.35 over the same meshes) while looking
healthy. That fix improved the simplex path too, from ~0.77 to ~1.006.

Reference: Zienkiewicz & Zhu, IJNME **33** (1992) 1331–1364; Barlow, IJNME **10** (1976) 243–251.

**Refining on your own criterion.** `adapt=` marks on the Zienkiewicz–Zhu recovery estimator by
default. Production AMR codes mark on *physical* quantities instead — a density gradient, an
interface, vorticity — and `criterion=` takes any traced expression, with **no test function** (a
criterion is a field, not an equation):

```python
ui = u.bind(x=xi, y=yi)

remesh(criterion=jno.np.sqrt(ui.x**2 + ui.y**2))              # gradient / shock detector
remesh(criterion=phi * (1.0 - phi))                            # phase-field interface
remesh(criterion=jno.np.abs(uy.x - ux.y))                      # 2-D vorticity
remesh(criterion=d.by_region({"weld": 1.0, "plate": 0.0}))     # refine one material
```

It is assembled against the problem's own test function, normalised by the lumped mass to a nodal
field — so the scale is the criterion, not `criterion x cell volume`, and `theta` means the same thing
on a coarse and a fine mesh — then integrated per cell and marked as usual. Everything else on the
spec is unchanged, so a criterion composes with `theta`, `refine_factor`, `max_dofs` and the quad
rebuild path below. Measured on a thin diagonal ridge: cells on the feature come out **1.85x** smaller
than off it on triangles and **1.92x** on quads.

A Löhner-style detector `|D²u|/|Du|` needs `order >= 2` — a P1 Hessian is identically zero, so at
order 1 it would evaluate to nothing.

**h-adaptivity on quadrilaterals** works, by a different mechanism than on simplices. mmg adapts
triangles and tets by local edge split/collapse/swap and has no quad analogue — but a quad mesh in
jNO *is* a triangulation gmsh recombined, and the size field driving it is already part of the
`Shape` plan. So the remesh stage **rebuilds the plan** at the marked size field:

```python
d = jno.Shape.polygon(L_SHAPE, size=0.12).quad().domain()
u = fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=4))     # refines at the corner, stays all-quad
```

Measured on the L-shape's re-entrant corner, against uniform refinement at matched DOFs: adapting is
**1.23×** more accurate on quads and **1.31×** on triangles (Ritz energy against a fine reference).
The refined cells go where they should — mean cell size near the corner is 1.5× smaller than far
from it, against 1.19× for the triangle control.

Three things it is not. It is a **global** remesh, so the mesh does not nest and each round costs a
full gmsh rebuild. It needs a **geometry to rebuild from**, so a quad mesh loaded from a `.msh` file
refuses by name. And it cannot refine a **`.structured()`** plan, whose resolution is its cell counts
rather than a size field — that would silently return the same mesh, so it refuses too. Recombination
purity is checked on every round rather than assumed.

**Tying two hexahedral domains** across a non-matching interface works. A tie constrains each
secondary node to a weighted combination of the main face's nodes, and those weights were barycentric
(triangle) shape functions — which a hexahedron's *quadrilateral* facet has none of, so the tie
refused rather than interpolate a quad from three of its four nodes.

A quad facet's map is **bilinear** and has no closed-form inverse, so its reference coordinates come
from a Newton inversion (`fem_lagrange._invert_tensor_map`, the same inverse the quad solution
transfer uses). Measured on two hex blocks meeting at different resolutions (16 facets against 9), the
**linear patch test passes to 5.6e-16** — solved directly, because the default iterative solver stops
at 4.1e-04 in float32 and would hide a constraint error of that size.

The *integrated* (mortar) coupling still refuses on a quad facet: its rows clip triangles. Collocated
weights are linearly complete and are what a hex tie — and a hanging node — actually needs.

### Local refinement with hanging nodes (quadrilaterals and hexahedra)

`adapt=jno.solve.remesh()` refines a quad mesh by rebuilding its `Shape` plan at a finer size field,
which is global and needs a geometry to rebuild from. Splitting marked cells into four needs neither —
it works on a mesh loaded from a file, every old node survives, and it is the mechanism 3-D hexes will
use, since no all-hex mesher exists to remesh *to*.

The price is conformity: a split cell's edge midpoint is not a vertex of its unrefined neighbour, so
that node's value is not free. It is a **hanging node**, constrained to the coarse edge it lies on:

```python
u = fem.solve(adapt=jno.solve.refine(theta=0.4, max_iters=4))       # ZZ-driven, like remesh()
u = fem.solve(adapt=jno.solve.refine(criterion=jno.np.abs(ui.x)))   # or on a traced criterion
```

`refine` sits beside `remesh` rather than being a flag on it, because it is a different algorithm:
`remesh` re-runs the mesher at a finer size field (needs a geometry to rebuild from, and the new mesh
does not nest inside the old), while `refine` splits marked cells (local, keeps every node and its
value, works on a mesh loaded from a file). Marking is shared — `theta`, `criterion`, `max_iters`,
`max_dofs`, `tol`, `eps` all mean the same thing. There is no `refine_factor` (a split halves the cell
by construction) and no `anisotropic` (a split is isotropic, so there is no direction to stretch along;
that needs a simplex mesh).

To refine a specific set of cells outside the loop, call `refine_domain(domain, cell_ids)` from
`jno.utils.solver.fem_refine` — the loop is a marking strategy on top of it.

`u_hanging = Σᵢ wᵢ u_parentᵢ` is the same relation a periodic tie and a mortar coupling impose, so it
rides the **same prolongation** (`prolongation_from_ties`) and reaches `reduce_matrix_periodic` and the
`B(P)` block fusion with no path of its own — the reason this is a constraint *layer* and not a second
constraint mechanism. deal.II, MFEM and p4est all take this route.

On `-Δu = 1` over the unit square (exact centre `0.073671`, Timoshenko & Woinowsky-Krieger, *Theory of
Plates and Shells*, 2nd ed. 1959, Art. 30), refining the middle four cells of a 4×4 grid gives
`0.075063` against `0.077679` for that grid and `0.074598` for a uniform 8×8. Four rounds into the
centre converge monotonically to `3.3e-05`. Dropping *only* the constraint on the same mesh gives
`0.090093` — **11.9× worse, and worse than the coarse grid it was refined from**: without the tie the
refinement is not merely wasted but harmful.

Two things in jNO assumed conformity and had to change with it, both silent when wrong:

- **The boundary.** jNO derives it topologically — a facet belonging to exactly one cell — and across
  a 2:1 interface the coarse edge and both half-edges each belong to one cell. Left alone, the
  interface reads as boundary and a Dirichlet condition pins it *in the middle of the domain*:
  measured, 32 identity rows on a mesh whose perimeter is 16 nodes, and a centre value of `0.0194`.
- **Refining an already-refined mesh** must reuse the hanging node at an edge midpoint rather than make
  a second node there. Edge topology cannot see it (once non-conforming, the coarse edge and the two
  half-edges are *different* edges), and the duplicate splits the mesh along the interface while the
  area, the winding and the 2:1 balance all still check out.

#### Hexahedra

The same call refines a hex mesh, splitting each marked cell into 8. This is the **only** h-adaptivity
a hex mesh has, for the reason in the scope table above: there is no all-hex mesher to remesh to.

```python
d = jno.Shape.box(0, 0, 0, 1, 1, 1).structured(n=4).quad().domain()   # .quad() on a 3-D lattice = hexes
...
u = fem.solve(adapt=jno.solve.refine(theta=0.4, max_iters=3))         # the same slot, in 3-D
```

3-D adds a **second kind of constrained node**. A 2:1 face interface leaves the coarse face's four edge
midpoints hanging on 2 parents each *and* its centre hanging on all 4 corners at ¼ each — 18 constrained
nodes around a single refined interior hex (12 + 6). A hex face's map is bilinear, but its centre is the
parameter point (½, ½) where the Q1 weights are exactly ¼ however the face is warped, so no Newton
inversion is needed here (unlike the general non-matching tie above).

On `-Δu = 1` over the unit cube against a uniform 16³ reference:

| mesh | nodes | error |
|---|---|---|
| uniform 4³ | 125 | 6.0e-03 |
| uniform 8³ | 729 | 1.1e-03 |
| 4³ + 2 rounds into the centre | 1277 | **2.7e-04** |

A quarter of the uniform mesh's error for 1.75× its nodes. The constraints are satisfied exactly (the
DOFs are eliminated), and no hanging node acquires a hanging parent across the rounds.

The subtlety 3-D adds is in the **boundary**: a refined face on the domain surface splits into four
sub-faces that are still boundary, and whose corners include genuinely hanging edge midpoints (their
edges are shared with unrefined neighbours). Only a node with as many parents as the facet has vertices
— a face *centre* — proves a facet was covered. Treating every hanging node as proof of interiority
deleted 9 faces of the cube's own surface: 96 where the answer is 105.

Through the slot, on the same 3-D problem: 125 → 216 nodes over three rounds with the ZZ estimate
falling `5.2e-03 → 4.3e-03 → 3.6e-03`, and both hanging kinds present throughout with nothing chained.
Before this, a hexahedral mesh had **no** h-adaptive path at all.

**What it composes with, measured.** Scalar and vector fields (`value_shape=(d,)`, constrained
component by component), **coupled multifield** and **complex** problems (one prolongation block per
field — a complex field is two coupled real fields, and both halves are constrained), 2-D and 3-D, and
**any Lagrange order** on quadrilaterals (verified to order 3). Higher order needs the coarse edge's
higher-degree basis, and it changes which DOFs hang, not just their weights: jNO shares one DOF per coordinate, so the fine vertex at a coarse
edge's midpoint **is** that edge's own order-2 DOF — free, not hanging — while the DOFs that do hang sit
at the quarter points with weights `(0.375, −0.125, 0.75)`. Those negative weights are real; a quadratic
through three points is not a convex combination. On `-Δu = 1`, refining at order 2 gives `1.16e-05`
against `1.98e-05` for order 2 on the mesh it was refined from.

**Element families.** Lagrange only. Simplex-only families — the C¹ spaces (Argyris, Morley, Hermite)
and the H(div)/H(curl) ones (RT, N1curl) — are unreachable here rather than refused: they are defined on
triangles in jNO, and a triangular mesh cannot be locally refined at all (see below), so the combination
cannot arise. Asking for one on a quadrilateral mesh fails the same way with or without refinement.

**Limitations, measured.** Order 2 and above on **hexahedra** is refused: a hex's 2:1 interface also constrains
DOFs lying on a *face*, needing that face's 9-node basis rather than the edge basis. A hanging node **on
a tied or periodic interface** is refused — that composes two prolongations and their order changes the
answer; it is reachable in 3-D only (along a 2-D boundary an edge belongs to one cell, so refining it
makes its midpoint a real vertex and nothing hangs there). **Steady problems only** — the transient
driver carries state across each mesh change, and that transfer does not apply the hanging constraint,
so it would be violated on the first step after a split. Simplex meshes refuse by name and should use
`remesh`. Geometry is preserved exactly only for affine cells — a warped hexahedron's faces are
non-planar, so a 0.06 warp on a 0.25 cell moves the total volume by 3.9e-04, shrinking as the mesh
refines (the usual O(h²) straight-edge error); the constraint weights stay exact regardless. The split
is a Python loop over marked cells (27 lattice points each in 3-D), which suits the cell counts
adaptivity produces and is not tuned for refining a whole large mesh at once.

Volume terms, Dirichlet conditions and **surface terms all work on both cells** — Neumann, Robin
and flux integrals included.

A *named subset* of a hexahedral boundary is resolved by facet identity, not by its nodes. The
distinction is only visible when a tag omits a facet whose corners it keeps: a physical group naming
8 of the 9 facets of a face shares every corner of the ninth with its neighbours, and the all-nodes
mask puts that facet back. Measured as an applied load, that is `1.000` where the truth is `8/9`.
The same applies to `region.contains(p)` — a point-in-region query anywhere between nodes — which
tests the quadrilateral directly rather than a triangulation of it.

Hexahedra took extra machinery to get there, and it is worth knowing why. A quadrilateral's facet is
a straight edge (restricted to one edge a bilinear map is *linear*), so a single normal per facet is
exact. A hexahedron's facet is a **bilinear surface**: its normal turns across the facet, and its
area element varies with it. Both are therefore formed at each quadrature point by Nanson's formula,
from the same physical tangents the area element already needed. The check is the divergence
theorem, `∮ x·n dS = 3·Vol`, which is sensitive to the normal's direction *and* orientation at every
point: it holds to **machine precision (≈3e-16)** on deliberately warped meshes with non-planar
faces, and a single per-facet normal cannot pass it there.

### Mesh resolution for wave problems

Because N1E is **lowest order only** (`order=` is refused, above), the mesh is your *only* accuracy
knob on a wave problem — so it is worth knowing what a given resolution buys. Measured on the PEC cube
cavity, whose lowest mode `k² = 2π²` is analytic, with `h` expressed as **points per wavelength**
(`λ = 2π/k = √2` on the unit cube, `ppw = λ/h`):

| `h` | ppw | DOFs | mean lowest triplet | rel. error |
|-----|-----|------|--------------------|------------|
| 0.50 | 2.8 | 53 | 14.52 | 26% |
| 0.40 | 3.5 | 121 | 18.67 | 5.4% |
| 0.30 | 4.7 | 217 | 19.48 | 1.3% |
| 0.25 | 5.7 | 276 | 19.07 | 3.4% |
| 0.20 | 7.1 | 532 | 19.27 | 2.4% |
| 0.16 | 8.8 | 1083 | 19.46 | 1.4% |

The fitted rate is **2.1 in `h`**, consistent with the theoretical `O(h²)` for *eigenvalues* with
lowest-order edge elements (the *field* itself converges at `O(h)`). Extrapolating the fit: **~9 ppw for
1%** and **~28 ppw for 0.1%** eigenvalue error.

Read those numbers with two caveats, both real:

- **The sequence is not monotone** (0.25 is worse than 0.30). gmsh meshes at different `h` are
  unstructured and non-nested, and the mode is 3-fold degenerate, so the triplet mean wobbles by a
  percent or so between meshes. Treat the table as a trend, not a per-`h` guarantee — and do your own
  convergence check on your own geometry rather than reading a single number off a single mesh.
- **A cavity eigenvalue is the friendly case.** A *driven* problem at high frequency additionally
  suffers the **pollution effect**: holding `ppw` fixed as the domain grows in wavelengths does *not*
  hold the error fixed — it degrades with `k(kh)^{2p}` (Babuška & Sauter, *Is the pollution effect of
  the FEM avoidable for the Helmholtz equation considering high wave numbers?*, SIAM J. Numer. Anal.
  **34**(6), 2392–2423, 1997). So an electrically large problem needs *more* points per wavelength
  than a small one, and lowest order is where that bites hardest. If you need many wavelengths across
  the domain, budget accordingly.

---
