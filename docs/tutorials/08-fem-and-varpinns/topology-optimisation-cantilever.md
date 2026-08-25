# Topology Optimisation on a Deformable Mesh

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/topology_optimisation_cantilever.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The classic SIMP cantilever — minimise compliance at a $40\%$ volume fraction — except the **mesh
is a design variable too**. The optimiser drives one vector $[\rho, d_x, d_y]$: an element density
per triangle, plus the position of every interior node. The boundary is resolved by *moving nodes
onto it* rather than by refining until the staircase stops mattering. This is the method of
K. Jung and D.-N. Kim, *Density-based topology optimization using a deformable mesh*,
Computers & Structures (2025), [doi:10.1016/j.compstruc.2025.107879](https://doi.org/10.1016/j.compstruc.2025.107879).

## Three design variables, one optimiser

Densities live on elements (`space="P0"`, one dof per triangle, their eq. 12). Nodal coordinates
become parameters through `.trainable()`, which must be called *before* `jno.fem(...)` because the
assembler reads the coordinate registry when it builds. Both kinds are discovered and stepped
together by `jno.core`:

```python
xm, ym, _ = d.variable("mv", where=interior, split=True)
xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y")
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
rho = jno.np.parameter(s, name="rho")
rho.optimizer(jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0))
```

`jno.optimizers.mma` is the Method of Moving Asymptotes (Svanberg 1987) — a *sentinel* optimiser,
detected by `solve()` rather than being a plain optax transform, because it needs the individual
constraint values and gradients that a scalar-loss transform never sees.

## The constraints that make it legal

Free nodal movement will happily invert an element, so the mesh needs its own constraints — minimum
interior angle, and maximum/minimum element volume (their eq. 24/26/28), aggregated with a $p=50$
norm. `d.cell_angles()` and `d.cell_volume()` are trace nodes, so they are differentiable in the
nodal coordinates, which is exactly what makes them usable as constraints:

```python
g_ang = jno.fn(lambda th: ((2*jnp.pi - th) / (2*jnp.pi - THETA_MIN)).reshape(-1),
               [d.cell_angles()], name="g1").pnorm(50.0, normalize=True)
crux = jno.core([compliance, jno.le(volume, 1.0), jno.le(g_ang, 1.0), ...], domain=...)
```

`jno.le` marks a constraint the optimiser **handles but does not add to the loss**. Without it every
constraint doubles as a soft penalty and fights MMA's own dual handling. `normalize=True` matters
more than it looks: summing $N$ terms at $p=50$ overshoots the true maximum by a factor growing with
$N$, so an unnormalised aggregate reads as a violation from iteration 0 and freezes MMA before it
takes a step. (Measured at $+7.7\%$ on a coarser version of this problem, and it grows with the mesh.)

## Manufacturability: the patch filter and perimeter control

Two separate levers, and they enter the pipeline in two *different* ways.

The **patch filter** (their eq. 17–19) replaces the usual density filter. It drives to zero exactly
those elements whose neighbourhood makes the layout unbuildable — a one-node connection, or a lone
dense element — and leaves everything else alone. It is **non-local**, so it cannot live inside the
weak form, where the kernel sees one element at a time. It goes in as a reparameterisation:

```python
patch = d.patch_filter()
rho.constrain(lambda r: jnp.where(pmask, 1.0, patch(r)))   # physics sees the PHYSICAL density
```

**Perimeter control** is the feature-scale knob. $P$ sums the smoothed density jump across every
interior edge (Haber, Jog & Bendsøe 1996); holding it under a target $P^\*$ forbids the optimiser
from buying stiffness with ever-finer members. It enters as an **interior penalty**,
$R=-\beta\log(P^\*-P)$, not as a `jno.le` constraint:

```python
terms.append(jno.fn(lambda rr, bb: bb[0] * rr, [rho.perimeter(zeta=0.1).log_barrier(PSTAR), beta_p]))
callbacks.append(jno.optimizers.geometric_decay(beta_p, 0.997, start=BETA, minimum=BETA_MIN))
```

`log_barrier` extends the logarithm quadratically once $P$ comes within $\tau$ of the bound. This is
not a detail: a plain `log(max(P* - P, eps))` is **constant** above the bound, so its gradient there
is exactly zero and the barrier silently stops doing anything — the failure mode is a satisfied-looking
run whose perimeter sits far above target.

## Every sensitivity is automatic

The paper hand-derives $\partial C/\partial\rho$ and $\partial C/\partial d_{x_i} = -\sum_{j\in
S_i}(u^j)^{\mathsf T}(\partial k^j_0/\partial d_{x_i})u^j$ — the derivative of each element stiffness
with respect to its three nodes' coordinates — across two pages. None of that is written here.
`fem.solve()` is a differentiable trace node and trainable coordinates are ordinary parameters, so
MMA gets its gradients from AD.

## Always reanalyse — and calibrate the reanalysis

An optimiser that moves nodes can lower compliance either by improving the structure or by distorting
elements until they under-integrate strain energy, and **it cannot tell those apart from the inside**.
The check is to transfer the converged density to a clean, undistorted mesh and re-solve:

```python
rho_ref = d.transfer_cell_field(rho_f, d_ref, points=pts_f, outside=1e-3)   # points= -> deformed source
```

A raw gap is not by itself a distortion measurement — the reference mesh is finer, and a coarse mesh
is over-stiff on its own. The script therefore also runs a **control** carrying no design and no
distortion at all (uniform density on both meshes). Here that control shows only $+0.8\%$, so the
reference mesh is genuinely converged and the excess is attributable to the moved nodes. Skipping this
control is how a load-application bug or a plain refinement effect gets reported as a method result.

??? note "The reanalysis script that produces the numbers below"
    Kept separate from the optimisation run because it is a *check on* the result, not part of
    producing it. It is a **CLI tool**, not a runnable example — it takes the converged design and
    re-solves it on a clean mesh:

    ```bash
    python reanalysis.py --design <design>.npz [--h-fine 0.5] [--threshold 0.5]
    ```

    !!! warning "The optimisation script above does not currently write that `.npz`"
        So the two do not connect out of the box: you have to save `rho_f` yourself at the end of the
        run before this script can consume it. The numbers in the table below were produced that way.

    ```python
    --8<-- "tutorial_examples/08_fem_and_varpinns/reanalysis.py:code"
    ```

## Result

![Top: the optimised physical density on the deformed mesh, a cantilever truss with thick well-defined members. Middle: the optimised mesh with interior nodes coloured by how far they moved, showing movement concentrated along the structural members. Bottom left: bar chart of compliance on its own mesh, the value expected on a clean mesh after correcting for discretisation, and the value actually measured. Bottom right: histogram of element densities, strongly bimodal at 0 and 1.](/jNO/assets/topology_optimisation_cantilever.png)

4,226 elements, 4,408 dofs, 400 iterations, ~160 s on CPU:

| quantity | value |
|---|---|
| compliance, own deformed mesh | $78.45$ |
| compliance, clean mesh (16,824 elements) | $89.86$ &nbsp; (raw gap $+14.5\%$) |
| control, uniform density, no distortion | $661.28 \to 666.87$ &nbsp; ($+0.8\%$ discretisation) |
| **over-report attributable to the moved nodes** | $\mathbf{+13.6\%}$ |
| perimeter $P$ | $587.2$ against target $P^\*=650$ |
| volume fraction / $M_{nd}$ / inverted elements | $0.400$ / $0.075$ / $0$ |

Perimeter control earns its place. Running the same script with `PSTAR = 0.0` gives $P=849.2$,
$C=80.23$, $M_{nd}=0.127$ and an over-report of $+21.6\%$ — so constraining the perimeter produced a
design that is **more binary and substantially more honest about its own stiffness**, because it has
fewer fine features with which to farm discretisation error.

Do **not** read the accompanying drop in compliance ($80.23 \to 78.45$) as perimeter control being
free. Without a length-scale restriction the continuum problem is not well posed — that is exactly
why Ambrosio & Buttazzo introduced the perimeter constraint and Haber, Jog & Bendsøe implemented it —
and numerically the unregularised run fragments instead of converging. On the companion half-MBB
problem the uncontrolled run measurably fails to settle: doubling the iteration budget moved
compliance the *wrong* way, $209.9 \to 224.5$, with perimeter rising $1274 \to 1302$. An
uncontrolled baseline is therefore a poor reference point, and the honest claim here is that
perimeter control makes the problem tractable enough to converge — not that it costs nothing.

For scale, the paper reports $+17.6\%$ for conventional elements on its own cantilever and closes the
gap to $-0.5\%$ with the interpolation-cover enrichment of its 2026 follow-up
(Computers & Structures **331** (2026) 108403), which this tutorial does not implement.

## What to notice

- `.trainable()` on coordinates must precede `jno.fem(...)`.
- `space="P0"` gives one design value per element; `jno.np.parameter` sizes it from the symbol.
- `jno.le(...)` = constraint the optimiser sees but does not descend; `rho.constrain(...)` =
  reparameterisation the *physics* sees. They are not interchangeable — MMA only gets a multiplier
  for the former.
- A near-point traction must be applied over a span that the node predicate actually covers: `y < SPAN`
  drops the facet ending exactly at `SPAN`, which silently scales the load by $(\mathrm{SPAN}-h)/\mathrm{SPAN}$.
- `x64` is on — the $p=50$ aggregation needs the exponent range.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/topology_optimisation_cantilever.py:code"
```
