# Adaptivity — h, r and p on one problem

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

jNO adapts a discretisation three ways, all through the same `adapt=` slot on `fem.solve`:

- **h** — `jno.solve.remesh()` / `refine()`: **add** elements where the error is.
- **r** — `jno.solve.relocate()`: **move** a fixed node set down a differentiable objective.
- **p** — `jno.solve.enrich()`: **raise the local order**, leaving the mesh alone, by switching
  interpolation covers on at the marked nodes (`space="cover"`).

They are being compared, so they all run on the **same problem**, from the same coarse mesh, against the
same reference, measured by the same functional:

$$-\Delta u = f \quad\text{on the L-shape},\qquad u=0 \ \text{on } \partial\Omega$$

with $f$ a compact $C^3$ bump in the lower-right arm. That carries **two features of different
character at once** — the re-entrant corner's $r^{2/3}$ singularity, and a smooth localized bump — so one
run exercises both regimes. The boundary condition is homogeneous on purpose: a cover field's trace is
only the P1 interpolant of an *inhomogeneous* $g$, so $u=0$ keeps that documented scope limit out of a
comparison it would otherwise distort.

![The solution with its corner and bump; the h-refined mesh; the enriched node set; and all four runs on
one error-versus-DOF axis.](/jNO/assets/adaptive_l_shape.png)

## The result

| method | error $E_\text{ref}-E$ | active DOFs | |
|---|---|---|---|
| coarse start | 5.725e-03 | 144 | |
| **h** | 6.727e-04 | 561 | 8.5× lower |
| **r** | 9.147e-03 | 144 (+0) | **worse than the start** |
| **p** | 2.147e-04 | 313 (50 % enriched) | 26.7× lower |
| **h then p** | **5.773e-05** | 739 (70 %) | **99.2× lower** |

p reaches **3.1× lower error than h with 56 % of the DOFs**, and composing the two beats either alone.

!!! warning "The r row is worse than the start, and the objective is why"
    Relocation descends a *mesh functional*, and which one is valid depends on the problem. The Ritz
    functional is $J(v)=\tfrac12 a(v,v)-(f,v)$, and it is $J$ that satisfies
    $J_h-J_\text{exact}=\tfrac12\|u-u_h\|_E^2$. With **no** source, $J=E$ — so `objective="energy"`
    descends the error, and on the classic singular-mode L-shape it cuts that error by 55 % at fixed
    DOFs. **This** problem has a source, so $J_h=-E_h$ at the discrete solution: minimising the error
    means *maximising* $E$, and descending it walks away from the solution while flattening elements,
    which is the cheapest way to lower $\int|\nabla u|^2$. Measured: $E$ duly fell 0.12252 → 0.10788,
    the true error rose 3.6×, and the mesh's smallest angle collapsed **40.8° → 3.2°**.

    The tutorial therefore uses the default (arclength equidistribution), which targets resolution and
    keeps the mesh sane — smallest angle 40.8° → 31.8°. It still does not *help* here: the error rises
    monotonically from the very first iteration at every step size tested, so this problem is simply
    not r's regime. An assert now fails if relocation ever costs more than 40 % of the smallest angle,
    because "not tangled" is far too low a bar to catch this.

## Where each method spent its DOFs

This is the interesting half, and one problem with two features is what makes it visible:

```
p put its covers:       corner  75% | bump 100% | elsewhere  38%
h put its elements:     corner 0.0257 | bump 0.0135 | elsewhere 0.0289    (mean cell size)
```

Both concentrate on the **bump**, because at this resolution the bump dominates the error budget — the
smooth feature is simply where the error is. What separates them is the price: p buys its accuracy at
1.5 extra DOFs per marked node, h by adding nodes and elements outright.

The corner is what neither fully resolves alone, and it is why **h then p** wins: refine the mesh where
regularity is the limit, then raise the order where smoothness pays.

## Choosing between them

| the solution is… | reach for | why |
|---|---|---|
| singular (corner, crack, shock) | **h** | order cannot buy a rate the regularity does not support |
| smooth, resolved, the feature moves | **r** | no new DOFs; the mesh follows the feature |
| smooth, localized, under-resolved | **p** | order is far cheaper per DOF than nodes |

The first row is measured, not asserted: on the pure singular-mode L-shape (all error at the corner,
nothing smooth to enrich) h cuts the energy error by **72 %** while p manages **4 %** for 75 % more DOFs.
$r^{2/3}\notin H^2$, and a higher-order space buys its rate from smoothness the solution does not have.

## What to notice

- **All three are one argument.** `remesh()`, `relocate()` and `enrich()` all return an `AdaptSpec` for
  the same `adapt=` slot; the weak form — the physics — is untouched by the choice.
- **The functional has to be space-agnostic.** $E=\tfrac12\int|\nabla u_h|^2$ is read off the assembled
  form, `0.5 * u @ fem.eval(stiff, u)`, so P1, a relocated mesh and an enriched space are measured the
  same way. A geometric formula on the vertex values is **blind to p**: a cover's coefficients are the
  local gradient, so they move $u_h$ *between* nodes and barely move the nodal values. Measured that
  way, enriching 5 % of the nodes and enriching all of them are indistinguishable.
- **The reference must out-resolve everything measured against it.** A P1 reference is not good enough
  here — an enriched run beats a much larger P1 reference outright, which shows up as a *negative*
  error. The reference is a fully enriched (third-order) solve, and an assert fails if it ever fails to
  bound a run.
- **`theta` means the same thing everywhere** — Dörfler bulk marking, over cells for h and over **nodes**
  for p. For p it runs over the *unenriched* nodes only: splitting a cell drops its indicator, so an
  h-loop can re-rank the whole field each round, but enriching a node does not move a geometric
  criterion at all, and ranking globally would re-mark the same nodes forever.
- **`n_dofs` in a p-run's history is the ACTIVE count.** The padded layout gives every node its cover
  slots whether or not it is enriched, so the total never changes; an unenriched node has its slots
  pinned, and `max_dofs` budgets against the free count.
- **A fresh `enrich` starts from plain P1.** Calling it on a field that is already uniformly
  `space="cover"` therefore *removes* enrichment before adding it back selectively — the active DOF
  count goes **down** at that call (912 → 739 in the h-then-p run above). A fresh cover field carries
  no mask, and a loop that began fully enriched would have nothing left to select. Calling `enrich` a
  **second** time is the other case: it resumes from the mask the first call left, so two budgeted
  runs compose instead of the second undoing the first.
- **Two API frictions worth knowing.** `adapt=` does not take the `linear=` slot on a steady problem, so
  a non-default linear solver is passed positionally as `solve_fn`; and an `adapt=` driver returns the
  *vertex view*, which for a cover field is only the value slots — re-solve on the final space (the loop
  leaves `fem` bound to it) when you need the full coefficient vector.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py:code"
```
