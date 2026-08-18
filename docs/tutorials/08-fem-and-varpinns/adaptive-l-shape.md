# Adaptivity — h, r and p side by side

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

jNO adapts a discretisation three ways, all through the same `adapt=` slot on `fem.solve`:

- **h** — `jno.solve.remesh()` / `refine()`: **add** elements where the error is.
- **r** — `jno.solve.relocate()`: **move** a fixed node set down a differentiable objective.
- **p** — `jno.solve.enrich()`: **raise the local order**, leaving the mesh alone, by switching
  interpolation covers on at the marked nodes (`space="cover"`).

Which one wins is decided by the **regularity of the solution**, so this tutorial runs all three on two
problems that answer differently — a corner singularity, and smooth structures that are merely localized.

![Three mechanisms on two problems: h adds elements at the re-entrant corner, r slides a fixed node set
toward it, p switches covers on inside three smooth features; two error-versus-DOF axes show h winning
the singularity and p winning the smooth case.](/jNO/assets/adaptive_l_shape.png)

Everything is measured with one functional, $E=\tfrac12\int|\nabla u_h|^2$, read straight off the assembled
form with `fem.eval`, so it means the same thing for P1, for a relocated mesh and for an enriched space:

```python
def energy(fem, stiff, sol):
    return 0.5 * float(np.dot(sol, np.asarray(fem.eval(stiff, sol)).reshape(-1)))
```

That detail is load-bearing. The obvious alternative — a geometric formula on the vertex values — is
**blind to p**: a cover's extra coefficients are the local gradient, so they change $u_h$ *between* nodes
and barely move the nodal values. Measured that way, enriching 5 % of the nodes and enriching all of them
come out identical, and neither appears to beat P1.

## Part 1 — a singularity: the L-shape re-entrant corner

With Dirichlet data equal to the exact singular mode $u=r^{2/3}\sin(2\varphi/3)$ about $(0.5,0.5)$, $u$ is
harmonic, so **all** the error is that one corner.

| method | error $E-E_\text{ref}$ | DOFs | |
|---|---|---|---|
| coarse start | 4.459e-03 | 92 | |
| **h** | **1.244e-03** | 142 (+50) | 72 % lower |
| **r** | 1.991e-03 | 92 (+0) | 55 % lower |
| **p** | 4.291e-03 | 161 (68 % enriched) | 4 % lower |

**h owns this problem, and p is the wrong tool** — 75 % more DOFs bought a 4 % improvement. That is not a
defect, it is the regime: $r^{2/3}\notin H^2$, and a higher-order space buys its rate from smoothness the
solution does not have. A second, jNO-specific reason compounds it here: this problem's Dirichlet data is
**inhomogeneous**, and a cover field's trace is only the P1 interpolant of $g$ (the tangential covers pin
to zero, not to $\mathrm{d}g/\mathrm{d}s$) — a documented scope limit of the element.

## Part 2 — smooth, but localized

Three compactly-supported structures on a plate: an oscillatory **packet**, a broad **dome** carrying the
*largest* amplitude, and a narrow **spike**. Each is $C^3$ and exactly zero outside its own disc, so $u=0$
on the wall is exact and Part 1's inhomogeneous-trace limitation never enters.

| method | error $E_\text{ref}-E$ | DOFs | |
|---|---|---|---|
| coarse start | 1.097e+00 | 2812 | |
| **h** | 3.065e-01 | 3529 | 3.6× lower |
| **r** | 3.458e+00 | 2812 (+0) | **worse than the start** |
| **p** | **2.139e-02** | 4618 (30 % enriched) | **51× lower** |

p is **14× more accurate than h** here for 1.3× the DOFs, and the enrichment map shows why it is cheap:
it covers the three features and spends nothing on the rest of the plate. The selection order across
rounds is packet → spike → dome — the dome has the largest amplitude and is chosen **last**, because what
an error estimator ranks is resolution demand, not size.

!!! warning "The r row is worse than the start, and the tutorial says so"
    Relocation degrades this problem with either objective (energy $12.23\to9.87$, equidistribution
    $\to7.13$). What is *not* established is why: `relocate()` is known to write back a mesh that differs
    from the geometry its own loop validated ([FhG-IISB/jNO#114](https://github.com/FhG-IISB/jNO/issues/114)),
    so this row may be measuring that defect rather than the method. Read it as **unresolved**, not as
    "r-adaptivity does not work on smooth problems".

## Choosing between them

| the solution is… | reach for | why |
|---|---|---|
| singular (corner, crack, shock) | **h** | order cannot buy a rate the regularity does not support |
| smooth, resolved, the feature moves | **r** | no new DOFs; the mesh follows the feature |
| smooth, localized, under-resolved | **p** | order is far cheaper per DOF than nodes |

## What to notice

- **All three are one argument.** `remesh()`, `relocate()` and `enrich()` all return an `AdaptSpec` for
  the same `adapt=` slot; the physics — the weak form — is untouched by the choice.
- **`theta` means the same thing everywhere** — Dörfler bulk marking, over cells for h and over **nodes**
  for p. For p the marking runs over the *unenriched* nodes only: splitting a cell drops its indicator, so
  an h-loop can re-rank the whole field each round, but enriching a node does not move a geometric
  criterion at all, and ranking globally would re-mark the same nodes forever.
- **`n_dofs` in a p-run's history is the ACTIVE count.** The padded layout gives every node its cover
  slots whether or not it is enriched, so the total never changes; an unenriched node simply has its
  slots pinned, and `max_dofs` budgets against the free count.
- **`adapt=` does not take the `linear=` slot** on a steady problem, so a non-default linear solver is
  passed positionally as `solve_fn` — which every realistic-size adaptive run needs.
- **An `adapt=` driver returns the vertex view.** For P1 that is the whole coefficient vector; for a
  cover field it is only the value slots, so a functional computed from the returned array sees just the
  P1 part. Re-solve on the final space (the loop leaves `fem` bound to it) when you need the full vector.
- **The reference must out-resolve everything measured against it.** A P1 reference is not good enough
  here: the enriched run beat a 17.8k-DOF P1 reference outright, which showed up as a *negative* error.
  The reference is a fully enriched solve instead.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py:code"
```
