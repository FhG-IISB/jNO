# Stabilised Navier–Stokes: equal-order P1/P1 (SUPG + PSPG)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/navier_stokes_stabilised_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Taylor–Hood (P2 velocity / P1 pressure) is inf-sup stable, and it is what the other flow tutorials
use. It is also expensive: in 3-D a P2 velocity carries roughly four times the DOFs of P1 on the same
mesh. **Equal-order P1/P1 is not a stable pair** — used naively its pressure carries a spurious mode —
but residual-based stabilisation compensates, and in jNO that stabilisation is not a feature you
enable. It is terms you write.

$$(\mathbf u\!\cdot\!\nabla)\mathbf u - \nu\,\Delta\mathbf u + \nabla p = 0,\qquad \nabla\!\cdot\mathbf u = 0$$

The exact solution is Kovasznay (1948), so every number here is checked against a closed form.

## The two pieces the formula needs

Neither is a stabilisation feature; they are what the formula is made of.

**A vector Laplacian.** The momentum *strong* residual is
$\mathbf r_M = (\mathbf u\!\cdot\!\nabla)\mathbf u - \nu\Delta\mathbf u + \nabla p$, so
`jno.np.laplacian` has to accept a vector field.

**An element metric.** $\tau$ has to know the element's *shape*, not just its size — on a
boundary-layer cell the streamwise and wall-normal directions are not interchangeable. `dom.cell_size`
is $|\det J|^{1/d}$, an isotropic scalar that cannot see stretch at all. `dom.cell_metric` is the
covariant metric $G = J^{-\mathsf T}J^{-1}$, per quadrature point:

```python
G  = d.cell_metric
gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)      # aᵀ G a
tau = jno.lag((gG(ub) + C_I * NU**2 * inner(G, G, n_contract=2)) ** -0.5)
```

`trace(G)` and `inner(G, G, n_contract=2)` read as written — `trace` takes the last two axes and
`inner` aligns on the trailing ones.

!!! tip "`jno.lag` on `tau`"
    $\tau$ is a *coefficient*, not part of the equation. Differentiating through $\mathbf u^\mathsf T G\mathbf u$
    hands Newton a tangent it gains nothing from. `jno.lag` freezes it within each linearisation.

## SUPG and PSPG are two terms, and their signs differ

This is the part that is easy to get wrong, so it is worth stating plainly.

They are the *same* strong residual weighted by two *different* test perturbations — but they are two
separate terms in the list, because **an additive term carries exactly one test field**: the test
field is what names the equation block. And each term joins an equation that already has a sign
convention, so it must carry that convention:

```python
r_m = adv(gu, ub) - NU * lap(u, [xi, yi]) + gp        # the strong residual

momentum   = momentum   + tau * inner(adv(gv, ub), r_m, n_contract=1)   # SUPG → momentum   (+)
continuity = continuity - tau * inner(gq, r_m, n_contract=1)            # PSPG → continuity (−)
```

Momentum is written $+(\mathbf u\!\cdot\!\nabla\mathbf u,\mathbf v)$, so SUPG is `+`. Continuity is
written $-(q,\nabla\!\cdot\mathbf u)$, so PSPG is `−`. Getting that sign wrong is not a small error:
with `+` on PSPG, Newton diverges outright.

## What it buys, measured

Kovasznay at $\mathrm{Re}=20$, equal-order P1/P1, against the closed form:

| | velocity error | pressure error |
|---|---|---|
| no stabilisation, `ms=0.11` | 2.53e-02 | 4.09e-01 |
| **SUPG + PSPG**, `ms=0.11` | 3.07e-02 | **9.42e-02** |

A **4.3× better pressure** — which is the whole point, since the pressure is what the unstable pair
gets wrong — for a slightly worse velocity. That trade is expected and is why the term exists.
Under refinement the observed rates are **1.78 (velocity) / 1.70 (pressure)**, in line with theory
for a PSPG-stabilised equal-order pair.

## Scope — stated up front

- **`tau` here is the steady form.** A transient run adds a $(2/\Delta t)^2$ term inside the root.
  That term assumes the *fixed* grid from `domain(time=…)`, so it is wrong under
  `jno.solve.adaptive()`, where the step size is chosen per step.
- **grad-div / LSIC is not included.** The usual coefficient $\tau_C = 1/(\mathrm{tr}(G)\,\tau_M)$ made
  both errors *worse* at this Reynolds number (velocity 6.14e-02 → 8.36e-02, pressure 1.78e-01 →
  2.06e-01), so its calibration is left open rather than shipped as though it were verified. The term
  is one line if you want it: `tau_c * div(gu) * div(gv)`.
- **The default solver will not do.** Use `jno.solve.newton(direct=True)`; the matrix-free default
  goes NaN on a cold start from rest here.
- **`dom.cell_metric` is native 2-D/3-D volume terms only** — like `dom.cell_size`. A 1-D form or a
  non-nodal element family refuses by name.
- This tutorial does **not** claim a turbulence model, a free surface, or a compressible path. jNO's
  fluid scope is still laminar incompressible; see [Formulations](../../fem/formulations.md).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/navier_stokes_stabilised_2d.py:code"
```
