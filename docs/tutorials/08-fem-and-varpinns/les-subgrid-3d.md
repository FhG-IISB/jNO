# An LES subgrid model, inside a real 3-D solve

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/les_subgrid_3d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A large-eddy simulation replaces the scales the mesh cannot carry with an **eddy viscosity**
$\nu_t(\nabla\mathbf u)$ added to the molecular one. In jNO that needs no API at all: the filter width
is `d.cell_size`, the model is arithmetic on `grad(u)`, and it goes in the term list like any other
coefficient. Three of them:

| model | | |
|---|---|---|
| **Smagorinsky** (1963) | $\nu_t = (C_s\Delta)^2\lvert S\rvert$ | the original |
| **Vreman** | $\nu_t = c\sqrt{B_\beta/(\mathbf g\!:\!\mathbf g)}$ | *Phys. Fluids* **16** (2004) 3670, eq. 5 |
| **WALE** | $\nu_t = (C_w\Delta)^2\,(S_d\!:\!S_d)^{3/2}\big/\big((S\!:\!S)^{5/2}+(S_d\!:\!S_d)^{5/4}\big)$ | Nicoud & Ducros, *Flow Turb. Combust.* **62** (1999) 183, eq. 13 |

All three are written through **second invariants** — $\mathrm{tr}(\cdot)$ and $\mathrm{tr}(\cdot^2)$ —
rather than by naming tensor components, so one spelling serves 2-D and 3-D.

## The property that separates them

A subgrid model exists to represent turbulence. In **simple shear**, $\mathbf u = (z,0,0)$, there is
no turbulence to represent, and the correct $\nu_t$ is **zero**. The velocity gradient there is
nilpotent, which makes Vreman's $B_\beta$ and WALE's $S_d\!:\!S_d$ vanish identically. Smagorinsky's
$\lvert S\rvert$ does not — it is non-zero in *any* shear — so it manufactures eddy viscosity
throughout a laminar boundary layer. That is the defect Van Driest wall damping exists to patch.

Measured on a **solved** plane Couette flow (not a prescribed gradient), 6×6×6 tets, Re = 100:

| model | $\max \nu_t/\nu$ | |
|---|---|---|
| no model | 0 | |
| **Vreman** | **1.13 × 10⁻⁶** | vanishes (cancellation floor) |
| **WALE** | **1.99 × 10⁻¹⁹** | vanishes |
| Smagorinsky | **8.03 × 10⁻²** | **70,890× Vreman's** |

## …but they are not inert

A laminar *cavity* is not a case any of these models claims to vanish on — it has genuine 3-D strain
with recirculation, so $B_\beta$ and $S_d\!:\!S_d$ are legitimately non-zero. Same mesh, same Re:

| model | $\max \nu_t/\nu$ | change in the answer |
|---|---|---|
| Vreman | 1.61 × 10⁻¹ | 8.1 × 10⁻³ |
| WALE | 2.58 × 10⁻¹ | 8.3 × 10⁻³ |
| Smagorinsky | 5.19 × 10⁻¹ | 5.6 × 10⁻² |

So the separation in the first table is a property of the **flow structure**, not of the
implementation, and it is worth stating plainly: these models are not "off in laminar flow". They are
off in *simple shear*, which is what a wall boundary layer looks like — and that is exactly where it
matters.

## Two things that will bite

!!! danger "`nu_t` must be lagged"
    $\nu_t$ is a square root, so its slope at $\mathbf u = 0$ is **infinite** and Newton diverges from
    a rest state outright. `jno.lag` freezes it within each linearisation — the same Picard treatment
    a Carman–Kozeny mushy-zone drag needs.

    ```python
    nu_t   = jno.lag(eddy_viscosity(model, grad(u, ax), d.cell_size))
    momentum = inner(adv(gu, ub), vv, 1) + (nu + nu_t) * inner(gu, gv, 2) - pp * div(gv)
    ```

!!! danger "The invariants must be clamped"
    $B_\beta$ and $S_d\!:\!S_d$ are non-negative in exact arithmetic and are computed as a
    **difference of nearly equal numbers**. Measured in pure shear: $B_\beta$ lands at $1.4\times10^{-20}$
    where it should be 0, from a relative cancellation of $2.7\times10^{-16}$. Unclamped, one excursion
    below zero is not a small error — `sqrt` and `**1.5` return **NaN**.

    ```python
    clamp = lambda z: where(z > 0.0, z, 0.0)     # part of the model, not a tidy-up
    ```

    It is also why the first table is judged *relative* to Smagorinsky rather than against an absolute
    floor: Vreman's $1.1\times10^{-6}$ **is** its zero.

## Scope — what this is not

This is a wall-bounded **laminar** demonstration that the models behave as designed inside a Newton
solve, and that they cost little. It is **not validated LES**. That needs a turbulent benchmark
against DNS — a wall-resolved channel, or decaying isotropic turbulence against Comte-Bellot–Corrsin
spectra — neither of which has been run here. A triply-periodic case (Taylor–Green) would also need a
periodic *vector* field, which jNO has no test for today.
