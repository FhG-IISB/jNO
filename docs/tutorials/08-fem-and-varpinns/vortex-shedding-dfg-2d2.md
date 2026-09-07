# DFG 2D-2: vortex shedding at Re = 100

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/vortex_shedding_dfg_2d2.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The [2D-1 tutorial](navier-stokes-cylinder-dfg.md) solves the *steady* cylinder at Re = 20. Drive it
three times harder and the wake goes unstable: a von Kármán street, shedding at a frequency the
benchmark publishes. This is the external check on the stabilised-flow work — it exercises the vector
Laplacian, `dom.cell_metric` and `jno.solve.bdf2()` in one run, against a number nobody in this repo
chose.

$$\partial_t\mathbf u + (\mathbf u\!\cdot\!\nabla)\mathbf u - \nu\Delta\mathbf u + \nabla p = 0,\qquad \mathrm{St} = \frac{f D}{U_\text{mean}},\qquad \mathrm{St}_\text{ref}\approx 0.30$$

Configuration is Schäfer & Turek (1996), benchmark 2D-2.

![Vorticity through one shedding cycle: alternating vortices form on the cylinder and convect down the
channel as a von Karman street.](../../assets/vortex_shedding_dfg_2d2.gif)

Vorticity `w = dv/dx - du/dy`, computed from the P1 element gradients — the exact derivative of the
velocity the run solved for, one value per cell. Near field only (`x < 1.4`); the colour range is set
from the **wake**, so the attached shear layer on the cylinder (which reaches `|w| ~ 380`) saturates.

## Why the Strouhal number and not the drag

The 2D-1 tutorial reads drag and lift as the **reaction** conjugate to the cylinder's no-slip
constraint: `fem.eval` assembles the momentum residual with no essential elimination, and the sum over
the constrained DOFs is the force. That is the accurate route to a force from an FE solution — and it
is **steady-only**. The transient path publishes no free (pre-Dirichlet) residual, so `fem.eval`
refuses here.

The Strouhal number needs no forces. It is the frequency of the transverse velocity at a fixed point
in the wake, read straight out of the trajectory — so this benchmark is reachable anyway.

## Measured

Equal-order P1/P1, BDF2, quasi-static `tau`. `St` converges **monotonically from below** under both
mesh and step refinement:

![Left: the transverse velocity at a wake probe, growing from rest into a limit cycle. Right: the
Strouhal number against DOF count for two step sizes, approaching the published reference
band.](../../assets/vortex_shedding_dfg_2d2_strouhal.png)

| mesh size | DOFs | `dt` | scheme | St | vs 0.30 |
|---|---|---|---|---|---|
| 0.040 | 2,778 | 0.005 | BDF2 | 0.2660 | −11.3 % |
| 0.030 | 4,512 | 0.005 | BDF2 | 0.2730 | −9.0 % |
| 0.022 | 7,767 | 0.005 | BDF2 | 0.2752 | −8.3 % |
| 0.016 | 13,812 | 0.005 | BDF2 | 0.2827 | −5.8 % |
| 0.022 | 7,767 | **0.0025** | BDF2 | 0.2862 | −4.6 % |
| **0.016** | **13,812** | **0.0025** | **BDF2** | **0.2899** | **−3.4 %** |

Halving `dt` at fixed mesh moves `St` about as much as refining the mesh at fixed `dt`, so **neither**
error dominates — this run is coarse in both, and the remaining 3.4 % is consistent with that rather
than with a defect in the formulation.

## BDF2 versus backward Euler, on a real problem

Same mesh, same step, only the time scheme changed:

| scheme | St |
|---|---|
| backward Euler (θ = 1) | 0.2712 |
| **BDF2** | **0.2752** |

Backward Euler is more damped, so it under-predicts the shedding frequency. This is the L-stable
first-order scheme losing to the L-stable second-order one on exactly the quantity that a damped
scheme distorts — the [time-scheme note](../../solvers.md#transient-problems) makes the same argument
on a heat problem, and here it costs 1.5 % of a benchmark number.

## One combination that is refused, and correctly

A fully consistent transient `tau` puts `u_t` in the momentum strong residual. That makes the
stabilisation a **state-dependent mass** `c(u)·u_t`, and `jno.solve.bdf2()` refuses it by name: such a
mass is assembled as a residual against *one* previous state, so BDF2's second level has nowhere to
go. The residual here is therefore quasi-static — a real approximation, named as one. Backward Euler
would accept the consistent form; that trade has not been measured.

## Scope

- **St is still 3.4 % low** at the finest configuration run, and converging. This is not a converged
  benchmark result and is not presented as one.
- **Drag and lift are not reported** — see above. That is a jNO capability gap, not a choice.
- The wake probe sits at one point; `St` from zero crossings over the second half of the trajectory
  assumes the transient has died out by then, which the swing amplitude supports but does not prove.
- `tau` carries `2/Δt` and so assumes the fixed grid from `domain(time=…)`. It is wrong under
  `jno.solve.adaptive()`.
- `linear=jno.solve.lu(backend="host")` is not cosmetic: the device sparse factorisation ran out of
  cuSolver memory at 4,512 DOFs on an 8 GB card, and the host route was also **3.8× faster** here.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/vortex_shedding_dfg_2d2.py:code"
```
