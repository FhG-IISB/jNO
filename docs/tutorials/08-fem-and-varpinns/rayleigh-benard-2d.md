# A 2D pot heated from below — Rayleigh–Bénard convection (heat + flow)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/rayleigh_benard_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Heat a layer of fluid from below and, above a critical temperature difference, it stops merely
conducting: hot fluid becomes buoyant and rises, cold fluid sinks, and the layer organises into
**rolling convection cells**. This is the canonical multiphysics problem — incompressible flow coupled
to heat transport — and it falls out of one `jno.fem([...])`.

## The Boussinesq model — three fields, two-way coupling

$$
\begin{aligned}
\partial_t u + (u\cdot\nabla)u &= -\nabla p + \mathrm{Pr}\,\nabla^2 u + \mathrm{Pr}\,\mathrm{Ra}\,T\,\hat e_y &&\text{(momentum + buoyancy)}\\
\nabla\cdot u &= 0 &&\text{(incompressible)}\\
\partial_t T + u\cdot\nabla T &= \nabla^2 T &&\text{(heat: advected + diffused)}
\end{aligned}
$$

Velocity $u$ (P2), pressure $p$ (P1) and temperature $T$ (P1) are three coupled fields. The coupling
is genuinely **two-way**, and that is what makes it multiphysics rather than two solves glued together:

- **buoyancy** $\mathrm{Pr}\,\mathrm{Ra}\,T$ feeds temperature into the momentum balance — a *linear*
  cross term, like the exchange term in the two-temperature example;
- **advection** $u\cdot\nabla T$ feeds the flow into the heat balance — a product of two *different*
  unknowns, so the whole system is **nonlinear** and routes through the coupled Newton path (the same
  machinery as the Navier–Stokes convective term $(u\cdot\nabla)u$).

```python
u, v = d.fem_symbols(value_shape=(2,), order=2)   # velocity (P2)
p, q = d.fem_symbols(order=1)                       # pressure (P1)
T, s = d.fem_symbols(order=1)                        # temperature (P1)

momentum = (u.t·v) + (u·∇)u·v + Pr·∇u:∇v - p·div v - Pr*Ra*T·v_y   # buoyancy: T -> momentum
continuity = q · div u
energy   = T.t·s + (u·∇T)·s + ∇T·∇s                                # advection: u -> heat
fem = jno.fem([momentum, continuity, energy, *walls, *initial_condition])
```

`fem.solve()` marches it implicitly (backward-Euler + Newton per step, internally) and returns the
forward trajectory; we watch the rolls grow from rest.

## A pot: no-slip walls, hot floor, cold lid

- **no-slip** on every wall — `u(walls) - 0` (one all-component vector Dirichlet);
- a **hot floor / cold lid**, with the conductive profile held on the walls — `T(walls) - (1 - y)`;
- the fluid starts **at rest** from the conductive state plus a tiny perturbation that seeds the
  instability — `u(initial) - 0` (an all-component vector initial condition) and `T(initial) - T0`.

The all-component vector `u=0` Dirichlet and initial condition both exercise multifield BC/IC paths
that this work completed (a vector field's BC/IC no longer has to be spelled out per component).

## The result

![A rectangular pot, temperature shown hot (red) at the floor to cold (blue) at the lid, with the
computed velocity arrows forming counter-rotating convection rolls — hot plumes rising, cold plumes
sinking.](/jNO/assets/rayleigh_benard_2d.gif)

Temperature is the colour (hot floor → cold lid); the arrows are the **computed** velocity. From rest,
the conductive layer is unstable: small perturbations grow into rising hot plumes and sinking cold
ones, settling into steady counter-rotating rolls. Both fields are the **actual finite-element
solution** — nothing painted in.

## What to notice

- **One coupled solve.** Velocity, pressure and temperature are assembled and Newton-solved together;
  the buoyancy and advection cross-terms are just ordinary terms in the weak form.
- **Convection genuinely onsets.** The script starts from rest (`max|u| = 0`) and asserts the flow
  grows and carries hot fluid upward — the instability is computed, not imposed.
- **Generality by reuse.** This is the nonlinear coupled-multifield path (Navier–Stokes + two
  temperatures, combined): vector + scalar fields, mixed P2/P1 orders, transient, on a real domain.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/rayleigh_benard_2d.py:code"
```
