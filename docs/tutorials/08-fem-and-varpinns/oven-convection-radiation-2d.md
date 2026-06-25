# A Heated Box: Conduction + Convection + Radiation

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/oven_convection_radiation_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A rectangular box of air holds two solid cylinders — a **hot** one (a heater) low on the left and a
**cold** one high on the right — while the outer walls leak heat to a cooler outside. Heat moves between
the cylinders and out through the walls by *three* coupled routes, all in one `jno.fem([...])`:
**conduction** (`lap T`), **convection** (buoyancy lifts warm air off the hot cylinder and drops chilled
air off the cold one — the Boussinesq model, two interacting plumes), and **radiation** (the two
cylinders and the four walls form a grey-body enclosure; each surface radiates across the transparent
air to every other it can *see*, and the cylinders occlude each other).

![Heated box: temperature + convection plumes, and the three-mode heat budget](/jNO/assets/oven_convection_radiation_2d.png)

$$
\nabla\!\cdot u = 0,\quad
\partial_t u + (u\!\cdot\!\nabla)u = -\nabla p + \mathrm{Pr}\nabla^2 u + \mathrm{Pr}\,\mathrm{Ra}\,\theta\,e_y,\quad
\partial_t\theta + u\!\cdot\!\nabla\theta = \nabla^2\theta\;(+\ \text{radiation}).
$$

## Robin walls: the box leaks heat

The outer walls are neither fixed-temperature nor perfectly insulated — they lose heat to a cooler
outside by **Newton cooling**, a Robin condition $-\partial_\theta/\partial n = \mathrm{Bi}\,(\theta -
\theta_\text{ext})$. It enters the weak form as a natural surface term on the wall tag:

```python
robin = Bi * (T.bind(x=xo, y=yo) - T_ext) * sT.bind(x=xo, y=yo)   # heat leaves where the wall is warm
```

## The enclosure is the meshed fluid, with two obstacles

The cylinders are holes in the fluid mesh and the walls bound it, so radiation crosses the *meshed*
interior — element normals point **inward** (`inward=True`). The two cylinders also block rays, which the
view-factor builder handles by occlusion; a coarse curved mesh under-resolves the factors, so
`enforce_closure=True` rescales `F` to satisfy closure and reciprocity exactly (van Leersum 1989):

```python
gap = d.enclosure(["hot", "cold", "outer"], inward=True, enforce_closure=True)
gap.check()                                  # closure ~1e-16, reciprocity ~1e-17
```

Temperature is the last field, so its DOFs are `w[off[2]:]` — exactly what `gap.field`/`gap.load` expect.
Each time step freezes the radiosity from the start-of-step wall temperatures (operator splitting:
robust, exact at steady state) and applies it as a consistent surface load, zeroed on the pinned
cylinder rows:

```python
def q_rad(Tdofs):                            # grey-body radiosity, net flux leaving each wall element
    Tk = gap.field(Tdofs) + tau              # absolute non-dim temperature (T**4 is not offset-invariant)
    J  = jnp.linalg.solve(eye - rho[:, None] * F, eps * Tk**4)
    return J - F @ J

def rad_load(w):
    return jnp.zeros(fem.dofs).at[off[2]:].set(N_rc * gap.load(q_rad(w[off[2]:]), size=nT) * free_T)
```

## What each physics does — validated, nothing painted in

Marched from a cold start with backward Euler + Newton, the box develops a two-plume circulation. The
checks audit the computed field directly:

* **convection** develops vigorously ($\max|u|\approx35$), with a warm plume above the hot cylinder;
* **radiation** is energy-conserving over the closed enclosure ($\sum_i A_i q_i \approx 0$ to machine
  precision) and the signs are physical — the hot cylinder **emits**, the cold one and the walls
  **absorb** (right panel above);
* **Robin walls** carry net heat **out** of the box to the cooler outside.

The animation is the *computed* temperature with the *computed* velocity arrows as the box warms; only
the cylinder outlines are drawn, to mark the boundary conditions.

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4–5 (the net-radiation / radiosity
method for diffuse-grey enclosures).
