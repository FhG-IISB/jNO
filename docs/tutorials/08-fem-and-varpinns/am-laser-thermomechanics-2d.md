# Additive Manufacturing: Scanning-Laser Thermo-Mechanics

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/am_laser_thermomechanics_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A laser sweeps across a clamped metal plate — one pass of a powder-bed or directed-energy build. The
moving hot spot conducts heat into the plate (and loses some to the surroundings); the locally hot
material wants to expand but the cool material around it holds it back, so a **thermal stress** field
travels with the beam. Two physics in one coupled `jno.fem`:

$$
\partial_t T = \nabla^2 T - \mathrm{Bi}\,T + Q_\text{laser}(x,y,t),\qquad
\nabla\!\cdot\sigma = 0,\quad \sigma = \lambda\,\mathrm{tr}(\varepsilon)\,I + 2\mu\,\varepsilon - \beta\,(T-T_\text{ref})\,I .
$$

![Travelling temperature + von Mises stress, and the Rosenthal wake comparison](/jNO/assets/am_laser_thermomechanics_2d.png)

## Temperature drives stress — a linear cross-coupling

Temperature (P1) and displacement (P2 vector) go into one monolithic system. The coupling is the
thermal-expansion term `- beta*(Tb - T_ref)*trace(ep)` — temperature feeds the mechanical balance
exactly the way Boussinesq buoyancy feeds the momentum balance:

```python
T, sT = d.fem_symbols(names=("T", "sT"), order=1)
u, phi = d.fem_symbols(value_shape=(2,), names=("u", "phi"), order=2)
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])

thermal = Tb.t*sb + (Tb.x*sb.x + Tb.y*sb.y) + Bi*Tb*sb          # heat (the laser is added per-step)
mech    = lam*trace(eu)*trace(ep) + 2*mu*inner(eu, ep, n_contract=2) - beta*(Tb - T_ref)*trace(ep)
fem     = jno.fem([thermal, mech, u(xb, yb) - 0.0, T(ci[0],ci[1]) - 0.0, u(ci[0],ci[1]) - 0.0])
```

## The moving laser is a per-step load — and the operator is constant

A *time-dependent* source can't live in the weak form, but with a bring-your-own stepper that's no
obstacle: the whole problem is **linear**, so its operator `M + dt·A` never changes. Factor it **once**
with a sparse LU and back-substitute every step — the laser is simply a different right-hand side
(a Gaussian centred at the current beam position, scattered as a lumped nodal load):

```python
lu = splu((M + dt*A).tocsc())                                  # factor ONCE
for step in range(nsteps):
    cx = x0 + v_scan * (step+1)*dt                             # beam moves
    g  = (2*P/(pi*r0**2)) * exp(-2*((xT-cx)**2 + (yT-y0)**2)/r0**2)
    load = zeros(fem.dofs); load[:nT] = g * nodal_area         # lumped source on the T block
    w  = lu.solve(M @ w + dt*load)                             # back-substitute (no re-factor)
```

The entire 50-step coupled solve runs in a fraction of a second.

## Validated against Rosenthal — and honest about plasticity

The deposited laser energy balances the stored heat plus surface loss (to ~1 %); the constrained hot
zone is in **compression** (`tr σ < 0`); and the trailing thermal **wake** matches the shape of the
classic **Rosenthal** moving-point-source solution (right panel — the finite spot, finite plate, and
surface loss make it slightly cooler than the idealised point source).

Pure thermo-elasticity shows the *transient* stress that travels with the beam and relaxes once the
plate re-cools uniformly. **Permanent residual stress and warping require plasticity** (a J2
return-mapping step) — the natural extension; we deliberately do not call the elastic stress "residual".

Reference: D. Rosenthal, *Trans. ASME* **68**:849 (1946) — the theory of moving heat sources.
