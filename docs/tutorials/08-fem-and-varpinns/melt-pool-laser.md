# A laser melt pool: melting, thermal stress, and the distortion left behind

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/melt_pool_laser.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A laser crosses a steel plate at 0.15 m/s. Metal melts behind it, the pool travels with the beam, and
once the beam leaves the pool freezes — locking in a stress that was never relieved. This tutorial is
that whole chain as **one term list, marched once**:

> laser → conduction with latent heat → a melt pool → thermo-elastic stress → cooling → **residual distortion**

$$\rho\,c_{\text{eff}}(T)\,\dot T = \nabla\!\cdot(k\nabla T), \qquad \nabla\!\cdot\boldsymbol\sigma = 0$$

$$\boldsymbol\sigma = 2G\,\boldsymbol\varepsilon(\mathbf w) + \lambda\,\mathrm{tr}\,\boldsymbol\varepsilon(\mathbf w)\,\mathbf I - (3\lambda + 2G)\,\alpha\,(T - T_0)\,\mathbf I$$

Nothing is passed to `fem.solve()` but solver slots. The phase change, the moving heat source and the
coupling are all *terms*.

## Three ideas carry the model

**Latent heat is an apparent heat capacity.** Melting absorbs 270 kJ/kg that never shows up as a
temperature rise. Rather than tracking an interface, put it in the specific heat as
$c_{\text{eff}} = c_p + L_f\,\mathrm{d}f_\ell/\mathrm{d}T$, where the liquid fraction $f_\ell$ is a
smooth ramp through the mushy range. That is a formula, so it goes straight in:

```python
fl     = 0.5 * (1.0 + tanh((Ti - T_MID) / T_HALF))      # liquid fraction
dfl    = 0.5 * (1.0 - tanh((Ti - T_MID) / T_HALF) ** 2) / T_HALF
c_eff  = CP + L_FUS * dfl                                # the latent-heat spike
```

!!! warning "The time step has to resolve the spike"
    $c_{\text{eff}}$ is a narrow peak — 40 K wide here. If $T$ jumps past the mushy range inside one
    step, the latent heat is simply *missed* and the pool runs hot. Measured on a coarse march (15
    steps over 4 ms): **4868 K**, against 2750 K once the step is small enough. It is not a stability
    failure, so nothing warns you; the answer is just wrong.

**A liquid carries no shear.** The moduli are scaled by the solid fraction squared, so the melt goes
soft and cannot support the thermal strain it is under. This is what makes the model produce a
*residual* stress rather than a reversible expansion — and it is the reason the answer at the end is
not zero:

```python
s_deg = jno.lag((1.0 - fl) ** 2 + E_MIN)     # E_MIN keeps the block invertible in the melt
sigma = 2.0 * G_EL * s_deg * eps(w) + LAM * s_deg * trace(eps(w)) * I2 - (3*LAM + 2*G_EL) * s_deg * eps_th
```

**Radiation is written as a heat-transfer coefficient.** $\varepsilon\sigma(T^4 - T_0^4)$ factors
exactly into $h_{\text{rad}}(T)\,(T - T_0)$ with $h_{\text{rad}} = \varepsilon\sigma(T^2+T_0^2)(T+T_0)$.
Lagging *that* is stable where the raw quartic is not — measured, one overshooting Newton iterate on
$T^4$ sent the solve to $2.8\times10^{8}$ K.

## The mesh has to be graded, and `size=` takes a callable

The pool is **~50 µm deep in a 400 µm domain**. A uniform mesh fine enough to resolve it spends almost
every element on cold metal that only conducts. `size=` accepts a function of position, which becomes
a gmsh mesh-size callback:

```python
# THREE arguments, not two: gmsh calls a size function as f(x, y, z) whatever the dimension.
h_of = lambda x, y, z: H_FINE + (H_COARSE - H_FINE) * min(1.0, max(0.0, (LY - y) / BAND))
d = jno.shape.rect(0.0, 0.0, LX, LY, size=h_of).domain(time=(0.0, T_END, NSTEPS + 1))
```

| mesh | nodes | through the pool depth |
|---|---|---|
| uniform 16 µm | 2,296 | **3 cells** — a recirculation cannot live in that |
| uniform 4 µm | 35,226 | 12 cells |
| **graded 4 → 30 µm** | **2,035** | **12 cells** |

Same near-surface resolution as the uniform 4 µm mesh for **17× fewer nodes**, and *fewer* nodes than
the 16 µm mesh that resolved nothing.

!!! danger "`f(x, y)` is the natural thing to write, and it is wrong"
    The callback passes three coordinates in 2-D as well as 3-D. A two-argument function raises inside
    gmsh's C callback, where it used to surface as `Wrong mesh element size lc = 0 (lcmin = 0,
    lcmax = 1e+22)` — naming neither the callable, nor its signature, nor the shape. Worse, whether it
    surfaced at all depended on gmsh's global state: the same wrong function raised in a fresh process
    and meshed *silently* in one that had already built other meshes. jNO now probes the callable
    before registering it and refuses by name. Note also that meshing is **lazy** — `.domain()` alone
    never reaches the callback.

## What it produces

2,035 nodes / **6,105 DOFs**, 2000 steps of 10 µs over 20 ms — **121 s**.

| | |
|---|---|
| peak temperature | **2754 K** at 5.06 ms (solidus 1690 K; steel boils near 3100 K, so this stays in conduction mode) |
| melt pool | 352 µm long × **49 µm deep — 12 cells through the depth**, 371 molten nodes |
| beam leaves the domain | 6.0 ms |
| pool state at 20 ms | **frozen** — 450 K, zero molten nodes |
| peak distortion | **13.07 µm** at 6.00 ms (surface +6.20 µm) |
| **residual distortion, cooled** | **1.24 µm**, surface −0.22 to +1.04 µm |

About 10% of the peak deformation is locked in permanently. That is the number the model exists to
produce: the melt could not carry stress, so what froze in is a mismatch, not a spring.

!!! success "Two checks it has to pass"
    **Physics.** Free thermal expansion over the heated depth is $\alpha\,\Delta T\,L$, with $L$ the
    thermal diffusion length $\sqrt{\alpha t} = 185$ µm at the peak instant. That bounds the surface
    rise at **≤ 6.38 µm**; the computed rise is **6.20 µm**.

    **Mesh independence.** The same model on a *uniform* 16 µm mesh (9,184 DOFs, 3 cells through the
    pool) gives a residual of 1.246 µm and a peak of 13.076 µm, against **1.239 µm** and **13.074 µm**
    graded — agreement to 0.6% and 0.02%. The grading buys pool resolution, not a different answer.

## Scope — what this model is not

**Conduction mode only.** The top surface stays flat, so there is no depression, no humping, no
keyhole. No evaporation and no recoil pressure.

??? note "Melt convection: what was measured, and why it is not switched on"
    A real pool convects. Surface tension falls with temperature for a clean metal
    ($\mathrm{d}\gamma/\mathrm{d}T < 0$), so the free surface is dragged from the hot centre outward —
    the Marangoni roll that makes a conduction-mode pool wide and shallow. Measured from the converged
    temperature field here, the traction is **6370 Pa** ($\partial T/\partial x = 1.59\times10^{7}$ K/m).
    It is the dominant driver at this scale; Boussinesq buoyancy is negligible and was measured to be.

    Adding it needs a velocity and a pressure, a Carman–Kozeny drag $A(T)\mathbf u$ to freeze the
    solid, and PSPG/SUPG stabilisation. Three things were established:

    * **A steady, resolved pool converges at molecular viscosity.** Refining a steady probe with the
      same drag and traction: 16 µm **fails**, 8 µm gives 10.78 m/s, 6 µm 9.33 m/s, 4 µm 8.97 m/s. An
      earlier conclusion here — that the pool is turbulent at $Re\approx9000$ so a laminar model has
      nothing to converge to — was **wrong**; the failure was under-resolution, and the effective
      viscosity used to work around it was unnecessary.
    * **The Carman–Kozeny constant sets the conditioning.** $C = 10^{7}$ puts $10^{10}$ in the solid
      against a viscous scale of ~$4\times10^{2}$, and the transient march will not step through the
      melt onset. At $C = 10^{5}$–$10^{6}$ it marches in about a minute.
    * **The coupled transient is still not trustworthy.** With the Marangoni traction on, the march
      either fails to finish or converges to a thermally wrong state — 907 K where conduction alone
      gives 2490 K at the same instant, with velocities of order $10^{-5}$ m/s that cannot explain the
      difference. That is an unexplained defect, not a modelling choice, so convection is left out
      rather than shipped looking plausible.

    The failure is at least *legible*: a transient march now reports the step it failed at
    (`step 37 of 120, t = 0.00038` — the instant the first cell melts), instead of returning a finite,
    plausible trajectory.
