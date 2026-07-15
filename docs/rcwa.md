# Rigorous Coupled-Wave Analysis (RCWA)

`jno.rcwa` is an **optional** solver for **periodic, layered** electromagnetic structures — the
canonical case being an extruded metasurface unit cell (a patterned dielectric slab between two
semi-infinite ambients). Unlike [`jno.fem`](fem.md), which discretises the whole volume, RCWA is
*semi-analytic in the propagation direction*: each layer is expanded in a truncated in-plane Fourier
basis, solved by an eigenmode decomposition, and the layers are stitched with a scattering matrix. For
a periodic slab that is far cheaper than a full 3-D complex-Helmholtz solve.

It is built on [`fmmax`](https://github.com/facebookresearch/fmmax), a differentiable JAX Fourier
Modal Method, imported **lazily** — the core `jno` install does not pull it in. Enable the backend with
the `rcwa` extra:

```bash
pip install jax-neural-operators[rcwa]      # or:  pixi run -e rcwa ...
```

## The front door — infer the problem from a jNO constraint list

Hand `jno.rcwa` the **same constraint list you would give [`jno.fem`](fem.md)** (or an already-built
`FEM`). Nothing else is required — it reads everything out of the traced problem:

```python
rc  = jno.rcwa(constraints, orders=300)
sol = rc.solve()                 # period, layers, permittivity, wavelength, incidence all inferred
sol.efficiency("T")              # transmitted power fraction
sol.order(+1, 0)                 # a specific diffraction order

rc.spec                          # the inferred RcwaSpec — inspectable WITHOUT fmmax
```

What is inferred from the problem, and from where:

| inferred | source in the list / domain |
|---|---|
| **periodicity + period `(Px,Py)`** | the Floquet ties `u(left)-u(right)`, `u(front)-u(back)` — **absent ⇒ raise**, never assumed |
| **super/substrate ambients** | the two z-normal radiation faces |
| **permittivity** | the `K0²·ε` coefficient recovered from the scalar Helmholtz volume term, sampled along z, then grouped by [`detect_layers`](#layer-detection) |
| **wavelength / `k0`** | the coefficient's value in the vacuum superstrate (`k0 = √coeff`); pass `wavelength=` to override |
| **incident wave** (lit face + angle `k_in`) | the assembled forcing `b` — a constant-phase source ⇒ normal incidence |
| **tensor permittivity `ε̂`** | an `inner(ε̂ @ u, v)` mass term (a 3×3 `MatrixView`) — birefringence, polarization conversion |
| **in-plane PML** | a complex coordinate stretch `S = 1 + iσ/k` in the stiffness coefficients — see below |
| **internal source** | a `- f·v` / `- inner(J, v)` volume forcing — a dipole / Gaussian emitter — see below |

The permittivity is recovered by splitting the volume weak form `∇u·∇v − K0²·ε·u·v`: the stiffness
summands carry trial/test inside `Jacobian` nodes, the mass summand carries them as bare values, so
dropping the `u·v` factor leaves `K0²·ε`. Because it is the same traced coefficient, its dependence on
a trainable `jno.np.parameter` carries through, so inverse design flows unchanged. Only `orders` (a
numerical truncation choice) is genuinely the user's; `wavelength` is an optional override for the case
where no ambient is vacuum.

Because RCWA solves the *infinitely periodic* problem, a finite aperture with plain absorbing side
walls and **no ties** is **rejected** rather than silently periodicised. To model an *isolated*
scatterer, keep the ties and add an in-plane PML frame (below) — the standard periodic-supercell trick.

## In-plane PML — an isolated scatterer from a periodic supercell

A [Perfectly Matched Layer](https://en.wikipedia.org/wiki/Perfectly_matched_layer) is a complex
coordinate stretch `S = 1 + iσ/k` (σ ramps up in an absorbing frame, `0` in the physical core). Written
into the **scalar Helmholtz** volume term it appears as anisotropic **stiffness** coefficients (and a
matching mass coefficient):

```python
Sx, Sy = 1 + 1j*sx/K0, 1 + 1j*sy/K0                       # sx, sy ramp near the x / y walls
vol = (  (Sy/Sx)*(ui.x*vi.x) + (Sx/Sy)*(ui.y*vi.y) + (Sx*Sy)*(ui.z*vi.z)   # uniaxial stretch, Sz = 1
       - K0**2 * (Sx*Sy) * eps * (u*vi) )
constraints = [vol, absorbing_top, absorbing_incident_bottom, u_left-u_right, u_front-u_back]
sol = jno.rcwa(constraints, orders=200).solve()
```

The front door reads the stretch `Λ = diag(Sy/Sx, Sx/Sy, Sx·Sy)` straight off the stiffness
coefficients and forms the Maxwell uniaxial PML — a **diagonal `ε̂` *and* `μ̂`** (`ε̂ = ε·Λ`, `μ̂ = Λ`) —
solved with `fmmax`'s general anisotropic eigensolve. The **Floquet ties stay** (`fmmax` is inherently
periodic); the absorbing frame just makes the supercell walls non-coupling, so light diffracted toward a
neighbour is absorbed instead of recirculated, and the cell behaves like a single isolated scatterer.
The traced stretch is honoured **exactly** — any σ profile, not a fixed built-in one.

A design `jno.np.parameter` on the **scatterer** inside the PML supercell **is differentiable** — the PML
layers are re-derived from it (`ε̂ = ε·Λ`, `μ̂ = Λ`), so `jax.grad` flows and you can inverse-design an
*isolated* structure. Scope: a **uniaxial** (diagonal) **in-plane** stretch on the **scalar** Helmholtz
term. An off-diagonal stretch or a z-stretch (meaningless for RCWA — the S-matrix already gives
outgoing-wave z-boundaries) each **raise**.

## Internal sources — dipole / Gaussian emitters

An emitter is authored the **same way you drive `jno.fem` or a PINN** — as a forcing term in the residual,
not a special object. A scalar monopole is `- f·v`; a vector dipole is `- inner(J, v)` (with `J` a
current-density vector giving the orientation):

```python
profile = jno.np.exp(-(((xi-x0)**2 + (yi-y0)**2 + (zi-z0)**2) / (2*w**2)))   # localized emitter
vol = ui.x*vi.x + ui.y*vi.y + ui.z*vi.z - K0**2*eps*(u*vi) - A*profile*vi     # `- A·profile·v` forcing
sol = jno.rcwa([vol, absorbing_top, absorbing_bottom, u_left-u_right, u_front-u_back]).solve()
sol.power("up"), sol.power("down")     # power radiated into each ambient
sol.extraction("up")                   # directionality = up / (up + down)
```

The front door detects the trial-free / test-present volume summand, **localizes** it (centroid → point
`dirac_delta_source` vs Gaussian `gaussian_source(fwhm)`; which z-layer it sits in), splits the stack at the
source plane, and drives `fmmax`'s `amplitudes_for_source`. A higher-index substrate correctly biases
emission downward (substrate-emission enhancement — the LED/OLED extraction physics). The **amplitude,
orientation, lateral location, z-position and Gaussian width — plus the design ε — are all differentiable**
(`jax.grad` of extraction or emitted power flows through), so you can inverse-design the environment around
an emitter, optimize *where* it sits, or recover an unknown source strength. Only the discrete choices
(which layer, point-vs-Gaussian) are frozen at construction.

Scope: scalar Helmholtz, the source must lie in a **finite (contrast-defined) layer**, `k_in` is nudged off
the singular Γ-point (a single Bloch point — full Brillouin-zone averaging is future work), and a boundary
plane-wave incidence together with an internal source **raises** (author one excitation). Purcell / LDOS
(total emitted power ÷ a homogeneous-medium reference) is not wired yet.

## The backend — explicit layers

`jno.Rcwa` takes a hand-built stack directly; this is what the front door constructs internally:

```python
import numpy as np
inf = np.inf
rc  = jno.Rcwa([(inf, 1.0), (0.4, 4.0), (inf, 1.0)],   # [(thickness, eps), ...] super -> substrate
               period=(0.6, 0.6), orders=200, wavelength=1.03, assume_periodic=True)
sol = rc.solve()
```

## Layer detection

`jno.rcwa.detect_layers(E, z)` groups a z-sampled permittivity `E` of shape `(Nz, Ng, Ng)` into RCWA
layers, with the two ambients marked semi-infinite. Continuous z-variation (an inclined or curved
geometry) **raises** unless you opt in with `slices=N` to staircase it.

## Never fails silently

Every inference is validated and raises `RcwaError` (or `ImportError` for the missing backend) with a
concrete fix:

| condition | guard |
|---|---|
| `fmmax` not installed | `ImportError` pointing at the `[rcwa]` extra |
| no Floquet ties (finite aperture) | raises — author periodic side walls |
| periodicity in only one of x / y | raises — add the missing tie |
| no z-normal ambient faces found | raises — tag the top/bottom faces |
| `eps` varies continuously in z | raises unless `slices=N` |
| no source / forcing in the problem | raises — nothing to illuminate with |
| forcing spread over a z-range, not one face | raises — put the incident wave on one ambient |
| patterned grid under-resolves `orders` | raises with the required grid size (Nyquist) |
| energy `T + R > 1` after a solve | raises — raise `orders` |

Two `fmmax` conventions are baked in so callers never rediscover them: the Poynting flux is summed
*with sign* (an `abs()` sum over-counts and can report `T+R>1`), and the `JONES_DIRECT_FOURIER`
factorization is the default (the naive rule converges poorly for high-contrast dielectrics such as
a-Si). Correctness is checked against the analytic Fresnel/transfer-matrix transmittance in the tests.

## Roadmap

- an `as_precond` path so RCWA can serve as the layered-background preconditioner `M⁻¹ = A₀⁻¹` for the
  large 3-D FEM complex-Helmholtz solve, once the FEM↔grid transfer is wired;
- a differentiable PML path (a design `jno.np.parameter` routed through a PML supercell — the forward
  solve works today, the re-sample does not yet).

## References

- M. G. Moharam & T. K. Gaylord, *Rigorous coupled-wave analysis of planar-grating diffraction*,
  J. Opt. Soc. Am. **71**, 811 (1981).
- M. G. Moharam, D. A. Pommet, E. B. Grann & T. K. Gaylord, *Stable implementation of the enhanced
  transmittance matrix approach*, J. Opt. Soc. Am. A **12**, 1077 (1995).
- W. C. Chew & W. H. Weedon, *A 3D perfectly matched medium from modified Maxwell's equations with
  stretched coordinates*, Microw. Opt. Technol. Lett. **7**, 599 (1994) — the complex coordinate stretch.
- Z. S. Sacks, D. M. Kingsland, R. Lee & J.-F. Lee, *A perfectly matched anisotropic absorber for use as
  an absorbing boundary condition*, IEEE Trans. Antennas Propag. **43**, 1460 (1995) — the uniaxial `ε̂`/`μ̂` PML.

## API

::: jno.rcwa.rcwa

::: jno.rcwa.RcwaSpec

::: jno.rcwa.Rcwa

::: jno.rcwa.detect_layers
