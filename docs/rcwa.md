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
`FEM`) together with the permittivity expression `eps`; it reads everything else out of the traced
problem:

```python
rc  = jno.rcwa(constraints, eps, orders=300, wavelength=1.0)
sol = rc.solve()                 # period, layers, ambients, source, incidence all inferred
sol.efficiency("T")              # transmitted power fraction
sol.order(+1, 0)                 # a specific diffraction order

rc.spec                          # the inferred RcwaSpec — inspectable WITHOUT fmmax
```

What is inferred from the problem, and from where:

| inferred | source in the list / domain |
|---|---|
| **periodicity + period `(Px,Py)`** | the Floquet ties `u(left)-u(right)`, `u(front)-u(back)` — **absent ⇒ raise**, never assumed |
| **super/substrate ambients** | the two z-normal radiation faces |
| **layer stack** | `eps` sampled along z, then grouped by [`detect_layers`](#layer-detection) |
| **incident wave** (lit face + angle `k_in`) | the assembled forcing `b` — a constant-phase source ⇒ normal incidence |

Two arguments stay explicit, on purpose:

- **`eps`** — isolating one coefficient sub-tree from an *assembled* weak form is not supported yet, so
  the permittivity expression is passed directly (you already hold it — it is the design variable).
  Because it is the same traced `eps`, its dependence on a trainable `jno.np.parameter` carries through,
  so inverse design flows unchanged.
- **`wavelength`** — a bare-float `k0` is not an identifiable node in the trace, so `λ` is passed in.

Because RCWA solves the *infinitely periodic* problem, a finite aperture (absorbing side walls, no
ties) is **rejected** rather than silently periodicised — exactly the difference between a supercell
and a finite metasurface.

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
- auto-inference of `eps` (a general `eps.at(coords)` primitive) and of `k0`, so both become optional.

## References

- M. G. Moharam & T. K. Gaylord, *Rigorous coupled-wave analysis of planar-grating diffraction*,
  J. Opt. Soc. Am. **71**, 811 (1981).
- M. G. Moharam, D. A. Pommet, E. B. Grann & T. K. Gaylord, *Stable implementation of the enhanced
  transmittance matrix approach*, J. Opt. Soc. Am. A **12**, 1077 (1995).

## API

::: jno.rcwa.rcwa

::: jno.rcwa.RcwaSpec

::: jno.rcwa.Rcwa

::: jno.rcwa.detect_layers
