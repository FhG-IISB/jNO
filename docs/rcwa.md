# Rigorous Coupled-Wave Analysis (RCWA)

`jno.rcwa` is an **optional** solver for **periodic, layered** electromagnetic structures — the
canonical case being an extruded metasurface unit cell (a patterned dielectric slab between two
semi-infinite ambients). Unlike [`jno.fem`](fem.md), which discretises the whole volume, RCWA is
*semi-analytic in the propagation direction*: each layer is expanded in a truncated in-plane Fourier
basis, solved by an eigenmode decomposition, and the layers are stitched together with a scattering
matrix. For a periodic slab that is far cheaper than a full 3-D complex-Helmholtz solve.

It is built on [`fmmax`](https://github.com/facebookresearch/fmmax), a differentiable JAX Fourier
Modal Method, which is imported **lazily** — the core `jno` install does not pull it in. Enable the
backend with the `rcwa` extra:

```bash
pip install jax-neural-operators[rcwa]      # or:  pixi run -e rcwa ...
```

Constructing an `Rcwa` without `fmmax` installed raises a clear `ImportError` telling you to install
the extra — it never silently no-ops.

## Usage

```python
import numpy as np
import jno

inf = np.inf
# a uniform slab (n = 2) between air ambients — the simplest sanity case
rc = jno.rcwa(
    [(inf, 1.0), (0.4, 4.0), (inf, 1.0)],   # [(thickness, eps), ...] super -> substrate
    period=(0.6, 0.6),                      # in-plane unit-cell period
    orders=200,                             # Fourier truncation (high contrast needs hundreds)
    wavelength=1.03,
    assume_periodic=True,                   # you must affirm the cell is a true period/supercell
)
sol = rc.solve()                            # inc=None -> a normal-incidence plane wave
sol.efficiency("T")                         # transmitted power fraction (all propagating orders)
sol.efficiency("R")                         # reflected power fraction
sol.order(+1, 0)                            # efficiency into a specific diffraction order
```

A patterned layer takes a 2-D `eps` array sampled on the unit-cell grid instead of a scalar; the grid
must be fine enough to resolve `orders` (a Nyquist guard enforces this).

## Never fails silently

Every inference is validated and raises `RcwaError` (or `ImportError` for the missing backend) with a
concrete fix, so a physically wrong result is never returned quietly:

| condition | guard |
|---|---|
| `fmmax` not installed | `ImportError` pointing at the `[rcwa]` extra |
| cell not affirmed periodic | must pass `assume_periodic=True` (a finite aperture is not periodic) |
| non-positive period / empty layers / bad thickness | raises with the offending value |
| patterned grid under-resolves `orders` | raises with the required grid size (Nyquist) |
| wavelength unknown | raises — it is never defaulted |
| energy `T + R > 1` after a solve | raises — the modal solve has not converged (raise `orders`) |

Two `fmmax` conventions are baked in so callers never rediscover them: the Poynting flux is summed
*with sign* (an `abs()` sum over-counts and can report `T+R>1`), and the `JONES_DIRECT_FOURIER`
factorization is the default (the naive rule converges poorly for high-contrast dielectrics such as
a-Si). Correctness is checked against the analytic Fresnel/transfer-matrix transmittance in the tests.

## Status and roadmap

`jno.rcwa` currently exposes the **forward engine** with an *explicit* layer stack. Two increments are
planned:

- an `eps`-driven front-end `jno.rcwa(eps, orders=...)` that samples a jNO permittivity expression on a
  grid (via a new `eps.at(coords)` primitive), **auto-derives the layers**, and threads the design
  gradient through for inverse design;
- `Rcwa.as_precond(...)` as a background (`M⁻¹ = A₀⁻¹`) preconditioner for the large 3-D FEM
  complex-Helmholtz solve, once the FEM↔grid transfer is wired. Until then it raises rather than
  returning a no-op preconditioner.

## References

- M. G. Moharam & T. K. Gaylord, *Rigorous coupled-wave analysis of planar-grating diffraction*,
  J. Opt. Soc. Am. **71**, 811 (1981).
- M. G. Moharam, D. A. Pommet, E. B. Grann & T. K. Gaylord, *Stable implementation of the enhanced
  transmittance matrix approach*, J. Opt. Soc. Am. A **12**, 1077 (1995).

## API

::: jno.rcwa.Rcwa

::: jno.rcwa.rcwa
