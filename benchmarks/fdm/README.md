# `jno.fdm` benchmarks

Each script solves one problem written as its mathematics and checks it against an exact solution or a
published benchmark. The docstring states the question and the oracle, with its source. Run one with:

```bash
python benchmarks/fdm/<script>.py
```

Measured on CPU (float64), 2026-09-22, on branch `fix/fdm-silent-bugs`:

| script | problem | result |
|---|---|---|
| `ns_kovasznay.py` | steady Navier–Stokes, Re = 40, vector velocity | u second order (1.92); p converging (1.70, 1.55) |
| `ns_taylor_green.py` | time-dependent Navier–Stokes, BDF2 | u 2.27 / 2.18; p 1.88 / 1.94 |
| `ns_cavity.py` | lid-driven cavity, Re = 100 | max \|u − Ghia\| 0.014 (33²), 0.0019 (65²) |
| `boussinesq.py` | natural convection, marched to steady state | 33²: Ra 10³ Nu 1.121 (ref. 1.118), u 3.628 (3.649), v 3.669 (3.697); Ra 10⁴ Nu 2.291 (2.243), u 15.92 (16.18), v 19.25 (19.62) |
| `elasticity.py` | linear elasticity, vector displacement | second order (2.02, 2.01) |
| `electrostatics.py` | layered dielectric ε = 1 \| 10 | exact (1e-11 at h = 0.1; was 0.14 before the conservative form) |
| `maxwell_tm.py` | TM cavity mode, first-order E–H system and wave form | both second order; the first-order system has ~2.5× the error |
| `inverse_viscosity.py` | recover ν from a Navier–Stokes velocity field | ν = 0.025 to relative error 4e-9 (with `linear=jno.solve.lu()`) |

Known limits these exposed, with the details in each docstring:
- **The default solver for coupled nonlinear systems** (matrix-free, unpreconditioned) is weak on
  saddle-point problems. The inverse's gradient solve fails with it, and steady Boussinesq doesn't converge
  from a zero initial guess, so the benchmark marches instead.
- **The Boussinesq pressure** is only weakly determined (smallest Jacobian singular value 5.7e-4 on 9²).
  Velocity and temperature are right, but the pressure keeps drifting in a march until the flow is steady.
- **The cavity** stops converging above Re ≈ 300 on 65²: central convection at a cell Reynolds number above
  2. `jno.fd(upwind=u, order=k)` is the tool for that, but it isn't benchmarked here yet.
