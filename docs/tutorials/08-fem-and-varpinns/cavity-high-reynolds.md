# How far the stabilised pair goes: the cavity to Re = 5000

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/cavity_high_reynolds.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The [stabilised-flow tutorial](navier-stokes-stabilised-2d.md) verifies equal-order P1/P1 against a
closed form at **Re = 20**. Stabilisation exists for convection-dominated flow, so that is a
correctness check, not an envelope. This run measures the envelope, and checks the answer against the
inf-sup-**stable** pair rather than against itself.

## Two things carry the run

**A regularised lid.** The classical cavity drives the lid at constant speed, putting a discontinuity
— and a pressure singularity — in each top corner. Newton from rest does not survive it: measured, it
fails at **Re = 100** already. `16x²(1−x)²` is the standard fix.

**Continuation in Re.** Even regularised, a cold solve at Re = 1000 drives the iterate somewhere its
own tangent is exactly singular. `nu` is left as a `jno.np.parameter`, so the sweep is a solver slot,
not a hand-written loop:

```python
nu = jno.np.parameter((1,), name="nu")          # in the form
...
fem.solve(nonlinear=jno.solve.newton(direct=True),
          linear=jno.solve.lu(backend="host"),
          continuation=jno.solve.continuation(nu=[1/re for re in ladder], keep="all"))
```

One `fem.solve`, not one per rung — the form compiles once and `nu` arrives as a runtime argument.
And `continuation` **raises** on the first rung that fails, so arriving at the top *is* the
convergence statement: there is no quietly-unconverged rung.

## Measured

![Speed with streamlines at Re = 100, 1000 and 5000: the primary vortex centre migrates toward the
geometric centre and the corner eddies grow.](../../assets/cavity_high_reynolds.png)

48×48 structured mesh, 7,203 DOFs. The primary vortex strengthens monotonically with Re — the classic
cavity signature, and the vortex centre migrates toward the middle as the corner eddies grow:

| Re | min `u_x` on x = 0.5 | `u_y` range on y = 0.5 |
|---|---|---|
| 100 | −0.1635 | −0.1954 … +0.1394 |
| 400 | −0.2340 | −0.3280 … +0.2109 |
| 1000 | −0.2761 | −0.3751 … +0.2604 |
| 2000 | −0.2985 | −0.3930 … +0.2879 |
| 5000 | −0.3312 | −0.4176 … +0.3277 |

## Is the equal-order answer right?

P1/P1 violates the inf-sup condition and works only because PSPG compensates, so the check is
Taylor–Hood P2/P1 — the stable pair — on the same mesh, at Re = 1000:

| | max deviation | as % of profile range |
|---|---|---|
| `u_x` along x = 0.5 | 0.0093 | **0.7 %** |
| `u_y` along y = 0.5 | 0.0084 | **0.7 %** |

for **2.9× fewer DOFs** (7,203 vs 21,219). In 3-D that ratio grows — a P2 tet carries 10 nodes to
P1's 4.

!!! warning "This is not a Ghia comparison"
    Ghia, Ghia & Shin (*J. Comput. Phys.* **48** (1982) 387) tabulate the cavity with a **constant**
    lid. The regularisation changes the driving profile, so these numbers are not comparable to that
    table, and they are not presented as validation against it. This is an *internal*
    cross-validation: the new pair against the established one, same mesh, same problem.

## Scope

- Re = 5000 is where this run stopped, not a proven ceiling — nothing was tried above it.
- Steady solutions only. The physical cavity is unsteady well below Re = 5000; a converged steady
  solution at high Re is a solution of the steady equations, not a claim about the physics.
- The mesh is fixed at 48×48. No mesh-convergence study was run, so the *values* are resolution-
  dependent even where the two discretisations agree with each other.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/cavity_high_reynolds.py:code"
```
