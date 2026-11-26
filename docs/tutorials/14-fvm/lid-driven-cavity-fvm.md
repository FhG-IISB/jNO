# Lid-driven cavity (FVM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/14_fvm/lid_driven_cavity_fvm.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Incompressible Navier-Stokes at Re = 100 with the classical constant lid, compared with Ghia, Ghia & Shin (1982).

## Result

On $32 \times 32$ cells the centre-line velocity is within $3.2\times10^{-3}$ of Ghia et al., and continuity holds to $3\times10^{-13}$.

## The code

```python
--8<-- "tutorial_examples/14_fvm/lid_driven_cavity_fvm.py:code"
```

## What to notice

- `jno.fvm` finds the constraint by structure. The continuity equation has no term in its own unknown $p$, so it gets the Rhie-Chow mass flux and $p$ is its multiplier.
- The pressure gradient `jnn.stack([pi.x, pi.y], axis=-1)` is read in Green-Gauss form, which is conservative.
- `nonlinear=jno.solve.simple()` and `precond=jno.precond.saddle(schur="simple")` solve the same system. The default is Newton on the assembled tangent.

See [Finite Volume Method](../../fvm.md) for how each term is read.
