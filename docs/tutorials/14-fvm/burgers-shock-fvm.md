# Burgers shock (FVM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/14_fvm/burgers_shock_fvm.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

An inviscid Burgers shock from $u = 1 \mid 0$, marched with `jno.solve.erk("ssprk3", cfl=0.5)`.

## Result

The shock is at $x_0 + t/2$, the Rankine-Hugoniot position, to within one cell, and the solution stays within $[0, 1]$.

## The code

```python
--8<-- "tutorial_examples/14_fvm/burgers_shock_fvm.py:code"
```

## What to notice

- The conservation form `(ui*ui/2).d(x, ...)` is what moves the shock at the right speed. `ui*ui.x` is evaluated with the volume gradient and warns that it is not conservative.
- `cfl=` chooses the substeps of every step from the spectral radius of the first-order operator: the classical Courant number for upwinding.
- SSPRK3 with a TVD limiter keeps the total variation from growing.

See [Finite Volume Method](../../fvm.md) for how each term is read.
