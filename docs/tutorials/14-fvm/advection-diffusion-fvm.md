# Advection-diffusion and limiters (FVM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/14_fvm/advection_diffusion_fvm.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A boundary layer at a cell Péclet number of 2.5: $u' - 0.01u'' = 0$, $u(0) = 0$, $u(1) = 1$, with the numerical fluxes and limiters of `jno.fv`.

## Result

The unlimited second-order reconstruction undershoots to $-0.09$. First order, van Albada, minmod and HLL with van Leer all stay in $[0, 1]$.

## The code

```python
--8<-- "tutorial_examples/14_fvm/advection_diffusion_fvm.py:code"
```

## What to notice

- The convective term is written in divergence form, `ui.d(x, scheme=jno.fv(...))`. The scheme rides on the outer derivative, like a `jno.fd(...)` stencil.
- Rusanov (the default) is exact upwinding for a linear flux; with `recon="none"` it is first order.
- The default van Albada limiter is smooth, so Newton converges on a steady problem. Use minmod or Barth-Jespersen for explicit marches.

See [Finite Volume Method](../../fvm.md) for how each term is read.
