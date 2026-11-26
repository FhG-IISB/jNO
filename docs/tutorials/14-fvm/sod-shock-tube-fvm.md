# Sod shock tube (FVM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/14_fvm/sod_shock_tube_fvm.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The Euler equations of an ideal gas as the flux of a vector unknown $(\rho, \rho u, E)$, with no Euler-specific code.

## Result

At $h = 0.01$ the density $L^1$ error against the exact Riemann solution is $5.7\times10^{-3}$ with Rusanov, $4.6\times10^{-3}$ with HLL and $4.5\times10^{-3}$ with Roe. The mass is constant to $10^{-12}$.

## The code

```python
--8<-- "tutorial_examples/14_fvm/sod_shock_tube_fvm.py:code"
```

## What to notice

- The flux is written as a stacked vector, `jnn.stack([...], axis=-1)`, with one row of components per point.
- Every numerical flux reads the Jacobian of the flux you write: Rusanov uses its spectral radius, HLL its extreme eigenvalues, and Roe the matrix $|A| = \sqrt{A^2 + \delta^2 I}$.
- With no boundary condition, the ends are plain outflow.

See [Finite Volume Method](../../fvm.md) for how each term is read.
