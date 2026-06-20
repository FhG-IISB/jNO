# Vector Ginzburg–Landau (nonlinear vector field)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/ginzburg_landau_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The Ginzburg–Landau equation $-\Delta\mathbf{u} + (|\mathbf{u}|^2 - 1)\mathbf{u} = \mathbf{f}$ underlies
superconductivity and pattern formation. The unknown is a two-component **vector** field, and the
reaction $(|\mathbf{u}|^2-1)\mathbf{u}$ couples the components through $|\mathbf{u}|^2 = \mathbf{u}\cdot\mathbf{u}$
— a cubic in which the unknown is contracted with itself.

## The weak form

`jno.fem` detects the self-contraction, routes the system to its nonlinear coupled operator, and
the Jacobian comes from autodiff:

```python
react = (inner(ub, ub, n_contract=1) - 1.0) * inner(ub, vv, n_contract=1)   # (|u|^2 - 1) u . v
weak = inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=2) + react - (f1 * vv[0] + f2 * vv[1])
fem = jno.fem([weak, u(xb, yb) - (0.0, 0.0)])
```

![Computed Ginzburg–Landau vector-field magnitude.](/jNO/assets/ginzburg_landau_2d.png)

## What to notice

- A single vector unknown (`value_shape=(2,)`) with the two components coupled through `|u|^2`.
- Verified by the method of manufactured solutions (recovery to relative $L^2 \approx 6\times10^{-4}$);
  the convergence order of this exact problem is certified in `tests/test_fem_convergence.py`.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/ginzburg_landau_2d.py"
```
