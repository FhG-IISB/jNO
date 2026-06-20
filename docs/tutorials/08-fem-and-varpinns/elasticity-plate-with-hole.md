# Elasticity: a plate with a hole under tension (complex domain)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/elasticity_plate_with_hole.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Vector linear elasticity on a domain with a hole. The unknown is the displacement
$\mathbf{u}=(u_x,u_y)$ (a `fem_symbols(value_shape=(2,))` field) with P2 elements; the isotropic
plane-stress bilinear form is

$$a(\mathbf u,\boldsymbol\varphi)=\lambda\,(\nabla\!\cdot\mathbf u)(\nabla\!\cdot\boldsymbol\varphi)+2\mu\,\boldsymbol\varepsilon(\mathbf u)\!:\!\boldsymbol\varepsilon(\boldsymbol\varphi).$$

## Complex domain, refined at the hole

The plate-with-hole is one shapely line; a named `ring` around the hole is meshed finer, where the
strain is steep:

```python
hole = Point(0, 0).buffer(0.3)
ring = Point(0, 0).buffer(0.62).difference(hole)
dom = jno.domain({"plate": box(-1, -1, 1, 1).difference(hole).difference(ring), "ring": ring})
dom = dom.build_mesh(0.12, sizes={"ring": 0.045})
```

## The elasticity form, in the symbols of the math

`symgrad`, `trace`, and `inner` write $\boldsymbol\varepsilon=\tfrac12(\nabla u+\nabla u^{\mathsf T})$
and the bilinear form directly:

```python
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
fem = jno.fem([weak - fx * phi.bind(x=xi, y=yi)[0],
               u(xb, yb)[0] - ux_star(xb, yb),
               u(xb, yb)[1] - uy_star(xb, yb)])
sol = fem.solve(solve_fn=lambda A, b: lineax.linear_solve(lineax.MatrixLinearOperator(A), b).value)
```

## The result

![Left: the undeformed plate with a circular hole. Right: the plate stretched in x by the imposed
tension, with the circular hole deformed to an ellipse (elongated along the pull, contracted across
it).](/jNO/assets/elasticity_plate_with_hole.png)

Under the x-tension the plate elongates and the circular hole deforms to an ellipse — stretched
along the pull, contracted across it (Poisson effect).

## What to notice

- **Complex geometry + local refinement**: `box.difference(hole)` and a `ring` refined
  independently — exactly the moves you need for real parts.
- **Vector fields read like the math**: `symgrad`/`trace`/`inner` give $\boldsymbol\varepsilon$ and
  the bilinear form with no index bookkeeping; P2 (`order=2`) avoids shear locking.
- **Bring your own solver** (`lineax`) through `solve_fn`.
- **Verified by the method of manufactured solutions:** a known uniaxial-tension field
  $\mathbf u^\*$ (with a sinusoidal perturbation, so the gate is genuinely mesh-convergent rather
  than exactly polynomial) is imposed on the whole boundary and recovered to rel-L2 $\sim7\times10^{-6}$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/elasticity_plate_with_hole.py"
```
