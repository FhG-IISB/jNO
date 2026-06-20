# Kovasznay Flow (steady Navier–Stokes)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/kovasznay_flow_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Kovasznay (1948) found a rare **closed-form** solution of the *full nonlinear* incompressible
Navier–Stokes equations — the laminar wake behind a 2-D grid — which makes it the canonical
**verification** problem for an incompressible flow solver: there is an exact field to compare
against. At $Re=1/\nu=40$,

$$u = 1 - e^{\lambda x}\cos 2\pi y,\quad v = \tfrac{\lambda}{2\pi}e^{\lambda x}\sin 2\pi y,\quad
p = \tfrac12(1-e^{2\lambda x}),\quad \lambda = \tfrac{Re}{2}-\sqrt{\tfrac{Re^2}{4}+4\pi^2}.$$

## The weak form

The convective term $(\mathbf{u}\cdot\nabla)\mathbf{u}$ — the unknown contracted with itself — makes
the form nonlinear, so `jno.fem` returns a coupled residual operator with an autodiff Jacobian,
solved by Newton on the inf–sup-stable Taylor–Hood pair (P2 velocity / P1 pressure):

```python
conv = inner(gu, ub, n_contract=1)  # (u.grad) u
momentum = inner(conv, vv, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * trace(gv)
fem = jno.fem([momentum, -qq * trace(gu),
               u(xb, yb)[0] - bx, u(xb, yb)[1] - by,   # analytic velocity on the boundary
               p(xpn, ypn) - p0])                       # single-node pressure pin
```

![Computed Kovasznay speed and streamlines, the laminar wake behind a grid.](/jNO/assets/kovasznay_flow_2d.png)

## What to notice

- The convective nonlinearity routes the whole system to the nonlinear coupled operator; the
  Jacobian is autodiffed and Newton converges from rest.
- The recovered velocity matches the analytic Kovasznay field to relative $L^2 \approx 8\times10^{-5}$
  — the order of accuracy of this solve is certified in `tests/test_fem_convergence.py`.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/kovasznay_flow_2d.py"
```
