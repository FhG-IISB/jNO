# Bratu Problem (nonlinear combustion)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/bratu_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The Bratu (Gelfand) problem $-\Delta u = \lambda e^{u}$ with $u=0$ on the boundary is a classic model
of a thermal explosion: an exponential heat-release reaction competes with diffusion. Solutions
exist only for $\lambda \le \lambda_c \approx 6.81$ on the unit square, in two branches — and the
stable lower branch is found by Newton from rest.

## The weak form

The $e^{u}$ term is nonlinear in the unknown, so `jno.fem` returns a residual operator
(`fem.residual` / `fem.jacobian`) rather than a linear `A, b`:

```python
# grad u . grad v - lambda e^u v = 0 ; the e^u term makes the form nonlinear
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - lam * jno.np.exp(ui) * vi, u(xb, yb) - 0.0])
```

![Computed lower-branch Bratu solution, a smooth combustion bump.](/jNO/assets/bratu_2d.png)

## What to notice

- `jno.np.exp(ui)` of a trial function is detected as a nonlinearity → the Newton residual route.
- There is no closed form in 2-D, so the prediction is verified by **mesh convergence**: the peak
  value of the solution is mesh-independent (coarse vs fine agree to $\sim2\times10^{-3}$).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/bratu_2d.py"
```
