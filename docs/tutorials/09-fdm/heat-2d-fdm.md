# Transient Heat 2D (FDM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/09_fdm/heat_2d_fdm.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The heat equation $u_t = \nu\,\Delta u$ on the unit square with $u = 0$ on the boundary and
$u_0 = \sin(\pi x)\sin(\pi y)$, whose exact solution decays as
$u = e^{-2\nu\pi^2 t}\sin(\pi x)\sin(\pi y)$. Solved through `jno.fdm` by the method of lines.

## Result

![Animation of the sine bump decaying towards zero over time on the unit square.](/jNO/assets/heat_2d_fdm.gif)

![Left: the jno.fdm field at t=0.5. Middle: the signed error against the analytic decay. Right: a log-log spatial-refinement study of the relative L2 error at the final time.](/jNO/assets/heat_2d_fdm.png)

The initial $\sin(\pi x)\sin(\pi y)$ bump decays smoothly under the method-of-lines march (peak
$1 \to \approx 0.61$ by $t = 0.5$). At the final time the field matches the analytic decay to
rel-$L^2 \approx 9.1\times10^{-3}$, and re-solving at three mesh sizes (200 steps each) shows the
spatial error falling at second order (fitted slope $\approx 1.94$, right).

## The initial condition is a constraint

As in `jno.fem`, a problem is transient exactly when it carries an initial condition — and the IC is
*found from the constraints* (`u(xi, yi) - u0`, with `xi, yi` the `"initial"` region), never a config
flag. The time window and step count come from `domain.time = (t0, t1, n)`, and the `u.t` term marks
the time derivative:

```python
d = jno.Shape.rect(0, 0, 1, 1, size=0.06).domain(time=(0.0, 0.5, 200))
x, y, t   = d.variable("interior", split=True)     # temporal Variable t
xi, yi, _ = d.variable("initial",  split=True)     # the t = t0 slice
ui = u.bind(x=x, y=y, t=t)

traj = jno.fdm([
    ui.t - nu * (ui.d2(x) + ui.d2(y)),                 # u_t = nu * Delta u
    u(xb, yb) - 0.0,                                   # Dirichlet
    u(xi, yi) - jnn.sin(np.pi*xi) * jnn.sin(np.pi*yi), # initial condition
]).solve()
```

## What to notice

- `.solve()` marches by the **method of lines**, reusing the same semidiscrete time-stepper `jno.fem`
  uses — no new time-integration code — and returns the trajectory `(n_steps, N)`.
- `t_span` and the number of steps are inferred from `domain.time`; the IC supplies the initial state.
- The final field is audited against the analytic decay
  (rel-$L^2 \approx 9\times10^{-3}$ at $t = 0.5$).

## Full script

```python
--8<-- "tutorial_examples/09_fdm/heat_2d_fdm.py:code"
```
