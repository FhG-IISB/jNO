# Vibrating Membrane (2-D wave, second order in time)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/wave_membrane_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A square drum head, clamped on all four edges, plucked into its fundamental mode and released:
$u_{tt}=c^2\Delta u$ with $u=0$ on the boundary. This is the first **second-order-in-time** FEM
problem — the unknown carries a *second* time derivative `ui.tt`, which `jno.fem` auto-reduces to a
first-order system in $y=[u,\,v{=}u_t]$.

## A second time derivative + two initial conditions

`ui.tt` makes the form second order; a second-order problem needs **two** initial conditions —
displacement and velocity. The velocity IC binds the `"initial"`-slice coordinates **and time**
`ti0` (the `.t` derivative carries its region on the temporal variable):

```python
xi0, yi0, ti0 = d.variable("initial", split=True)
ui0 = u.bind(x=xi0, y=yi0, t=ti0)
weak = ui.tt * vi + C**2 * (ui.x * vi.x + ui.y * vi.y)      # ∫ u_tt φ + c² ∫ ∇u·∇φ = 0
fem  = jno.fem([weak, u(xb, yb) - 0.0,
                u(xi0, yi0) - jno.fn(pluck, [xi0, yi0]),    # displacement IC
                ui0.t - 0.0])                               # velocity IC (at rest)
```

## Step with θ=½, not backward Euler

The assembled block is the usual transient block (`fem.M`, `fem.operator.A`, `fem.state0`), with
state $y=[u;v]$ of size $2N$ — split with `fem.offsets` (`[0, N, 2N]`). **Unlike** the parabolic
(first-order) case, integrate with the energy-conserving **trapezoidal rule** (θ=½); plain backward
Euler would spuriously damp the membrane:

```python
M, A = dense(fem.M), dense(fem.operator.A)
lhs, rhs = M + 0.5 * dt * A, M - 0.5 * dt * A        # θ=½: (M+½dtA) y⁺ = (M−½dtA) y
y = np.asarray(fem.state0)
for _ in range(n_steps):
    y = np.linalg.solve(lhs, rhs @ y)
```

(Or call `fem.solve()`, which applies θ=½ for you.)

## What to notice

- `ui.tt` triggers the second-order route; `ui.t` is the velocity. The state is `y = [u; v]`.
- The centre antinode tracks the analytic standing wave $\sin(\pi x)\sin(\pi y)\cos(\omega t)$,
  $\omega=c\pi\sqrt2$, to ~1% over a full period.
- The amplitude after one period is conserved ($\approx 1$) — the trapezoidal rule does not bleed
  energy from an undamped wave (Newmark average-acceleration; Newmark 1959).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/wave_membrane_2d.py"
```
