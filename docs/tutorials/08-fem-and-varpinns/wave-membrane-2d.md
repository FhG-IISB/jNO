# Vibrating Membrane (2-D wave, second order in time)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/wave_membrane_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
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

## `fem.solve()` steps with θ=½, not backward Euler

The augmented block has state $y=[u;v]$ of size $2N$ — split with `fem.offsets` (`[0, N, 2N]`).
`fem.solve()` integrates it itself with the energy-conserving **trapezoidal rule** (θ=½) — **not**
plain backward Euler, which (unlike the parabolic first-order case) would spuriously damp an undamped
membrane. For a transient solve the result is a differentiable *trace node*, so we read the trajectory
(one $y$ per step) through a minimal crux and split it into displacement and velocity:

```python
N   = fem.offsets[1]                                 # y = [u; v]; first N = displacement
sol = fem.solve()                                    # θ=½ trapezoidal stepping, done inside
state = np.asarray(jno.core([sol.mse]).eval([sol]))  # (n_steps, 2N) trajectory of y = [u; v]
traj, V = state[:, :N], state[:, N:]                 # displacement and velocity histories
```

The energy check $E=\tfrac12 v^\top M v+\tfrac12 u^\top K u$ still pulls the mass/stiffness blocks
straight from `fem.M` and `fem.operator.A`.

## What to notice

- `ui.tt` triggers the second-order route; `ui.t` is the velocity. The state is `y = [u; v]`.
- The centre antinode tracks the analytic standing wave $\sin(\pi x)\sin(\pi y)\cos(\omega t)$,
  $\omega=c\pi\sqrt2$, to ~1% over a full period.
- The amplitude after one period is conserved ($\approx 1$) — the trapezoidal rule does not bleed
  energy from an undamped wave (Newmark average-acceleration; Newmark 1959).

## Result

![Animation of the membrane displacement over one period on a symmetric red-blue scale, the fundamental mode oscillating between positive and negative.](/jNO/assets/wave_membrane_2d.gif)

The fundamental mode swings between its positive and negative extremes and back over one period, on a colour scale held fixed and symmetric across every frame.

![Line plot of the centre-node displacement from jNO versus the analytic cosine over one period; the two curves overlap.](/jNO/assets/wave_membrane_2d.png)

The centre antinode tracks the analytic standing wave $\sin(\pi x)\sin(\pi y)\cos(\omega t)$ to rel-$L^2\approx1.2\times10^{-2}$ over the full period, and the amplitude is conserved ($\approx1$) — the trapezoidal ($\theta=\tfrac12$) rule does not bleed energy from the undamped drum.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/wave_membrane_2d.py:code"
```
