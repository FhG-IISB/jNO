# Full-Waveform Inversion: recover a wave speed (2nd-order time)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/inverse_wave_speed_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The inverse sibling of the [vibrating membrane](wave-membrane-2d.md). A wave

$$u_{tt} = c^2\,\Delta u,\qquad u = 0 \text{ on } \partial\Omega,\qquad u(0)=\sin(\pi x)\sin(\pi y),\quad u_t(0)=0,$$

travels through a medium whose speed $c^2$ is **unknown**. Given the observed displacement history
$u_\text{obs}(t)$ — a "seismogram" — we recover $c^2$ by differentiating the **time integration
itself**.

## The parameter rides the weak form; `fem.solve()` differentiates the march

```python
c2  = jno.np.parameter((1,), name="c2")          # the unknown coefficient
fem = jno.fem([ui.tt * vi + c2 * (ui.x * vi.x + ui.y * vi.y),
               u(xb, yb) - 0.0, u(xi0, yi0) - u0, ui0.t - 0.0])
```

For a second-order form `fem.solve()` returns the trajectory marched with the energy-conserving
trapezoidal (θ=½) rule, reducing $M_2\ddot u + K u = 0$ to the first-order augmented block in
$y=[u,\,v{=}u_t]$. When a coefficient is a `jno.np.parameter` the block is **re-formed from the
parameter each step**, and the gradient flows through the whole scan back to $c^2$ — no `custom_root`,
no hand-written adjoint. The same mechanism recovers a density on the `ui.tt` term or a shear modulus
in a vector (elastodynamic) form: the machinery behind full-waveform inversion and elastography.

```python
c2.initialize(jax.nn.initializers.constant(1.0))     # start at the wrong speed
c2.optimizer(optax.adam(5e-2))
crux = jno.core([(fem.solve() - u_obs).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
crux.solve(220)                                       # fit c² to the seismogram
```

## What to notice

- A wrong speed makes the wave oscillate at the wrong frequency, so the misfit is **sharply
  informative** — the optimizer has a clean gradient to follow.
- The recovered trajectory lands back on top of the data; $c^2$ is recovered to well under 1%.
- Second-order **soft modes need float64** (see the [ringing cantilever](elastodynamics-cantilever-2d.md));
  `jno.fem` warns if a `u_tt` form is assembled without `jax_enable_x64`.

## Result

![Receiver seismogram: the observed displacement history for the true wave speed, the (too slow) history at the wrong starting speed, and the recovered history, which coincides with the observed one.](/jNO/assets/inverse_wave_speed_2d.png)

The wave at the wrong starting speed (dotted) oscillates too slowly; after fitting through the
differentiable `fem.solve()`, the recovered trajectory (dashed) coincides with the observed
seismogram (solid) and $c^2$ is recovered to $\approx0\%$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/inverse_wave_speed_2d.py:code"
```
