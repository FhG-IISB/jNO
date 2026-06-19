# Bring your own integrator: a transient solve stepped with diffrax

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/transient_diffrax_heat_spreader.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

`fem.solve()` is a convenience, not a requirement. For a transient problem `jno.fem` hands you the
**semidiscrete block** `M u̇ + A u = c` as `fem.operator`, and you can integrate it with any solver
you like. Here we step it with [diffrax](https://docs.kautz.dev/diffrax/) — an adaptive, stiff-aware
`Kvaerno5` — on a complex domain: a **heat spreader**, a plate with two insulated bores, starting
cold and driven by a constant heat source until it settles into steady state.

## The block is yours — turn it into a diffrax term

`block.as_diffrax()` rewrites the semidiscrete system as `u̇ = M⁻¹(c − A u)` and wraps it in a diffrax
`ODETerm`. You drive the solve; jno never does:

```python
import diffrax

def diffrax_solve(block, args, save_ts):
    db = block.as_diffrax(args=args)
    sol = diffrax.diffeqsolve(
        db.term, diffrax.Kvaerno5(), t0=save_ts[0], t1=save_ts[-1],
        dt0=save_ts[1] - save_ts[0], y0=db.state0,
        saveat=diffrax.SaveAt(ts=save_ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
    )
    return sol.ys

block = fem.operator                       # the FeaxTimeBlock (M, A, state0, dt)
traj  = diffrax_solve(block, {}, save_ts)  # your integrator — no fem.solve()
```

The same `solve_fn` signature `(block, args, save_ts) -> ys` also plugs straight into
`fem.solve(solve_fn=diffrax_solve)`, so the diffrax adjoint carries gradients through to any
`jno.np.parameter` for a transient *inverse* problem.

## The result

![Animation: the plate starts cold; a constant heat source switches on near the left, heating a
region that grows and wraps around the two insulated bores, settling into a steady temperature
pattern that is hot near the source and cold at the far end.](/jNO/assets/transient_diffrax_heat_spreader.gif)

The source heats the plate from a cold start; the temperature builds up, flows **around** the two
insulated bores, and settles into steady state (the diffrax trajectory is the actual computed field
at each frame).

## What to notice

- **You never call `fem.solve()`** — `fem.operator` exposes `M`, `A`, the constant forcing, `state0`,
  and `dt`, and `as_diffrax()` adapts them to diffrax. optimistix, lineax, or a hand-written
  `lax.scan` stepper fit the same way.
- **Verified without an analytic solution.** The diffrax trajectory agrees with the default
  backward-Euler to rel-L2 $\sim10^{-3}$, and the field is at steady state by the final frame (the
  last snapshots stop changing, $\sim6\times10^{-4}$).
- **Complex geometry, unchanged workflow:** the two bores are just `box.difference(...)` circles;
  the insulated (Neumann) condition needs no boundary term at all.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/transient_diffrax_heat_spreader.py"
```
