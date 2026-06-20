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

## The pieces are yours — build a diffrax term

`fem` hands you the semidiscrete pieces directly — `fem.M` (dense mass), `fem.operator.A`
(stiffness), the forcing `c`, and `fem.state0`. You form `u̇ = M⁻¹(c − A u)` and wrap it in a diffrax
`ODETerm` yourself; jno never does:

```python
import diffrax

def diffrax_solve(M, A, c, state0, save_ts):
    def rhs(t, u, _args):
        return jnp.linalg.solve(M, c - A @ u)        # u̇ = M⁻¹(c − A u)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Kvaerno5(), t0=save_ts[0], t1=save_ts[-1],
        dt0=save_ts[1] - save_ts[0], y0=state0,
        saveat=diffrax.SaveAt(ts=save_ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
    )
    return sol.ys

M, A = fem.M, dense(fem.operator.A)                  # dense mass + stiffness; c is the forcing vector
traj  = diffrax_solve(M, A, c, fem.state0, save_ts)  # your integrator — no fem.solve()
```

Wrap it as a `(block, args, save_ts) -> ys` callable — reading `block.M` / `block.A` inside — to
pass straight into `fem.solve(solve_fn=...)`, so the diffrax adjoint carries gradients through to any
`jno.np.parameter` for a transient *inverse* problem.

## The result

![Animation: the plate starts cold; a constant heat source switches on near the left, heating a
region that grows and wraps around the two insulated bores, settling into a steady temperature
pattern that is hot near the source and cold at the far end.](/jNO/assets/transient_diffrax_heat_spreader.gif)

The source heats the plate from a cold start; the temperature builds up, flows **around** the two
insulated bores, and settles into steady state (the diffrax trajectory is the actual computed field
at each frame).

## What to notice

- **You never call `fem.solve()`** — `fem.M`, `fem.operator.A`, the constant forcing, and
  `fem.state0` are all you need; you adapt them to diffrax yourself. optimistix, lineax, or a
  hand-written `lax.scan` stepper fit the same way.
- **Verified without an analytic solution.** The diffrax trajectory agrees with the default
  backward-Euler to rel-L2 $\sim10^{-3}$, and the field is at steady state by the final frame (the
  last snapshots stop changing, $\sim6\times10^{-4}$).
- **Complex geometry, unchanged workflow:** the two bores are just `box.difference(...)` circles;
  the insulated (Neumann) condition needs no boundary term at all.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/transient_diffrax_heat_spreader.py"
```
