# Stokes Channel (Poiseuille) Flow — coupled FEM

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/stokes_channel_poiseuille.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Steady incompressible **Stokes flow** in a channel, $-\mu\Delta u+\nabla p=0,\ \nabla\!\cdot u=0$,
solved with an inf-sup-stable **Taylor-Hood** pair (P2 velocity, P1 pressure). Fully-developed
flow has the exact parabolic Poiseuille profile, recovered here to machine precision.

## Two coupled fields + a pressure pin

Each field comes from its own `fem_symbols` call (P2 velocity, P1 pressure); the form carries
one momentum and one continuity term. Pure-Dirichlet velocity leaves the pressure defined only
up to a constant; `p.pin()` removes that null space by gauge-fixing one arbitrary DOF (it is
gauge-fixing, not a boundary condition — so no coordinates are named):

```python
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)   # P2 velocity
p, q = d.fem_symbols(names=("p", "q"), order=1)                     # P1 pressure
fem = jno.fem([mu * inner(gu, gv, n_contract=2) - pp * trace(gv),   # momentum
               -qq * trace(gu),                                     # continuity
               u(xb, yb)[0] - u_profile(yb), u(xb, yb)[1] - 0.0,    # velocity BCs
               p.pin()])                                            # gauge-fix the pressure
```

Per-field blocks of the solution are sliced with `fem.problem.offset`.

## What to notice

- Coupled systems = one `fem_symbols` call per field, then momentum + continuity terms.
- Taylor-Hood = `order=2` velocity + `order=1` pressure; gauge-fix the pressure with `p.pin()`.
- Recovers the exact parabola $u_x=\tfrac{G}{2\mu}y(H-y)$ and $\nabla\!\cdot u\approx 0$ to $\sim 10^{-15}$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/stokes_channel_poiseuille.py"
```
