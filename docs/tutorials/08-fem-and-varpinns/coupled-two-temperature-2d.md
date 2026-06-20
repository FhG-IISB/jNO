# Coupled multiphysics on a complex domain (two-temperature heat exchange)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/coupled_two_temperature_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A genuine multiphysics problem on a domain that looks like an engineering part, not a unit square.
The **two-temperature** (local thermal non-equilibrium) model carries a solid temperature $T_s$ and
a fluid temperature $T_f$ on the same domain, coupled by interphase heat exchange $h(T_s-T_f)$:

$$-k_s\,\Delta T_s + h\,(T_s-T_f) = f_s, \qquad -k_f\,\Delta T_f - h\,(T_s-T_f) = f_f.$$

It exercises four things a serious FEM solve needs — and each is a one-liner in jno.

## A real CSG domain with named, independently-refined regions

The domain is a plate with a circular **cooling channel**, built with shapely set arithmetic. A
named annulus `ring` hugs the channel and is meshed **finer** than the `bulk` — different regions,
different mesh sizes, in one `build_mesh`:

```python
channel = Point(1.0, 0.5).buffer(0.28)
ring    = Point(1.0, 0.5).buffer(0.5).difference(channel).intersection(box(0, 0, 2, 1))
dom = jno.domain({"bulk": box(0, 0, 2, 1).difference(channel).difference(ring), "ring": ring})
dom = dom.build_mesh(0.06, sizes={"ring": 0.025})    # coarse bulk, fine ring
```

## Two coupled fields, assembled as one block

Each field is its own `fem_symbols` pair; the cross term `h*(s - f)` couples them. `jno.fem`
assembles the whole thing as one block system:

```python
Ts, qs = dom.fem_symbols(names=("Ts", "qs"))
Tf, qf = dom.fem_symbols(names=("Tf", "qf"))
fem = jno.fem([
    k_s * (s.x*vs.x + s.y*vs.y) + h*(s - f)*vs - f_s*vs,   # solid energy balance
    k_f * (f.x*vf.x + f.y*vf.y) - h*(s - f)*vf - f_f*vf,   # fluid energy balance
    Ts(xb, yb) - Ts_star(xb, yb),                          # boundary data (outer wall + channel)
    Tf(xb, yb) - Tf_star(xb, yb),
])
```

## Your own solver — you never call the built-in

`fem.solve(solve_fn=...)` takes **your** `(A, b) -> u`. Here it is `lineax`; jno writes no solver:

```python
import lineax
sol = fem.solve(solve_fn=lambda A, b: lineax.linear_solve(lineax.MatrixLinearOperator(A), b).value)
off = fem.problem.offset                       # per-field slices of the coupled vector
Th_s, Th_f = sol[off[0]:off[1]], sol[off[1]:]
```

## The result

![Solid temperature T_s (one central lobe), fluid temperature T_f (two lobes — a genuinely
different field), and the exchange field T_s−T_f, all on the plate with its cooling channel; the
mesh is visibly finer in the ring around the
channel.](/jNO/assets/coupled_two_temperature_2d.png)

$T_s$ and $T_f$ are genuinely different fields; the third panel is the computed exchange
$T_s-T_f$ — the local driving force for heat transfer between the phases.

## What to notice

- **Complex geometry is free:** shapely CSG in, a meshed `jno.domain` out — `box.difference(...)`
  for the channel, a named `ring` refined independently of the `bulk`.
- **Multi-field coupling** is just more residual terms; `fem.problem.offset` slices the block
  solution back into per-field vectors.
- **Bring your own solver** through `solve_fn` (here `lineax`) — and the same hook accepts an
  `optimistix` Newton (nonlinear) or a `diffrax` stepper (transient). `.solve()`'s default is only a
  convenience.
- **Verified by the method of manufactured solutions:** impose a known $T_s^\*,T_f^\*$ on the full
  boundary and recover it — rel-L2 $\sim 3\times10^{-5}$, the standard correctness gate for a FEM
  code.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/coupled_two_temperature_2d.py"
```
