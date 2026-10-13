# Channel at a Prescribed Flow Rate (constant unknowns)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/channel_flow_rate.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Stokes flow through a channel is usually driven by a pressure you choose. Here the **flow rate** `Q` is
prescribed instead, and the inlet pressure that delivers it is solved for. That pressure is a single
number, so it is a **constant unknown**: `P = d.unknown.scalar(constant=True)`.

## The unknowns and their test functions

```python
u = d.unknown.vector(2, order=2)                   # Taylor-Hood velocity
p = d.unknown.scalar(name="p")                     # pressure
P = d.unknown.scalar(constant=True, name="P_in")   # the inlet pressure: one value
v, q, W = u.test(), p.test(), P.test()
```

Each test function is derived from its unknown, so space, shape and order always match.

## The term list

```python
fem = jno.fem([
    inner(grad(u), grad(v), n_contract=2) - pi * div(v),   # momentum
    -qi * div(u),                                          # continuity
    P * (-v_in[0]),                                        # inlet traction -P n
    (u_in[0] - Q / H) * W,                                 # the flow rate: an integral row
    u(xw, yw) - 0.0, u(xi, yi)[1] - 0.0,
])
```

`P` multiplies the inlet traction like any coefficient. Its test function `W` is one everywhere, so the
term `(u_in[0] - Q/H) * W` integrated over the inlet is the equation `∫ u_x dy = Q`. Nothing else is
needed: the constant is assembled as one extra DOF after the fields.

## What to notice

- The answer is the Poiseuille value `P = 12 Q L / H³` to round-off; the velocity is the parabola and the
  pressure the linear profile, both in the Taylor-Hood space, so they are exact too.
- `P` has no diagonal entry of its own — its only equation is the `W` row — so this is a bordered saddle
  system and is solved directly (`linear=jno.solve.lu()`).
- The same constant can instead be **tied** to a field on a region, `u(region) - U`, which makes the
  field uniform there with a solved-for value. See [Boundary conditions](../../fem/boundary-conditions.md#tying-a-region-to-a-constant-uregion-u).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/channel_flow_rate.py:code"
```
