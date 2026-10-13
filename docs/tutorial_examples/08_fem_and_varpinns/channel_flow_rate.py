# --8<-- [start:code]
"""A channel driven at a prescribed flow rate: the inlet pressure is a constant unknown.

    -Δu + ∇p = 0,  ∇·u = 0   on [0, L] x [0, H]
    u = 0 on the walls, u_y = 0 on the inlet, a do-nothing outlet,
    inlet traction -P n with P unknown, and  ∫_inlet u_x dy = Q.

The pressure needed to push Q through the channel is not given -- it is solved for. It is ONE number,
so it is a constant unknown: `d.unknown.scalar(constant=True)`. Its trial multiplies the inlet traction
and its test function, which is one everywhere, turns a weak term on the inlet into the flow-rate
equation. Closed form (Poiseuille, unit viscosity): P = 12 Q L / H³.
"""

import jax
import numpy as np

import jno

jax.config.update("jax_enable_x64", True)  # the assembler builds in float64

L, H, Q = 2.0, 1.0, 0.3

d = jno.shape.rect(0.0, 0.0, L, H).structured(n=6).domain()
d.tag("walls", lambda x, y: (y < 1e-9) | (y > H - 1e-9))
x, y = d.variable("interior", split=True)[:2]
xi, yi = d.variable("left", split=True)[:2]  # the inlet
xw, yw = d.variable("walls", split=True)[:2]
print(jno.info(d))

u = d.unknown.vector(2, order=2)  # Taylor-Hood velocity
p = d.unknown.scalar(name="p")  # and pressure
P = d.unknown.scalar(constant=True, name="P_in")  # the inlet pressure: one value
v, q, W = u.test(), p.test(), P.test()

grad = lambda w: jno.np.jacobian(w, [x, y])  # noqa: E731
div = lambda w: jno.np.trace(grad(w))  # noqa: E731
inner = jno.np.inner
pi, qi = p.bind(x=x, y=y), q.bind(x=x, y=y)
u_in, v_in = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)

fem = jno.fem(
    [
        inner(grad(u), grad(v), n_contract=2) - pi * div(v),  # momentum
        -qi * div(u),  # continuity
        P * (-v_in[0]),  # inlet traction -P n, with n = (-1, 0)
        (u_in[0] - Q / H) * W,  # the flow rate through the inlet: an integral row
        u(xw, yw) - 0.0,  # no slip
        u(xi, yi)[1] - 0.0,  # inflow normal to the inlet
    ]
)
print(jno.info(fem))
sol = fem.solve(linear=jno.solve.lu())  # a bordered saddle system: solve it directly

P_in = float(np.asarray(sol)[fem.blocks[fem.block_index(P)]][0])
exact = 12 * Q * L / H**3
print(f"\ninlet pressure P = {P_in:.12f}   Poiseuille 12 Q L / H^3 = {exact:.12f}")
assert abs(P_in - exact) < 1e-10
# --8<-- [end:code]
