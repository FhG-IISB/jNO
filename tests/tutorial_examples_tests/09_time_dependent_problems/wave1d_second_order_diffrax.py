"""
04 - 1D wave equation: second-order strong form to Diffrax by manual reduction

Problem
-------
    u_tt - c^2 u_xx = 0

For the solver showcase we use one modal coordinate:

    q'' + omega^2 q = 0

Exact solution
--------------
    q(t) = q0 cos(omega t) + v0/omega sin(omega t)

Showcases
---------
- second-order symbolic residual
- expr.assemble(target="diffrax", second_order="manual")
- manual first-order reduction
- DiffraxBlock with time_order=2
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import jno

jax.config.update("jax_enable_x64", True)


omega = 2.0 * jnp.pi
q0 = 1.0
v0 = 0.0
T_END = 1.0
DT0 = 1e-3


def exact_q(t):
    return q0 * jnp.cos(omega * t) + (v0 / omega) * jnp.sin(omega * t)


# ---------------------------------------------------------------------
# jNO time domain and symbolic second-order residual
# ---------------------------------------------------------------------

domain = jno.domain(
    constructor=jno.domain.line(x_range=(0.0, 1.0), mesh_size=0.25),
    time=(0.0, T_END, 21),
    compute_mesh_connectivity=False,
)

x, t = domain.variable("interior", split=True)

q = domain.variable(
    "q_state",
    sample=np.asarray([[q0]], dtype=np.float64),
)

# Symbolic second-order oscillator residual:
#     q_tt + omega^2 q = 0
q_t = jno.np.grad(q, t)
q_tt = jno.np.grad(q_t, t)
pde = q_tt + (omega**2) * q


# ---------------------------------------------------------------------
# Manual first-order reduced system
# ---------------------------------------------------------------------


def rhs_manual(t, y, args):
    q_val, v_val = y[0], y[1]
    return jnp.asarray(
        [
            v_val,
            -(omega**2) * q_val,
        ],
        dtype=y.dtype,
    )


state0 = jnp.asarray([q0, v0], dtype=jnp.float64)

block = pde.assemble(
    target="diffrax",
    state_expr=q,
    second_order="manual",
    rhs=rhs_manual,
    state0=state0,
    state_names=("q", "v"),
    t0=0.0,
    t1=T_END,
    dt0=DT0,
)

print("\n" + "=" * 70)
print("1D wave/modal oscillator: second-order strong-form Diffrax")
print("=" * 70)
print("backend          :", block.backend)
print("form             :", block.form)
print("time_order       :", block.time_order)
print("rewritten_system :", block.rewritten_system)
print("metadata         :", block.metadata)

assert block.time_order == 2
assert block.term is not None


# ---------------------------------------------------------------------
# Diffrax solve
# ---------------------------------------------------------------------

save_ts = jnp.linspace(0.0, T_END, 201)

sol = diffrax.diffeqsolve(
    block.term,
    diffrax.Tsit5(),
    t0=block.t0,
    t1=block.t1,
    dt0=block.dt0,
    y0=block.state0,
    saveat=diffrax.SaveAt(ts=save_ts),
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
    max_steps=200000,
)

q_num = jnp.asarray(sol.ys[:, 0]).reshape(-1)
q_ref = exact_q(save_ts).reshape(-1)

rel_l2 = float(jnp.linalg.norm(q_num - q_ref) / (jnp.linalg.norm(q_ref) + 1e-14))
max_abs = float(jnp.max(jnp.abs(q_num - q_ref)))

print("\nComparison")
print("-" * 70)
print(f"relative L2 error : {rel_l2:.6e}")
print(f"max abs error     : {max_abs:.6e}")

assert rel_l2 < 1e-6
