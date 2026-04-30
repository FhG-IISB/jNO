"""
Strong-form reaction-only Allen-Cahn ODE to Diffrax.

Problem
-------
    u_t + alpha (u^3 - u) = s(t)

or equivalently:

    u_t = -alpha (u^3 - u) + s(t)

Purpose
-------
This example showcases the strong-form time route:

    symbolic jNO residual
        -> pde.assemble(target="diffrax")
        -> DiffraxBlock
        -> diffrax.diffeqsolve(...)

The assembled jNO DiffraxBlock is compared against a hand-written Diffrax RHS.
The problem is intentionally reaction-only, not reaction-diffusion.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import diffrax

import jno
import jno.numpy as jnn

jax.config.update("jax_enable_x64", True)


# ============================================================
# Output folder
# ============================================================

OUT_DIR = "priority2_reaction_ode"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Problem setup
# ============================================================

alpha = 2.0
u0 = 0.25

T_END = 4.0
N_SAVE = 401
DT0 = 1.0e-3


def forcing_sym(t):
    """Symbolic forcing used in the jNO traced residual."""
    return 0.50 * jnn.sin(2.0 * jnn.pi * t)


def forcing_num(t):
    """Plain JAX forcing used in the hand-written reference RHS."""
    return 0.50 * jnp.sin(2.0 * jnp.pi * t)


# ============================================================
# Domain
# ============================================================

domain = jno.domain( constructor=jno.domain.line(x_range=(0.0, 1.0), mesh_size=0.25),time=(0.0, T_END, 21), compute_mesh_connectivity=False,)

# For a 1D space-time domain this returns (x, t).
x, t = domain.variable("interior", split=True)
del x

# State placeholder for the strong-form ODE.
# This attaches a tensor tag to the domain context and returns a TensorTag.
u = domain.variable(
    "u_state",
    sample=np.asarray([[u0]], dtype=np.float64),
)


# ============================================================
# Strong-form ODE residual
#
#     u_t + alpha(u^3 - u) - s(t) = 0
#
# Therefore:
#
#     u_t = -alpha(u^3 - u) + s(t)
# ============================================================

pde = jno.np.grad(u, t) + alpha * (u * u * u - u) - forcing_sym(t)


# ============================================================
# Assemble symbolic strong form to DiffraxBlock
# ============================================================

t0_assemble = time.perf_counter()

block = pde.assemble(
    target="diffrax",
    state_expr=u,
    time_var=t,
    state0=jnp.asarray(u0, dtype=jnp.float64),
    t0=0.0,
    t1=T_END,
    dt0=DT0,
)

t1_assemble = time.perf_counter()


print("\n" + "=" * 72)
print("Reaction-only Allen-Cahn ODE: strong form -> Diffrax")
print("=" * 72)
print("Returned object :", type(block).__name__)
print("backend         :", block.backend)
print("form            :", block.form)
print("time_order      :", block.time_order)
print("t0, t1, dt0     :", block.t0, block.t1, block.dt0)
print("metadata        :", block.metadata)

if block.term is None or block.rhs is None:
    raise RuntimeError(
        "Strong-form Diffrax lowering did not produce a usable Diffrax term."
    )

if block.backend != "diffrax":
    raise RuntimeError(f"Expected backend='diffrax', got {block.backend!r}.")

if block.time_order != 1:
    raise RuntimeError(f"Expected first-order block, got time_order={block.time_order}.")


# ============================================================
# Extra RHS sanity check
# ============================================================


def rhs_ref(t, y, args):
    """Hand-written reference RHS."""
    del args
    return -alpha * (y**3 - y) + forcing_num(t)


probe_ts = jnp.asarray([0.0, 0.3, 1.0, 2.5], dtype=jnp.float64)
probe_us = jnp.asarray([0.25, 0.5, -0.2, 0.9], dtype=jnp.float64)

rhs_jno_vals = jnp.asarray(
    [block.rhs(tt, uu, None) for tt, uu in zip(probe_ts, probe_us)]
)
rhs_ref_vals = jnp.asarray(
    [rhs_ref(tt, uu, None) for tt, uu in zip(probe_ts, probe_us)]
)

rhs_max_abs_err = float(jnp.max(jnp.abs(rhs_jno_vals - rhs_ref_vals)))


# ============================================================
# Solve assembled jNO Diffrax block
# ============================================================

save_ts = jnp.linspace(0.0, T_END, N_SAVE)
saveat = diffrax.SaveAt(ts=save_ts)

solver = diffrax.Tsit5()
stepsize_controller = diffrax.PIDController(rtol=1.0e-7, atol=1.0e-9)

t0_jno = time.perf_counter()

sol_jno = diffrax.diffeqsolve(
    block.term,
    solver,
    t0=block.t0,
    t1=block.t1,
    dt0=block.dt0 if block.dt0 is not None else DT0,
    y0=block.state0,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=200000,
)

t1_jno = time.perf_counter()

u_jno = jnp.asarray(sol_jno.ys).reshape(-1)
t_hist = jnp.asarray(sol_jno.ts)


# ============================================================
# Reference solve with hand-written RHS
# ============================================================

t0_ref = time.perf_counter()

sol_ref = diffrax.diffeqsolve(
    diffrax.ODETerm(rhs_ref),
    solver,
    t0=0.0,
    t1=T_END,
    dt0=DT0,
    y0=jnp.asarray(u0, dtype=jnp.float64),
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=200000,
)

t1_ref = time.perf_counter()

u_ref = jnp.asarray(sol_ref.ys).reshape(-1)


# ============================================================
# Metrics
# ============================================================

abs_err = jnp.abs(u_jno - u_ref)

max_abs_err = float(jnp.max(abs_err))
rel_l2 = float(
    jnp.linalg.norm(u_jno - u_ref)
    / (jnp.linalg.norm(u_ref) + 1.0e-14)
)

final_abs_err = float(jnp.abs(u_jno[-1] - u_ref[-1]))

print("\n" + "=" * 72)
print("Comparison against hand-written reference RHS")
print("=" * 72)
print(f"assemble time        : {t1_assemble - t0_assemble:.6f} s")
print(f"jno Diffrax solve    : {t1_jno - t0_jno:.6f} s")
print(f"ref Diffrax solve    : {t1_ref - t0_ref:.6f} s")
print(f"RHS max abs error    : {rhs_max_abs_err:.6e}")
print(f"solution max abs     : {max_abs_err:.6e}")
print(f"solution rel L2      : {rel_l2:.6e}")
print(f"final abs error      : {final_abs_err:.6e}")
print(f"initial value        : {float(u_jno[0]):.10f}")
print(f"final value (jno)    : {float(u_jno[-1]):.10f}")
print(f"final value (ref)    : {float(u_ref[-1]):.10f}")


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

axes[0].plot(
    np.asarray(t_hist),
    np.asarray(u_ref),
    label="Reference hand-written RHS",
    linewidth=2,
)
axes[0].plot(
    np.asarray(t_hist),
    np.asarray(u_jno),
    "--",
    label="jNO strong form -> Diffrax",
    linewidth=2,
)
axes[0].set_ylabel("u(t)")
axes[0].set_title("Reaction-only Allen-Cahn ODE")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].semilogy(
    np.asarray(t_hist),
    np.asarray(abs_err) + 1.0e-18,
    label="|u_jno - u_ref|",
    linewidth=2,
)
axes[1].set_xlabel("t")
axes[1].set_ylabel("absolute error")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "reaction_only_ac_ode_compare.png")
plt.savefig(plot_path, dpi=180)
plt.close(fig)

print(f"\nSaved comparison plot to: {plot_path}")


# ============================================================
# Pass/fail guard
# ============================================================

if not np.isfinite(max_abs_err):
    raise RuntimeError("Non-finite solution error encountered.")

if not np.isfinite(rhs_max_abs_err):
    raise RuntimeError("Non-finite RHS error encountered.")

if rhs_max_abs_err > 1.0e-10:
    raise RuntimeError(
        f"RHS mismatch too large: {rhs_max_abs_err:.3e}. "
        "Check strong-form lowering/sign convention."
    )

if max_abs_err > 1.0e-8:
    raise RuntimeError(
        f"Solution mismatch too large: {max_abs_err:.3e}. "
        "Check strong-form lowering or Diffrax solve settings."
    )

print("\nPASS: strong-form Diffrax lowering matches the hand-written reference.")