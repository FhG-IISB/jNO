"""
02 - 2D heat equation: strong PINN + weak FEAX-time + Diffrax + FEAX pipeline

Problem
-------
    u_t - nu Δu = 0

Exact solution
--------------
    u(x,y,t) = exp(-2 pi^2 nu t) sin(pi x) sin(pi y)

Showcases
---------
- strong-form PINN reference
- weak.assemble(target="feax_time")
- FeaxTimeBlock linear payload
- block.as_diffrax()
- block.as_feax_pipeline()
"""

import os
import time
import numpy as np

import jax
import jax.numpy as jnp
import optax
import diffrax
import foundax
from feax.solvers.time_solver import run as feax_run

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs

jax.config.update("jax_enable_x64", False)


PI = jnp.pi
nu = 0.10
T_END = 0.20
N_T = 21

TEST_MODE = os.getenv("JNO_TUTORIAL_TEST_MODE", "").lower() in {"1", "true", "yes"}


def pick(default, test):
    return test if TEST_MODE else default


def exact_u_sym(x, y, t):
    return (
        jnn.exp(-2.0 * jnn.pi**2 * nu * t)
        * jnn.sin(jnn.pi * x)
        * jnn.sin(jnn.pi * y)
    )


def exact_u_jax(x, y, t):
    return (
        jnp.exp(-2.0 * PI**2 * nu * t)
        * jnp.sin(PI * x)
        * jnp.sin(PI * y)
    )


# ---------------------------------------------------------------------
# Strong-form PINN
# ---------------------------------------------------------------------

def train_pinn_reference():
    domain = jno.domain(constructor=jno.domain.rect(mesh_size=pick(0.20, 0.30)), time=(0.0, T_END, pick(N_T, 5)), compute_mesh_connectivity=False,)

    x, y, t = domain.variable("interior", split=True)
    x0, y0, t0 = domain.variable("initial", split=True)

    net = jno.nn.wrap(foundax.mlp(
            3,
            hidden_dims=pick(48, 16),
            num_layers=pick(4, 2),
            activation=jax.nn.tanh,
            key=jax.random.PRNGKey(0),))

    net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(pick(1000, 10), 1, 1e-3, 1e-5),)

    u_pinn = net(x, y, t) * x * (1.0 - x) * y * (1.0 - y)
    u0_pinn = net(x0, y0, t0) * x0 * (1.0 - x0) * y0 * (1.0 - y0)

    pde = jno.np.grad(u_pinn, t) - nu * jno.np.laplacian(u_pinn, [x, y])
    ini = u0_pinn - jnn.sin(jnn.pi * x0) * jnn.sin(jnn.pi * y0)

    print("\n" + "=" * 70)
    print("Training strong PINN reference")
    print("=" * 70)

    crux = jno.core([pde.mse, ini.mse], domain)
    crux.solve(pick(1000, 10))

    u_pred, u_true = crux.eval([u_pinn, exact_u_sym(x, y, t)])
    rel = float(jnp.linalg.norm(u_pred - u_true) / (jnp.linalg.norm(u_true) + 1e-8))
    print(f"PINN domain relative L2 error: {rel:.6e}")

    return crux, net


# ---------------------------------------------------------------------
# Weak FEAX-time solve
# ---------------------------------------------------------------------

def run_case(mesh_size=0.12, diffrax_dt0=1e-4, feax_dt=1e-3):
    crux, net = train_pinn_reference()

    fem_domain = jno.domain(constructor=jno.domain.rect(mesh_size=pick(mesh_size, 0.25)),
        time=(0.0, T_END, pick(N_T, 7)),
        compute_mesh_connectivity=False,
    )

    fem_domain.init_fem(
        element_type="TRI3",
        quad_degree=3,
        bcs=[
            fem_domain.dirichlet(["left", "right", "bottom", "top"], 0.0),
        ],
        fem_solver=True,
    )

    u_h, phi = fem_domain.fem_symbols()
    xg, yg, tg = fem_domain.variable("fem_gauss", split=True)

    u_t = jno.np.grad(u_h, tg)
    u_x = jno.np.grad(u_h, xg)
    u_y = jno.np.grad(u_h, yg)

    phi_x = jno.np.grad(phi, xg)
    phi_y = jno.np.grad(phi, yg)

    weak = u_t * phi + nu * (u_x * phi_x + u_y * phi_y)

    coords = np.asarray(fem_domain.mesh.points)[:, :2]
    x_nodes = jnp.asarray(coords[:, 0:1], dtype=jnp.float32)
    y_nodes = jnp.asarray(coords[:, 1:2], dtype=jnp.float32)

    u0_nodes = exact_u_jax(x_nodes, y_nodes, 0.0).reshape(-1)

    t0_asm = time.perf_counter()

    block = weak.assemble(
        target="feax_time",
        linear=True,
        state0=u0_nodes,
        initial_conditions={"u(x,y,0)": "sin(pi x) sin(pi y)"},
        mode="implicit",
    )

    t1_asm = time.perf_counter()

    print("\nReturned weak transient object:", type(block).__name__)
    print("backend    :", block.backend)
    print("time_order :", block.time_order)
    print("mode       :", block.mode)
    print("is_linear  :", block.is_linear())
    print("metadata   :", block.metadata)

    # Diffrax route
    dblock = block.as_diffrax()

    save_ts = jnp.linspace(0.0, T_END, pick(41, 9))
    sol = diffrax.diffeqsolve(
        dblock.term,
        diffrax.Tsit5(),
        t0=dblock.t0,
        t1=dblock.t1,
        dt0=diffrax_dt0,
        y0=dblock.state0,
        saveat=diffrax.SaveAt(ts=save_ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        max_steps=200000,
    )

    u_diffrax_final = jnp.asarray(sol.ys[-1]).reshape(-1)

    # FEAX pipeline route
    center_xy = np.array([0.5, 0.5], dtype=np.float32)
    center_idx = int(np.argmin(np.sum((coords - center_xy[None, :]) ** 2, axis=1)))

    pblock = block.as_feax_pipeline(scheme="backward_euler", monitor_index=center_idx,  compile_step=True,)

    time_cfg = pblock.make_time_config( dt=feax_dt, print_every=pick(20, 1), save_every=10**9, )

    feax_result = feax_run(pblock.pipeline, pblock.mesh,time_cfg, )

    u_feax_final = jnp.asarray(feax_result.final_state).reshape(-1)

    # PINN / exact comparison at FEM nodes
    t_nodes = jnp.full_like(x_nodes, T_END)
    u_exact = exact_u_jax(x_nodes, y_nodes, t_nodes).reshape(-1)

    u_pinn = crux.eval( net(x_nodes, y_nodes, t_nodes) * x_nodes * (1.0 - x_nodes) * y_nodes * (1.0 - y_nodes), domain=None, )
    u_pinn = jnp.asarray(u_pinn).reshape(-1)

    rel_pinn = float(jnp.linalg.norm(u_pinn - u_exact) / (jnp.linalg.norm(u_exact) + 1e-8))
    rel_diffrax = float(jnp.linalg.norm(u_diffrax_final - u_exact) / (jnp.linalg.norm(u_exact) + 1e-8))
    rel_feax = float(jnp.linalg.norm(u_feax_final - u_exact) / (jnp.linalg.norm(u_exact) + 1e-8))

    print("\nFinal-time errors")
    print("-" * 70)
    print(f"PINN     relative L2 : {rel_pinn:.6e}")
    print(f"Diffrax  relative L2 : {rel_diffrax:.6e}")
    print(f"FEAX-BE  relative L2 : {rel_feax:.6e}")
    print(f"assemble time        : {t1_asm - t0_asm:.3f} s")

    return {
        "pinn_l2": rel_pinn,
        "diffrax_l2": rel_diffrax,
        "feax_l2": rel_feax,
    }


if __name__ == "__main__":
    results = run_case()
    print("\nSummary:", results)