"""
03 - Nonlinear Allen-Cahn: strong PINN + weak FEAX-time + Diffrax + FEAX pipeline

Problem
-------
    u_t - kappa Δu + alpha (u^3 - u) = s(x,y,t)

Manufactured solution
---------------------
    moving tanh front with zero boundary envelope

Showcases
---------
- nonlinear transient weak form
- FeaxTimeBlock nonlinear residual/Jacobian route
- block.as_diffrax()
- block.as_feax_pipeline()
- strong PINN comparison
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


kappa = 0.015
alpha = 2.0
beta = 12.0
anis = 0.75
T_END = 0.20
N_T = 21

TEST_MODE = os.getenv("JNO_TUTORIAL_TEST_MODE", "").lower() in {"1", "true", "yes"}


def pick(default, test):
    return test if TEST_MODE else default


def _front_sym(t):
    xc = 0.35 + 0.10 * jnn.sin(2.0 * jnn.pi * t)
    yc = 0.55 - 0.08 * jnn.cos(2.0 * jnn.pi * t)
    r = 0.18 + 0.03 * jnn.sin(jnn.pi * t)
    xct = 0.20 * jnn.pi * jnn.cos(2.0 * jnn.pi * t)
    yct = 0.16 * jnn.pi * jnn.sin(2.0 * jnn.pi * t)
    rt = 0.03 * jnn.pi * jnn.cos(jnn.pi * t)
    return xc, yc, r, xct, yct, rt


def _front_jax(t):
    xc = 0.35 + 0.10 * jnp.sin(2.0 * jnp.pi * t)
    yc = 0.55 - 0.08 * jnp.cos(2.0 * jnp.pi * t)
    r = 0.18 + 0.03 * jnp.sin(jnp.pi * t)
    xct = 0.20 * jnp.pi * jnp.cos(2.0 * jnp.pi * t)
    yct = 0.16 * jnp.pi * jnp.sin(2.0 * jnp.pi * t)
    rt = 0.03 * jnp.pi * jnp.cos(jnp.pi * t)
    return xc, yc, r, xct, yct, rt


def exact_u_sym(x, y, t):
    xc, yc, r, _, _, _ = _front_sym(t)
    E = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    P = (x - xc) ** 2 + anis * (y - yc) ** 2 - r**2
    return E * jnn.tanh(beta * P)


def exact_u_jax(x, y, t):
    xc, yc, r, _, _, _ = _front_jax(t)
    E = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    P = (x - xc) ** 2 + anis * (y - yc) ** 2 - r**2
    return E * jnp.tanh(beta * P)


def source_term_sym(x, y, t):
    xc, yc, r, xct, yct, rt = _front_sym(t)

    E = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    Ex = 16.0 * (1.0 - 2.0 * x) * y * (1.0 - y)
    Ey = 16.0 * x * (1.0 - x) * (1.0 - 2.0 * y)
    Exx = -32.0 * y * (1.0 - y)
    Eyy = -32.0 * x * (1.0 - x)

    Px = 2.0 * (x - xc)
    Py = 2.0 * anis * (y - yc)
    Pxx = 2.0
    Pyy = 2.0 * anis
    Pt = -2.0 * (x - xc) * xct - 2.0 * anis * (y - yc) * yct - 2.0 * r * rt

    P = (x - xc) ** 2 + anis * (y - yc) ** 2 - r**2
    Z = beta * P
    T = jnn.tanh(Z)
    S = 1.0 - T**2

    Tx = beta * S * Px
    Ty = beta * S * Py
    Tt = beta * S * Pt

    Txx = beta * S * Pxx - 2.0 * beta**2 * T * S * Px**2
    Tyy = beta * S * Pyy - 2.0 * beta**2 * T * S * Py**2

    u = E * T
    ut = E * Tt
    uxx = Exx * T + 2.0 * Ex * Tx + E * Txx
    uyy = Eyy * T + 2.0 * Ey * Ty + E * Tyy

    return ut - kappa * (uxx + uyy) + alpha * (u * u * u - u)


# ---------------------------------------------------------------------
# Strong PINN
# ---------------------------------------------------------------------

def train_pinn_reference():
    domain = jno.domain(
        constructor=jno.domain.rect(mesh_size=pick(0.18, 0.25)),
        time=(0.0, T_END, pick(N_T, 7)),
        compute_mesh_connectivity=False,
    )

    x, y, t = domain.variable("interior", split=True)
    x0, y0, t0 = domain.variable("initial", split=True)

    net = jno.nn.wrap(
        foundax.mlp(
            3,
            hidden_dims=pick(64, 16),
            num_layers=pick(5, 2),
            activation=jax.nn.tanh,
            key=jax.random.PRNGKey(0),
        )
    )

    net.optimizer(
        optax.adam(1),
        lr=lrs.warmup_cosine(pick(1500, 20), 1, 1e-3, 1e-5),
    )

    envelope = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    envelope0 = 16.0 * x0 * (1.0 - x0) * y0 * (1.0 - y0)

    u_pinn = net(x, y, t) * envelope
    u0_pinn = net(x0, y0, t0) * envelope0

    pde = (
        jno.np.grad(u_pinn, t)
        - kappa * jno.np.laplacian(u_pinn, [x, y])
        + alpha * (u_pinn * u_pinn * u_pinn - u_pinn)
        - source_term_sym(x, y, t)
    )
    ini = u0_pinn - exact_u_sym(x0, y0, t0)

    print("\n" + "=" * 70)
    print("Training nonlinear Allen-Cahn PINN reference")
    print("=" * 70)

    crux = jno.core([pde.mse, ini.mse], domain)
    crux.solve(pick(1500, 20))

    return crux, net


# ---------------------------------------------------------------------
# Weak FEAX-time nonlinear solve
# ---------------------------------------------------------------------

def run_case(mesh_size=0.12, diffrax_dt0=2e-3, feax_dt=1e-2):
    crux, net = train_pinn_reference()

    domain = jno.domain(
        constructor=jno.domain.rect(mesh_size=pick(mesh_size, 0.25)),
        time=(0.0, T_END, pick(N_T, 7)),
        compute_mesh_connectivity=False,
    )

    domain.init_fem(
        element_type="TRI3",
        quad_degree=3,
        bcs=[
            domain.dirichlet(["left", "right", "bottom", "top"], 0.0),
        ],
        fem_solver=True,
    )

    u, phi = domain.fem_symbols()
    xg, yg, tg = domain.variable("fem_gauss", split=True)

    u_t = jno.np.grad(u, tg)
    u_x = jno.np.grad(u, xg)
    u_y = jno.np.grad(u, yg)

    phi_x = jno.np.grad(phi, xg)
    phi_y = jno.np.grad(phi, yg)

    weak = (
        u_t * phi
        + kappa * (u_x * phi_x + u_y * phi_y)
        + alpha * (u * u * u - u) * phi
        - source_term_sym(xg, yg, tg) * phi
    )

    coords = np.asarray(domain.mesh.points)[:, :2]
    x_nodes = jnp.asarray(coords[:, 0:1], dtype=jnp.float32)
    y_nodes = jnp.asarray(coords[:, 1:2], dtype=jnp.float32)
    u0_nodes = exact_u_jax(x_nodes, y_nodes, 0.0).reshape(-1)

    t0_asm = time.perf_counter()

    block = weak.assemble(
        target="feax_time",
        state0=u0_nodes,
        initial_conditions={"u(x,y,0)": "moving tanh front"},
        mode="implicit",
    )

    t1_asm = time.perf_counter()

    print("\nReturned weak transient object:", type(block).__name__)
    print("backend      :", block.backend)
    print("time_order   :", block.time_order)
    print("mode         :", block.mode)
    print("is_nonlinear :", block.is_nonlinear())
    print("metadata     :", block.metadata)

    assert block.is_nonlinear()

    # Diffrax route
    dblock = block.as_diffrax()

    save_ts = jnp.linspace(0.0, T_END, pick(9, 5))
    sol = diffrax.diffeqsolve(
        dblock.term,
        diffrax.Tsit5(),
        t0=dblock.t0,
        t1=dblock.t1,
        dt0=diffrax_dt0,
        y0=dblock.state0,
        saveat=diffrax.SaveAt(ts=save_ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        max_steps=20000,
    )

    u_diffrax_final = jnp.asarray(sol.ys[-1]).reshape(-1)

    # FEAX pipeline route
    monitor_xy = np.array([0.5, 0.5], dtype=np.float32)
    monitor_idx = int(np.argmin(np.sum((coords - monitor_xy[None, :]) ** 2, axis=1)))

    pblock = block.as_feax_pipeline(
        scheme="backward_euler",
        monitor_index=monitor_idx,
        newton_tol=1e-5,
        newton_maxiter=6,
        newton_damping=1.0,
        compile_step=True,
    )

    time_cfg = pblock.make_time_config(
        dt=feax_dt,
        print_every=pick(5, 1),
        save_every=10**9,
    )

    feax_result = feax_run(
        pblock.pipeline,
        pblock.mesh,
        time_cfg,
    )

    u_feax_final = jnp.asarray(feax_result.final_state).reshape(-1)

    # Exact / PINN comparison
    t_nodes = jnp.full_like(x_nodes, T_END)
    u_exact = exact_u_jax(x_nodes, y_nodes, t_nodes).reshape(-1)

    u_pinn = crux.eval(
        net(x_nodes, y_nodes, t_nodes)
        * 16.0
        * x_nodes
        * (1.0 - x_nodes)
        * y_nodes
        * (1.0 - y_nodes),
        domain=None,
    )
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