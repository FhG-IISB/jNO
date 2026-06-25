"""Tests for SemidiscreteTimeBlock.step — the one-step advance primitive.

block.step is the brick the default integrator now scans over and the basis for operator
splitting. Covers: it reproduces the analytic heat decay (linear theta-step), it actually solves
the backward-Euler root (nonlinear Newton path), and it advances a periodic block in reduced space.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nonlinear=False):
    """Transient heat on the unit square, IC sin(pi x) sin(pi y), homogeneous Dirichlet."""
    dom = jno.domain(
        constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=12, ny=12),
        time=(0.0, 0.02, 11),
    )
    dom.tag("bnd", lambda x, y: (x < 1e-6) | (x > 1 - 1e-6) | (y < 1e-6) | (y > 1 - 1e-6))
    u, phi = dom.fem_symbols()
    xi, yi, ti = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bnd", split=True)
    ci = dom.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    weak = ui.t * vi + 1.0 * (ui.x * vi.x + ui.y * vi.y)
    if nonlinear:
        weak = weak + ui * ui * ui * vi  # cubic reaction -> nonlinear transient block
    fem = jno.fem([weak, u(xb, yb) - 0.0, ic])
    return fem


def test_block_step_linear_heat_decay(_x64):
    """Manually stepping with block.step reproduces the analytic decay exp(-2 nu pi^2 t)."""
    fem = _heat()
    block = fem.operator
    u = jnp.asarray(block.state0)
    t, dt = float(block.t0), float(block.dt)
    for _ in range(round((block.t1 - block.t0) / dt)):
        u = block.step(u, t, dt)
        t += dt
    pts = np.asarray(fem.points)
    analytic = np.exp(-2.0 * np.pi**2 * block.t1) * np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
    rel = float(np.linalg.norm(analytic - np.asarray(u)) / np.linalg.norm(analytic))
    assert rel < 5e-2


def test_block_step_matches_default_integrator(_x64):
    """A scan of block.step equals the default integrator's trajectory (one definition of stepping)."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    fem = _heat()
    block = fem.operator
    dt = float(block.dt)
    ys = np.asarray(_default_transient_integrate(block, {}, save_ts=jnp.asarray([block.t1])))
    u = jnp.asarray(block.state0)
    t = float(block.t0)
    for _ in range(round((block.t1 - block.t0) / dt)):
        u = block.step(u, t, dt)
        t += dt
    assert np.allclose(np.asarray(u), ys[-1], atol=1e-8, rtol=1e-6)


def test_block_step_nonlinear_solves_backward_euler_root(_x64):
    """For a nonlinear block, block.step actually drives the backward-Euler residual G(u_next) to 0."""
    fem = _heat(nonlinear=True)
    block = fem.operator
    assert block.is_nonlinear()
    u0 = jnp.asarray(block.state0)
    t, dt = float(block.t0), float(block.dt)
    u1 = block.step(u0, t, dt)
    M = block.mass(t + dt, {})
    G = (jnp.asarray(M.todense() if hasattr(M, "todense") else M) @ (u1 - u0)) / dt + jnp.asarray(
        block.residual(u1, t + dt, {})
    ).reshape(-1)
    assert float(jnp.linalg.norm(G)) < 1e-6 * (float(jnp.linalg.norm(u1)) + 1.0)


def test_block_step_periodic_reduced_space(_x64):
    """block.step advances a periodic block in its reduced DOF space; the mode decays correctly."""
    import jno.jnp_ops as jnn

    n = 12
    dom = jno.domain(
        constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=n, ny=n),
        time=(0.0, 0.02, 11),
        compute_mesh_connectivity=False,
    )
    for nm, pred in {
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > 1 - 1e-6,
        "bottom": lambda x, y: y < 1e-6,
        "top": lambda x, y: y > 1 - 1e-6,
    }.items():
        dom.tag(nm, pred)
    u, phi = dom.fem_symbols()
    xi, yi, ti = dom.variable("interior", split=True)
    ci = dom.variable("initial", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ic = jnn.sin(2 * np.pi * ci[0]) * jnn.sin(2 * np.pi * ci[1])
    fem = jno.fem(
        [
            ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y),
            u(xl, yl) - u(xr, yr),
            u(xb, yb) - u(xt, yt),
            u(ci[0], ci[1]) - ic,
        ]
    )
    block = fem.operator
    assert block.metadata.get("periodic") is True
    w = jnp.asarray(block.state0)  # reduced DOFs
    t, dt = float(block.t0), float(block.dt)
    w0_norm = float(jnp.linalg.norm(w))
    for _ in range(round((block.t1 - block.t0) / dt)):
        w = block.step(w, t, dt)
        t += dt
    # the 2pi-mode decays as exp(-8 pi^2 nu t); it must shrink but stay finite and nonzero
    decay = np.exp(-8.0 * np.pi**2 * 0.1 * block.t1)
    assert 0.0 < float(jnp.linalg.norm(w)) < w0_norm
    assert abs(float(jnp.linalg.norm(w)) / w0_norm - decay) < 0.1
