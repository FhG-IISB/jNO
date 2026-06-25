"""Transient FEM *construction* for non-homogeneous Dirichlet and source/forcing.

`jno.fem` is matrices-only: for a transient weak form it assembles the faithful
semidiscrete operators — mass ``M``, spatial operator ``A``, load ``c`` (``affine_bias``),
and time-dependent forcing ``f(t)`` (``forcing_vector_fn``) — and the *user* marches them.
These tests assert the **assembly is faithful** (structure of ``M``/``c``/``f``) and confirm
it with a throwaway few-line backward-Euler ``(M + dt·A) w_next = M·w + dt·(c + f(t_next))``;
that loop is verification, not a jno feature.

For non-homogeneous Dirichlet and forced problems the assembly zeros ``M``'s Dirichlet rows
(a constrained DOF carries no time derivative) and exposes the load ``c`` and forcing ``f(t)``,
so source-driven and non-homogeneous problems march faithfully.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64, so these tests opt into x64 per-test. The session default is
    x64-off (see tests/conftest.py); save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _march(fem):
    """Canonical backward-Euler over the assembled block: the only correct stepper for a
    non-homogeneous / forced transient (the old ``(M+dt·A)w=M·w`` shortcut drops c and f)."""
    M, A = _dense(fem.M), _dense(fem.operator.A)
    c = np.asarray(fem.operator.affine_bias).reshape(-1)
    f = fem.operator.forcing_vector_fn
    w = np.asarray(fem.state0).copy()
    dt, t = float(fem.dt), float(fem.t0)
    for _ in range(round((fem.t1 - fem.t0) / dt)):
        t += dt
        rhs = M @ w + dt * c
        if f is not None:
            rhs = rhs + dt * np.asarray(f(t)).reshape(-1)
        w = np.linalg.solve(M + dt * A, rhs)
    return w


def test_transient_nonhomog_dirichlet_single_field():
    # Heat u_t = lap u with u=1 held on the boundary, IC 0 -> relaxes to u == 1.
    # (Previously silently wrong: the mass kept identity Dirichlet rows + the stepper
    # dropped the load, so the boundary never moved off the IC.)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.5, 51))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 1.0, u(ci[0], ci[1]) - 0.0])
    assert fem.is_transient and fem.is_linear
    # construction check: the mass carries NO time derivative on Dirichlet DOFs.
    # Dirichlet DOFs = the boundary nodes (scalar P1: one DOF per node).
    pts = np.asarray(fem.field_points[0])
    rows = np.where(np.asarray(jax.vmap(d._make_tag_location_fn("boundary"))(jax.numpy.asarray(pts))).reshape(-1))[0]
    assert rows.size > 0 and np.allclose(_dense(fem.M)[rows], 0.0)
    # and the load c carries g=1 on those rows
    assert np.allclose(np.asarray(fem.operator.affine_bias).reshape(-1)[rows], 1.0)
    w = _march(fem)
    assert np.abs(w - 1.0).max() < 5e-3  # u -> 1 everywhere


def test_coupled_transient_nonhomog_dirichlet():
    # Coupled diffusion, u=1 and p=2 held on the boundary -> u==1, p==2.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.5, 51))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pi, qi = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x + ui.y * vi.y,
            pi.t * qi + pi.x * qi.x + pi.y * qi.y,
            u(xb, yb) - 1.0,
            p(xb, yb) - 2.0,
            u(ci[0], ci[1]) - 0.0,
            p(ci[0], ci[1]) - 0.0,
        ]
    )
    n = int(np.asarray(d.mesh.points).shape[0])
    assert fem.is_transient and fem.is_linear and fem.dofs == 2 * n
    w = _march(fem)
    assert np.abs(w[:n] - 1.0).max() < 5e-3 and np.abs(w[n:] - 2.0).max() < 5e-3


def test_coupled_transient_source_recovers():
    # Coupled source: zero-flux (natural Neumann) so the fields stay spatially uniform and
    # the system reduces to the ODEs u_t=-u+2 (constant source), p_t=-p+u (coupling) ->
    # analytic u=2(1-e^{-t}), p=2-2 e^{-t}(1+t), mesh-independent. The +2 source is a
    # standalone load term (no trial), exercising the block forcing_vector_fn path.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 2.0, 201))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pi, qi = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui * vi - 2.0 * vi,  # u_t = -u + 2 (standalone source)
            pi.t * qi + pi * qi - u.bind(x=xi, y=yi, t=ti) * qi,  # p_t = -p + u (coupling)
            u(ci[0], ci[1]) - 0.0,
            p(ci[0], ci[1]) - 0.0,
        ]
    )
    n = int(np.asarray(d.mesh.points).shape[0])
    # The +2 constant load is time-independent, so it lands in the (constant) affine bias, not the
    # time-varying forcing_vector_fn (which carries only the per-step increment; a time-dependent
    # source is exercised by test_coupled_transient_time_dependent_source). It is u-block only.
    c0 = np.asarray(fem.operator.affine_bias).reshape(-1)
    assert np.linalg.norm(c0[:n]) > 0 and np.allclose(c0[n:], 0.0)
    w = _march(fem)
    u_ex = 2.0 * (1.0 - np.exp(-fem.t1))
    p_ex = 2.0 - 2.0 * np.exp(-fem.t1) * (1.0 + fem.t1)
    assert abs(w[:n].mean() - u_ex) / u_ex < 1e-2 and w[:n].std() < 1e-8
    assert abs(w[n:].mean() - p_ex) / p_ex < 1e-2 and w[n:].std() < 1e-8


def test_coupled_transient_time_dependent_source():
    # Time-dependent coupled source: u_t = -u + e^{-t} (resonant) -> u = t e^{-t};
    # p_t = -p + u -> p = (t^2/2) e^{-t}. Spatially uniform (zero-flux), mesh-independent.
    # Exercises a temporal coefficient inside the block kernel (forcing evaluated at run-time t).
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 1.0, 101))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pi, qi = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui * vi - jno.np.exp(-1.0 * ti) * vi,  # u_t = -u + e^{-t}
            pi.t * qi + pi * qi - u.bind(x=xi, y=yi, t=ti) * qi,  # p_t = -p + u
            u(ci[0], ci[1]) - 0.0,
            p(ci[0], ci[1]) - 0.0,
        ]
    )
    n = int(np.asarray(d.mesh.points).shape[0])
    f = fem.operator.forcing_vector_fn
    assert f is not None and not np.allclose(np.asarray(f(0.1)), np.asarray(f(0.9)))  # genuinely time-varying
    w = _march(fem)
    u_ex = fem.t1 * np.exp(-fem.t1)
    p_ex = (fem.t1**2 / 2.0) * np.exp(-fem.t1)
    assert abs(w[:n].mean() - u_ex) / u_ex < 2e-2 and w[:n].std() < 1e-8
    assert abs(w[n:].mean() - p_ex) / p_ex < 2e-2 and w[n:].std() < 1e-8


def test_transient_stokes_dae_recovers():
    # Transient (unsteady) Stokes: pressure has no p_t -> algebraic (DAE) field, which gets a
    # ZERO mass block; the single-node pressure pin (domain.point_region) makes the
    # (M + dt A) saddle solvable. Manufactured u=(x,-y) (steady, so a time-CONSTANT velocity
    # Dirichlet -> no time-varying BC needed) and p=e^{-t} x (decaying), driven by the
    # t-dependent body force f=e^{-t}(1,0). P2 velocity / P1 pressure -> exact recovery.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.3, 31))
    d.point_region("ppin", (0.0, 0.0))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
    p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure (no p_t -> algebraic)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    ci = d.variable("initial", split=True)
    gu, gv = jno.np.grad(u, [xi, yi]), jno.np.grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    mom = (
        jno.np.inner(ui.t, vi, n_contract=1)
        + jno.np.inner(gu, gv, n_contract=2)
        - pp * jno.np.trace(gv)
        - jno.np.exp(-1.0 * ti) * vi[0]  # body force f = e^{-t} (1, 0)
    )
    cont = -qq * jno.np.trace(gu)
    fem = jno.fem(
        [
            mom,
            cont,
            u(xb, yb)[0] - xb,
            u(xb, yb)[1] - (-1.0 * yb),
            p(xpn, ypn) - 0.0,  # pressure pin (the manufactured p = e^{-t} x is 0 at the origin)
            u(ci[0], ci[1])[0] - ci[0],
            u(ci[0], ci[1])[1] - (-1.0 * ci[1]),
        ]
    )
    assert fem.is_transient and fem.is_linear
    off = fem.offsets  # [0, n_vel, n_total]
    nu = off[1] - off[0]
    M = _dense(fem.M)
    assert np.allclose(M[nu:, nu:], 0.0)  # DAE: pressure carries a zero mass block
    assert np.abs(M[:nu, :nu]).max() > 0.0  # velocity mass present
    w = _march(fem)
    pts_v = np.asarray(fem.field_points[0])
    pts_p = np.asarray(fem.field_points[1])
    uu = w[off[0] : off[1]].reshape(-1, 2)
    pr = w[off[1] :]
    u_ex = np.stack([pts_v[:, 0], -pts_v[:, 1]], axis=-1)
    p_ex = np.exp(-fem.t1) * pts_p[:, 0]
    assert np.linalg.norm(uu - u_ex) / np.linalg.norm(u_ex) < 1e-9
    assert np.linalg.norm(pr - p_ex) / np.linalg.norm(p_ex) < 1e-8


def test_coupled_transient_time_varying_dirichlet():
    # General time-varying Dirichlet g(x,t) in a coupled block: u carries u=x+t on the
    # boundary (time-varying) with a constant source 1 (u_t = lap u + 1 -> u=x+t); p carries
    # the constant Dirichlet p=y (steady). Both linear in space AND time -> P1-exact and
    # backward-Euler-exact, so recovery is ~machine precision. jno builds a JIT-traceable
    # forcing_vector_fn(t) (the Dirichlet lift via replace_vals + the parametric residual);
    # the user marches the canonical stepper.
    import jax

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.4, 41))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pi, qi = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x + ui.y * vi.y - 1.0 * vi,  # u_t = lap u + 1 -> u=x+t
            pi.t * qi + pi.x * qi.x + pi.y * qi.y,  # p_t = lap p -> p=y (steady)
            u(xb, yb) - (xb + tb),  # time-varying Dirichlet u = x + t
            p(xb, yb) - yb,  # constant Dirichlet p = y
            u(ci[0], ci[1]) - ci[0],  # IC u = x
            p(ci[0], ci[1]) - ci[1],  # IC p = y
        ]
    )
    n = int(np.asarray(d.mesh.points).shape[0])
    assert fem.is_transient and fem.is_linear and fem.dofs == 2 * n
    f = fem.operator.forcing_vector_fn
    assert f is not None
    # the time-varying Dirichlet load is genuinely t-varying and JAX-traceable (replace_vals path)
    jax.jit(lambda t: f(t))(0.1)
    assert not np.allclose(np.asarray(f(0.1)), np.asarray(f(0.3)))
    w = _march(fem)
    cc = np.asarray(d.mesh.points)[:, :2]
    u_ex, p_ex = cc[:, 0] + fem.t1, cc[:, 1]
    assert np.linalg.norm(w[:n] - u_ex) / np.linalg.norm(u_ex) < 1e-9  # u = x + t
    assert np.linalg.norm(w[n:] - p_ex) / np.linalg.norm(p_ex) < 1e-9  # p = y
