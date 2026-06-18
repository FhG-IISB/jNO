"""Transient FEM *construction* for non-homogeneous Dirichlet and source/forcing.

`jno.fem` is matrices-only: for a transient weak form it assembles the faithful
semidiscrete operators — mass ``M``, spatial operator ``A``, load ``c`` (``affine_bias``),
and time-dependent forcing ``f(t)`` (``forcing_vector_fn``) — and the *user* marches them.
These tests assert the **assembly is faithful** (structure of ``M``/``c``/``f``) and confirm
it with a throwaway few-line backward-Euler ``(M + dt·A) w_next = M·w + dt·(c + f(t_next))``;
that loop is verification, not a jno feature.

Earlier the feax (2D/3D) transient path left identity rows on ``M`` and never exposed the
load, so only homogeneous, source-free problems were faithful. The fix zeros ``M``'s
Dirichlet rows (a constrained DOF carries no time derivative) and exposes ``c``/``f``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import box  # noqa: E402


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
    # construction check: the mass carries NO time derivative on Dirichlet DOFs
    rows = np.asarray(fem.domain._feax_bc.bc_rows).reshape(-1)
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
    assert fem.operator.forcing_vector_fn is not None
    f0 = np.asarray(fem.operator.forcing_vector_fn(0.0)).reshape(-1)
    assert np.linalg.norm(f0[:n]) > 0 and np.allclose(f0[n:], 0.0)  # source lands in u-block only
    w = _march(fem)
    u_ex = 2.0 * (1.0 - np.exp(-fem.t1))
    p_ex = 2.0 - 2.0 * np.exp(-fem.t1) * (1.0 + fem.t1)
    assert abs(w[:n].mean() - u_ex) / u_ex < 1e-2 and w[:n].std() < 1e-8
    assert abs(w[n:].mean() - p_ex) / p_ex < 1e-2 and w[n:].std() < 1e-8
