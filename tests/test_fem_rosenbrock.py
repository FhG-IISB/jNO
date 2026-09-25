"""jno.solve.rosenbrock(): linearly implicit Rosenbrock-W schemes (Rang & Angermann 2005; Verwer et al. 1999).

Oracles: the ORDER against a fine SDIRK3 reference on the same mesh, L-stability, exactness where the discrete
solution is exact (a DAE boundary row u = x + t), and agreement with the Newton-based schemes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import box

import jno

sys.path.insert(0, str(Path(__file__).parent))
from test_fem_sdirk import _final, _heat  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("method, order", [("ros2", 2), ("ros34pw2", 3)])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_rosenbrock_has_its_order(method, order, nonlinear):
    ref = _final(_heat(320, nonlinear=nonlinear), time=jno.solve.sdirk(3))
    errs = [float(np.linalg.norm(_final(_heat(n, nonlinear=nonlinear), time=jno.solve.rosenbrock(method)) - ref)) for n in (5, 10, 20)]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert rates[-1] > order - 0.35, f"{method}: rates {rates} from errors {errs}"


def test_ros34pw2_is_l_stable():
    cn = np.asarray(_heat(8, T=0.5, h=0.14, ic_one=True).solve(time=jno.solve.theta(0.5)).fn())
    ro = np.asarray(_heat(8, T=0.5, h=0.14, ic_one=True).solve(time=jno.solve.rosenbrock()).fn())
    assert cn.min() < -0.5
    assert abs(ro.min()) < 0.1 * abs(cn.min()) and np.abs(ro[-1]).max() < 0.2 * np.abs(cn[-1]).max()


def test_a_time_varying_dirichlet_row_is_exact():
    """u = x + t: the boundary rows are algebraic (index-1 DAE); ros34pw2 is consistent for them."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.2, 5))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - (xb + tb), u(ci[0], ci[1]) - ci[0]])
    traj = np.asarray(fem.solve(time=jno.solve.rosenbrock()).fn())
    pts = np.asarray(d.mesh.points)[:, 0]
    t = np.linspace(0.0, 0.2, traj.shape[0])
    np.testing.assert_allclose(traj, pts[None, :] + t[:, None], atol=1e-9)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_solver_slots_see_the_stage_matrix(nonlinear):
    base = _final(_heat(10, nonlinear=nonlinear), time=jno.solve.rosenbrock())
    for kw in (
        dict(linear=jno.solve.cg(tol=1e-12), precond=jno.precond.jacobi()),
        dict(linear=jno.solve.lu(backend="host")),
    ):
        got = _final(_heat(10, nonlinear=nonlinear), time=jno.solve.rosenbrock(), **kw)
        np.testing.assert_allclose(got, base, rtol=1e-7, atol=1e-10)


def test_adaptive_and_gradient():
    ref = _final(_heat(320), time=jno.solve.sdirk(3))
    ad = _final(_heat(10), time=jno.solve.rosenbrock().adaptive(rtol=1e-7, atol=1e-9))
    assert np.linalg.norm(ad - ref) < 1e-5 * np.linalg.norm(ref)

    fem = _heat(6, T=0.05, nonlinear=True)
    blk = fem.operator
    save = jnp.linspace(blk.t0, blk.t1, 3)
    sch = jno.solve.rosenbrock()

    def loss(s):
        import dataclasses

        b = dataclasses.replace(blk, state0=s * blk.state0)
        return jnp.sum(sch.integrate(b, {}, save) ** 2)

    g = float(jax.grad(loss)(1.3))
    fd = float((loss(1.3 + 1e-6) - loss(1.3 - 1e-6)) / 2e-6)
    np.testing.assert_allclose(g, fd, rtol=1e-6)


def test_refusals():
    with pytest.raises(ValueError, match="method"):
        jno.solve.rosenbrock("ros5")
    with pytest.raises(ValueError, match="linearly implicit"):
        _heat(5, nonlinear=True).solve(time=jno.solve.rosenbrock(), nonlinear=jno.solve.newton()).fn()
    # state-dependent mass c(u) u_t
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 0.1, 4))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([(1 + ui * ui) * ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 0.5])
    with pytest.raises(NotImplementedError, match="state-dependent mass"):
        fem.solve(time=jno.solve.rosenbrock()).fn()
