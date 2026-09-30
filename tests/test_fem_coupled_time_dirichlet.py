"""A time-varying Dirichlet value ``g(x, t)`` on a NONLINEAR transient -- coupled (multi-field) or single field.

The Dirichlet value is a formula in the boundary variable's ``t`` (``u(xb, yb) - g(xb, yb, tb)``), so nothing
new is passed anywhere. The nonlinear residual replaces each constrained row by ``u[d] - g(x_d, t)`` with ``t``
the time the residual is called at, which is the time the step or stage lands on in every scheme.

Oracles:

* **Exactly representable** -- ``u = x + t``, ``w = y - 2t`` under a nonlinear coupling: linear in space (P1
  exact) and in time (exact in every scheme with stage order >= 1: theta, BDF2, SDIRK, Rosenbrock). The boundary
  value therefore has to be imposed at exactly the right time; one step late puts every boundary node ``dt`` off.
* **Taylor--Green vortex** (Navier--Stokes, Taylor--Hood P2/P1) with the exact velocity imposed on every wall as
  a function of ``t``. Error against the analytic solution under ``h`` refinement, the BDF2 temporal order
  against a same-mesh reference, and the same order on the periodic box (no Dirichlet data at all), whose
  error is the scalar BDF2 recursion's error for ``y' = -2 nu y``.
* **Differential** -- a linear Stokes form built twice, once as it is (the linear block, whose time-varying
  Dirichlet lift is the established path) and once with a negligible convective term that routes it through the
  nonlinear residual path. The two trajectories agree to round-off in every scheme.
* **Runtime vs hard-coded** -- a trainable parameter inside the value (``u(wall) - a*x*sin(3t)``) set at runtime
  marches bit-identically to the same number written into the form; ``jax.grad`` w.r.t. it matches central
  differences; and ``jno.core`` recovers it from the trajectory it produced.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
sin, cos, exp = jno.np.sin, jno.np.cos, jno.np.exp


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _newton():
    return dict(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=1e-13), linear=jno.solve.lu(backend="host"))


SCHEMES = {
    "theta1": (lambda: jno.solve.theta(1.0), True),
    "theta0.5": (lambda: jno.solve.theta(0.5), True),
    "bdf2": (jno.solve.bdf2, True),
    "sdirk2": (lambda: jno.solve.sdirk(2), True),
    "sdirk3": (lambda: jno.solve.sdirk(3), True),
    "ros34pw2": (jno.solve.rosenbrock, False),  # linearly implicit: no Newton to configure
    "ros2": (lambda: jno.solve.rosenbrock("ros2"), False),
}


def _solve(fem, scheme, **extra):
    make, newton = SCHEMES[scheme]
    kw = _newton() if newton else dict(linear=jno.solve.lu(backend="host"))
    kw.update(extra)
    return np.asarray(fem.solve(time=make(), **kw).fn())


# ---------------------------------------------------------------------------------------------------------------
# exactly representable ramp
# ---------------------------------------------------------------------------------------------------------------

T_RAMP, N_RAMP = 0.2, 4


def _ramp_coupled(late=0.0, k=None):
    """``u = x + t``, ``w = y - 2t`` solve a coupled system with a bilinear and a cubic coupling. ``late`` shifts
    the Dirichlet data back in time (``g(t - late)``): what imposing ``g(t_n)`` for ``g(t_{n+1})`` would produce."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, T_RAMP, N_RAMP + 1))
    u, v = d.fem_symbols(names=("u", "v"))
    w, q = d.fem_symbols(names=("w", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    wi, qi = w.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    ue, we = xi + ti, yi - 2.0 * ti
    kk = 1.0 if k is None else k
    fu = 1.0 + ue * we  # u_t - lap u + k u w   (the source is written with k = 1)
    fw = -2.0 + we**3 + ue * ue  # w_t - lap w + w^3 + u^2
    tl = tb - late
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x + ui.y * vi.y + kk * ui * wi * vi - fu * vi,
            wi.t * qi + wi.x * qi.x + wi.y * qi.y + (wi**3 + ui * ui) * qi - fw * qi,
            u(xb, yb) - (xb + tl),  # a ramp in t on every wall
            w(xb, yb) - (yb - 2.0 * tl),
            u(ci[0], ci[1]) - ci[0],
            w(ci[0], ci[1]) - ci[1],
        ]
    )
    return fem, u, w


def _ramp_exact(fem, u, w):
    ts = np.linspace(0.0, T_RAMP, N_RAMP + 1)[:, None]
    pu = np.asarray(fem.field_points[fem.block_index(u)])
    pw = np.asarray(fem.field_points[fem.block_index(w)])
    return pu[:, 0][None, :] + ts, pw[:, 1][None, :] - 2.0 * ts


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_coupled_nonlinear_ramp_is_exact_in_every_scheme(scheme):
    """Every scheme lands each step (and each stage) on the exact ramp, so the trajectory is exact to round-off at
    every saved time -- boundary rows AND interior, which sees the boundary's rate through the mass columns."""
    fem, u, w = _ramp_coupled()
    assert fem.is_transient and not fem.is_linear and len(fem.offsets) == 3
    traj = _solve(fem, scheme)
    ue, we = _ramp_exact(fem, u, w)
    err_u = np.abs(traj[:, fem.blocks[fem.block_index(u)]] - ue).max()
    err_w = np.abs(traj[:, fem.blocks[fem.block_index(w)]] - we).max()
    assert max(err_u, err_w) < 1e-9, f"{scheme}: max error u {err_u:.2e}, w {err_w:.2e}"


def test_coupled_nonlinear_ramp_is_exact_in_3d():
    """The same exactness on tetrahedra: nothing in the imposition is dimension-specific."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=3).domain(time=(0.0, T_RAMP, N_RAMP + 1))
    u, v = d.fem_symbols(names=("u", "v"))
    w, q = d.fem_symbols(names=("w", "q"))
    xi, yi, zi, ti = d.variable("interior", split=True)
    xb, yb, zb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi, t=ti), v.bind(x=xi, y=yi, z=zi, t=ti)
    wi, qi = w.bind(x=xi, y=yi, z=zi, t=ti), q.bind(x=xi, y=yi, z=zi, t=ti)
    dot = lambda a, b: a.x * b.x + a.y * b.y + a.z * b.z  # noqa: E731
    ue, we = xi + ti, yi - 2.0 * ti
    fem = jno.fem(
        [
            ui.t * vi + dot(ui, vi) + ui * wi * vi - (1.0 + ue * we) * vi,
            wi.t * qi + dot(wi, qi) + (wi**3 + ui * ui) * qi - (-2.0 + we**3 + ue * ue) * qi,
            u(xb, yb, zb) - (xb + tb),
            w(xb, yb, zb) - (yb - 2.0 * tb),
            u(ci[0], ci[1], ci[2]) - ci[0],
            w(ci[0], ci[1], ci[2]) - ci[1],
        ]
    )
    ue_n, we_n = _ramp_exact(fem, u, w)
    for scheme in ("bdf2", "sdirk3"):
        traj = _solve(fem, scheme)
        err_u = np.abs(traj[:, fem.blocks[fem.block_index(u)]] - ue_n).max()
        err_w = np.abs(traj[:, fem.blocks[fem.block_index(w)]] - we_n).max()
        assert max(err_u, err_w) < 1e-9, f"3-D {scheme}: u {err_u:.2e}, w {err_w:.2e}"


@pytest.mark.parametrize("scheme", ["bdf2", "sdirk3"])
def test_data_one_step_late_is_visible(scheme):
    """The negative control for the exactness above: the same problem with the boundary data one step late (what
    imposing ``g(t_n)`` where ``g(t_{n+1})`` is due would produce) is off by O(dt), eight orders above it."""
    dt = T_RAMP / N_RAMP
    fem, u, w = _ramp_coupled(late=dt)
    traj = _solve(fem, scheme)
    ue, _ = _ramp_exact(fem, u, w)
    err = np.abs(traj[-1, fem.blocks[fem.block_index(u)]] - ue[-1]).max()
    assert err > 0.5 * dt, f"{scheme}: one-step-late data changed the answer by only {err:.2e} (dt = {dt})"


@pytest.mark.parametrize("kind", ["reaction", "mass"])
def test_single_field_nonlinear_ramp_is_exact(kind):
    """The single-field path takes the same branch: a cubic reaction, and a state-dependent mass ``c(u) u_t``
    (whose mass residual must also drop the time-varying rows)."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, T_RAMP, N_RAMP + 1))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ue = xi + yi + ti
    if kind == "reaction":
        form = ui.t * vi + ui.x * vi.x + ui.y * vi.y + ui**3 * vi - (1.0 + ue**3) * vi
    else:
        form = (1.0 + ui * ui) * ui.t * vi + ui.x * vi.x + ui.y * vi.y - (1.0 + ue * ue) * vi
    fem = jno.fem([form, u(xb, yb) - (xb + yb + tb), u(ci[0], ci[1]) - (ci[0] + ci[1])])
    assert not fem.is_linear
    pts = np.asarray(fem.points)
    exact = (pts[:, 0] + pts[:, 1])[None, :] + np.linspace(0.0, T_RAMP, N_RAMP + 1)[:, None]
    for scheme in ("bdf2", "sdirk3"):
        traj = _solve(fem, scheme)
        assert np.abs(traj - exact).max() < 1e-9, f"{kind}/{scheme}: {np.abs(traj - exact).max():.2e}"


def test_gradient_through_the_march_matches_finite_differences():
    """Reverse mode through the march -- the per-step Newton is a ``custom_root`` -- with the time-varying rows in
    place, w.r.t. a runtime parameter of the coupled nonlinear form."""
    k = jno.np.reshape(jno.np.parameter((1,), name="k"), ())
    fem, u, w = _ramp_coupled(k=k)
    blk = fem.operator
    assert blk.is_nonlinear() and "k" in blk.runtime_parameter_exprs
    save = jnp.linspace(blk.t0, blk.t1, 3)
    for sch in (jno.solve.bdf2(), jno.solve.sdirk(3)):

        def loss(kv, sch=sch):
            return jnp.sum(sch.integrate(blk, {"k": jnp.reshape(kv, (1,))}, save) ** 2)

        g = float(jax.grad(loss)(1.3))
        fd = float((loss(1.3 + 1e-6) - loss(1.3 - 1e-6)) / 2e-6)
        assert abs(g) > 1e-3, "the parameter must actually move the trajectory"
        np.testing.assert_allclose(g, fd, rtol=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# Navier-Stokes: Taylor-Green vortex
# ---------------------------------------------------------------------------------------------------------------

NU, T_TG = 1.0, 0.5


def _taylor_green(n, nst, *, periodic=False, late=0.0, eps=None):
    """``u = (sin x cos y, -cos x sin y) e^{-2 nu t}`` on Taylor-Hood P2/P1, pressure pinned at the centre.

    Walled (``[0, pi]^2``): the exact velocity on every wall as a function of ``t``, ``late`` shifting it back in
    time. Periodic (``[0, 2 pi]^2``): ties on both pairs of faces, no Dirichlet data. ``eps`` scales the
    convective term (``0`` = Stokes, assembled as a linear block)."""
    L = 2 * np.pi if periodic else np.pi
    d = jno.shape.rect(0, 0, L, L).structured(n=n).domain(time=(0.0, T_TG, nst + 1))
    d.point_region("ppin", (L / 2, L / 2))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xn, yn = d.variable("ppin", split=True)[:2]
    ci = d.variable("initial", split=True)
    X = [xi, yi]
    ub, vv = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    gu, gv = grad(u, X), grad(v, X)
    conv = inner(inner(gu, ub, n_contract=1), vv, 1)  # (u . grad) u on the unknown
    mom = inner(ub.t, vv, 1) + NU * inner(gu, gv, 2) - pp * trace(gv)
    mom = mom + (conv if eps is None else (eps * conv if eps else 0.0))
    terms = [mom, -qq * trace(gu), p(xn, yn) - 0.0]
    if periodic:
        e = 1e-9
        for name, pred in {
            "xlo": lambda x, y: x < e,
            "xhi": lambda x, y: x > L - e,
            "ylo": lambda x, y: y < e,
            "yhi": lambda x, y: y > L - e,
        }.items():
            d.tag(name, pred)
        face = lambda tag: d.variable(tag, split=True)[:2]  # noqa: E731
        for a, b in (("xlo", "xhi"), ("ylo", "yhi")):
            terms += [u(*face(a)) - u(*face(b)), p(*face(a)) - p(*face(b))]
    else:
        xb, yb, tb = d.variable("boundary", split=True)[:3]
        decay = exp(-2 * NU * (tb - late))
        terms += [u(xb, yb)[0] - sin(xb) * cos(yb) * decay, u(xb, yb)[1] - (-cos(xb) * sin(yb) * decay)]
    terms += [u(*ci)[0] - sin(ci[0]) * cos(ci[1]), u(*ci)[1] - (-cos(ci[0]) * sin(ci[1]))]
    return jno.fem(terms), u


def _velocity(fem, u, state):
    return np.asarray(state)[fem.blocks[fem.block_index(u)]].reshape(-1, 2)


def _tg_error(fem, u, traj):
    pv = np.asarray(fem.field_points[fem.block_index(u)])
    x, y = pv[:, 0], pv[:, 1]
    ue = np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], axis=1) * np.exp(-2 * NU * T_TG)
    return float(np.linalg.norm(_velocity(fem, u, traj[-1]) - ue) / np.linalg.norm(ue))


def test_taylor_green_driven_walls_converge_in_h():
    """Navier-Stokes with the exact, decaying velocity imposed on all four walls: the error against the analytic
    field is small and falls at the P2 rate under h refinement (dt fixed small enough not to mask it)."""
    errs = []
    for n in (4, 8):
        fem, u = _taylor_green(n, 16)
        assert fem.is_transient and not fem.is_linear
        errs.append(_tg_error(fem, u, _solve(fem, "bdf2")))
    assert errs[1] < 5e-4, f"n=8: {errs[1]:.2e}"  # measured 2.7e-4
    assert errs[0] / errs[1] > 6.0, f"h-refinement: {errs}"  # measured 13 (3.5e-3 -> 2.7e-4)


def test_taylor_green_default_and_jfnk_match_direct_newton():
    """The default Newton (assembled tangent) and matrix-free JFNK reach the same march as sparse-direct Newton."""
    fem, u = _taylor_green(4, 4)
    ref = _solve(fem, "bdf2")
    tol = dict(rtol=1e-10, atol=1e-12)
    with pytest.warns(UserWarning, match="saddle-point"):  # the default Krylov on a saddle warns; it still converges
        dflt = np.asarray(fem.solve(time=jno.solve.bdf2(), nonlinear=jno.solve.newton(**tol)).fn())
    assert np.abs(dflt - ref).max() < 1e-7
    jfnk = np.asarray(fem.solve(time=jno.solve.sdirk(2), nonlinear=jno.solve.newton(**tol)).fn())
    assert np.abs(jfnk - _solve(fem, "sdirk2")).max() < 1e-7
    assert _tg_error(fem, u, ref) < 5e-2


def _bdf2_time_errors(n, steps, ref_steps=64, **kw):
    """Relative time error of the final velocity against a same-mesh, ``ref_steps`` BDF2 reference (so the
    spatial error cancels), for each step count in ``steps``."""

    def final(nst, **kw2):
        fem, u = _taylor_green(n, nst, **kw2)
        return _velocity(fem, u, _solve(fem, "bdf2")[-1])

    ref = final(ref_steps, **{k: v for k, v in kw.items() if k != "late"})
    out = {}
    for nst in steps:
        extra = dict(kw, late=kw["late"] / nst) if "late" in kw else kw
        out[nst] = float(np.linalg.norm(final(nst, **extra) - ref) / np.linalg.norm(ref))
    return out


def test_taylor_green_bdf2_is_second_order_with_driven_walls():
    """BDF2 stays second order in time with every wall driven by g(t). With the wall data one step late -- what
    imposing ``g(t_n)`` where ``g(t_{n+1})`` is due would do -- it drops to first order, and at 16 steps the
    error is ~900x larger."""
    e = _bdf2_time_errors(6, (8, 16))
    late = _bdf2_time_errors(6, (8, 16), late=T_TG)  # late = one step: T/nst
    rate, rate_late = np.log2(e[8] / e[16]), np.log2(late[8] / late[16])
    assert rate > 1.8, f"walled BDF2 temporal rate {rate:.2f} (errors {e})"  # measured 2.3
    assert rate_late < 1.3, f"one-step-late data should be first order, rate {rate_late:.2f}"  # measured 1.05
    assert late[16] > 200 * e[16], f"late data {late[16]:.2e} vs correct {e[16]:.2e}"  # measured 930x


def test_periodic_taylor_green_bdf2_error_is_the_scalar_recursion():
    """The cross-check without any Dirichlet data: on the periodic box the whole field is one eigenmode, so the
    BDF2 time error is the scalar recursion's for ``y' = -2 nu y`` (measured to 0.5 %) -- the same second order
    the walled problem shows above."""

    def bdf2_scalar(nst, lam=2 * NU):
        h = T_TG / nst
        y = [1.0, 1.0 / (1.0 + lam * h)]  # backward-Euler start, as jno.solve.bdf2
        for _ in range(nst - 1):
            y.append((4 * y[-1] - y[-2]) / (3.0 + 2.0 * lam * h))
        return y[-1]

    e = _bdf2_time_errors(8, (8, 16), periodic=True)
    for nst, got in e.items():
        want = abs(bdf2_scalar(nst) - bdf2_scalar(64)) / abs(bdf2_scalar(64))
        assert abs(got / want - 1.0) < 0.03, f"periodic nst={nst}: {got:.4e} vs scalar {want:.4e}"


@pytest.mark.parametrize("scheme", ["bdf2", "sdirk3", "theta0.5", "ros34pw2"])
def test_nonlinear_path_reproduces_the_linear_block(scheme):
    """Stokes with the Taylor-Green wall data, assembled as the linear block (the established time-varying
    Dirichlet lift) and again through the nonlinear residual (a 1e-30 convective term). Same march to round-off --
    which also pins the mass COLUMNS of the time-varying rows: zeroing them (as a constant condition does) was
    measured to move the answer by 8e-3 (8x8 mesh, 8 BDF2 steps)."""
    lin, u = _taylor_green(4, 6, eps=0)
    nl, _ = _taylor_green(4, 6, eps=1e-30)
    assert lin.is_linear and not nl.is_linear
    a = np.asarray(lin.solve(time=SCHEMES[scheme][0](), linear=jno.solve.lu(backend="host")).fn())
    b = _solve(nl, scheme)
    assert np.abs(a - b).max() < 1e-10, f"{scheme}: {np.abs(a - b).max():.2e}"


# ---------------------------------------------------------------------------------------------------------------
# a trainable parameter inside the time-varying value: identify boundary data
# ---------------------------------------------------------------------------------------------------------------

A_TRUE = 1.7


def _driven_wall(a, nonlinear, coupled=True):
    """``u = a x sin(3t)`` on every wall, zero IC; the second field (coupled) is held at zero on the walls and
    driven through the coupling. ``a`` is a number (hard-coded) or a ``jno.np.parameter`` (set at runtime)."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, 0.2, 5))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    lap = lambda p, q: p.x * q.x + p.y * q.y  # noqa: E731
    wall = u(xb, yb) - a * xb * sin(3.0 * tb)
    if not coupled:
        return jno.fem([ui.t * vi + lap(ui, vi) + (ui**3 * vi if nonlinear else 0.0), wall, u(ci[0], ci[1]) - 0.0]), u
    w, q = d.fem_symbols(names=("w", "q"))
    wi, qi = w.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    cpl = (ui * wi * vi + (wi**3 - ui) * qi) if nonlinear else (ui * qi - wi * vi)
    form = ui.t * vi + lap(ui, vi) + wi.t * qi + lap(wi, qi) + cpl
    return jno.fem([form, wall, w(xb, yb) - 0.0, u(ci[0], ci[1]) - 0.0, w(ci[0], ci[1]) - 0.0]), u


def _amp():
    return jno.np.reshape(jno.np.parameter((1,), name="a"), ())


def _march(blk, scheme, args, save):
    return SCHEMES[scheme][0]().integrate(blk, args, save, linear_solve=None, nonlinear_solve=None)


@pytest.mark.parametrize("coupled", [True, False], ids=["coupled", "single"])
@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_parameter_in_time_varying_value_equals_the_hard_coded_march(nonlinear, coupled):
    """``u(wall) - a*x*sin(3t)`` with ``a`` a runtime parameter used to refuse at build on every path. Set at
    runtime it must march exactly the trajectory the hard-coded ``a`` does, in every scheme, and the wall must
    carry ``a*x*sin(3t)`` itself (a stiffly accurate scheme holds it to round-off)."""
    fp, u = _driven_wall(_amp(), nonlinear, coupled)
    fh, _ = _driven_wall(A_TRUE, nonlinear, coupled)
    bp, bh = fp.operator, fh.operator
    assert "a" in bp.runtime_parameter_exprs and bp.is_nonlinear() == nonlinear
    save = jnp.linspace(bp.t0, bp.t1, 3)
    blk = fp.blocks[fp.block_index(u)] if coupled else slice(None)
    pts = np.asarray(fp.field_points[fp.block_index(u)] if coupled else fp.points)
    wall = np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 1) | np.isclose(pts[:, 1], 0) | np.isclose(pts[:, 1], 1)
    g = A_TRUE * pts[wall, 0][None, :] * np.sin(3.0 * np.asarray(save))[:, None]
    for scheme in ("theta1", "bdf2", "sdirk3", "ros2"):
        yp = np.asarray(_march(bp, scheme, {"a": jnp.asarray([A_TRUE])}, save))
        yh = np.asarray(_march(bh, scheme, {}, save))
        assert np.abs(yp - yh).max() < 1e-12, f"{scheme}: runtime a vs hard-coded a: {np.abs(yp - yh).max():.2e}"
        if scheme != "ros2":  # ros2 is not stiffly accurate -- see the wall-value test below
            err = np.abs(yp[:, blk][:, wall] - g).max()
            assert err < 1e-10, f"{scheme}: wall value off by {err:.2e}"
    # and the parameter does move the answer (a wrong key would fall back to the stored value silently)
    y2 = np.asarray(_march(bp, "bdf2", {"a": jnp.asarray([2.0 * A_TRUE])}, save))
    assert np.abs(y2 - np.asarray(_march(bh, "bdf2", {}, save))).max() > 0.5


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_gradient_wrt_boundary_amplitude_matches_finite_differences(nonlinear):
    """Reverse mode through the march w.r.t. the boundary amplitude, the loss read on the INTERIOR nodes only
    (so the derivative has to travel through the PDE, not just the wall rows): central differences to 1e-6."""
    fp, u = _driven_wall(_amp(), nonlinear)
    bp = fp.operator
    save = jnp.linspace(bp.t0, bp.t1, 3)
    pts = np.asarray(fp.field_points[fp.block_index(u)])
    inner_ = ~(np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 1) | np.isclose(pts[:, 1], 0) | np.isclose(pts[:, 1], 1))
    idx = np.arange(fp.dofs)[fp.blocks[fp.block_index(u)]][inner_]
    for scheme in ("bdf2", "sdirk3"):

        def loss(av, scheme=scheme):
            return jnp.sum(_march(bp, scheme, {"a": jnp.reshape(av, (1,))}, save)[-1, idx] ** 2)

        gr = float(jax.grad(loss)(A_TRUE))
        fd = float((loss(A_TRUE + 1e-6) - loss(A_TRUE - 1e-6)) / 2e-6)
        assert abs(gr) > 1e-3, "the amplitude must actually move the interior"
        np.testing.assert_allclose(gr, fd, rtol=1e-6, err_msg=f"{scheme}")


def test_boundary_amplitude_is_recovered_through_jno_core():
    """The inverse problem this enables, end to end: the amplitude of a driven wall recovered from the
    trajectory it produced, through ``jno.core`` (0.5 -> 1.7)."""
    import optax

    fh, _ = _driven_wall(A_TRUE, False, coupled=False)
    u_traj = np.asarray(fh.solve(time=jno.solve.bdf2()).fn())
    a = jno.np.parameter((1,), name="a")
    a.dtype(jnp.float64)
    a.initialize(jax.nn.initializers.constant(0.5))
    a.optimizer(optax.adam(5e-2))
    fem, _ = _driven_wall(jno.np.reshape(a, ()), False, coupled=False)
    crux = jno.core(
        [(fem.solve(time=jno.solve.bdf2()) - u_traj).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))})
    )
    crux.solve(300)
    got = float(np.asarray(crux.eval([a])).reshape(-1)[0])
    assert abs(got - A_TRUE) < 1e-5, f"recovered amplitude {got} (true {A_TRUE})"


# ---------------------------------------------------------------------------------------------------------------
# what still refuses
# ---------------------------------------------------------------------------------------------------------------


def test_refusals_name_the_combination():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=3).domain(time=(0.0, 0.1, 3))
    u, v = d.fem_symbols(names=("u", "v"))
    w, q = d.fem_symbols(names=("w", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    wi, qi = w.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    weak = [ui.t * vi + ui.x * vi.x + ui.y * vi.y + ui * wi * vi, wi.t * qi + wi.x * qi.x + wi.y * qi.y]
    ics = [u(ci[0], ci[1]) - 0.0, w(ci[0], ci[1]) - 0.0]
    g = jno.np.reshape(jno.np.parameter((1,), name="g"), ())
    # a trainable parameter inside a time-varying value builds on a first-order transient ...
    assert "g" in jno.fem(weak + [u(xb, yb) - g * tb, w(xb, yb) - 0.0] + ics).operator.runtime_parameter_exprs
    # ... but not on a u_tt form, whose block evaluates g(x, t) and its rate without the runtime args
    ci0 = u.bind(x=ci[0], y=ci[1], t=ci[2])
    with pytest.raises(NotImplementedError, match="BOTH runtime-parametric"):
        jno.fem([ui.tt * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - g * tb, u(ci[0], ci[1]) - 0.0, ci0.t - 0.0])
    # a parameter-valued Dirichlet on one field next to a time-varying one on another
    with pytest.raises(NotImplementedError, match="parameter-valued Dirichlet combined with a time-varying"):
        jno.fem(weak + [u(xb, yb) - g, w(xb, yb) - tb] + ics)
    # the old refusal of the plain combination is gone
    fem = jno.fem(weak + [u(xb, yb) - tb, w(xb, yb) - 0.0] + ics)
    assert fem.is_transient and not fem.is_linear

    # single field: nonlinear + time-varying Dirichlet builds; a runtime parameter in the form still refuses
    k = jno.np.reshape(jno.np.parameter((1,), name="kk"), ())
    base = [u(xb, yb) - tb, u(ci[0], ci[1]) - 0.0]
    assert not jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y + ui**3 * vi] + base).is_linear
    with pytest.raises(NotImplementedError, match="without a runtime parameter"):
        jno.fem([ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y) + ui**3 * vi] + base)
