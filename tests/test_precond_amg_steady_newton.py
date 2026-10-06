"""An unbuilt ``jno.precond.amg()`` in a STEADY nonlinear solve: frozen once, from the tangent at ``x0``.

Newton runs as a ``lax.while_loop``, so the tangent the inner linear solve sees is traced and pyamg's host
setup cannot read it: ``fem.solve(linear=jno.solve.fgmres(), precond=jno.precond.amg())`` over a nonlinear
problem died with "AMG setup needs a concrete matrix but got a traced one" (the matrix-free Newton with
"LinearOperator.dense(): a matvec-only operator cannot densify"). The march already froze such specs before
its scan; the steady solve now does the same before its loop: one hierarchy, built eagerly from the tangent
at the initial guess.

Oracles: a manufactured solution anchors the default solve, and the frozen-AMG solve must converge to the
same root (a preconditioner changes the Krylov speed, never Newton's root). For Navier-Stokes the oracle is
the sparse-direct Newton of the same system.
"""

import jax
import numpy as np
import pytest

import jno
from jno import jnp_ops as jnn

pytest.importorskip("pyamg", reason="jno.precond.amg() sets up with pyamg")

EPS = 1e-9
π = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def builds(monkeypatch):
    """Every AMG hierarchy built, and whether its matrix was concrete (a traced one cannot be)."""
    import jno.utils.solver.amg as amg

    seen, real = [], amg.build_hierarchy

    def spy(A, **kw):
        seen.append(not isinstance(getattr(A, "data", A), jax.core.Tracer))
        return real(A, **kw)

    monkeypatch.setattr(amg, "build_hierarchy", spy)
    return seen


def _diffusion(periodic=False, n=16):
    """``-div((1 + u^2) grad u) = f`` on the unit square, manufactured from

    * ``u = sin(pi x) sin(pi y)`` with u = 0 on the whole boundary, or
    * ``u = (1.5 + cos(2 pi x)) sin(pi y)``, periodic in x, u = 0 at y = 0 and y = 1."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).domain()
    d.tag("left", lambda x, y: x < EPS)
    d.tag("right", lambda x, y: x > 1 - EPS)
    if periodic:
        d.tag("held", lambda x, y: ((y < EPS) | (y > 1 - EPS)) & (x > EPS))
    else:
        d.tag("held", lambda x, y: (y < EPS) | (y > 1 - EPS) | (x < EPS) | (x > 1 - EPS))
    x, y = d.variable("interior", split=True)[:2]
    xh, yh, _ = d.variable("held", split=True)
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    if periodic:
        ue = (1.5 + jnn.cos(2 * π * x)) * jnn.sin(π * y)
        ux, uy = -2 * π * jnn.sin(2 * π * x) * jnn.sin(π * y), π * (1.5 + jnn.cos(2 * π * x)) * jnn.cos(π * y)
        lap = -4 * π**2 * jnn.cos(2 * π * x) * jnn.sin(π * y) - π**2 * ue
        exact = lambda X, Y: (1.5 + np.cos(2 * π * X)) * np.sin(π * Y)  # noqa: E731
    else:
        ue = jnn.sin(π * x) * jnn.sin(π * y)
        ux, uy = π * jnn.cos(π * x) * jnn.sin(π * y), π * jnn.sin(π * x) * jnn.cos(π * y)
        lap = -2 * π**2 * ue
        exact = lambda X, Y: np.sin(π * X) * np.sin(π * Y)  # noqa: E731
    f = -((1 + ue**2) * lap + 2 * ue * (ux**2 + uy**2))  # -div((1+u^2) grad u) = f
    κ = 1 + ub**2
    terms = [κ * (ub.x * vb.x + ub.y * vb.y) - f * vb, u(xh, yh) - 0.0]
    if periodic:
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        terms += [u(xl, yl) - u(xr, yr)]
    fem = jno.fem(terms)
    pts = np.asarray(fem.points)
    return fem, exact(pts[:, 0], pts[:, 1])


def _arr(out):
    """A solve's result as a flat array (a periodic nonlinear solve stays a lazy node: evaluate it)."""
    return np.asarray(out.fn() if hasattr(out, "fn") else out).reshape(-1)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


_FGMRES = dict(linear=jno.solve.fgmres(tol=1e-10))


@pytest.mark.parametrize("periodic", [False, True], ids=["dirichlet", "periodic"])
def test_amg_in_a_steady_newton_matches_the_default_solve(periodic, builds):
    fem, exact = _diffusion(periodic)
    ref = _arr(fem.solve())
    # anchor the oracle: P1 on a 16 x 16 grid (measured 2.2e-3 and 1.0e-2 max-norm)
    assert np.abs(ref - exact).max() < (2e-2 if periodic else 5e-3)
    got = _arr(fem.solve(precond=jno.precond.amg(), **_FGMRES))
    assert _rel(got, ref) < 1e-9
    assert builds == [True], "one hierarchy for the whole Newton solve, from a concrete tangent"


@pytest.mark.parametrize("nonlinear", ["newton-matrix-free", "picard"])
def test_drivers_that_are_not_handed_a_tangent_build_from_the_assembled_one(nonlinear, builds):
    """``newton(direct=False)`` / ``picard()`` linearize matrix-free; the hierarchy comes from the problem's
    own assembled tangent at ``x0`` (on the reduced space for the periodic case)."""
    driver = jno.solve.newton(direct=False) if nonlinear == "newton-matrix-free" else jno.solve.picard()
    for periodic in (False, True):
        fem, _exact = _diffusion(periodic)
        ref = _arr(fem.solve())
        builds.clear()
        got = _arr(fem.solve(precond=jno.precond.amg(), nonlinear=driver, **_FGMRES))
        assert _rel(got, ref) < 1e-9
        assert builds == [True]


def test_the_hierarchy_is_built_at_x0(builds):
    """Frozen where the solve starts: from a guess far from zero (the exact solution itself, where the
    tangent carries the ``2 u grad u . du`` term) it still converges to the same root."""
    fem, exact = _diffusion()
    ref = _arr(fem.solve())
    for s in (0.5, 1.0):
        builds.clear()
        got = _arr(fem.solve(precond=jno.precond.amg(), x0=s * exact, **_FGMRES))
        assert _rel(got, ref) < 1e-8
        assert builds == [True]


def test_ilu_is_built_once_the_same_way():
    """Any spec whose setup is host-side (scipy's incomplete LU here), not only AMG."""
    fem, _exact = _diffusion()
    ref = _arr(fem.solve())
    assert _rel(_arr(fem.solve(precond=jno.precond.ilu(), **_FGMRES)), ref) < 1e-9


def test_a_rebuild_cadence_is_refused_not_ignored():
    """``cached(amg(), refresh=k)`` would need the Newton ``while_loop`` cut into chunks; the setup is built
    once, so a cadence that would silently never fire is refused by name."""
    fem, _exact = _diffusion()
    with pytest.raises(NotImplementedError, match="refresh=3"):
        fem.solve(precond=jno.precond.cached(jno.precond.amg(), refresh=3), **_FGMRES)


# --- steady Navier-Stokes: triangular((u, amg()), (p, pressure mass)) on the Picard-lagged tangent ---------------

MU, L, H = 0.1, 2.0, 1.0


def _navier_stokes(periodic):
    """Taylor-Hood channel ``[0, L] x [0, H]``, no-slip walls, driven by a body force that varies along x
    (so the convection ``(u . grad) u`` is not zero), with the CONVECTING velocity lagged (``jno.lag``): the
    tangent is then the Oseen operator, whose velocity block is AMG-friendly (docs/solvers.md, "The
    momentum block: Picard, not Newton"). ``periodic``: tied in x; else no-slip at both ends too."""
    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    d = jno.shape.rect(0.0, 0.0, L, H).structured(n=8).domain()
    d.point_region("ppin", (L / 2, H / 2))
    d.tag("left", lambda x, y: x < EPS)
    d.tag("right", lambda x, y: x > L - EPS)
    if periodic:
        d.tag("wall", lambda x, y: ((y < EPS) | (y > H - EPS)) & (x > EPS))
    else:
        d.tag("wall", lambda x, y: (y < EPS) | (y > H - EPS) | (x < EPS) | (x > L - EPS))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    X = list(d.variable("interior", split=True)[:2])
    ub, vb = u.bind(x=X[0], y=X[1]), v.bind(x=X[0], y=X[1])
    pb, qb = p.bind(x=X[0], y=X[1]), q.bind(x=X[0], y=X[1])
    ᐁu, ᐁv = grad(u, X), grad(v, X)
    div = trace
    dot = lambda a, b: inner(a, b, n_contract=1)  # noqa: E731
    f = 1.0 + 0.5 * jno.np.sin(2 * π * X[0] / L)
    xw, yw, _ = d.variable("wall", split=True)
    xp, yp, _ = d.variable("ppin", split=True)
    terms = [
        MU * inner(ᐁu, ᐁv, n_contract=2) + dot(dot(ᐁu, jno.lag(ub)), vb) - pb * div(ᐁv) - f * vb[0] - 0.3 * f * vb[1],
        -qb * div(ᐁu),
        u(xw, yw)[0] - 0.0,
        u(xw, yw)[1] - 0.0,
        p(xp, yp) - 0.0,
    ]
    if periodic:
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        terms += [u(xl, yl) - u(xr, yr), p(xl, yl) - p(xr, yr)]
    return jno.fem(terms), u, p, pb, qb


@pytest.mark.parametrize("periodic", [False, True], ids=["closed", "periodic"])
@pytest.mark.parametrize("recipe", ["triangular", "saddle"])
def test_an_amg_velocity_block_on_a_steady_navier_stokes_matches_lu(recipe, periodic, builds):
    """``triangular((u, amg()), (p, pressure mass))`` spelled out, and ``saddle()`` (the same recipe, its
    momentum block an ``amg()`` it creates itself)."""
    fem, u, p, pb, qb = _navier_stokes(periodic)
    newton = jno.solve.newton(rtol=1e-11, atol=1e-12)
    ref = _arr(fem.solve(nonlinear=newton, linear=jno.solve.lu(backend="host")))
    steps = fem.stats["nonlinear"]["steps"] if fem.stats and fem.stats.get("nonlinear") else None
    if recipe == "triangular":
        spec = jno.precond.triangular((u, jno.precond.amg()), (p, jno.precond.form([(1.0 / MU) * pb * qb])))
    else:
        spec = jno.precond.saddle(mass_weight=1.0 / MU)
    builds.clear()
    got = _arr(fem.solve(nonlinear=newton, linear=jno.solve.fgmres(tol=1e-12, restart=200, maxiter=2000), precond=spec))
    assert _rel(got, ref) < 1e-9
    assert builds == [True], "the velocity block's hierarchy: once, from the concrete tangent at x0"
    if steps is not None:
        assert steps > 1, "the convection must actually make the problem nonlinear"


def test_a_frozen_preconditioner_refuses_a_sub_system_by_name():
    """``jno.solve.staggered`` solves each field group on its own; a preconditioner frozen from the whole
    tangent cannot be applied there, and says so instead of failing inside the Krylov loop."""
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext, _FrozenMarchPrecond

    frozen = _FrozenMarchPrecond(lambda r: r, "amg()", n=10)
    ctx = PrecondContext(LinearOperator.from_matvec(lambda r: r, shape=(4, 4)))
    with pytest.raises(ValueError, match="10 x 10 tangent, but is being applied to a 4 x 4"):
        frozen(ctx)
