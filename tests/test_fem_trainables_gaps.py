"""Trainables inside ``jno.fem([...])`` on the solve types that had no gradient test.

jNO promises that a trainable -- a ``jno.np.parameter``, a ``jno.nn.wrap(net)`` coefficient -- placed in
the term list is carried through ``fem.solve()`` and optimised by ``jno.core`` with the exact gradient
(implicit differentiation, never an unrolled solve). The existing checks cover steady single-field
linear/nonlinear solves and backward Euler, and most of them bypass the public path (they call
``op.residual`` + ``newton_krylov`` directly). This file covers the rest, every one through the public
path ``jno.core([loss]).solve(...)``:

1. mixed/saddle (Stokes refused -> the Navier-Stokes route), 2. eigenproblems, 3. the non-default
time schemes, 4. a network coefficient in 3-D, 5. network weights on a linear solve, 6. Neumann/Robin
boundary parameters, 7. staggered solves and history marches, 8. a network coefficient in ``jno.fdm``.

How the gradient is read off ``jno.core`` without reaching inside it: attach ``optax.sgd(lr)`` and take
ONE ``crux.solve(1)`` step. A single plain-SGD step moves the trainable by exactly ``-lr * dL/dθ``
(jNO's own learning-rate scale defaults to 1), so ``(θ0 - θ1) / lr`` IS the gradient jno.core
computed. It is compared with central finite differences of the same loss, each side evaluated by
``jno.core(...).eval([loss])`` after re-initialising the trainable -- nothing but the public API on
either side. Where the discrete answer is known in closed form (``u(k) = u(1)/k`` for a coefficient
multiplying the whole operator), the gradient is checked against that instead.

Run with x64 (assembly runs in float64).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("equinox")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402
from jno.trace import ModelWeights, Placeholder  # noqa: E402

grad, inner, trace = jno.np.grad, jno.np.inner, jno.np.trace
sin, PI = jno.np.sin, np.pi

# The losses below have no spatial Variable (the FEM solve is global), so crux gets an explicit
# one-point domain to drive its loop.
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64; set x64 per-test with save/restore (the flag is global)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ----------------------------------------------------------------------------------------------------
# helpers: trainables, and the gradient as jno.core computes it
# ----------------------------------------------------------------------------------------------------
def _const(c):
    return lambda *a, **kw: jnp.array([float(c)])


def _param(name, value):
    p = jno.np.parameter((1,), name=name).initialize(_const(value))
    p.dtype(jnp.float64)
    return p


class _Affine(eqx.Module):
    """A two-leaf 'network' ``a + b*x`` -- a genuine spatially varying coefficient whose leaves can each
    be finite-differenced."""

    a: jnp.ndarray
    b: jnp.ndarray

    def __call__(self, x, *rest):
        return self.a + self.b * jnp.asarray(x).reshape(-1, 1)


def _affine(a, b):
    return _Affine(a=jnp.asarray(float(a)), b=jnp.asarray(float(b)))


def _affine_net(a, b):
    net = jno.nn.wrap(_affine(a, b))
    net.dtype(jnp.float64)
    return net


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def _read_param(p):
    return lambda crux: _scalar(crux.eval([p]))


def _read_leaf(net, leaf):
    return lambda crux: float(getattr(crux.eval([ModelWeights(net)]), leaf))


def _crux_gradient(loss, trainable, read, lr):
    """dL/dθ exactly as jno.core computes it: one plain-SGD step moves θ by ``-lr * dL/dθ``.

    Returns ``(gradient, loss at θ0)``."""
    trainable.optimizer(optax.sgd(lr))
    crux = jno.core([loss], domain=_DUMMY)
    loss0 = _scalar(crux.eval([loss]))
    before = read(crux)
    crux.solve(1)
    return (before - read(crux)) / lr, loss0


def _loss_at(loss, trainable, init):
    """The loss as jno.core evaluates it, with the trainable re-initialised (``init`` is an initializer
    for a parameter or a module for a network)."""
    trainable.initialize(init)
    return _scalar(jno.core([loss], domain=_DUMMY).eval([loss]))


def _central_fd(loss, trainable, make_init, x0, h):
    return (_loss_at(loss, trainable, make_init(x0 + h)) - _loss_at(loss, trainable, make_init(x0 - h))) / (2 * h)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def _evaluate(out):
    """A solve result as an array: a deferred node (a transient march, a parametric solve) is evaluated
    through jno.core, a concrete array is passed through."""
    if isinstance(out, Placeholder):
        out = jno.core([out.mse], domain=_DUMMY).eval([out])
    return jnp.asarray(out)


def _recover(loss, p, steps):
    crux = jno.core([loss], domain=_DUMMY)
    crux.solve(steps)
    return _scalar(crux.eval([p]))


# ----------------------------------------------------------------------------------------------------
# 1. mixed / saddle: Stokes (refused, loudly) and the Navier-Stokes route
# ----------------------------------------------------------------------------------------------------
def _flow_terms(nu, *, convective, force_scale=1.0, size=0.3):
    """Taylor-Hood P2/P1 flow in the unit square, no-slip walls, pressure gauged by its mean, driven by
    the body force ``s*sin(pi x) sin(pi y) (1, 1/2)``."""
    d = jno.shape.rect(0, 0, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    X = [xi, yi]
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pb, qb = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    f = force_scale * sin(PI * xi) * sin(PI * yi)
    momentum = nu * inner(grad(u, X), grad(v, X), 2) - pb * trace(grad(v, X)) - f * vb[0] - 0.5 * f * vb[1]
    if convective:
        momentum = momentum + inner(inner(grad(u, X), ub, 1), vb, 1)
    return [momentum, -qb * trace(grad(u, X)), u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0, p.pin(mean=True)]


@pytest.mark.parametrize("where", ["viscosity", "body_force"])
def test_stokes_with_a_trainable_is_refused_loudly_with_a_route(where):
    """Linear Stokes carrying a trainable (viscosity, or a body-force scale) is a coupled LINEAR form with
    a runtime parameter, which has no parametric assembly. That must be a loud build-time refusal that
    names a working route (a nonlinear coupled form / a load-path march), never a silently frozen value.
    This pins the refusal; the gap itself -- no trainable in a linear saddle problem -- is reported."""
    s = _param("s", 1.0)
    kwargs = {"nu": s, "force_scale": 1.0} if where == "viscosity" else {"nu": 1.0, "force_scale": s}
    with pytest.raises(NotImplementedError, match="runtime parameter") as err:
        jno.fem(_flow_terms(convective=False, **kwargs))
    assert "NONLINEAR" in str(err.value) and "load-path march" in str(err.value)


def test_stokes_refusal_names_the_offending_parameter():
    """The refusal must say WHICH trainable put the form on the unsupported branch (house rule: name the
    offending input). With two coefficients in play, "it has a runtime parameter (True)" does not."""
    nu = _param("visc_mu", 1.0)
    with pytest.raises(NotImplementedError) as err:
        jno.fem(_flow_terms(nu, convective=False))
    assert "visc_mu" in str(err.value), f"the refusal does not name the parameter: {str(err.value)[:200]}"


def test_navier_stokes_viscosity_gradient_through_crux_matches_fd():
    """The nonlinear route the Stokes refusal points to: Navier-Stokes (Taylor-Hood, mean-gauged
    pressure) with a trainable viscosity. The gradient jno.core takes through the saddle-point Newton
    solve must equal central finite differences of the same loss (both through the public path)."""
    lu = jno.solve.lu  # the saddle point needs a direct inner solve (Jacobi cannot see a zero diagonal)
    u_obs = _evaluate(jno.fem(_flow_terms(0.1, convective=True)).solve(linear=lu()))
    nu = _param("nu", 0.13)
    fem = jno.fem(_flow_terms(nu, convective=True))
    assert not fem.is_linear and fem.operator.is_parametric
    loss = (fem.solve(linear=lu()) - u_obs).mse

    g, loss0 = _crux_gradient(loss, nu, _read_param(nu), lr=1.0)
    fd = _central_fd(loss, nu, _const, 0.13, 1e-5)
    assert loss0 > 0 and abs(g) > 0
    assert _rel(g, fd) < 1e-5, f"crux {g:.10e} vs FD {fd:.10e} (rel {_rel(g, fd):.2e})"


def test_navier_stokes_viscosity_recovered_via_crux():
    """End-to-end: recover nu = 0.1 from the velocity/pressure it produces, starting at 0.13."""
    u_obs = _evaluate(jno.fem(_flow_terms(0.1, convective=True)).solve(linear=jno.solve.lu()))
    nu = _param("nu", 0.13)
    nu.optimizer(optax.adam(2e-3))
    fem = jno.fem(_flow_terms(nu, convective=True))
    rec = _recover((fem.solve(linear=jno.solve.lu()) - u_obs).mse, nu, 60)
    assert abs(rec - 0.1) < 2e-3, f"recovered nu = {rec:.5f} (truth 0.1)"


# ----------------------------------------------------------------------------------------------------
# 2. eigenproblem with a trainable in the stiffness
# ----------------------------------------------------------------------------------------------------
def _laplace_pieces(size=0.25):
    d = jno.shape.rect(0, 0, 1, 1, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    X = [xi, yi]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    a = lambda k: k * inner(grad(u, X), grad(v, X), 1)  # noqa: E731  -- the bilinear form  k grad u . grad v
    return d, u, v, X, ui, vi, (xb, yb), a


def test_eigs_with_a_trainable_stiffness_is_differentiable_or_refused_clearly():
    """``k * grad u . grad v`` with a trainable k, Dirichlet walls: ``lam(k) = k * lam(1)`` exactly, so
    ``d lam / dk = lam(1)``. ``fem.eigs`` must either return eigenvalues that jno.core can evaluate and
    differentiate at the trainable's value, or refuse clearly (NotImplementedError/ValueError naming the
    parameter). An eager array at the stored value would silently freeze k; a crash is neither."""
    d, u, v, X, ui, vi, (xb, yb), a = _laplace_pieces()
    lam1, _ = jno.fem([a(1.0), u(xb, yb) - 0.0]).eigs(mass=[ui * vi], k=3)
    lam1 = np.asarray(lam1)

    k = _param("k_eig", 1.7)
    Kp = jno.fem([a(k), u(xb, yb) - 0.0])
    try:
        lam, _X = Kp.eigs(mass=[ui * vi], k=3)
    except (NotImplementedError, ValueError) as err:  # a clean refusal is acceptable -- and a gap
        assert "k_eig" in str(err) or "parameter" in str(err), f"unclear refusal: {err}"
        return
    assert isinstance(lam, Placeholder), (
        "fem.eigs on a trainable form returned a concrete value: the parameter was frozen at its stored "
        "value and no gradient can reach it"
    )
    loss = lam.mean
    g, loss0 = _crux_gradient(loss, k, _read_param(k), lr=1e-3)
    assert _rel(loss0, 1.7 * lam1.mean()) < 1e-10
    assert _rel(g, lam1.mean()) < 1e-8, f"d mean(lam)/dk = {g} vs lam(1) mean {lam1.mean()}"


def test_eigenvalue_derivative_through_the_parametric_operator():
    """The building blocks the missing ``fem.eigs`` path would compose: the parametric stiffness
    ``fem.operator.evaluate({k: ...})`` fed to ``jno.solve.eigs`` is differentiable in k, and
    ``d lam_i / dk = lam_i(1)`` exactly. All-Neumann (no Dirichlet rows, which ``evaluate`` would keep
    row-replaced), so the constant mode lam_0 = 0 and the checked modes are lam_1, lam_2."""
    d, u, v, X, ui, vi, _, a = _laplace_pieces()
    lam_ref, _ = jno.fem([a(1.0)]).eigs(mass=[ui * vi], k=3)
    k = _param("k_n", 1.0)
    op = jno.fem([a(k)]).operator
    M = jno.fem([ui * vi]).operator[0]
    solver = jno.solve.eigs(k=3)

    def lam(kv):
        A, _b = op.evaluate({"k_n": jnp.reshape(kv, (1,))})
        return solver(A, M)[0]

    np.testing.assert_allclose(np.asarray(lam(1.0)), np.asarray(lam_ref), rtol=1e-10, atol=1e-9)
    jac = np.asarray(jax.jacfwd(lam)(1.7))
    np.testing.assert_allclose(jac[1:], np.asarray(lam_ref)[1:], rtol=1e-8)


# ----------------------------------------------------------------------------------------------------
# 3. transient marches with the non-default time schemes
# ----------------------------------------------------------------------------------------------------
_SCHEMES = {
    "crank_nicolson": lambda: jno.solve.theta(0.5),
    "bdf2": lambda: jno.solve.bdf2(),
    "sdirk2": lambda: jno.solve.sdirk(order=2),
    "rosenbrock": lambda: jno.solve.rosenbrock(),
}


def _heat_terms(alpha, size=0.3, time=(0.0, 0.05, 6)):
    """``u_t = alpha * lap u``, u = 0 on the walls, u(0) = sin(pi x) sin(pi y)."""
    d = jno.shape.rect(0, 0, 1, 1, size=size).domain(time=time)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    X = [xi, yi]
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    u0 = sin(PI * ci[0]) * sin(PI * ci[1])
    return [ui.t * vi + alpha * inner(grad(u, X), grad(v, X), 1), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0]


@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_transient_scheme_diffusivity_gradient_through_crux_matches_fd(scheme):
    """A trainable diffusivity marched by a non-default scheme (Crank-Nicolson, BDF2, SDIRK2,
    Rosenbrock): the gradient jno.core takes through the whole march equals central FD of the same
    trajectory loss. Existing tests differentiate only the backward-Euler march."""
    u_obs = _evaluate(jno.fem(_heat_terms(1.0)).solve(time=_SCHEMES[scheme]()))
    assert u_obs.shape[0] == 6 and bool(jnp.all(jnp.isfinite(u_obs)))
    alpha = _param("alpha", 1.3)
    fem = jno.fem(_heat_terms(alpha))
    assert fem.is_transient and list(fem.operator.runtime_parameter_exprs) == ["alpha"]
    loss = (fem.solve(time=_SCHEMES[scheme]()) - u_obs).mse

    g, loss0 = _crux_gradient(loss, alpha, _read_param(alpha), lr=1.0)
    fd = _central_fd(loss, alpha, _const, 1.3, 1e-5)
    assert loss0 > 0 and abs(g) > 0
    assert _rel(g, fd) < 1e-5, f"{scheme}: crux {g:.10e} vs FD {fd:.10e} (rel {_rel(g, fd):.2e})"

    # ... and the march being differentiated IS the requested scheme: at alpha = 1 the parametric march
    # reproduces the constant-coefficient march of that scheme, which is measurably not backward Euler.
    u_be = _evaluate(jno.fem(_heat_terms(1.0)).solve())
    gap_to_be = float(jnp.mean((u_obs - u_be) ** 2))
    assert gap_to_be > 1e-9, f"{scheme} gave the backward-Euler trajectory (mse {gap_to_be:.2e})"
    assert _loss_at(loss, alpha, _const(1.0)) < 1e-6 * gap_to_be, "the parametric march ran a different scheme"


def test_bdf2_diffusivity_recovered_via_crux():
    """End-to-end recovery through a BDF2 march: alpha from 1.3 back to the truth 1.0."""
    u_obs = _evaluate(jno.fem(_heat_terms(1.0)).solve(time=jno.solve.bdf2()))
    alpha = _param("alpha", 1.3)
    alpha.optimizer(optax.adam(2e-2))
    fem = jno.fem(_heat_terms(alpha))
    rec = _recover((fem.solve(time=jno.solve.bdf2()) - u_obs).mse, alpha, 120)
    assert abs(rec - 1.0) < 1e-2, f"recovered alpha = {rec:.5f} (truth 1.0)"


# ----------------------------------------------------------------------------------------------------
# 4. a network coefficient in 3-D
# ----------------------------------------------------------------------------------------------------
def test_3d_net_coefficient_gradient_through_crux_matches_fd():
    """``(a + b x) grad u . grad v`` on a tetrahedral box (linear Poisson, walls clamped): the gradient
    jno.core takes to the network leaf b equals central FD of the same loss."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.45).domain()
    u, v = d.fem_symbols()
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    X = [xi, yi, zi]
    vi = v.bind(x=xi, y=yi, z=zi)
    a_form = lambda k: k * inner(grad(u, X), grad(v, X), 1)  # noqa: E731
    u_obs = _evaluate(jno.fem([a_form(1.0 + 0.5 * xi) - 1.0 * vi, u(xb, yb, zb) - 0.0]).solve())
    net = _affine_net(1.2, 0.1)
    fem = jno.fem([a_form(net(xi, yi, zi)) - 1.0 * vi, u(xb, yb, zb) - 0.0])
    assert fem.is_linear and fem.operator.is_parametric
    loss = (fem.solve() - u_obs).mse

    g_b, loss0 = _crux_gradient(loss, net, _read_leaf(net, "b"), lr=1.0)
    fd_b = _central_fd(loss, net, lambda b: _affine(1.2, b), 0.1, 1e-5)
    assert loss0 > 0 and abs(g_b) > 0
    assert _rel(g_b, fd_b) < 1e-6, f"crux {g_b:.10e} vs FD {fd_b:.10e} (rel {_rel(g_b, fd_b):.2e})"


# ----------------------------------------------------------------------------------------------------
# 5. network weights on a LINEAR solve, through the public path
# ----------------------------------------------------------------------------------------------------
def test_linear_net_coefficient_gradient_through_crux_matches_fd():
    """``(a + b x) grad u . grad v`` as a two-leaf network on a linear Poisson solve: the gradient
    jno.core takes to each leaf equals central FD of the same loss (public path both sides)."""
    d, u, v, (xi, yi), ui, vi, (xb, yb), a = _laplace_pieces()
    u_obs = _evaluate(jno.fem([a(1.0 + 0.5 * xi) - 1.0 * vi, u(xb, yb) - 0.0]).solve())
    net = _affine_net(1.2, 0.1)
    fem = jno.fem([a(net(xi, yi)) - 1.0 * vi, u(xb, yb) - 0.0])
    assert fem.is_linear and fem.operator.is_parametric
    loss = (fem.solve() - u_obs).mse

    net.optimizer(optax.sgd(1.0))  # one SGD step, lr 1: both leaves move by exactly -dL/dleaf
    crux = jno.core([loss], domain=_DUMMY)
    m0 = crux.eval([ModelWeights(net)])
    a0, b0 = float(m0.a), float(m0.b)
    crux.solve(1)
    m1 = crux.eval([ModelWeights(net)])
    g_a, g_b = a0 - float(m1.a), b0 - float(m1.b)

    h = 1e-5
    fd_a = _central_fd(loss, net, lambda a: _affine(a, 0.1), 1.2, h)
    fd_b = _central_fd(loss, net, lambda b: _affine(1.2, b), 0.1, h)
    assert _rel(g_a, fd_a) < 1e-6, f"leaf a: crux {g_a:.10e} vs FD {fd_a:.10e}"
    assert _rel(g_b, fd_b) < 1e-6, f"leaf b: crux {g_b:.10e} vs FD {fd_b:.10e}"


def test_linear_parametric_solve_honours_a_value_keyword():
    """``fem.solve(k=value)`` is the documented way to solve a parametric problem AT a value. On a LINEAR
    parametric form it must give the solution at that value -- here ``u(2) = u(1)/2`` exactly -- not
    return a node that silently solves at the parameter's stored value."""
    d, u, v, X, ui, vi, (xb, yb), a = _laplace_pieces(size=0.3)
    ref2 = np.asarray(jno.fem([a(2.0) - 1.0 * vi, u(xb, yb) - 0.0]).solve()).reshape(-1)
    k = _param("k_val", 1.0)
    fem = jno.fem([a(k) - 1.0 * vi, u(xb, yb) - 0.0])
    got = np.asarray(_evaluate(fem.solve(k_val=2.0))).reshape(-1)  # a deferred node must still solve at k=2
    np.testing.assert_allclose(got, ref2, rtol=1e-8, atol=1e-12, err_msg="fem.solve(k_val=2.0) ignored the value")


# ----------------------------------------------------------------------------------------------------
# 6. a scalar parameter in a Neumann or Robin boundary term
# ----------------------------------------------------------------------------------------------------
def _edge_terms(where, s):
    """Poisson on the unit square, clamped on the left edge; the right edge carries the parameter either
    as a Neumann flux ``-s v`` or a Robin reaction ``s u v``."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xl, yl, _ = d.variable("left", split=True)
    X = [xi, yi]
    vi = v.bind(x=xi, y=yi)
    ur, vr = u.bind(x=xr, y=yr), v.bind(x=xr, y=yr)
    edge = -s * vr if where == "neumann" else s * ur * vr
    return [inner(grad(u, X), grad(v, X), 1) - 1.0 * vi, edge, u(xl, yl) - 0.0]


@pytest.mark.parametrize("where, truth, start", [("neumann", 0.5, 0.8), ("robin", 2.0, 1.3)])
def test_boundary_term_parameter_gradient_through_crux_matches_fd(where, truth, start):
    """A trainable in a surface (Neumann flux / Robin reaction) integrand on Lagrange elements: the
    gradient jno.core takes equals central FD of the same loss, and the parameter actually changes the
    solution (the boundary kernel threads it rather than freezing the stored value)."""
    u_obs = _evaluate(jno.fem(_edge_terms(where, truth)).solve())
    s = _param("s_edge", start)
    fem = jno.fem(_edge_terms(where, s))
    assert fem.is_linear and list(fem.operator.runtime_parameter_exprs) == ["s_edge"]
    loss = (fem.solve() - u_obs).mse

    g, loss0 = _crux_gradient(loss, s, _read_param(s), lr=1.0)
    fd = _central_fd(loss, s, _const, start, 1e-5)
    assert loss0 > 1e-8, "the edge parameter does not move the solution -- the comparison would be vacuous"
    assert _rel(g, fd) < 1e-6, f"{where}: crux {g:.10e} vs FD {fd:.10e} (rel {_rel(g, fd):.2e})"
    assert _loss_at(loss, s, _const(truth)) < 1e-20, "the parametric solve at the truth is not the constant one"


# ----------------------------------------------------------------------------------------------------
# 7. staggered solves and history marches
# ----------------------------------------------------------------------------------------------------
def _coupled_terms(kc):
    """Two coupled fields, nonlinear in the first (so the coupled form assembles as a residual operator
    and may carry a runtime parameter): the pair is swept by ``jno.solve.staggered``."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.3).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    X = [xi, yi]
    a, phi = d.fem_symbols(names=("sa", "sphi"))
    b, chi = d.fem_symbols(names=("sb", "schi"))
    ai, pi_ = a.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    bi, qi = b.bind(x=xi, y=yi), chi.bind(x=xi, y=yi)
    terms = [
        inner(grad(a, X), grad(phi, X), 1) + 0.1 * (ai * ai) * pi_ - 10.0 * pi_ - kc * bi * pi_,
        inner(grad(b, X), grad(chi, X), 1) + bi * qi - 2.0 * ai * qi,
        a(xb, yb) - 0.0,
        b(xb, yb) - 0.0,
    ]
    return terms, (a, b)


def test_staggered_parameter_gradient_through_crux_matches_fd():
    """A coupling constant inside a coupled nonlinear form solved by ``jno.solve.staggered([a, b])``:
    the gradient jno.core takes (custom_root on the full residual) equals central FD of the same loss,
    and the staggered answer is the monolithic Newton answer (so the sweep really converged)."""
    terms, _ = _coupled_terms(0.4)
    u_obs = _evaluate(jno.fem(terms).solve())
    kc = _param("kc", 0.7)
    terms, (a, b) = _coupled_terms(jno.np.reshape(kc, ()))
    fem = jno.fem(terms)
    assert len(fem.blocks) == 2 and fem.operator.is_parametric
    stag = lambda: jno.solve.staggered([a, b], rtol=1e-11, atol=1e-12)  # noqa: E731

    u_st = _evaluate(fem.solve(nonlinear=stag()))
    u_nw = _evaluate(fem.solve())
    assert float(jnp.max(jnp.abs(u_st - u_nw))) < 1e-8 * float(jnp.max(jnp.abs(u_nw)))

    loss = (fem.solve(nonlinear=stag()) - u_obs).mse
    g, loss0 = _crux_gradient(loss, kc, _read_param(kc), lr=1.0)
    fd = _central_fd(loss, kc, _const, 0.7, 1e-5)
    assert loss0 > 0 and abs(g) > 0
    assert _rel(g, fd) < 1e-5, f"crux {g:.10e} vs FD {fd:.10e} (rel {_rel(g, fd):.2e})"


def _history_terms(k, n=4):
    """A load-path march whose state counts the steps: ``s.evolves(s.i(-1) + 1)``, so step n solves
    ``k * (-lap u) = 1 + n``. The march is linear in u and the state, so the trajectory at ``k`` is
    exactly the trajectory at 1 divided by k."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.3).domain(tau=(0.0, 1.0, n))
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    return [k * inner(grad(u, X), grad(phi, X), 1) - (1.0 + s.i(-1)) * phi, s.evolves(s.i(-1) + 1.0), u(*cw) - 0.0]


def test_history_march_parameter_gradient_through_crux_is_exact():
    """A trainable stiffness in a ``tau=`` history march (``.i(-1)`` + ``.evolves``), differentiated by
    jno.core through the whole march. Oracle in closed form: ``traj(k) = traj(1)/k`` exactly, so for
    ``L = mean((traj(k) - traj(1))^2)`` the gradient is ``mean(2 (T/k - T)(-T/k^2))``."""
    T = np.asarray(_evaluate(jno.fem(_history_terms(1.0)).solve()))
    assert T.shape[0] == 4 and np.abs(T).max() > 1e-3
    k = _param("k_hist", 1.6)
    fem = jno.fem(_history_terms(jno.np.reshape(k, ())))
    assert fem.operator.is_parametric
    loss = (fem.solve() - jnp.asarray(T)).mse

    g, loss0 = _crux_gradient(loss, k, _read_param(k), lr=1.0)
    kv = 1.6
    assert _rel(loss0, float(np.mean((T / kv - T) ** 2))) < 1e-8, "the march at k is not traj(1)/k"
    exact = float(np.mean(2.0 * (T / kv - T) * (-T / kv**2)))
    assert _rel(g, exact) < 1e-7, f"crux {g:.10e} vs closed form {exact:.10e} (rel {_rel(g, exact):.2e})"


# ----------------------------------------------------------------------------------------------------
# 8. a network coefficient in jno.fdm
# ----------------------------------------------------------------------------------------------------
def _fdm_problem(k_of_x):
    """Strong form ``-k(x) lap u = 1``, u = 0 on the walls, on an unstructured 2-D mesh."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    return jno.fdm([-k_of_x(x, y) * (ui.xx + ui.yy) - 1.0, u(xb, yb) - 0.0])


@pytest.mark.parametrize("kind", ["parameter", "network"])
def test_fdm_coefficient_gradient_through_crux_matches_fd(kind):
    """A trainable coefficient of a ``jno.fdm`` strong form -- a scalar ``jno.np.parameter`` (the
    covered case, kept as the control) and a ``jno.nn`` network leaf (the gap). The solve must be a
    deferred node and the gradient jno.core takes must equal central FD of the same loss."""
    u_obs = _evaluate(_fdm_problem(lambda x, y: 1.0 + 0.5 * x).solve())
    if kind == "parameter":
        tr = _param("k_fdm", 1.3)
        coeff, read, make_init, x0 = (lambda x, y: tr), _read_param(tr), _const, 1.3
    else:
        tr = _affine_net(1.3, 0.0)
        coeff, read, make_init, x0 = (lambda x, y: tr(x, y)), _read_leaf(tr, "b"), (lambda b: _affine(1.3, b)), 0.0
    tr.optimizer(optax.sgd(1.0))  # jno.fdm reads "trainable" off an attached optimizer, so attach it first
    node = _fdm_problem(coeff).solve()
    assert isinstance(node, Placeholder), (
        f"a trainable {kind} (optimizer attached) inside jno.fdm was solved eagerly at its stored value: "
        "the result is a constant array, so jno.core can never train it"
    )
    loss = (node - u_obs).mse

    g, loss0 = _crux_gradient(loss, tr, read, lr=1.0)
    fd = _central_fd(loss, tr, make_init, x0, 1e-5)
    assert loss0 > 0 and abs(g) > 0
    assert _rel(g, fd) < 1e-6, f"{kind}: crux {g:.10e} vs FD {fd:.10e} (rel {_rel(g, fd):.2e})"


def test_fdm_net_coefficient_recovered_via_crux():
    """Recover ``k(x) = 1 + 0.5 x`` (both network leaves) from the field it produces, through the FDM
    solve node and jno.core, starting from ``k = 1.3``."""
    u_obs = _evaluate(_fdm_problem(lambda x, y: 1.0 + 0.5 * x).solve())
    net = _affine_net(1.3, 0.0)
    net.optimizer(optax.adam(2e-2))
    node = _fdm_problem(lambda x, y: net(x, y)).solve()
    assert isinstance(node, Placeholder), "the FDM solve froze the trainable network at its stored weights"
    crux = jno.core([(node - u_obs).mse], domain=_DUMMY)
    crux.solve(300)
    m = crux.eval([ModelWeights(net)])
    assert abs(float(m.a) - 1.0) < 1e-2 and abs(float(m.b) - 0.5) < 2e-2, (
        f"recovered a={float(m.a):.4f}, b={float(m.b):.4f}"
    )
