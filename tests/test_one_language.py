"""One language, several front doors: the SAME written terms must mean the same problem on every path.

jNO's claim (docs/concepts.md, docs/fem/formulations.md "The trial may be a network", docs/index.md) is that
a problem is written once as traced math, and only the TRIAL changes between paths:

* weak form   -> ``jno.fem([...])`` with an FE trial ``u.bind(...)``  or  a network trial ``net(x, y)`` (VPINN);
* strong form -> ``jno.fdm([...])`` with a grid unknown ``d.unknown()`` or ``jno.core([res.mse])`` with a network (PINN).

The sharpest test of "the same terms" needs no training. Take a field ``u*`` that each discretisation represents
EXACTLY (affine for P1 test spaces; quadratic for second-order central differences), and hand it to both trials:
the FE trial as its nodal interpolant, the network trial as a tiny module that computes ``u*`` in closed form.
Both then see identical values and gradients at every quadrature point / node, so their residuals must agree to
round-off, whatever the nonlinearity. Two known, documented differences are accounted for, not hidden:

* the network-trial (VPINN) residual is divided by the nodal area ``∫|φ_i|`` (a loss scaling; ``= ∫φ_i`` for the
  non-negative P1 basis the comparisons use), and its Dirichlet test functions are masked to zero rather than
  replaced by constraint rows;
* FDM evaluates boundary nodes with one-sided stencils, and those rows are replaced by the boundary conditions
  in any solve, so the strong-form comparison is on interior nodes.

The slow tests at the end check the same claim at the level users see it: trained networks and solved
discretisations of one written problem agree with each other and with the exact solution.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("foundax", reason="foundax provides the MLPs of the trained tests")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

n = jno.np
grad, inner, jac, trace = n.grad, n.inner, n.jacobian, n.trace


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# --------------------------------------------------------------------------------------------------
# closed-form "networks": exact fields as jNO models. Shape-preserving on purpose: jNO calls a model on
# (N, 1) coordinate columns and also point by point for derivatives.
# --------------------------------------------------------------------------------------------------
class Affine2(eqx.Module):
    c: jnp.ndarray  # u = c0 + c1 x + c2 y

    def __call__(self, x, y):
        return self.c[0] + self.c[1] * x + self.c[2] * y


class Affine3(eqx.Module):
    c: jnp.ndarray  # u = c0 + c1 x + c2 y + c3 z

    def __call__(self, x, y, z):
        return self.c[0] + self.c[1] * x + self.c[2] * y + self.c[3] * z


class AffineVec2(eqx.Module):
    a: jnp.ndarray  # (2, 3): component k = a[k,0] + a[k,1] x + a[k,2] y

    def __call__(self, x, y):
        cols = [self.a[k, 0] + self.a[k, 1] * x + self.a[k, 2] * y for k in range(2)]
        return jnp.concatenate(cols, axis=-1)


class Quad2(eqx.Module):
    c: jnp.ndarray  # u = c0 + c1 x + c2 y + c3 x² + c4 xy + c5 y²

    def __call__(self, x, y):
        c = self.c
        return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y


def model(module):
    net = jno.nn.wrap(module)
    net.dtype(jnp.float64)
    return net


A2 = np.array([0.4, 1.3, -0.8])  # a generic affine field: none of its terms vanish
Q2 = np.array([1.0, 2.0, -1.0, 0.7, -0.4, 0.5])


def affine2(p, c=A2):
    return c[0] + c[1] * p[:, 0] + c[2] * p[:, 1]


def quad2(p, c=Q2):
    x, y = p[:, 0], p[:, 1]
    return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y


def on_edge(p, *edges):
    """Nodes of the unit square on the named edges ("left", "right", "bottom", "top"; default: all)."""
    edges = edges or ("left", "right", "bottom", "top")
    test = {
        "left": np.abs(p[:, 0]) < 1e-9,
        "right": np.abs(p[:, 0] - 1) < 1e-9,
        "bottom": np.abs(p[:, 1]) < 1e-9,
        "top": np.abs(p[:, 1] - 1) < 1e-9,
    }
    return np.logical_or.reduce([test[e] for e in edges])


# --------------------------------------------------------------------------------------------------
# the two weak-form residuals
# --------------------------------------------------------------------------------------------------
def fe_residual(fem, uvec):
    """The FE-trial residual at ``uvec`` (the operator's own rows; Dirichlet rows are constraint rows)."""
    uvec = jnp.asarray(uvec)
    if fem.is_linear:
        A, b = fem.operator
        return np.asarray(A @ uvec - jnp.asarray(b).reshape(-1))
    return np.asarray(fem.residual(uvec)).reshape(-1)


def network_residual(pde, d):
    """The network-trial residual vector, as ``jno.core`` evaluates it (the VPINN loss is its ``.mse``)."""
    return np.asarray(jno.core([pde.mse], domain=d).eval([pde])).reshape(-1)


def lumped_area(d, xi, yi, zi=None, order=1):
    """∫ φ_i over the domain, per basis function: for P1 (φ_i >= 0) the ∫|φ_i| the network-trial residual carries."""
    u, phi = d.fem_symbols(order=order)
    kw = dict(x=xi, y=yi) if zi is None else dict(x=xi, y=yi, z=zi)
    _, b = jno.fem([u.bind(**kw) * phi.bind(**kw) - 1.0 * phi.bind(**kw)]).operator
    return np.asarray(b).reshape(-1)


def assert_same_residual(R_fe, R_nn, area, free, *, ncomp=1):
    """R_nn = R_fe / ∫|φ_i| on every free row, to round-off relative to the residual's own size."""
    R_fe = R_fe.reshape(-1, ncomp)[free]
    R_nn = R_nn.reshape(-1, ncomp)[free]
    scaled = R_fe / (area[free][:, None] + 1e-12)  # jNO adds 1e-12 to the area (trace_evaluator)
    scale = max(np.abs(scaled).max(), 1e-300)
    assert scale > 1e-6, "the residual must be non-trivial for the comparison to mean anything"
    err = np.abs(R_nn - scaled).max() / scale
    assert err < 1e-11, f"FE-trial and network-trial residuals of the same terms differ: rel {err:.2e}"


# ==================================================================================================
# 1. Weak form: FE trial vs network trial, the same terms
#
# The forms carry a Robin term on the whole boundary and NO essential condition: an essential condition
# projects the FE trial's boundary DOFs onto its value before the residual is evaluated, and the network
# trial only accepts the zero value (documented), so with one the two trials would not see the same field.
# The essential condition gets its own test below.
# ==================================================================================================
def _square(size=0.25):
    d = jno.shape.rect(0, 0, 1, 1, size=size).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    return d, xi, yi, xb, yb


def _compare_weak(d, xi, yi, xb, yb, volume, *, order=1):
    """Build the same terms with both trials, evaluate at the exact affine field, compare every row."""
    u, phi = d.fem_symbols(order=order)
    vi, vb = phi.bind(x=xi, y=yi), phi.bind(x=xb, y=yb)
    net = model(Affine2(c=jnp.asarray(A2)))

    def terms(w_in, w_bd):
        return [volume(w_in, vi), 1.0 * w_bd * vb]

    fem = jno.fem(terms(u.bind(x=xi, y=yi), u.bind(x=xb, y=yb)))
    pde = jno.fem(terms(net(xi, yi), net(xb, yb)))
    pts = np.asarray(fem.field_points[0])
    everything = np.ones(len(pts), bool)
    assert_same_residual(fe_residual(fem, affine2(pts)), network_residual(pde, d), lumped_area(d, xi, yi), everything)
    return fem


def test_poisson_with_a_coordinate_source():
    d, xi, yi, xb, yb = _square()
    f = n.sin(3 * xi) * yi
    _compare_weak(d, xi, yi, xb, yb, lambda w, v: grad(w, xi) * grad(v, xi) + grad(w, yi) * grad(v, yi) - f * v)


def test_nonlinear_reaction_variable_coefficient_and_advection():
    """u³, exp(u), a coordinate-dependent diffusivity and an advection term, all in one form."""
    d, xi, yi, xb, yb = _square()
    k = 1.0 + xi * yi
    f = n.cos(2 * yi) + xi

    def volume(w, v):
        diffusion = k * (grad(w, xi) * grad(v, xi) + grad(w, yi) * grad(v, yi))
        return diffusion + (w**3 + 0.1 * n.exp(w)) * v + (1.0 + xi) * grad(w, xi) * v - f * v

    fem = _compare_weak(d, xi, yi, xb, yb, volume)
    assert not fem.is_linear


def test_a_flux_on_one_edge_and_robin_elsewhere():
    """A prescribed flux on the right edge next to the Robin term: both boundary channels of the residual."""
    d, xi, yi, xb, yb = _square()
    d.tag("right", lambda x, y: x > 1 - 1e-9)
    xr, yr, _ = d.variable("right", split=True)
    u, phi = d.fem_symbols()
    vi, vb, vr = phi.bind(x=xi, y=yi), phi.bind(x=xb, y=yb), phi.bind(x=xr, y=yr)
    net = model(Affine2(c=jnp.asarray(A2)))

    def terms(w_in, w_bd):
        return [grad(w_in, xi) * grad(vi, xi) + grad(w_in, yi) * grad(vi, yi) - 1.0 * vi, 2.0 * w_bd * vb, -(1.0 + yr) * vr]

    fem = jno.fem(terms(u.bind(x=xi, y=yi), u.bind(x=xb, y=yb)))
    pde = jno.fem(terms(net(xi, yi), net(xb, yb)))
    pts = np.asarray(fem.field_points[0])
    assert_same_residual(
        fe_residual(fem, affine2(pts)), network_residual(pde, d), lumped_area(d, xi, yi), np.ones(len(pts), bool)
    )


def _vector_case(coupling):
    d, xi, yi, xb, yb = _square()
    u, phi = d.fem_symbols(value_shape=(2,))
    X = [xi, yi]
    vi, vb = phi.bind(x=xi, y=yi), phi.bind(x=xb, y=yb)
    a = np.array([[0.3, 1.1, -0.6], [-0.2, 0.5, 0.9]])

    def terms(w, w_bd):
        J, Jv = jac(w, X), jac(vi, X)
        vol = inner(J, Jv, 2) + 2.0 * trace(J) * trace(Jv) - (1.0 + xi) * vi[1]
        return [vol + coupling(w, vi), inner(w_bd, vb, 1)]

    net = model(AffineVec2(a=jnp.asarray(a)))
    fem = jno.fem(terms(u.bind(x=xi, y=yi), u.bind(x=xb, y=yb)))
    pde = jno.fem(terms(net(xi, yi), net(xb, yb)))
    pts = np.asarray(fem.field_points[0])
    uvec = np.stack([a[k, 0] + a[k, 1] * pts[:, 0] + a[k, 2] * pts[:, 1] for k in range(2)], axis=1).reshape(-1)
    everything = np.ones(len(pts), bool)
    assert_same_residual(fe_residual(fem, uvec), network_residual(pde, d), lumped_area(d, xi, yi), everything, ncomp=2)


def test_vector_field_laplacian_and_grad_div():
    """A two-component field: vector Laplacian and grad-div (through the trace of the Jacobian)."""
    _vector_case(lambda w, v: 0.0 * v[0])


@pytest.mark.parametrize("spelling", ["ellipsis", "view"])
def test_vector_field_cross_component_coupling(spelling):
    """A component of the field in another component's equation, ``u[..., 1] * v[0]`` or ``u.vector[1] * v[0]``:
    the portable spellings (a bare ``u[1]`` indexes the POINT axis of a network's (points, components)
    output, as NumPy does). Both trials must give the same term; a network's component used to drop its axis
    and fail to broadcast against the test component."""
    pick = (lambda w: w[..., 1]) if spelling == "ellipsis" else (lambda w: w.vector[1])
    _vector_case(lambda w, v: -pick(w) * v[0])


def test_a_gradient_component_is_written_with_an_ellipsis():
    """``grad(u, [x, y])[..., 0]`` is the x-derivative on any trial, and gives exactly the ``ui.x`` solution. A bare
    ``grad(u, [x, y])[0]`` indexes the point axis, as in NumPy; on FE symbols it is refused at build time,
    naming the component spelling, rather than dying in the assembler on a raw broadcast error."""
    d, xi, yi, xb, yb = _square(0.2)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    g, gv = grad(ui, [xi, yi]), grad(vi, [xi, yi])
    ref = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0]).solve())
    got = np.asarray(jno.fem([g[..., 0] * gv[..., 0] + g[..., 1] * gv[..., 1] - 1.0 * vi, u(xb, yb) - 0.0]).solve())
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match=r"POINT axis.*\[\.\.\., i\]"):
        jno.fem([g[0] * gv[0] + g[1] * gv[1] - 1.0 * vi, u(xb, yb) - 0.0])


def test_three_dimensional_nonlinear_form():
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.4).domain()
    xi, yi, zi, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols()
    vi, vb = phi.bind(x=xi, y=yi, z=zi), phi.bind(x=xb, y=yb, z=zb)
    c = np.array([0.2, 1.0, -0.5, 0.7])

    def terms(w, w_bd):
        lap = grad(w, xi) * grad(vi, xi) + grad(w, yi) * grad(vi, yi) + grad(w, zi) * grad(vi, zi)
        return [lap + w**3 * vi - (1.0 + zi) * vi, 1.0 * w_bd * vb]

    net = model(Affine3(c=jnp.asarray(c)))
    fem = jno.fem(terms(u.bind(x=xi, y=yi, z=zi), u.bind(x=xb, y=yb, z=zb)))
    pde = jno.fem(terms(net(xi, yi, zi), net(xb, yb, zb)))
    pts = np.asarray(fem.field_points[0])
    uvec = c[0] + pts @ c[1:]
    assert_same_residual(
        fe_residual(fem, uvec), network_residual(pde, d), lumped_area(d, xi, yi, zi), np.ones(len(pts), bool)
    )


def test_an_essential_condition_differs_only_in_its_value():
    """The documented exception: on the network trial an essential term only DECLARES which test functions
    vanish (a non-zero value is refused), while the FE trial imposes its value. With the FE value set to the
    exact field's trace, the two residuals agree on every row whose test function does not vanish."""
    d, xi, yi, xb, yb = _square()
    u, phi = d.fem_symbols()
    vi = phi.bind(x=xi, y=yi)
    f = n.sin(3 * xi) * yi
    g = A2[0] + A2[1] * xb + A2[2] * yb

    def volume(w):
        return grad(w, xi) * grad(vi, xi) + grad(w, yi) * grad(vi, yi) + w**3 * vi - f * vi

    fem = jno.fem([volume(u.bind(x=xi, y=yi)), u(xb, yb) - g])
    net = model(Affine2(c=jnp.asarray(A2)))
    pde = jno.fem([volume(net(xi, yi)), u(xb, yb) - 0.0])
    pts = np.asarray(fem.field_points[0])
    assert_same_residual(fe_residual(fem, affine2(pts)), network_residual(pde, d), lumped_area(d, xi, yi), ~on_edge(pts))

    with pytest.raises(Exception, match="(?i)essential|dirichlet|value"):
        network_residual(jno.fem([volume(net(xi, yi)), u(xb, yb) - g]), d)


def test_the_exact_solution_is_a_root_of_the_network_loss_with_p2_test_functions():
    """With a quadratic manufactured solution and f = -Δu* + u*³ written out, u* is a root of the weak form
    for P2 test functions (every integrand is a polynomial the quadrature integrates exactly). So the
    network-trial residual of the exact network must vanish, as the FE-trial residual does.

    It did not while the network-trial residual divided every row by ∫φ_i: P2 vertex basis functions integrate
    to zero on a triangle, so round-off in those rows was multiplied by ~1e12. It now divides by ∫|φ_i| > 0."""
    d, xi, yi, xb, yb = _square(0.3)
    u, phi = d.fem_symbols(order=2)
    vi = phi.bind(x=xi, y=yi)
    c = Q2
    us = lambda x, y: c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y  # noqa: E731
    f = -(2 * c[3] + 2 * c[5]) + us(xi, yi) ** 3

    def volume(w):
        return grad(w, xi) * grad(vi, xi) + grad(w, yi) * grad(vi, yi) + w**3 * vi - f * vi

    fem = jno.fem([volume(u.bind(x=xi, y=yi)), u(xb, yb) - us(xb, yb)])
    pts = np.asarray(fem.field_points[0])
    free = ~on_edge(pts)
    R_fe = fe_residual(fem, quad2(pts))[free]
    assert np.abs(R_fe).max() < 1e-10, "u* must be a root of the FE-trial form (the oracle itself)"

    R_nn = network_residual(jno.fem([volume(model(Quad2(c=jnp.asarray(c)))(xi, yi)), u(xb, yb) - 0.0]), d)[free]
    assert np.all(np.isfinite(R_nn))
    assert np.abs(R_nn).max() < 1e-8, (
        f"the exact solution is not a root of the network loss: max |R| = {np.abs(R_nn).max():.2e}"
    )


# ==================================================================================================
# 2. Strong form: FDM unknown vs PINN network, the same terms
# ==================================================================================================
def _grid(m=8):
    d = jno.shape.rect(0, 0, 1, 1, size=1.0 / m).structured().domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    return d, xi, yi, xb, yb


def _strong(w, xi, yi, f):
    return -w.xx - w.yy + w**3 + (1.0 + xi) * w.x - f


def test_fdm_and_pinn_residuals_agree_at_the_nodes():
    """Central differences are exact on quadratics, so the FDM residual of u* and the PINN residual of an
    exact network are the same numbers at every interior node."""
    d, xi, yi, xb, yb = _grid()
    f = n.sin(3 * xi) * yi
    w = d.unknown()
    fdm = jno.fdm([_strong(w.bind(x=xi, y=yi), xi, yi, f), w(xb, yb) - 0.0])
    pts = np.asarray(d.built_mesh.points)[:, :2]
    R_fd = np.asarray(fdm._pde_residual_fn()(jnp.asarray(quad2(pts)))).reshape(-1)

    res = _strong(model(Quad2(c=jnp.asarray(Q2)))(xi, yi).scalar.bind(x=xi, y=yi), xi, yi, f)
    crux = jno.core([res.mse], domain=d)
    R_pinn = np.asarray(crux.eval([res])).reshape(-1)
    at = np.stack([np.asarray(v).reshape(-1) for v in crux.eval([xi, yi])], axis=1)
    np.testing.assert_allclose(at, pts, atol=1e-12, err_msg="PINN collocation points must be the FDM nodes")

    inside = ~on_edge(pts)
    scale = np.abs(R_pinn[inside]).max()
    assert scale > 1.0
    assert np.abs(R_fd[inside] - R_pinn[inside]).max() / scale < 1e-12


def test_fdm_reproduces_a_quadratic_exactly_and_it_is_a_pinn_root():
    """Manufactured: f = -Δu* + u*³ + (1+x) ∂u*/∂x with u* quadratic and Dirichlet data u*. FDM is exact on
    quadratics, so its SOLUTION is u* at the nodes; the same terms with the exact network have zero residual."""
    d, xi, yi, xb, yb = _grid()
    c = Q2
    us = lambda x, y: c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y  # noqa: E731
    ux = c[1] + 2 * c[3] * xi + c[4] * yi
    f = -(2 * c[3] + 2 * c[5]) + us(xi, yi) ** 3 + (1.0 + xi) * ux
    w = d.unknown()
    sol = np.asarray(jno.fdm([_strong(w.bind(x=xi, y=yi), xi, yi, f), w(xb, yb) - us(xb, yb)]).solve()).reshape(-1)
    pts = np.asarray(d.built_mesh.points)[:, :2]
    assert np.abs(sol - quad2(pts)).max() < 1e-9

    res = _strong(model(Quad2(c=jnp.asarray(c)))(xi, yi).scalar.bind(x=xi, y=yi), xi, yi, f)
    assert float(np.abs(np.asarray(jno.core([res.mse], domain=d).eval([res]))).max()) < 1e-11


def test_heat_equation_fdm_march_is_exact_and_the_space_time_pinn_residual_vanishes():
    """u* = (1 + t)(x² + y²)/4 solves u_t - Δu = f with f = (x² + y²)/4 - (1 + t): linear in t (backward Euler is
    exact) and quadratic in space (central differences are exact). The same term, u.t - u.xx - u.yy - f, is the
    FDM march and the space-time PINN residual."""
    d = jno.shape.rect(0, 0, 1, 1, size=1 / 8).structured().domain(time=(0.0, 0.5, 6))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    x0, y0, t0 = d.variable("initial", split=True)
    us = lambda x, y, t: (1 + t) * (x * x + y * y) / 4  # noqa: E731
    f = (xi * xi + yi * yi) / 4 - (1 + ti)

    def heat(w):
        return w.t - w.xx - w.yy - f

    w = d.unknown()
    traj = np.asarray(
        jno.fdm([heat(w.bind(x=xi, y=yi, t=ti)), w(xb, yb, tb) - us(xb, yb, tb), w(x0, y0, t0) - us(x0, y0, t0)]).solve()
    )
    pts = np.asarray(d.built_mesh.points)[:, :2]
    final = traj.reshape(traj.shape[0], -1)[-1]
    assert np.abs(final - us(pts[:, 0], pts[:, 1], 0.5)).max() < 1e-9

    class SpaceTime(eqx.Module):
        s: jnp.ndarray

        def __call__(self, x, y, t):
            return self.s * (1 + t) * (x * x + y * y) / 4

    net = model(SpaceTime(s=jnp.asarray(1.0)))
    res = heat(net(xi, yi, ti).scalar.bind(x=xi, y=yi, t=ti))
    assert float(np.abs(np.asarray(jno.core([res.mse], domain=d).eval([res]))).max()) < 1e-12


@pytest.mark.parametrize("path", ["fdm", "fem"])
def test_an_initial_condition_may_mention_the_time_coordinate(path):
    """``u(x0, y0, t0) - t0`` must start the march from the start time, here 0.3 (so neither 0 nor x passes).

    Writing a manufactured u*(x, y, t) once and using it for the boundary AND the initial condition is the
    common pattern. Both paths used to evaluate the initial value with spatial points only
    (``jno._fem._eval_value_node_at``), so ``t0`` read the x column and the start state was x, silently."""
    r = jno.shape.rect(0, 0, 1, 1, size=1 / 4)
    d = (r.structured() if path == "fdm" else r).domain(time=(0.3, 0.8, 3))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    x0, y0, t0 = d.variable("initial", split=True)
    if path == "fdm":
        w = d.unknown()
        wi = w.bind(x=xi, y=yi, t=ti)
        traj = np.asarray(jno.fdm([wi.t - wi.xx - wi.yy, w(xb, yb, tb) - 0.0, w(x0, y0, t0) - t0]).solve())
    else:
        u, v = d.fem_symbols()
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        out = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb, tb) - 0.0, u(x0, y0, t0) - t0]).solve()
        traj = np.asarray(out.fn() if hasattr(out, "fn") else out)
    start = traj.reshape(traj.shape[0], -1)[0]
    assert np.abs(start - 0.3).max() < 1e-12, f"the start state is not t0 = 0.3 everywhere: {start[:4]}"


# ==================================================================================================
# 3. The same problem through every front door, solved and trained (slow)
# ==================================================================================================
@pytest.mark.parametrize("derivative", ["first", "second"])
def test_a_pinn_residual_on_a_domain_that_already_carries_a_fem_problem(derivative):
    """docs/concepts.md: a PINN residual, a FEM solve and a data term can sit in one ``jno.core``. Building a
    ``jno.fem`` problem retags the term's coordinate Variables to its quadrature pool, in place; the same
    Variables in a PINN residual afterwards failed (KeyError 'fem_gauss' for a first derivative,
    AttributeError for a second). The PINN residual must be the same before and after the FEM problem exists."""
    import foundax

    d, xi, yi, xb, yb = _square(0.2)
    net = jno.nn(foundax.mlp(2, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
    up = (net(xi, yi) * (xi * (1 - xi) * yi * (1 - yi))).scalar.bind(x=xi, y=yi)
    res = up.x - 1.0 if derivative == "first" else -(up.xx + up.yy) - 1.0
    before = np.asarray(jno.core([res.mse], domain=d).eval([res])).reshape(-1)

    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])  # building it is enough

    after = np.asarray(jno.core([res.mse], domain=d).eval([res])).reshape(-1)
    assert np.all(np.isfinite(before)) and before.size > 1
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=0.0, err_msg="the PINN moved off its points")


@pytest.mark.slow
def test_fem_vpinn_fdm_and_pinn_agree_on_one_problem():
    """-Δu = 2π² sin πx sin πy, u = 0 on the boundary; exact u = sin πx sin πy. The weak form goes to the FE
    trial and to the network trial; the strong form to FDM and to a PINN. All four must reach the exact
    solution, and so each other, within their discretisation / training error."""
    import foundax
    import optax

    PI = np.pi
    exact = lambda p: np.sin(PI * p[:, 0]) * np.sin(PI * p[:, 1])  # noqa: E731

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    # weak form
    d, xi, yi, xb, yb = _square(0.08)
    f = 2 * PI**2 * n.sin(PI * xi) * n.sin(PI * yi)
    u, phi = d.fem_symbols()
    vi = phi.bind(x=xi, y=yi)

    def weak(w):
        return [grad(w, xi) * grad(vi, xi) + grad(w, yi) * grad(vi, yi) - f * vi, u(xb, yb) - 0.0]

    fem = jno.fem(weak(u.bind(x=xi, y=yi)))
    pts = np.asarray(fem.field_points[0])
    e_fem = rel(np.asarray(fem.solve()).reshape(-1), exact(pts))

    key = jax.random.PRNGKey(0)
    net = jno.nn(foundax.mlp(2, hidden_dims=32, num_layers=3, activation=jax.nn.tanh, key=key))
    net.optimizer(optax.adam(optax.exponential_decay(3e-3, 500, 0.5, end_value=1e-5)))
    trial = net(xi, yi) * (xi * (1 - xi) * yi * (1 - yi))
    vpinn = jno.core([jno.fem(weak(trial)).mse], domain=d)
    vpinn.solve(3000)
    u_vp = np.asarray(vpinn.eval([trial])).reshape(-1)
    xy = np.stack([np.asarray(v).reshape(-1) for v in vpinn.eval([xi, yi])], axis=1)
    e_vp = rel(u_vp, exact(xy))

    # strong form
    g, gx, gy, gbx, gby = _grid(24)
    fs = 2 * PI**2 * n.sin(PI * gx) * n.sin(PI * gy)

    def strong(w):
        return -w.xx - w.yy - fs

    w = g.unknown()
    gpts = np.asarray(g.built_mesh.points)[:, :2]
    e_fdm = rel(np.asarray(jno.fdm([strong(w.bind(x=gx, y=gy)), w(gbx, gby) - 0.0]).solve()).reshape(-1), exact(gpts))

    net2 = jno.nn(foundax.mlp(2, hidden_dims=32, num_layers=3, activation=jax.nn.tanh, key=key))
    net2.optimizer(optax.adam(optax.exponential_decay(3e-3, 500, 0.5, end_value=1e-5)))
    up = (net2(gx, gy) * (gx * (1 - gx) * gy * (1 - gy))).scalar.bind(x=gx, y=gy)
    pinn = jno.core([strong(up).mse], domain=g)
    pinn.solve(3000)
    e_pinn = rel(np.asarray(pinn.eval([up])).reshape(-1), exact(gpts))

    errs = {"FEM": e_fem, "VPINN": e_vp, "FDM": e_fdm, "PINN": e_pinn}
    assert e_fem < 1e-2 and e_fdm < 1e-2, errs
    assert e_vp < 3e-2 and e_pinn < 3e-2, errs


@pytest.mark.slow
def test_one_trainable_is_recovered_through_the_fem_solve_and_through_a_pinn():
    """The same unknown coefficient k in -k Δu = f, from the same observations of u, recovered two ways: through
    the differentiable FEM solve, and as a PINN inverse problem (network + k, residual + data loss)."""
    import foundax
    import optax

    PI = np.pi
    k_true = 2.5
    exact = lambda x, y: np.sin(PI * x) * np.sin(PI * y) / k_true  # noqa: E731
    const = lambda v: lambda *a, **kw: jnp.array([v])  # noqa: E731

    d, xi, yi, xb, yb = _square(0.1)
    f = 2 * PI**2 * n.sin(PI * xi) * n.sin(PI * yi)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    pts = np.asarray(d.built_mesh.points)[:, :2]

    k1 = jno.np.parameter((1,), name="k_fem").initialize(const(1.0))
    k1.optimizer(optax.adam(5e-2))
    fem = jno.fem([k1 * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
    u_obs_nodes = jnp.asarray(exact(pts[:, 0], pts[:, 1]))
    crux = jno.core([(fem.solve() - u_obs_nodes).mse], domain=d)
    crux.solve(400)
    k_fem = float(np.asarray(crux.eval([k1])).reshape(-1)[0])

    # the PINN gets its own domain object: a PINN collocated on a domain that already carries a FEM problem
    # fails (see test_a_pinn_residual_on_a_domain_that_already_carries_a_fem_problem)
    d, xi, yi, xb, yb = _square(0.1)
    f = 2 * PI**2 * n.sin(PI * xi) * n.sin(PI * yi)
    k2 = jno.np.parameter((1,), name="k_pinn").initialize(const(1.0))
    k2.optimizer(optax.adam(1e-2))
    net = jno.nn(foundax.mlp(2, hidden_dims=32, num_layers=3, activation=jax.nn.tanh, key=jax.random.PRNGKey(1)))
    net.optimizer(optax.adam(optax.exponential_decay(3e-3, 500, 0.5, end_value=1e-5)))
    up = (net(xi, yi) * (xi * (1 - xi) * yi * (1 - yi))).scalar.bind(x=xi, y=yi)
    data = n.sin(PI * xi) * n.sin(PI * yi) / k_true
    pinn = jno.core([(-k2 * (up.xx + up.yy) - f).mse, 100.0 * (up - data).mse], domain=d)
    pinn.solve(3000)
    k_pinn = float(np.asarray(pinn.eval([k2])).reshape(-1)[0])

    assert abs(k_fem - k_true) < 0.05 * k_true, f"through the FEM solve: k = {k_fem:.4f}"
    assert abs(k_pinn - k_true) < 0.05 * k_true, f"through the PINN: k = {k_pinn:.4f}"
