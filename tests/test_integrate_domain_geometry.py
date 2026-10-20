"""``expr.integrate(d)`` on a MOVING mesh: trainable node coordinates and the cell-geometry symbols.

With ``domain.variable(...).trainable()`` coordinates the integral follows the node positions (|det J|, the
quadrature points, every pushed-forward basis), so its gradient with respect to the nodes is the shape
derivative. ``d.cell_metric`` (G = J^-T J^-1) and ``d.cell_size`` resolve per cell inside it, so a mesh-quality
constraint is an integral: a p-norm of the condition number ``sqrt(tr G tr G^-1) / dim``.
"""

import numpy as np
import pytest

pytest.importorskip("gmsh")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _moving_rect():
    """a 2 x 1 rectangle with trainable x and y on every vertex, and a frozen P1 field f = 1 + x"""
    d = jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=0.25).domain()
    c = d.variable("interior", split=True)
    x, y = c[0], c[1]
    x.trainable(name="X"), y.trainable(name="Y")  # marks every vertex trainable
    f_sym, _g = d.fem_symbols(names=("f", "g"))
    P = np.asarray(d.mesh.points)
    f = f_sym.bind(x=x, y=y).freeze(1.0 + P[:, 0])
    return d, f, P


def _coords(node, P, Xv=None, Yv=None):
    """call the node's function with the coordinate inputs (the only inputs here: X then Y)"""
    Xv = P[:, 0] if Xv is None else Xv
    Yv = P[:, 1] if Yv is None else Yv
    return node.fn(jnp.asarray(Xv), jnp.asarray(Yv))


def test_the_integral_follows_the_nodes():
    """∫ f^2 over the rectangle is 26/3 at rest; stretching x by s keeps f's nodal values, so it becomes s * 26/3"""
    d, f, P = _moving_rect()
    node = (f * f).integrate(d)
    np.testing.assert_allclose(float(_coords(node, P)), 26.0 / 3.0, rtol=1e-12)
    np.testing.assert_allclose(float(_coords(node, P, Xv=1.5 * P[:, 0])), 1.5 * 26.0 / 3.0, rtol=1e-12)


def test_the_shape_derivative_matches_finite_differences():
    d, f, P = _moving_rect()
    node = (f * f).integrate(d)
    rng = np.random.default_rng(0)
    X0 = jnp.asarray(P[:, 0] + 0.02 * rng.normal(size=len(P)))
    g = np.asarray(jax.grad(lambda X: _coords(node, P, Xv=X))(X0))
    dv = rng.normal(size=len(P))
    h = 1e-6
    fd = (float(_coords(node, P, Xv=X0 + h * dv)) - float(_coords(node, P, Xv=X0 - h * dv))) / (2 * h)
    np.testing.assert_allclose(g @ dv, fd, rtol=1e-7)


def test_cell_metric_inside_the_integral():
    """∫ tr(G) dA = Σ_T |T| tr(G_T), G_T = J^-T J^-1 of each triangle (an independent host computation)"""
    d, f, P = _moving_rect()
    G = d.cell_metric
    node = jno.np.trace(G).integrate(d)
    C = np.asarray(d.mesh.cells_dict["triangle"])
    J = np.stack([P[C[:, 1], :2] - P[C[:, 0], :2], P[C[:, 2], :2] - P[C[:, 0], :2]], axis=2)
    K = np.linalg.inv(J)
    want = np.sum(0.5 * np.abs(np.linalg.det(J)) * np.einsum("tij,tij->t", K, K))
    np.testing.assert_allclose(float(_coords(node, P)), want, rtol=1e-12)


def test_a_mesh_quality_p_norm_sees_a_collapsing_cell():
    """the condition-number p-norm grows as one interior node is pushed onto its neighbour, and its gradient
    with respect to the nodes matches finite differences"""
    d, f, P = _moving_rect()
    G = d.cell_metric
    kappa = (jno.np.trace(G) * jno.np.trace(jno.np.inv(G))) ** 0.5 / 2.0
    p, area = 8.0, 2.0
    node = (kappa**p / area).integrate(d)
    Q = lambda X, Y: _coords(node, P, Xv=X, Yv=Y) ** (1.0 / p)  # noqa: E731
    inner = np.flatnonzero((P[:, 0] > 0.3) & (P[:, 0] < 1.7) & (P[:, 1] > 0.3) & (P[:, 1] < 0.7))
    i = inner[0]
    others = np.delete(np.arange(len(P)), i)
    j = others[np.argmin(np.linalg.norm(P[others] - P[i], axis=1))]
    q = []
    for t in (0.0, 0.5, 0.8):
        X = P[:, 0].copy()
        Y = P[:, 1].copy()
        X[i], Y[i] = (1 - t) * P[i, 0] + t * P[j, 0], (1 - t) * P[i, 1] + t * P[j, 1]
        q.append(float(Q(jnp.asarray(X), jnp.asarray(Y))))
    assert q[0] < q[1] < q[2], q
    rng = np.random.default_rng(1)
    X0, Y0 = jnp.asarray(P[:, 0]), jnp.asarray(P[:, 1])
    gx = np.asarray(jax.grad(lambda X: Q(X, Y0))(X0))
    dv = rng.normal(size=len(P))
    h = 1e-6
    fd = (float(Q(X0 + h * dv, Y0)) - float(Q(X0 - h * dv, Y0))) / (2 * h)
    np.testing.assert_allclose(gx @ dv, fd, rtol=1e-6)
