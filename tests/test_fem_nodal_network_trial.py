"""A neural operator's output declared a field on the mesh -- ``u = net(rho).field.bind(x=xi, y=yi)`` -- as
the trial of a ``jno.fem`` weak form.

An operator network (an FNO on the mesh's grid, a DeepONet at the nodes) predicts the FE coefficients of
the solution for a given input field. Written where the unknown goes (``u.x * vi.x``, exactly like the FE
trial), ``jno.fem`` reads it as the trial and returns the physics-informed loss of that prediction, the FE residual ``A u - b(rho)``, one per sample of a
batched ``S * domain``; ``.solve(precond=...)`` applies a preconditioner to it, ``P (A u - b)`` (Wen,
Mishra & Zeinhofer, arXiv:2609.36216). No solution data is involved.

The oracle is the ordinary FE solve of the same problem, written with the FE trial ``U`` and assembled
by the standard route: its matrix ``A``, load ``b(rho)`` and solution ``u* = A^-1 b`` per sample. The
"network" is that exact solution map scaled by ``w``, so for ``w = 1`` the residual must vanish and for
``w = 1/2`` it must be ``-b/2`` (and the exactly preconditioned residual ``-u*/2``) -- on every sample.
The grid is deliberately not square, so a transposed grid convention cannot pass.

Run in float64: the comparisons are against a direct solve (round-off ~1e-12).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

grad = jno.np.grad
S, NX, NY = 3, 4, 3  # samples; cells along x (length 2) and y (length 1): a (5, 4) node grid


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


class _Linear(eqx.Module):
    """``u = w * G rho + c`` on the grid: a stand-in operator network with a known output."""

    G: jax.Array
    w: jax.Array
    c: jax.Array = 0.0

    def __call__(self, rho):
        return (self.w * (self.G @ rho.reshape(-1)) + self.c).reshape(rho.shape)


def _grid_maps(points):
    """Node i sits at grid index (ix[i], iy[i]); the grid is indexed [ix, iy] ("ij", x first)."""
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    return np.searchsorted(xs, points[:, 0]), np.searchsorted(ys, points[:, 1])


def _oracle(g=0.0, n=(NX, NY), size=(2.0, 1.0)):
    """The same Poisson problem with the FE trial on a single domain: A, b(rho), and the grid maps."""
    d = jno.shape.rect(0, 0, *size).structured(n=n).quad().domain(compute_mesh_connectivity=True)
    U, V = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = U.bind(x=xi, y=yi), V.bind(x=xi, y=yi)
    f = jno.np.parameter(V, name="f")
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, U(xb, yb) - g])
    pts = np.asarray(fem.points)
    nodes = pts.shape[0]
    residual = lambda u, r: np.asarray(fem.residual(jnp.asarray(u), {"f": jnp.asarray(r)}))  # noqa: E731
    b = lambda r: -residual(np.zeros(nodes), r)  # noqa: E731   A is affine-free: r(0) = -b
    A = np.stack([residual(e, np.zeros(nodes)) + b(np.zeros(nodes)) for e in np.eye(nodes)], axis=1)
    ix, iy = _grid_maps(pts)
    return A, b, ix, iy, pts


def _problem(w, *, g=0.0, n=(NX, NY), size=(2.0, 1.0), seed=0, boundary_offset=0.0):
    """Batched domain, per-sample forcings, the network-trial form, and the oracle's answers."""
    A, b, ix, iy, pts = _oracle(g, n, size)
    nodes = pts.shape[0]
    to_grid = np.zeros((nodes, nodes))
    to_grid[np.ravel_multi_index((ix, iy), (n[0] + 1, n[1] + 1)), np.arange(nodes)] = 1.0  # node -> grid slot
    B = np.stack([b(e) for e in np.eye(nodes)], axis=1) - b(np.zeros(nodes))[:, None]  # linear part of b
    G = to_grid @ np.linalg.solve(A, B) @ to_grid.T  # the exact solution map, grid -> grid (for g = 0)

    rho = np.random.default_rng(seed).normal(size=(S, n[0] + 1, n[1] + 1, 1))
    d = S * jno.shape.rect(0, 0, *size).structured(n=n).quad().domain(compute_mesh_connectivity=True)
    U, V = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    vi = V.bind(x=xi, y=yi)
    f = d.variable("_rho", rho[:, None, None])
    on_b = np.zeros((n[0] + 1, n[1] + 1))
    on_b[[0, -1], :] = on_b[:, [0, -1]] = 1.0  # the grid's boundary nodes: garbage there must not matter
    net = jno.nn.wrap(_Linear(jnp.asarray(G), jnp.asarray(w), jnp.asarray(boundary_offset * on_b.ravel())))
    u = net(f).field.bind(x=xi, y=yi)
    pde = jno.fem([u.x * vi.x + u.y * vi.y - f * vi, U(xb, yb) - g])

    rho_nodes = rho[:, ix, iy, 0]  # (S, nodes)
    u_star = np.stack([np.linalg.solve(A, b(r)) for r in rho_nodes])
    b_all = np.stack([b(r) for r in rho_nodes])
    return pde, d, A, b_all, u_star, rho_nodes, pts


def _eval(term, d, samples=S):
    return np.asarray(jno.core([term.mse]).eval([term], domain=d)).reshape(samples, -1)


def _boundary(pts, size=(2.0, 1.0)):
    x, y = pts[:, 0], pts[:, 1]
    return np.isclose(x, 0) | np.isclose(x, size[0]) | np.isclose(y, 0) | np.isclose(y, size[1])


def test_the_exact_solution_operator_has_zero_residual_on_every_sample():
    """Breaks if the grid is mapped to the nodes transposed, if a sample sees another sample's forcing, or
    if the forcing does not reach the load at all."""
    pde, d, _A, b_all, _u, _rho, _pts = _problem(w=1.0)
    r = _eval(pde.solve(), d)
    assert np.abs(r).max() < 1e-10 * np.abs(b_all).max(), np.abs(r).max()


def test_a_wrong_prediction_has_the_fe_residual_of_that_prediction():
    """Half the exact solution leaves A u/2 - b = -b/2 on the free rows: a residual that is not the FE one
    (a missing load, another quadrature, the wrong sign) fails here even where the exact map passes."""
    pde, d, _A, b_all, _u, _rho, pts = _problem(w=0.5)
    r = _eval(pde.solve(), d)
    free = ~_boundary(pts)
    np.testing.assert_allclose(r[:, free], -0.5 * b_all[:, free], atol=1e-10 * np.abs(b_all).max())


def test_the_essential_condition_is_imposed_on_the_prediction_not_trained_for():
    """The network's boundary values never reach the loss (the essential rows are imposed exactly, like the
    hard boundary of an FE solve), and ``pde.field`` is the prediction with the prescribed values set. The
    network here is exact inside and wrong (+0.7) on the boundary. Breaks if the boundary rows stay in the
    loss as ``s (u_d - g)``, or if ``field`` keeps the network's boundary values."""
    pde, d, _A, b_all, u_star, _rho, pts = _problem(w=1.0, boundary_offset=0.7)
    on_b = _boundary(pts)
    assert np.abs(_eval(pde.solve(), d)).max() < 1e-10 * np.abs(b_all).max()
    field = _eval(pde.field, d)
    np.testing.assert_allclose(field[:, ~on_b], u_star[:, ~on_b], atol=1e-10)
    np.testing.assert_allclose(field[:, on_b], 0.0, atol=1e-12)

    pde1, d1, *_ = _problem(w=1.0, g=1.5, boundary_offset=0.7)
    np.testing.assert_allclose(_eval(pde1.field, d1)[:, on_b], 1.5, atol=1e-12)


def test_an_exact_preconditioner_turns_the_residual_into_the_error():
    """With P = A^-1 the preconditioned residual is ``u - u*`` on the free rows: the loss measures the error
    itself (the point of the paper). Breaks if P is built from another matrix or applied to another vector."""
    pde, d, A, _b, u_star, _rho, pts = _problem(w=0.5)
    e = _eval(pde.solve(precond=jno.precond.inner(jno.solve.lu())), d)
    free = ~_boundary(pts)
    np.testing.assert_allclose(e[:, free], -0.5 * u_star[:, free], atol=1e-9 * np.abs(u_star).max())


def test_a_multigrid_preconditioner_brings_the_residual_within_a_v_cycle_of_the_error():
    """One V(2,2) cycle reduces the error of A^-1 r by about 0.1 (measured 0.12 on this operator): the
    preconditioned residual is the error to that accuracy, where the plain residual is off by cond(A)."""
    pde, d, _A, _b, u_star, _rho, pts = _problem(w=0.5, n=(16, 16), size=(1.0, 1.0))
    e = _eval(pde.solve(precond=jno.precond.gmg()), d)
    free = ~_boundary(pts, (1.0, 1.0))
    err = -0.5 * u_star[:, free]
    assert np.linalg.norm(e[:, free] - err) < 0.3 * np.linalg.norm(err)


def test_a_preconditioner_on_a_data_dependent_operator_is_refused_by_name():
    """A forcing that multiplies the trial makes A differ per sample; one P built for all of them would be
    silently wrong for most, so it is refused (the plain residual stays available)."""
    d = 2 * jno.shape.rect(0, 0, 1, 1).structured(n=4).quad().domain(compute_mesh_connectivity=True)
    U, V = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    vi = V.bind(x=xi, y=yi)
    k = d.variable("_k", np.ones((2, 1, 1, 5, 5, 1)))
    u = jno.nn.wrap(_Linear(jnp.eye(25), jnp.asarray(1.0)))(k).field.bind(x=xi, y=yi)
    pde = jno.fem([k * (u.x * vi.x + u.y * vi.y) - 1.0 * vi, U(xb, yb) - 0.0])
    _eval(pde.solve(), d, samples=2)  # the plain residual is per sample and fine
    with pytest.raises(NotImplementedError, match="per sample"):
        pde.solve(precond=jno.precond.gmg())


def _small(samples=2):
    d = samples * jno.shape.rect(0, 0, 1, 1).structured(n=4).quad().domain(compute_mesh_connectivity=True)
    U, V = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = d.variable("_f", np.ones((samples, 1, 1, 5, 5, 1)))
    net = jno.nn.wrap(_Linear(jnp.eye(25), jnp.asarray(1.0)))
    return U, V.bind(x=xi, y=yi), xi, yi, xb, yb, f, net


def test_a_form_nonlinear_in_the_network_trial_is_refused_by_name():
    U, vi, xi, yi, xb, yb, f, net = _small()
    u = net(f).field.bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="linear"):
        jno.fem([u.x * vi.x + u.y * vi.y + u**3 * vi - f * vi, U(xb, yb) - 0.0])


def test_an_ad_gradient_of_a_network_of_data_is_refused_not_zero():
    """``grad(net(rho), x)`` with ``rho`` data is exactly 0 by autodiff -- the silent wrong answer the
    ``.field`` declaration exists to avoid. Breaks if the guard is removed or no longer names the fix."""
    _U, _vi, xi, _yi, _xb, _yb, f, net = _small()
    with pytest.raises(ValueError, match=r"field\.bind"):
        grad(net(f), xi)
    with pytest.raises(ValueError, match=r"field\.bind"):
        net(f).d(xi)


def test_a_network_of_the_coordinates_keeps_its_autodiff_gradient():
    """The guard must not touch a pointwise network: ``grad(net(x, y), x)`` is a genuine derivative."""
    _U, _vi, xi, yi, _xb, _yb, _f, _net = _small()

    class _XY(eqx.Module):
        def __call__(self, x, y):
            return x * y

    g = grad(jno.nn.wrap(_XY())(xi, yi), xi)  # builds without raising
    assert g is not None


def test_an_undeclared_network_of_data_as_trial_is_refused_by_name():
    """Without ``.field`` the same network, in a value term, would silently become a VPINN trial with zero
    derivatives; ``jno.fem`` refuses it and names the declaration."""
    U, vi, _xi, _yi, xb, yb, f, net = _small()
    with pytest.raises(ValueError, match=r"field\.bind"):
        jno.fem([net(f) * vi - f * vi, U(xb, yb) - 0.0])
