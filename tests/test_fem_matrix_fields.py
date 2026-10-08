"""Matrix-valued (rank-2) unknowns in a native FEM weak form.

A field declared with ``value_shape=(n, m)`` carries ``n*m`` components per node, stored row-major
(component ``(i, j)`` is flat index ``i*m + j``). Its test function is ``n*m`` basis functions per node,
each a one-hot ``(n, m)`` matrix, so ``inner(S, T, n_contract=2)``, ``trace(T)`` and ``T[i, j]`` all leave
the per-DOF component axis exactly as they do for a vector field.

Oracles (all closed form, none a restatement of the output):

* L2 projection of a polynomial tensor field the space contains -> nodally exact.
* Componentwise Laplace with a harmonic polynomial tensor as Dirichlet data -> nodally exact, in 2-D
  (P1, P2) and 3-D (P1); the same answer from a whole-tensor and from a per-entry Dirichlet.
* Steady transport ``(b . grad) S = 0`` with an inflow profile -> the profile is carried unchanged.
* Jaumann simple shear (Dienes 1979, Acta Mech. 32, 217): for u = (g y, 0) a particle entering at x = 0
  with zero stress has, after the time t = x/(g y) it took to get there, s_xy = G sin(g t),
  s_xx = -s_yy = G (1 - cos g t). The steady Eulerian form ``(u . grad) S - (W S - S W) = 2 G D`` must
  converge to it.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

J = jno.np
inner = J.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def mat2(a, b, c, d):
    """The 2x2 matrix expression [[a, b], [c, d]]."""
    return J.stack([J.stack([a, b], axis=-1), J.stack([c, d], axis=-1)], axis=-2)


def _square(n=4):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).domain(compute_mesh_connectivity=False)
    d.tag("left", lambda x, y: x < 1e-9)
    return d


def _harmonic2(order):
    """A tensor whose entries are harmonic polynomials of degree <= order (so the space contains it)."""
    if order == 1:
        return lambda X, Y: (1 + X, 2 * Y, X - Y, 3 + 0 * X)
    return lambda X, Y: (1 + X * X - Y * Y, 2 * X * Y, X - Y, 3 + X * Y)


@pytest.mark.parametrize("order", [1, 2])
def test_l2_projection_is_exact(order):
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), order=order)
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    ex = _harmonic2(order)
    a, b, c, e = ex(x, y)
    # entry-wise source: exercises T[i, j] on the test function
    fem = jno.fem([inner(Si, Ti, n_contract=2) - (a * Ti[0, 0] + b * Ti[0, 1] + c * Ti[1, 0] + e * Ti[1, 1])])
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2, 2)
    P = np.asarray(fem.points)
    E = np.stack(ex(P[:, 0], P[:, 1]), -1).reshape(-1, 2, 2)
    assert fem.dofs == 4 * P.shape[0]
    np.testing.assert_allclose(U, E, atol=1e-11)
    # the same source written as one matrix contracted with the test matrix
    fem2 = jno.fem([inner(Si, Ti, n_contract=2) - inner(mat2(a, b, c, e), Ti, n_contract=2)])
    np.testing.assert_allclose(np.asarray(fem2.solve(linear=jno.solve.lu())).reshape(-1, 2, 2), E, atol=1e-11)


def test_trace_source_projects_to_identity():
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"))
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    U = jno.fem([inner(Si, Ti, n_contract=2) - J.trace(Ti)]).solve(linear=jno.solve.lu())
    np.testing.assert_allclose(np.asarray(U).reshape(-1, 2, 2), np.broadcast_to(np.eye(2), (25, 2, 2)), atol=1e-12)


def _laplace2(order, per_entry):
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), order=order)
    ex = _harmonic2(order)
    terms = [inner(J.jacobian(S, [x, y]), J.jacobian(T, [x, y]), n_contract=3)]
    if per_entry:
        a, b, c, e = ex(xb, yb)
        Sb = S.bind(x=xb, y=yb)
        terms += [Sb[0, 0] - a, Sb[0, 1] - b, Sb[1, 0] - c, Sb[1, 1] - e]
    else:
        terms += [S(xb, yb) - mat2(*ex(xb, yb))]
    fem = jno.fem(terms)
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2, 2)
    P = np.asarray(fem.points)
    return U, np.stack(ex(P[:, 0], P[:, 1]), -1).reshape(-1, 2, 2), fem


@pytest.mark.parametrize("order", [1, 2])
def test_tensor_laplace_whole_tensor_dirichlet(order):
    U, E, _ = _laplace2(order, per_entry=False)
    np.testing.assert_allclose(U, E, atol=1e-11)


@pytest.mark.parametrize("order", [1, 2])
def test_tensor_laplace_per_entry_dirichlet(order):
    U, E, fem = _laplace2(order, per_entry=True)
    np.testing.assert_allclose(U, E, atol=1e-11)
    # entries are labelled by their flat index, never as if they were x/y/z axes
    assert {"dirichlet@boundary[0]", "dirichlet@boundary[1]", "dirichlet@boundary[2]", "dirichlet@boundary[3]"} <= set(
        fem.classification
    )


def test_single_entry_dirichlet_leaves_the_others_free():
    """Pin only S[1, 0] on the left; the projection fixes the rest. Entry (1, 0) is flat component 2 -- the
    one a vector reading would have called 'z'."""
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"))
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    fem = jno.fem([inner(Si, Ti, n_contract=2) - J.trace(Ti), S.bind(x=xl, y=yl)[1, 0] - 5.0])
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2, 2)
    P = np.asarray(fem.points)
    left = P[:, 0] < 1e-9
    np.testing.assert_allclose(U[left, 1, 0], 5.0, atol=1e-12)
    np.testing.assert_allclose(U[:, 0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(U[:, 1, 1], 1.0, atol=1e-12)
    np.testing.assert_allclose(U[:, 0, 1], 0.0, atol=1e-12)


def test_vector_dirichlet_beyond_three_components():
    """A 4-component vector field pins component 3 -- the old parser stopped at 'z'. Every component is
    pinned: an unpinned one is a pure-Neumann Laplace block, singular (the GPU LU says so; the host LU
    returned a value for it anyway)."""
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    u, v = d.fem_symbols(value_shape=(4,), names=("u", "v"))
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    pins = [u(xl, yl)[k] - g for k, g in enumerate((1.0, -1.0, 0.5, 2.0))]
    fem = jno.fem([inner(J.jacobian(u, [x, y]), J.jacobian(v, [x, y]), n_contract=2), *pins])
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 4)
    del ui, vi
    np.testing.assert_allclose(U, np.broadcast_to([1.0, -1.0, 0.5, 2.0], U.shape), atol=1e-12)


def test_matrix_row_is_not_a_dirichlet_component():
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"))
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    with pytest.raises(ValueError, match="names every index"):
        jno.fem([inner(Si, Ti, n_contract=2) - J.trace(Ti), S.bind(x=xl, y=yl)[0] - 1.0])


def test_matrix_view_indexing_and_transpose():
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"))
    Si = S.bind(x=x, y=y)
    with pytest.raises(TypeError, match="indexed by an entry"):
        Si[0, 1, 0]
    # S^T projected: the (0, 1) entry of the answer is the (1, 0) entry of the data
    Ti = T.bind(x=x, y=y)
    fem = jno.fem([inner(Si.T, Ti, n_contract=2) - (x * Ti[0, 1] + 2.0 * Ti[1, 0])])
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2, 2)
    P = np.asarray(fem.points)
    np.testing.assert_allclose(U[:, 1, 0], P[:, 0], atol=1e-11)
    np.testing.assert_allclose(U[:, 0, 1], 2.0, atol=1e-11)


@pytest.mark.parametrize("order", [1, 2])
def test_tensor_laplace_3d(order):
    d = jno.domain(jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5))
    x, y, z = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    S, T = d.fem_symbols(value_shape=(3, 3), names=("S", "T"), order=order)
    coeffs = np.arange(9.0).reshape(3, 3)

    def ex(X, Y, Z):  # entry (i, j) = c_ij + x_k (P1) or c_ij + x_k x_{k+1} (P2): harmonic either way
        xs = (X, Y, Z)
        k = lambda i, j: (i + j) % 3  # noqa: E731
        if order == 1:
            return [[coeffs[i, j] + xs[k(i, j)] for j in range(3)] for i in range(3)]
        return [[coeffs[i, j] + xs[k(i, j)] * xs[(k(i, j) + 1) % 3] for j in range(3)] for i in range(3)]

    G = J.stack([J.stack(row, axis=-1) for row in ex(xb, yb, zb)], axis=-2)
    fem = jno.fem([inner(J.jacobian(S, [x, y, z]), J.jacobian(T, [x, y, z]), n_contract=3), S(xb, yb, zb) - G])
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 3, 3)
    P = np.asarray(fem.points)
    E = np.stack([np.stack(r, -1) for r in ex(P[:, 0], P[:, 1], P[:, 2])], -2)
    np.testing.assert_allclose(U, E, atol=1e-10)


def test_inflow_profile_is_transported_unchanged():
    """(b . grad) S = 0 with b = (1, 0) and S(0, y) = S_in(y): S = S_in(y) everywhere. P2 holds the
    quadratic profile exactly, so the SUPG-stabilised discrete solution is nodally exact."""
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), order=2)
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    tau = 0.5 * 0.25
    fem = jno.fem([inner(Si.x, Ti + tau * Ti.x, n_contract=2), S(xl, yl) - mat2(1 + yl * yl, yl, -yl, 3 + 0 * yl)])
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2, 2)
    Y = np.asarray(fem.points)[:, 1]
    E = np.stack([1 + Y * Y, Y, -Y, 3 + 0 * Y], -1).reshape(-1, 2, 2)
    np.testing.assert_allclose(U, E, atol=1e-11)


def _jaumann_error(n, g=1.0, G=1.0):
    d = jno.shape.rect(0.0, 0.5, 1.0, 1.0).structured(n=n).domain(compute_mesh_connectivity=False)
    d.tag("inflow", lambda x, y: x < 1e-9)
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("inflow", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), order=2)
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    W = J.array([[0.0, g / 2], [-g / 2, 0.0]])  # spin of u = (g y, 0)
    D = J.array([[0.0, g / 2], [g / 2, 0.0]])  # rate of deformation
    transport = lambda A: g * y * A.x  # noqa: E731  (u . grad) A
    R = transport(Si) - (W @ Si - Si @ W) - 2 * G * D
    tau = (0.5 / n) / (2 * g)
    fem = jno.fem([inner(R, Ti + tau * transport(Ti), n_contract=2), S(xl, yl) - 0.0])
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2, 2)
    P = np.asarray(fem.points)
    t = P[:, 0] / (g * P[:, 1])
    s_xy, s_xx = G * np.sin(g * t), G * (1 - np.cos(g * t))
    E = np.stack([s_xx, s_xy, s_xy, -s_xx], -1).reshape(-1, 2, 2)
    return np.abs(U - E).max()


def test_jaumann_simple_shear_converges():
    e4, e8 = _jaumann_error(4), _jaumann_error(8)
    assert e8 < 2e-3
    assert e4 / e8 > 4.0  # at least second order


# ---------------------------------------------------------------------------------------------------------
# symmetric=True: n(n+1)/2 stored values per node, the full matrix in every expression
# ---------------------------------------------------------------------------------------------------------


def _jaumann(n, symmetric, g=1.0, G=1.0):
    d = jno.shape.rect(0.0, 0.5, 1.0, 1.0).structured(n=n).domain(compute_mesh_connectivity=False)
    d.tag("inflow", lambda x, y: x < 1e-9)
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("inflow", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), order=2, symmetric=symmetric)
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    L = J.array([[0.0, g], [0.0, 0.0]])  # grad u for u = (g y, 0)
    D, W = (L + L.T) / 2, (L - L.T) / 2
    transport = lambda A: g * y * A.x  # noqa: E731
    R = transport(Si) - (W @ Si - Si @ W) - 2 * G * D
    tau = (0.5 / n) / (2 * g)
    fem = jno.fem([inner(R, Ti + tau * transport(Ti), n_contract=2), S(xl, yl) - 0.0])
    U = np.asarray(fem.solve(linear=jno.solve.lu()))
    return fem, U


def _unpack2(U):
    """Stored (xx, xy, yy) -> full 2x2 per node."""
    s = U.reshape(-1, 3)
    return np.stack([s[:, 0], s[:, 1], s[:, 1], s[:, 2]], -1).reshape(-1, 2, 2)


def test_symmetric_jaumann_matches_full_storage_with_fewer_dofs():
    fem_f, U_f = _jaumann(6, symmetric=False)
    fem_s, U_s = _jaumann(6, symmetric=True)
    n_nodes = np.asarray(fem_f.points).shape[0]
    assert fem_f.dofs == 4 * n_nodes and fem_s.dofs == 3 * n_nodes
    np.testing.assert_allclose(_unpack2(U_s), U_f.reshape(-1, 2, 2), atol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_symmetric_laplace_and_per_entry_dirichlet(order):
    """Whole-tensor and per-entry Dirichlet land on the stored values; S[1, 0] and S[0, 1] are one value."""
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), order=order, symmetric=True)
    ex = (
        (lambda X, Y: (1 + X, 2 * Y, 2 * Y, 3 + 0 * X))
        if order == 1
        else (lambda X, Y: (X * X - Y * Y, X * Y, X * Y, 3 + X))
    )
    lap = inner(J.jacobian(S, [x, y]), J.jacobian(T, [x, y]), n_contract=3)
    P = None
    for bcs in (
        [S(xb, yb) - mat2(*ex(xb, yb))],
        [
            S.bind(x=xb, y=yb)[0, 0] - ex(xb, yb)[0],
            S.bind(x=xb, y=yb)[1, 0] - ex(xb, yb)[2],
            S.bind(x=xb, y=yb)[1, 1] - ex(xb, yb)[3],
        ],
    ):
        fem = jno.fem([lap] + bcs)
        U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 3)
        P = np.asarray(fem.points)
        E = np.stack(ex(P[:, 0], P[:, 1]), -1)[:, [0, 1, 3]]
        assert fem.dofs == 3 * P.shape[0]
        np.testing.assert_allclose(U, E, atol=1e-11)


def test_symmetric_3d():
    d = jno.domain(jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5))
    x, y, z = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    out = {}
    for sym in (False, True):
        S, T = d.fem_symbols(value_shape=(3, 3), names=("S", "T"), symmetric=sym)
        xs = (xb, yb, zb)
        rows = [[float(i + j) + xs[(i + j) % 3] for j in range(3)] for i in range(3)]  # symmetric, harmonic
        G = J.stack([J.stack(r, axis=-1) for r in rows], axis=-2)
        fem = jno.fem([inner(J.jacobian(S, [x, y, z]), J.jacobian(T, [x, y, z]), n_contract=3), S(xb, yb, zb) - G])
        out[sym] = (fem.dofs, np.asarray(fem.solve(linear=jno.solve.lu())), np.asarray(fem.points))
    (nf, Uf, P), (ns, Us, _) = out[False], out[True]
    assert nf == 9 * P.shape[0] and ns == 6 * P.shape[0]
    iu, ju = np.triu_indices(3)
    np.testing.assert_allclose(Us.reshape(-1, 6), Uf.reshape(-1, 3, 3)[:, iu, ju], atol=1e-10)
    E = np.stack([np.stack([i + j + P[:, (i + j) % 3] for j in range(3)], -1) for i in range(3)], -2)
    np.testing.assert_allclose(Uf.reshape(-1, 3, 3), E, atol=1e-10)


def test_symmetric_refuses_an_asymmetric_wall_value():
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), symmetric=True)
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    with pytest.raises(ValueError, match="not symmetric"):
        jno.fem([inner(Si, Ti, n_contract=2) - J.trace(Ti), S(xl, yl) - mat2(1.0 + 0 * yl, yl, 0 * yl, 1.0 + 0 * yl)])


def test_symmetric_needs_a_square_matrix():
    d = _square()
    with pytest.raises(ValueError, match="square matrix"):
        d.fem_symbols(value_shape=(2,), symmetric=True)
    with pytest.raises(ValueError, match="square matrix"):
        d.fem_symbols(value_shape=(2, 3), symmetric=True)


def test_symmetric_nonlinear_matches_full():
    """A form nonlinear in S (S + |S|^2 S = F, pointwise): the Newton path sees the same matrix either way."""
    out = {}
    for sym in (False, True):
        d = _square()
        x, y = d.variable("interior", split=True)[:2]
        S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), symmetric=sym)
        Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
        F = mat2(1 + x, y, y, 2 + 0 * x)
        fem = jno.fem([inner(Si + inner(Si, Si, n_contract=2) * Si - F, Ti, n_contract=2)])
        out[sym] = np.asarray(fem.solve(nonlinear=jno.solve.newton(rtol=1e-12, atol=1e-13)))
    np.testing.assert_allclose(_unpack2(out[True]), out[False].reshape(-1, 2, 2), atol=1e-10)


@pytest.mark.parametrize("symmetric", [False, True])
def test_matrix_transient_decay_with_initial_tensor(symmetric):
    """S_t + S = 0, S(0) = S0: Crank-Nicolson multiplies by r = (1 - dt/2)/(1 + dt/2) each step, exactly."""
    n_t = 21
    d = jno.shape.rect(0, 0, 1, 1).structured(n=3).domain(time=(0.0, 1.0, n_t))
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), symmetric=symmetric)
    V = d.variable("interior", split=True)
    Sb, Tb = S.bind(x=V[0], y=V[1], t=V[2]), T.bind(x=V[0], y=V[1], t=V[2])
    ci = d.variable("initial", split=True)
    one = 1.0 + 0 * ci[0]
    fem = jno.fem(
        [inner(Sb.t, Tb, n_contract=2) + inner(Sb, Tb, n_contract=2), S(*ci) - mat2(one, 2 * one, 2 * one, 3 * one)]
    )
    traj = np.asarray(fem.solve(time=jno.solve.theta(0.5)).fn())
    dt = 1.0 / (n_t - 1)
    r = (1 - dt / 2) / (1 + dt / 2)
    stored = [1.0, 2.0, 3.0] if symmetric else [1.0, 2.0, 2.0, 3.0]
    np.testing.assert_allclose(
        traj[-1].reshape(-1, len(stored)), np.broadcast_to(np.array(stored) * r ** (n_t - 1), (16, len(stored))), rtol=1e-10
    )


def test_symmetric_outside_the_native_assembler_is_refused():
    d = jno.domain(constructor=jno.domain.line(mesh_size=0.25))
    (x,) = d.variable("interior", split=True)[:1]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), symmetric=True)
    with pytest.raises(NotImplementedError, match="native 2-D/3-D Lagrange"):
        jno.fem([inner(S.bind(x=x), T.bind(x=x), n_contract=2) - J.trace(T.bind(x=x))])


def test_a_matrix_unknown_squared_is_nonlinear():
    """S + c S @ S = F, pointwise, with F built from a known non-symmetric S0: the solution is S0.

    `S @ S` used to classify LINEAR (matmul was a linear wrapper whatever its arguments), and the linear
    path builds its operator at S = 0, where the quadratic term has no slope -- the term vanished. A
    coefficient matrix times the unknown, A @ S, stays linear."""
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"))
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    S0 = np.array([[0.3, 0.6], [-0.2, 0.1]])
    c = 0.7
    F = S0 + c * S0 @ S0
    for square in (Si @ Si, J.matmul(Si, Si)):
        fem = jno.fem([inner(Si + c * square - J.array(F), Ti, n_contract=2)])
        assert fem._mode == "nonlinear"
        U = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-13, atol=1e-14))).reshape(-1, 2, 2)
        np.testing.assert_allclose(U, np.broadcast_to(S0, U.shape), atol=1e-12)
    A = J.array([[2.0, 1.0], [0.0, 1.0]])
    assert jno.fem([inner(A @ Si - J.array(F), Ti, n_contract=2)])._mode == "linear"


@pytest.mark.parametrize("symmetric", [False, True])
def test_a_position_dependent_tensor_dirichlet_value(symmetric):
    """S = (1 + y) M on every edge, -Δ S = 0 inside: the exact S = (1 + y) M is linear, so P1 holds it to
    round-off. Each edge of the n = 2 grid has three nodes: the batch evaluation used to multiply a (3, 1)
    coordinate column by M, which raises for this 2 x 2 M and, for a 3 x 3 one, returned a single matrix
    that was read as a constant. Evaluated point by point, it is the value at each node."""
    M = np.array([[1.0, 0.4], [0.4, -0.5]]) if symmetric else np.array([[1.0, 0.4], [-0.3, -0.5]])
    d = _square(n=2)
    x, y = d.variable("interior", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), symmetric=symmetric)
    terms = [inner(J.jacobian(S, [x, y]), J.jacobian(T, [x, y]), n_contract=3)]
    for edge in ("left", "right", "bottom", "top"):
        xe, ye = d.variable(edge, split=True)[:2]
        terms.append(S(xe, ye) - (1.0 + ye) * J.array(M))
    fem = jno.fem(terms)
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, *((3,) if symmetric else (2, 2)))
    P = np.asarray(fem.points)
    want = (1.0 + P[:, 1])[:, None, None] * M
    if symmetric:
        want = want[:, [0, 0, 1], [0, 1, 1]]
    np.testing.assert_allclose(U, want, atol=1e-12)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_a_per_point_scalar_times_a_constant_matrix(nonlinear):
    """`s(x) * M` with M a constant 3 x 3, on a 2-D mesh (a 3 x 3 field there is legitimate: a stress with an
    out-of-plane component). Per point, a scalar is (n_quad,) or (n_quad, 1) and M has no quadrature axis,
    so the product used to fail to broadcast -- or, with as many quadrature points as M has rows, silently
    returned a 3 x 3. Oracles: the L2 projection of (1 + x + 2y) M is that field (P1 holds it), and
    S + c tr(S) I = F with F built from S0 = (1 + x) M returns S0, which exercises `tr(S) * I`, the
    deviator's scalar-times-identity, on the Newton path."""
    d = _square()
    x, y = d.variable("interior", split=True)[:2]
    S, T = d.fem_symbols(value_shape=(3, 3), names=("S", "T"))
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    M = np.array([[1.0, 0.4, 0.0], [-0.3, -0.5, 0.2], [0.1, 0.0, 0.7]])
    I3 = J.array(np.eye(3))
    P = None
    if not nonlinear:
        fem = jno.fem([inner(Si - (1.0 + x + 2.0 * y) * J.array(M), Ti, n_contract=2)])
        U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 3, 3)
        P = np.asarray(fem.points)
        want = (1.0 + P[:, 0] + 2.0 * P[:, 1])[:, None, None] * M
    else:
        c = 0.3
        S0 = (1.0 + x) * J.array(M)
        F = S0 + c * J.trace(S0) * J.trace(S0) * I3
        fem = jno.fem([inner(Si + c * J.trace(Si) * J.trace(Si) * I3 - F, Ti, n_contract=2)])
        assert fem._mode == "nonlinear"
        U = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-13, atol=1e-14))).reshape(-1, 3, 3)
        P = np.asarray(fem.points)
        want = (1.0 + P[:, 0])[:, None, None] * M
    np.testing.assert_allclose(U, want, atol=1e-11)
