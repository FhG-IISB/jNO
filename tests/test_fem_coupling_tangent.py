"""An assembled tangent for a problem with a nonlocal Coupling.

An enclosure coupling declares exactly which DOFs it reads and writes, so ``jno.fem`` assembles its Jacobian as a
dense block on those DOFs next to the sparse local tangent. That makes ``newton(direct=True)`` (sparse LU, forward
and adjoint) available on radiation problems. A coupling that declares nothing keeps the matrix-free default, and
``direct=True`` then materialises the whole Jacobian densely, with a warning that states its size.

Oracles: ``jax.jacfwd`` of the full residual for the tangent; the same root from three independent routes; central
finite differences for the gradient through the direct Newton.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIGMA = 5.670374419e-8


def _rings():
    """Two concentric annuli with a radiating gap between them: hot inside, cold outside (as in the
    enclosure tests). Returns the pieces to write the problem with."""
    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import Point

    r0, r1, r2, r3 = 0.10, 0.20, 0.25, 0.35
    ring = lambda a, b: Point(0, 0).buffer(b, 16).difference(Point(0, 0).buffer(a, 16))  # noqa: E731
    d = jno.domain(ring(r0, r1).union(ring(r2, r3)), mesh_size=0.45)
    rad = lambda x, y: jnp.hypot(x, y)  # noqa: E731
    d.tag("hot", lambda x, y: jnp.abs(rad(x, y) - r0) < 4e-2)
    d.tag("cold", lambda x, y: jnp.abs(rad(x, y) - r3) < 4e-2)
    d.tag("inner_gap", lambda x, y: jnp.abs(rad(x, y) - r1) < 4e-2)
    d.tag("outer_gap", lambda x, y: jnp.abs(rad(x, y) - r2) < 4e-2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xh, yh, _ = d.variable("hot", split=True)
    xc, yc, _ = d.variable("cold", split=True)
    gap = d.enclosure(["inner_gap", "outer_gap"])
    F = jnp.asarray(gap.view_factor)
    eps = gap.emissivity({"inner_gap": 0.8, "outer_gap": 0.6})
    Q = (jnp.diag(F.sum(1)) - F) @ jnp.linalg.solve(jnp.eye(gap.size) - (1 - eps)[:, None] * F, jnp.diag(eps * SIGMA))
    conduction = lambda k: k * (ui.x * vi.x + ui.y * vi.y)  # noqa: E731
    walls = [u(xh, yh) - 1000.0, u(xc, yc) - 300.0]
    return d, u, gap, Q, conduction, walls


def _fem_declared(k=20.0):
    d, u, gap, Q, conduction, walls = _rings()
    return jno.fem([conduction(k), gap.load(Q @ gap.field(u) ** 4), *walls]), gap, d


def _fem_undeclared(k=20.0):
    """The same physics as a bare function: a Coupling that says nothing about the DOFs it touches."""
    d, u, gap, Q, conduction, walls = _rings()
    n = len(d.mesh.points)
    return jno.fem([conduction(k), lambda w: gap.load(Q @ gap.field(w) ** 4, size=n), *walls])


def test_enclosure_coupling_declares_exactly_the_nodes_it_touches():
    d, u, gap, Q, _c, _w = _rings()
    c = gap.load(Q @ gap.field(u) ** 4)
    np.testing.assert_array_equal(c._support, gap.nodes)
    # an expression that cannot know what it reads declares nothing
    assert jno.Coupling(lambda w: w)._support is None


def test_assembled_tangent_matches_jacfwd_of_the_residual():
    fem, gap, d = _fem_declared()
    op = fem._op
    assert op.jacobian is not None, "a coupling with a declared support must give the operator a tangent"
    u = jnp.asarray(np.random.default_rng(0).uniform(300.0, 1000.0, len(d.mesh.points)))
    J = np.asarray(op.jacobian(u, None).todense())
    J_ref = np.asarray(jax.jacfwd(lambda w: op.residual(w, None))(u))
    np.testing.assert_allclose(J, J_ref, rtol=1e-10, atol=1e-10 * np.abs(J_ref).max())
    # and the coupling block lives on the enclosure nodes only: outside them J is the local tangent
    off = np.setdiff1d(np.arange(J.shape[0]), gap.nodes)
    A_loc = np.asarray(jax.jacfwd(lambda w: op.residual(w, None))(u))[np.ix_(off, off)]
    np.testing.assert_allclose(J[np.ix_(off, off)], A_loc, rtol=1e-12, atol=1e-12 * np.abs(A_loc).max())


def test_direct_newton_default_and_dense_fallback_reach_the_same_root(capfd):
    fem, _gap, _d = _fem_declared()
    x0 = jnp.full(fem.dofs, 600.0)
    u_lu = np.asarray(fem.solve(x0=x0, nonlinear=jno.solve.newton(direct=True, rtol=0.0, atol=1e-8)))
    u_def = np.asarray(fem.solve(x0=x0, nonlinear=jno.solve.newton(rtol=0.0, atol=1e-8)))
    capfd.readouterr()
    u_dense = np.asarray(_fem_undeclared().solve(x0=x0, nonlinear=jno.solve.newton(direct=True, rtol=0.0, atol=1e-8)))
    out = capfd.readouterr().out
    assert "materialised DENSE" in out and "925 x 925" in out, out
    assert 300.0 - 1e-6 <= u_lu.min() and u_lu.max() <= 1000.0 + 1e-6
    np.testing.assert_allclose(u_def, u_lu, rtol=1e-9)
    np.testing.assert_allclose(u_dense, u_lu, rtol=1e-9)


def test_gradient_through_the_direct_newton_matches_finite_differences():
    """d(mean T)/d(k) through newton(direct=True): the implicit-function gradient solves J^T at the root with
    the assembled tangent; it must match a central difference of two forward solves."""
    d, u, gap, Q, conduction, walls = _rings()
    k = jno.np.parameter((1,), name="k")
    fem = jno.fem([conduction(k), gap.load(Q @ gap.field(u) ** 4), *walls])
    nl = jno.solve.newton(direct=True, rtol=0.0, atol=1e-10)
    x0 = jnp.full(fem.dofs, 600.0)
    mean_T = lambda kv: jnp.mean(fem.solve(x0=x0, nonlinear=nl, k=jnp.reshape(kv, (1,))))  # noqa: E731
    k0, h = 20.0, 1e-3
    g = float(jax.grad(mean_T)(k0))
    g_fd = (float(mean_T(k0 + h)) - float(mean_T(k0 - h))) / (2 * h)
    assert np.isfinite(g) and abs(g) > 0
    assert abs(g - g_fd) <= 1e-5 * abs(g_fd), (g, g_fd)


def test_bare_direct_newton_without_a_tangent_goes_dense_and_says_so(capfd):
    """As a bare driver (no assembled tangent threaded in) ``direct=True`` materialises the Jacobian densely,
    announces its size, and solves."""
    w = jno.solve.newton(direct=True)(lambda w: w**3 + w - 2.0, jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(w), 1.0, rtol=1e-8)  # Newton stops at |r| < atol = 1e-8
    assert "materialised DENSE" in capfd.readouterr().out


def test_direct_newton_gradient_is_exact_when_the_iterative_tangent_solve_misses():
    """The implicit-diff tangent re-solves by sparse LU when the configured solve misses. Here the solve is a
    single Jacobi sweep: good enough to drive Newton to the root (slowly), far too loose for the gradient, which
    must still be exactly ``J^-T`` applied to the cotangent."""
    import jax.experimental.sparse as jsparse

    from jno.utils.solver.newton_krylov import newton_direct

    rng = np.random.default_rng(1)
    n = 6
    A = np.diag(np.full(n, 4.0)) + rng.uniform(-0.5, 0.5, (n, n))  # non-symmetric, diagonally dominant
    A = jnp.asarray(A)
    jac = lambda w: jsparse.BCOO.fromdense(A + jnp.diag(3.0 * w**2), nse=n * n)  # noqa: E731
    jacobi_sweep = lambda J, b: b / J.todense().diagonal()  # noqa: E731

    def solve(b):
        return newton_direct(lambda w: A @ w + w**3 - b, jac, jnp.zeros(n), linear_solve=jacobi_sweep,
                             rtol=0.0, atol=1e-12, max_steps=500)  # fmt: skip

    b = jnp.asarray(rng.uniform(1.0, 2.0, n))
    u = solve(b)
    g = np.asarray(jax.grad(lambda bb: jnp.sum(solve(bb)))(b))
    J = np.asarray(A) + np.diag(3.0 * np.asarray(u) ** 2)
    np.testing.assert_allclose(g, np.linalg.solve(J.T, np.ones(n)), rtol=1e-10)
