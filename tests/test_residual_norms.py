"""``r.norm(space)``: function-space norms of residuals (L2, H1, H^-1, H^s), the backend picked from the domain.

Oracles are analytic: r = sin(pi x) sin(pi y) is a Dirichlet eigenfunction of -Laplace with eigenvalue 2 pi^2
and ||r||_L2^2 = 1/4, so ||r||^2 in H^s (norm (scale - Laplace)^s) is (1/4) (scale + 2 pi^2)^s. On a structured
lattice the nodal values of r are an exact eigenvector of the discrete Dirichlet Laplacian too, with eigenvalue
lambda_h = 2 * 4 n^2 sin^2(pi / (2 n)), so the grid backend must reproduce (1/4) (scale + lambda_h)^s to
round-off. On unstructured meshes the P1 values converge at second order.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
import foundax  # noqa: E402
import jax  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402

PI = np.pi
sin, cos = jno.np.sin, jno.np.cos


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _value(node, dom):
    crux = jno.core([node], domain=dom)
    return float(np.asarray(crux.eval([node], domain=dom)).reshape(-1)[0])


def _mesh(h):
    return jno.shape.rect(0, 0, 1, 1, size=h).domain()


def _lattice(n):
    return jno.shape.rect(0, 0, 1, 1).structured(n=(n, n)).domain()


def _eigen(dom):
    x, y, _ = dom.variable("interior", split=True)
    return x, y, sin(PI * x) * sin(PI * y)


def _rate(err_coarse, err_fine, h_coarse, h_fine):
    return np.log(abs(err_coarse) / abs(err_fine)) / np.log(h_coarse / h_fine)


def _lam_h(n, dims=2):
    return dims * 4 * n**2 * np.sin(PI / (2 * n)) ** 2


# ---- the pointwise norms keep their meaning ------------------------------------------------------------------


def test_space_none_keeps_the_pointwise_vector_and_matrix_norms():
    """Breaks if `.norm()` without a space stops being the per-point norm (guard for existing behaviour)."""
    dom = _lattice(4)
    x, y, _ = dom.variable("interior", split=True)
    v = jno.np.vector(3 * x, 4 * x)
    got = np.asarray(jno.core([v.norm().mse], domain=dom).eval([v.norm()], domain=dom)).reshape(-1)
    xs = np.asarray(jno.core([x.mse], domain=dom).eval([x], domain=dom)).reshape(-1)
    np.testing.assert_allclose(got, 5 * np.abs(xs), rtol=1e-14)


def test_scalar_pointwise_norm_is_the_absolute_value():
    """Breaks if a scalar's `.norm()` is not |r| per point (or reduces over the points)."""
    dom = _lattice(4)
    x, y, _ = dom.variable("interior", split=True)
    r = x - 0.5
    crux = jno.core([r.mse], domain=dom)
    got, xs = (np.asarray(v).reshape(-1) for v in crux.eval([r.norm(), x], domain=dom))
    np.testing.assert_allclose(got, np.abs(xs - 0.5), rtol=1e-14)


def test_matrix_norm_keeps_the_positional_ord_and_takes_a_space():
    """Breaks if `M.norm("nuc")` is read as a space, or if a matrix field's Sobolev norm is not the sum over
    its entries (Frobenius)."""
    dom = _lattice(16)
    x, y, r = _eigen(dom)
    A = jno.np.stack([jno.np.vector(r, 0 * r), jno.np.vector(0 * r, 2 * r)], axis=-2).matrix
    nuc = np.asarray(jno.core([r.mse], domain=dom).eval([A.norm("nuc")], domain=dom)).reshape(-1)
    rs = np.asarray(jno.core([r.mse], domain=dom).eval([r], domain=dom)).reshape(-1)
    np.testing.assert_allclose(nuc, 3 * np.abs(rs), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(_value(A.norm("H-1") ** 2, dom), 5 * _value(r.norm("H-1") ** 2, dom), rtol=1e-12)
    with pytest.raises(ValueError, match="Frobenius"):
        A.norm("H-1", ord="nuc")


def test_l2_squared_is_the_mse_bit_for_bit():
    """Breaks if `norm("l2")**2` is not fused into the squared form (a sqrt then a square rounds)."""
    dom = _lattice(8)
    x, y, _ = dom.variable("interior", split=True)
    r = sin(PI * x) * y + 0.3
    assert _value(r.norm("l2") ** 2, dom) == _value(r.mse, dom)


# ---- mesh backend ----------------------------------------------------------------------------------------------


def test_L2_on_a_mesh_uses_the_nodal_measure():
    """Breaks if the L2 norm drops the nodal measure or uses the wrong one (the value goes)."""
    errs = []
    for h in (0.1, 0.05):
        dom = _mesh(h)
        _, _, r = _eigen(dom)
        errs.append(_value(r.norm("L2") ** 2, dom) - 0.25)
    # the vertex rule converges faster than O(h^2) here (measured 1e-3 and 1e-5 relative); a wrong measure
    # (unweighted mean, cell areas not shared out) is off by far more than 1e-3
    assert abs(errs[0]) < 1e-3 * 0.25 and abs(errs[1]) < abs(errs[0]), errs


def test_H_minus_1_dirichlet_on_a_mesh_matches_the_eigenvalue_at_second_order():
    """Breaks if R = M r, the (M + K) solve, the Dirichlet restriction or the scaling is wrong."""
    want = 0.25 / (1 + 2 * PI**2)
    errs = []
    for h in (0.1, 0.05):
        dom = _mesh(h)
        _, _, r = _eigen(dom)
        node = r.norm("H-1") ** 2
        errs.append(_value(node, dom) - want)
        assert node.backend == "mesh"
    assert abs(errs[-1]) < 2e-2 * want, errs
    assert 1.5 < _rate(*errs, 0.1, 0.05) < 2.6, errs


def test_H_minus_1_seminorm_with_scale_zero():
    """Breaks if `scale` is ignored: with scale=0 the norm is (1/4) / (2 pi^2), not (1/4) / (1 + 2 pi^2)."""
    dom = _mesh(0.05)
    _, _, r = _eigen(dom)
    want = 0.25 / (2 * PI**2)
    # P1 discretisation error at h = 0.05: about (pi h)^2 / 4 < 1 %
    assert abs(_value(r.norm("H-1", scale=0.0) ** 2, dom) / want - 1) < 1e-2


def test_H_minus_1_neumann_on_a_cosine():
    """Breaks if bc="neumann" still removes the boundary nodes (cos does not vanish there)."""
    dom = _mesh(0.05)
    x, y, _ = dom.variable("interior", split=True)
    r = cos(PI * x) * cos(PI * y)
    want = 0.25 / (1 + 2 * PI**2)
    assert abs(_value(r.norm("H-1", bc="neumann") ** 2, dom) / want - 1) < 1e-2


def test_H1_on_a_mesh():
    """Breaks if H1 drops the gradient part: (1/4) (1 + 2 pi^2), not 1/4."""
    dom = _mesh(0.05)
    _, _, r = _eigen(dom)
    want = 0.25 * (1 + 2 * PI**2)
    assert abs(_value(r.norm("H1") ** 2, dom) / want - 1) < 1e-2


# ---- grid backend ----------------------------------------------------------------------------------------------


def test_grid_backend_is_taken_on_a_lattice_and_is_exact_for_the_discrete_eigenvector():
    """Breaks if a lattice silently takes the mesh path, or if the transform, the eigenvalues or the
    weights of the grid backend are off (the oracle is exact to round-off)."""
    n = 32
    dom = _lattice(n)
    _, _, r = _eigen(dom)
    node = r.norm("H-1") ** 2
    got = _value(node, dom)
    assert node.backend == "grid"
    np.testing.assert_allclose(got, 0.25 / (1 + _lam_h(n)), rtol=1e-12)


def test_grid_fractional_order():
    """Breaks if the exponent s is not applied as a real power of (scale + lambda)."""
    n = 16
    dom = _lattice(n)
    _, _, r = _eigen(dom)
    np.testing.assert_allclose(_value(r.norm("H^-0.5") ** 2, dom), 0.25 * (1 + _lam_h(n)) ** -0.5, rtol=1e-12)


def test_grid_and_mesh_backends_agree_on_the_same_lattice():
    """Breaks if the two backends measure different norms (scaling, bc, which nodes count)."""
    dom = _lattice(32)
    _, _, r = _eigen(dom)
    grid, mesh = r.norm("H-1") ** 2, r.norm("H-1", method="mesh") ** 2
    g, m = _value(grid, dom), _value(mesh, dom)
    assert (grid.backend, mesh.backend) == ("grid", "mesh")
    # both are O(h^2) approximations of the same number; h = 1/32
    assert abs(g / m - 1) < 1e-2


def test_vector_field_norm_is_the_sum_over_components():
    """Breaks if a vector residual is reduced over its components before the norm (or only one is used)."""
    dom = _lattice(16)
    x, y, r = _eigen(dom)
    scalar = _value(r.norm("H-1") ** 2, dom)
    vector = _value(jno.np.vector(r, 2 * r).norm("H-1") ** 2, dom)
    np.testing.assert_allclose(vector, 5 * scalar, rtol=1e-12)


def test_3d_box_on_grid_and_mesh():
    """Breaks if the backends assume two dimensions."""
    n = 8
    dom = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=(n, n, n)).domain()
    x, y, z, _ = dom.variable("interior", split=True)
    r = sin(PI * x) * sin(PI * y) * sin(PI * z)
    np.testing.assert_allclose(_value(r.norm("H-1") ** 2, dom), 0.125 / (1 + _lam_h(n, 3)), rtol=1e-12)
    want = 0.125 / (1 + 3 * PI**2)
    errs = []
    for m in (8, 16):
        box = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=(m, m, m)).domain()
        xm, ym, zm, _ = box.variable("interior", split=True)
        rm = sin(PI * xm) * sin(PI * ym) * sin(PI * zm)
        errs.append(_value(rm.norm("H-1", method="mesh") ** 2, box) / want - 1)
    # P1 on the lattice's tetrahedra: second order with a large constant (measured -13 % at h = 1/8)
    assert abs(errs[1]) < 5e-2 and 1.7 < _rate(*errs, 1 / 8, 1 / 16) < 2.3, errs


# ---- boundary, refusals ----------------------------------------------------------------------------------------


def test_boundary_L2_uses_the_boundary_measure():
    """Breaks if a boundary residual is weighted with volume weights: int_boundary 1 ds = 4 on the unit square."""
    dom = _mesh(0.1)
    xb, yb, _ = dom.variable("boundary", split=True)
    np.testing.assert_allclose(_value((0 * xb + 1).norm("L2") ** 2, dom), 4.0, rtol=1e-12)


def test_refusals_name_the_way_out():
    dom = _mesh(0.1)
    _, _, r = _eigen(dom)
    with pytest.raises(ValueError, match="H-1"):  # lists the known spaces
        r.norm("W2")
    with pytest.raises(NotImplementedError, match="planned"):
        r.norm("H-1", method="sample")
    xb, yb, _ = dom.variable("boundary", split=True)
    with pytest.raises(NotImplementedError, match="boundary"):
        _value((xb * yb).norm("H-1") ** 2, dom)
    pts = jno.domain.from_array({"interior": np.random.default_rng(0).random((50, 2))})
    xp, yp = pts.variable("interior", split=True)[:2]
    with pytest.raises(NotImplementedError, match="mesh"):
        _value((xp * yp).norm("H-1") ** 2, pts)


# ---- training ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["grid", "mesh"])
def test_the_cached_operator_survives_a_second_trace(method):
    """Breaks if the cached backend holds arrays made inside the first trace: evaluating the norm and then
    training on it compiles twice, and a leaked tracer raises UnexpectedTracerError."""
    dom = _lattice(8)
    x, y, ue = _eigen(dom)
    net = jno.nn(foundax.mlp(2, 1, hidden_dims=8, num_layers=1, key=jax.random.PRNGKey(0)))
    node = (net(x, y) * x * (1 - x) * y * (1 - y) - ue).norm("H-1", method=method) ** 2
    net.optimizer(optax.adam(1e-3))
    crux = jno.core([node], domain=dom)
    before = float(np.asarray(crux.eval([node], domain=dom)).reshape(-1)[0])
    crux.solve(5)
    after = float(np.asarray(crux.eval([node], domain=dom)).reshape(-1)[0])
    assert np.isfinite(after) and after != before


def test_the_mesh_solve_defaults_to_amg_and_takes_a_given_preconditioner():
    """Breaks if the default mesh H^-1 solve is not AMG-preconditioned when pyamg is there (measured: Jacobi-CG
    up to 10x slower per training step on the GPU), or if `precond=` is not passed through."""
    pytest.importorskip("pyamg")
    dom = _lattice(8)
    _, _, r = _eigen(dom)
    default, given = r.norm("H-1", method="mesh") ** 2, r.norm("H-1", method="mesh", precond=jno.precond.jacobi()) ** 2
    np.testing.assert_allclose(_value(default, dom), _value(given, dom), rtol=1e-9)  # CG to 1e-10
    precs = sorted(type(op.precond).__name__ for op in dom._norm_operator_cache.values())
    assert precs == ["_AMG", "_Jacobi"]


def test_a_pinn_trains_on_the_H_minus_1_loss():
    """Breaks if the norm is not differentiable through jno.core (the error would not drop)."""
    dom = _lattice(16)
    x, y, ue = _eigen(dom)
    net = jno.nn(foundax.mlp(2, 1, hidden_dims=32, num_layers=2, key=jax.random.PRNGKey(0)))
    u = net(x, y) * x * (1 - x) * y * (1 - y)
    r = -u.laplacian(x, y) - 2 * PI**2 * ue
    net.optimizer(optax.adam(1e-2))
    crux = jno.core([r.norm("H-1") ** 2], domain=dom)

    def err():
        a, b = (np.asarray(v).reshape(-1) for v in crux.eval([u, ue], domain=dom))
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    e0 = err()
    crux.solve(300)
    assert err() < 0.1 * e0
