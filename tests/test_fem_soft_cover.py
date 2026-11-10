"""Soft enrichment (prototype): a continuous per-node weight on interpolation covers.

The contract under test (see ``jno/utils/solver/fem_soft_cover.py``):

* ``w = 1`` everywhere IS the full cover space; ``w = 0`` off a mask and 1 on it IS the hard mask the
  existing ``jno.solve.enrich`` path builds, up to the penalty floor;
* the solution and its energy are differentiable in ``w`` (AD agrees with finite differences);
* at ``w = 0`` the gradient equals the closed-form diagonal hierarchical indicator;
* one compiled program serves every ``w``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.fem_soft_cover import (
    W_MIN,
    cover_dof_index,
    hierarchical_indicator,
    operator_diagonal,
    ritz_energy,
    soft_solve,
)
from jno.utils.solver.linear import sparse_lu_solve


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _problem(mask=None, size=0.15):
    """-lap u = f on the unit square, u = 0 on the boundary, one scalar cover field."""
    import jno

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    if mask is not None:
        d._fem_enriched_nodes = np.asarray(mask, dtype=bool)
    u, phi = d.fem_symbols(space="cover")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    # a source with curvature, so the covers carry something
    f = 20.0 * xi * (1.0 - xi) * yi + 1.0
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])
    A, b = fem.operator
    n = int(np.asarray(d.mesh.points).shape[0])
    idx = cover_dof_index(n, 2)
    return A, jnp.asarray(b).reshape(-1), idx, n


@pytest.fixture(scope="module")
def full():
    jax.config.update("jax_enable_x64", True)
    A, b, idx, n = _problem()
    return A, b, idx, operator_diagonal(A, idx), n


def test_w_one_is_the_full_cover_space(full):
    A, b, idx, c, n = full
    u_soft = np.asarray(soft_solve(A, b, idx, c, jnp.ones(n)))
    u_full = np.asarray(sparse_lu_solve(A, b))
    assert np.linalg.norm(u_soft - u_full) <= 1e-12 * np.linalg.norm(u_full)


def test_w_zero_off_a_mask_is_the_hard_mask(full):
    A, b, idx, c, n = full
    mask = np.random.default_rng(0).random(n) < 0.3
    u_soft = np.asarray(soft_solve(A, b, idx, c, jnp.asarray(mask, dtype=float)))
    Am, bm, _, _ = _problem(mask)
    u_hard = np.asarray(sparse_lu_solve(Am, bm))
    rel = np.linalg.norm(u_soft - u_hard) / np.linalg.norm(u_hard)
    assert rel < 100 * W_MIN, f"soft at w in {{0,1}} vs the hard mask: {rel:.2e}"
    # the energies agree too, measured with the same full-space operator (both fields live in it)
    e_soft, e_hard = float(ritz_energy(A, b, u_soft)), float(ritz_energy(A, b, u_hard))
    assert abs(e_soft - e_hard) < 100 * W_MIN * abs(e_hard)


def test_energy_gradient_matches_finite_differences(full):
    A, b, idx, c, n = full
    w0 = jnp.asarray(np.random.default_rng(1).uniform(0.2, 0.8, n))

    def pi(w):
        return ritz_energy(A, b, soft_solve(A, b, idx, c, w))

    g = np.asarray(jax.grad(pi)(w0))
    # Fourth-order central difference at a moderate step. A per-node derivative is ~1e-3 of Pi, so a
    # second-order step small enough to beat truncation (1e-6) is rounding-limited at ~1e-6 relative;
    # measured at h=1e-3 the fourth-order stencil agrees with AD to 2.5e-10.
    eps = 1e-3

    def fd(i):
        p = [float(pi(w0.at[i].add(k * eps))) for k in (2, 1, -1, -2)]
        return (-p[0] + 8 * p[1] - 8 * p[2] + p[3]) / (12 * eps)

    for i in np.argsort(-np.abs(g))[:6]:  # the components that matter, not zero ones on the boundary
        ref = fd(i)
        assert abs(g[i] - ref) <= 1e-7 * abs(ref), f"node {i}: AD {g[i]:.6e} vs FD {ref:.6e}"
    assert np.all(g <= 1e-14), "more enrichment can never raise the Ritz energy (Galerkin nesting)"


def test_gradient_at_zero_is_the_hierarchical_indicator(full):
    A, b, idx, c, n = full

    def pi(w):
        return ritz_energy(A, b, soft_solve(A, b, idx, c, w))

    w0 = jnp.zeros(n)
    g = np.asarray(jax.grad(pi)(w0))
    u_p1 = soft_solve(A, b, idx, c, w0)
    h = np.asarray(hierarchical_indicator(A, b, idx, c, u_p1))
    live = np.abs(h) > 1e-8 * np.abs(h).max()
    assert live.sum() > n // 4, "too few nodes with a non-trivial indicator to test anything"
    rel = np.abs(g[live] - h[live]) / np.abs(h[live])
    assert rel.max() < 1e-4, f"AD gradient at w=0 vs sum r^2/A_kk: worst rel {rel.max():.2e}"


def test_one_compiled_program_serves_every_w(full):
    A, b, idx, c, n = full
    f = jax.jit(lambda w: ritz_energy(A, b, soft_solve(A, b, idx, c, w)))
    g = jax.jit(jax.grad(lambda w: ritz_energy(A, b, soft_solve(A, b, idx, c, w))))
    rng = np.random.default_rng(2)
    for _ in range(5):
        w = jnp.asarray(rng.random(n))
        f(w).block_until_ready()
        g(w).block_until_ready()
    assert f._cache_size() == 1 and g._cache_size() == 1


# ---------------------------------------------------------------- decoupled switch-off (compacted pattern)


def _mixed_w(n, seed=3):
    """Half the nodes exactly switched off, half on with fractional weights."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.2, 0.9, n)))


def test_compact_off_nodes_reproduce_the_hard_mask_exactly(full):
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    mask = np.random.default_rng(0).random(n) < 0.3
    u_c = np.asarray(compact_solver(A, idx, c)(jnp.asarray(mask, dtype=float), b))
    Am, bm, _, _ = _problem(mask)
    u_h = np.asarray(sparse_lu_solve(Am, bm))
    assert np.linalg.norm(u_c - u_h) <= 1e-12 * np.linalg.norm(u_h)


def test_compact_matches_the_penalty_model_on_fractional_weights(full):
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    w = _mixed_w(n)
    u_c = np.asarray(compact_solver(A, idx, c)(w, b))
    u_p = np.asarray(soft_solve(A, b, idx, c, w))
    assert np.linalg.norm(u_c - u_p) < 100 * W_MIN * np.linalg.norm(u_p)


def test_compact_gradient_is_the_one_sided_limit_of_the_penalty_model(full):
    """Off nodes get their derivative from the custom JVP; it must agree with the coupled penalty model,
    whose gradient at w = 0 is an ordinary derivative, to O(w_min). On nodes go through the ordinary
    implicit derivative of the penalty and must agree as well."""
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    w = _mixed_w(n)
    solve = compact_solver(A, idx, c)
    g_c = np.asarray(jax.grad(lambda w_: ritz_energy(A, b, solve(w_, b)))(w))
    g_p = np.asarray(jax.grad(lambda w_: ritz_energy(A, b, soft_solve(A, b, idx, c, w_)))(w))
    live = np.abs(g_p) > 1e-8 * np.abs(g_p).max()
    rel = np.abs(g_c - g_p)[live] / np.abs(g_p[live])
    assert rel.max() < 1e-3, f"compact vs penalty gradient: worst rel {rel.max():.2e}"
    # a non-energy objective exercises the full tangent (not only the energy's stationarity shortcut)
    q = jnp.asarray(np.random.default_rng(4).standard_normal(A.shape[0]))
    h_c = np.asarray(jax.grad(lambda w_: jnp.dot(q, solve(w_, b)) ** 2)(w))
    h_p = np.asarray(jax.grad(lambda w_: jnp.dot(q, soft_solve(A, b, idx, c, w_)) ** 2)(w))
    assert np.linalg.norm(h_c - h_p) < 1e-3 * np.linalg.norm(h_p)


def test_compact_is_differentiable_in_the_load(full):
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    w, solve = _mixed_w(n), compact_solver(A, idx, c)
    gb_c = np.asarray(jax.grad(lambda bb: ritz_energy(A, bb, solve(w, bb)))(b))
    gb_p = np.asarray(jax.grad(lambda bb: ritz_energy(A, bb, soft_solve(A, bb, idx, c, w)))(b))
    assert np.linalg.norm(gb_c - gb_p) < 1e-3 * np.linalg.norm(gb_p)


def test_compact_runs_on_a_factor_once_backend_that_reads_the_padding(full):
    """PARDISO builds its matrix from the raw BCOO entries, so the entries the compact operator parks at
    the out-of-bounds padding index must be ignored there, as JAX's own conversion does. Same solution
    and same gradient as the default solve."""
    from jno.utils.solver.fem_soft_cover import compact_solver
    from jno.utils.solver.linear import _pardiso_available, pardiso_lu_solve

    if not _pardiso_available():
        pytest.skip("pypardiso not installed")
    A, b, idx, c, n = full
    w = _mixed_w(n)
    ref, par = compact_solver(A, idx, c), compact_solver(A, idx, c, linear_solve=pardiso_lu_solve)
    np.testing.assert_allclose(np.asarray(par(w, b)), np.asarray(ref(w, b)), rtol=1e-9, atol=1e-12)
    q = jnp.asarray(np.random.default_rng(5).standard_normal(A.shape[0]))
    g_ref = np.asarray(jax.grad(lambda w_: jnp.dot(q, ref(w_, b)) ** 2)(w))
    g_par = np.asarray(jax.grad(lambda w_: jnp.dot(q, par(w_, b)) ** 2)(w))
    assert np.linalg.norm(g_par - g_ref) < 1e-8 * np.linalg.norm(g_ref)


def _functional_grads(solve, b, data, w, params=None):
    """Solution, and the gradients of a non-energy functional wrt the weights and a scale on the values."""
    q = jnp.asarray(np.random.default_rng(6).standard_normal(b.shape[0]))
    f = lambda w_, t: jnp.dot(q, solve(w_, b, data=t * data, params=params)) ** 2  # noqa: E731
    gw, gt = jax.grad(f, argnums=(0, 1))(w, 1.0)
    return np.asarray(solve(w, b, data=data, params=params)), np.asarray(gw), float(gt)


def test_compact_cap_gathers_the_live_entries_and_refuses_an_overflow(full):
    """A fixed-length prefix of the live entries gives the same solve and gradients; too small a cap is NaN."""
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    w = _mixed_w(n)
    A_sw, _, _ = compact_solver(A, idx, c).switched(w, b)
    live = int((np.asarray(A_sw.indices)[: A.nse, 0] < A.shape[0]).sum())
    ref = _functional_grads(compact_solver(A, idx, c), b, A.data, w)
    got = _functional_grads(compact_solver(A, idx, c, cap=live + 7), b, A.data, w)
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(got[1], ref[1], rtol=1e-8, atol=1e-12 * np.abs(ref[1]).max())
    assert abs(got[2] - ref[2]) < 1e-8 * abs(ref[2])
    assert compact_solver(A, idx, c, cap=live + 7).switched(w, b)[0].nse == live + 7 + idx.size
    assert np.all(np.isnan(np.asarray(compact_solver(A, idx, c, cap=live - 1)(w, b))))


def test_compact_with_only_the_active_couplings_and_a_matrix_free_full_product(full):
    """A caller may assemble only the couplings the switched operator keeps (plus every cover diagonal) and
    hand the full operator's product over separately: same solve, same gradients."""
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    w = _mixed_w(n)
    ind = np.asarray(A.indices)
    cov_off = np.zeros(A.shape[0], dtype=bool)
    cov_off[np.asarray(idx)[np.asarray(w) <= 0.0].reshape(-1)] = True
    touches = (cov_off[ind[:, 0]] | cov_off[ind[:, 1]]) & (ind[:, 0] != ind[:, 1])
    data_active = jnp.where(jnp.asarray(touches), 0.0, A.data)
    full_matvec = lambda t, v: t * (A @ v)  # noqa: E731  -- params = the same scale the values carry
    ref = _functional_grads(compact_solver(A, idx), b, A.data, w)
    got = _functional_grads(compact_solver(A, idx, full_matvec=full_matvec), b, data_active, w, params=jnp.asarray(1.0))
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(got[1], ref[1], rtol=1e-8, atol=1e-12 * np.abs(ref[1]).max())
    assert abs(got[2] - ref[2]) < 1e-8 * abs(ref[2])
    # without the full product, the active-only values give the WRONG one-sided gradient
    bad = _functional_grads(compact_solver(A, idx), b, data_active, w)
    assert np.linalg.norm(bad[1] - ref[1]) > 1e-3 * np.linalg.norm(ref[1])


def test_compact_drops_the_switched_off_entries_from_the_factorisation(full):
    """The pattern the factorisation sees shrinks with the active set, while the arrays keep their length
    and one compiled program serves every mask."""
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    solve = compact_solver(A, idx, c)
    seen = lambda w_: int(np.sum(np.asarray(solve.switched(w_, b)[0].indices)[:, 0] < A.shape[0]))  # noqa: E731
    n_all_on, n_all_off = seen(jnp.ones(n)), seen(jnp.zeros(n))
    assert n_all_off < 0.5 * n_all_on, f"switching every node off kept {n_all_off} of {n_all_on} entries"
    f = jax.jit(lambda w_: solve(w_, b))
    rng = np.random.default_rng(5)
    for _ in range(4):
        f(jnp.asarray(np.where(rng.random(n) < 0.5, 0.0, 1.0))).block_until_ready()
    assert f._cache_size() == 1


def test_compact_matvec_is_the_switched_operator_and_reads_only_the_active_prefix(full):
    from jno.utils.solver.fem_soft_cover import compact_matvec, compact_solver

    A, b, idx, c, n = full
    w = _mixed_w(n)
    A_sw, _, _ = compact_solver(A, idx, c).switched(w, b)
    prepare, matvec = compact_matvec(A, idx, c, chunk=256)
    st = prepare(w)
    x = jnp.asarray(np.random.default_rng(6).standard_normal(A.shape[0]))
    assert np.allclose(np.asarray(matvec(st, x)), np.asarray(A_sw @ x), rtol=1e-13, atol=1e-13)
    # the loop covers the active entries and at most one chunk more
    n_live = int(np.sum(np.asarray(A_sw.indices)[:, 0] < A.shape[0])) - int(np.size(idx))  # minus the diagonal adds
    assert int(st["n_chunks"]) == -(-n_live // 256)
    # one program across masks
    mv = jax.jit(matvec)
    for seed in range(3):
        mv(prepare(_mixed_w(n, seed)), x).block_until_ready()
    assert mv._cache_size() == 1


def test_compact_scale_from_the_operator_is_differentiable_in_the_operator(full):
    """With ``c=None`` the penalty scale is A_kk of the CURRENT operator; the gradient w.r.t. the stored
    values (a design parameter's path) must include that dependence. Checked against finite differences
    along a random direction in ``data``, at fractional weights where the penalty is active."""
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, idx, c, n = full
    w = _mixed_w(n)
    solve = compact_solver(A, idx)  # scale from the operator
    u_fixed = np.asarray(compact_solver(A, idx, c)(w, b))
    assert np.allclose(np.asarray(solve(w, b)), u_fixed, rtol=1e-12, atol=1e-14), "c=None must equal c=A_kk at A"
    rng = np.random.default_rng(7)
    ddir = jnp.asarray(rng.standard_normal(A.data.shape)) * jnp.abs(A.data)  # relative perturbation
    ddir = 0.5 * (ddir + ddir)  # keep it simple; symmetry is not needed for the derivative check

    def obj(t):
        return jnp.dot(b, solve(w, b, data=A.data + t * ddir))

    g = float(jax.grad(obj)(0.0))
    eps = 1e-4
    p = [float(obj(k * eps)) for k in (2, 1, -1, -2)]
    fd = (-p[0] + 8 * p[1] - 8 * p[2] + p[3]) / (12 * eps)
    assert abs(g - fd) <= 1e-6 * abs(fd), f"d/d(data) AD {g:.8e} vs FD {fd:.8e}"


def _elastic(mask=None, size=0.2):
    """Plane-stress cantilever on a vector cover field: clamped left edge, load at the right mid-edge."""
    import jno

    d = jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=size).domain()
    d.tag("left", lambda x, y: x < 1e-9)
    d.tag("tip", lambda x, y: (x > 2.0 - 1e-9) & (np.abs(y - 0.5) < 0.1 + 1e-9))
    if mask is not None:
        d._fem_enriched_nodes = np.asarray(mask, dtype=bool)
    u, phi = d.fem_symbols(value_shape=(2,), space="cover")
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xt, yt, _ = d.variable("tip", split=True)
    sym, tr, inner = jno.np.symgrad, jno.np.trace, jno.np.inner
    eu, ep = sym(u, [xi, yi]), sym(phi, [xi, yi])
    lam, mu = 0.3 / (1 - 0.09), 1.0 / 2.6
    fem = jno.fem(
        [
            lam * tr(eu) * tr(ep) + 2 * mu * inner(eu, ep, n_contract=2),
            u(xl, yl) - (0.0, 0.0),
            -1.0 * inner(jnp.array([0.0, -1.0]), phi.bind(x=xt, y=yt), n_contract=1),
        ]
    )
    A, b = fem.operator
    return A, jnp.asarray(b).reshape(-1), int(np.asarray(d.mesh.points).shape[0])


def test_vector_cover_layout_matches_the_assemblers_hard_mask():
    from jno.utils.solver.fem_soft_cover import compact_solver

    A, b, n = _elastic()
    idx = cover_dof_index(n, 2, n_comp=2)
    assert idx.shape == (n, 4) and A.shape[0] == n * 6
    mask = np.random.default_rng(8).random(n) < 0.3
    u_c = np.asarray(compact_solver(A, idx)(jnp.asarray(mask, dtype=float), b))
    Am, bm, _ = _elastic(mask)
    u_h = np.asarray(sparse_lu_solve(Am, bm))
    assert np.linalg.norm(u_c - u_h) <= 1e-11 * np.linalg.norm(u_h)
