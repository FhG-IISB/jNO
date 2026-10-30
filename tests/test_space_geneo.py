"""jno.space.geneo(): GenEO coarse vectors per overlapping Schwarz subdomain (Spillane et al., Numer. Math. 126,
2014).

Oracles: the device's batched local eigenproblems against ``scipy.linalg.eigh(A_N, B)`` per subdomain, with
``A_N`` summed on the host from the cell matrices (themselves checked against the operator in
test_fem_element_matrices.py); the iteration counts it exists for (a stiffness jump cut by the subdomains: GenEO
holds where rigid-body modes do not); the same operator through ``schwarz(coarse=)`` and ``Q >> S >> Q``; the
refusals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
from jax.experimental import sparse as jsparse

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

pytest.importorskip("pymetis", reason="Schwarz subdomains use METIS: the [metis] extra")

R = jno.shape.rect


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _layered(contrast, h=0.1):
    """A clamped 4 x 1 block of four layers, E alternating 1e4 and 1e4 / contrast (the tutorial's problem)."""
    soft, nu = 1e4 / contrast, 0.3
    d = (
        (
            R(0, 0.00, 4, 0.25).name("layer0").attach(E=1e4)
            + R(0, 0.25, 4, 0.50).name("layer1").attach(E=soft)
            + R(0, 0.50, 4, 0.75).name("layer2").attach(E=1e4)
            + R(0, 0.75, 4, 1.00).name("layer3").attach(E=soft)
        )
        .sized(h)
        .domain()
    )
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace
    lam, mu = d.E * nu / ((1 + nu) * (1 - 2 * nu)), d.E / (2 * (1 + nu))
    fem = jno.fem([2 * mu * ddot(ε(u_), ε(v_)) + lam * tr(ε(u_)) * tr(ε(v_)) + 0.1 * v_[1], u(xl, yl) - (0.0, 0.0)])
    return fem, (x, y)


def _iterations(fem, precond, ref):
    sol = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-8, maxiter=3000), precond=precond))
    assert np.abs(sol - ref).max() < 1e-5 * np.abs(ref).max()
    return fem.stats["linear_iterations"]["total"]


def test_the_local_eigenproblems_are_scipys():
    from jno.utils.solver.deflation import element_triplets
    from jno.utils.solver.geneo import _local_eigs, neumann_tables
    from jno.utils.solver.schwarz import local_blocks

    fem, _ = _layered(1e4, h=0.2)
    A = fem._op[0]
    S = jno.precond.schwarz(parts=6, coarse=False).build(A, fem)
    pat = S._pattern
    r, c, v, cell = (np.asarray(a) for a in element_triplets(fem))
    n, p = pat.n, pat.p
    idx = np.asarray(pat.idx)
    valid = idx < n
    mult = np.bincount(idx[valid], minlength=n)
    D = np.where(valid, 1.0 / np.maximum(mult[np.minimum(idx, n - 1)], 1), 1.0)
    nev = 8
    w, V = _local_eigs(
        local_blocks(pat, A)[0],
        *neumann_tables(pat, r, c, cell),
        jnp.asarray(v),
        jnp.asarray(D),
        jnp.asarray(~valid, float),
        tau=0.2,
        nev=nev,
    )
    w, V = np.asarray(w), np.asarray(V)

    Ad = np.asarray(A.todense())
    free = np.asarray(pat.part) < p
    cdofs = [set(r[cell == e]) for e in range(cell.max() + 1)]
    for i in range(p):
        s = idx[i][valid[i]]
        inset = np.zeros(n, bool)
        inset[s] = True
        inside = [e for e, ds in enumerate(cdofs) if all(inset[d] or not free[d] for d in ds)]
        t = np.isin(cell, inside) & free[r] & free[c]
        AN = sp.coo_matrix((v[t], (r[t], c[t])), shape=(n, n)).toarray()[np.ix_(s, s)]
        Di = np.diag(1.0 / mult[s])
        lam = sla.eigh(AN, Di @ Ad[np.ix_(s, s)] @ Di, eigvals_only=True)[:nev]
        np.testing.assert_allclose(w[i], lam, rtol=1e-7, atol=1e-10)
        assert np.abs(V[i][~valid[i]]).max(initial=0.0) == 0.0  # padding carries nothing


@pytest.mark.parametrize("parts", [16, 64])
def test_geneo_holds_the_iterations_across_a_stiffness_jump_where_rigid_modes_do_not(parts):
    fem, (x, y) = _layered(1e4)
    ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    S = jno.precond.schwarz(parts=parts, coarse=False)
    Qr = jno.precond.coarse([(1, 0), (0, 1), (-y, x)], parts=S)
    g = jno.space.geneo()
    Qg = jno.precond.coarse(g, parts=S)
    rigid, geneo = _iterations(fem, Qr >> S >> Qr, ref), _iterations(fem, Qg >> S >> Qg, ref)
    assert geneo < 0.6 * rigid and geneo <= 50, (rigid, geneo)
    assert g.counts.size == S._pattern.p and g.counts.max() > 3  # more than the three rigid modes where cut

    homogeneous, (x, y) = _layered(1)
    ref = np.asarray(homogeneous.solve(linear=jno.solve.lu()))
    S = jno.precond.schwarz(parts=parts, coarse=False)
    Qg = jno.precond.coarse(jno.space.geneo(), parts=S)
    assert _iterations(homogeneous, Qg >> S >> Qg, ref) <= 1.3 * geneo  # the contrast no longer costs


def _matrix(spec, fem):
    spec.prepare(fem)
    A = fem._op[0]
    ap = materialize_precond(spec, PrecondContext(LinearOperator(A), fem))
    return np.column_stack([np.asarray(ap(jnp.asarray(e))) for e in np.eye(A.shape[0])])


def test_two_level_schwarz_with_geneo_is_the_coarse_space_around_the_one_level_method():
    fem, _ = _layered(1e4, h=0.2)
    two = jno.precond.schwarz(parts=6, coarse=jno.space.geneo(nev=6))
    S = jno.precond.schwarz(parts=6, coarse=False)
    Q = jno.precond.coarse(jno.space.geneo(nev=6), parts=S)
    B_two, B_alg = _matrix(two, fem), _matrix(Q >> S >> Q, fem)
    np.testing.assert_array_equal(np.asarray(two._pattern.part), np.asarray(S._pattern.part))
    np.testing.assert_allclose(B_alg, B_two, rtol=1e-9, atol=1e-9 * np.abs(B_two).max())


def test_what_geneo_cannot_serve_raises():
    fem, _ = _layered(1, h=0.25)
    with pytest.raises(ValueError, match="parts= must be that preconditioner"):
        fem.solve(precond=jno.precond.coarse(jno.space.geneo(), parts=4))
    with pytest.raises(ValueError, match="parts= must be that preconditioner"):
        fem.solve(precond=jno.precond.coarse(jno.space.geneo()))
    with pytest.raises(ValueError, match="per subdomain"):
        fem.solve(linear=jno.solve.cg(), precond=jno.precond.amg(nullspace=jno.space.geneo()))
    with pytest.raises(ValueError, match="positive"):
        jno.space.geneo(tau=0)
    with pytest.raises(ValueError, match="positive int"):
        jno.space.geneo(nev=0)


def _cracked_pair(h=0.1):
    """Two problems on ONE plate: intact, and with an AT2 damage band across it (stiffness 1e-6 in the crack)."""
    d = R(0.0, 0.0, 1.0, 1.0).sized(h).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace

    def problem(g):
        u, v = d.fem_symbols(value_shape=(2,))
        u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
        return jno.fem([g * (2.0 * ddot(ε(u_), ε(v_)) + 1.5 * tr(ε(u_)) * tr(ε(v_))) - 0.1 * v_[1], u(xb, yb) - (0.0, 0.0)])

    damage = jno.np.exp(-jno.np.abs(y - 0.5) / 0.05)
    return problem(1.0), problem((1.0 - damage) ** 2 + 1e-6)


@pytest.mark.parametrize("traced", [False, True])
def test_the_space_recomputed_from_another_operator_is_that_operators_space(traced):
    from jno.utils.solver.deflation import coarse_apply, coarse_factor, element_triplets
    from jno.utils.solver.geneo import geneo_values

    intact, cracked = _cracked_pair()
    old = jno.precond.coarse(jno.space.geneo(nev=6), parts=jno.precond.schwarz(parts=6, coarse=False))
    new = jno.precond.coarse(jno.space.geneo(nev=6), parts=jno.precond.schwarz(parts=6, coarse=False))
    old.prepare(intact)
    new.prepare(cracked)
    A = cracked._op[0]
    spat = old.parts._pattern
    np.testing.assert_array_equal(np.asarray(spat.part), np.asarray(new.parts._pattern.part))  # one structure

    def recompute(A, vals):
        return geneo_values(old._geneo, spat, A, vals)[:3]

    vals = element_triplets(cracked)[2]
    null, W, kmask = jax.jit(recompute)(A, vals) if traced else recompute(A, vals)
    pat = old._pattern._replace(null=null, W=W, kmask=kmask)
    assert int(np.asarray(pat.kmask).sum()) > int(np.asarray(old._pattern.kmask).sum())  # the crack adds modes

    def Q(p):
        fwd = coarse_apply(p, coarse_factor(p, A))[0]
        return np.asarray(jax.vmap(fwd, in_axes=1, out_axes=1)(jnp.eye(A.shape[0])))

    Q_new = Q(new._pattern)
    np.testing.assert_allclose(Q(pat), Q_new, rtol=1e-9, atol=1e-9 * np.abs(Q_new).max())


def _periodic_layers(contrast, n=24):
    """Elasticity on a 2 x 1 strip of four horizontal layers (E alternating 1 and 1/contrast), periodic in x (the
    left face tied to the right one), clamped at the bottom: the solve runs on the REDUCED system P^T A P."""
    eps = 1e-9
    soft = 1.0 / contrast
    d = (
        (
            R(0, 0.00, 2, 0.25).name("l0").attach(E=1.0)
            + R(0, 0.25, 2, 0.50).name("l1").attach(E=soft)
            + R(0, 0.50, 2, 0.75).name("l2").attach(E=1.0)
            + R(0, 0.75, 2, 1.00).name("l3").attach(E=soft)
        )
        .sized(1.0 / n)
        .domain()
    )
    d.tag("left", lambda x, y: x < eps)
    d.tag("right", lambda x, y: x > 2 - eps)
    d.tag("bot", lambda x, y: (y < eps) & (x > eps))
    x, y, _ = d.variable("interior", split=True)
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace
    xb, yb, _ = d.variable("bot", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    lam, mu = d.E * 0.3 / (1.3 * 0.4), d.E / 2.6
    return jno.fem(
        [
            2 * mu * ddot(ε(u_), ε(v_)) + lam * tr(ε(u_)) * tr(ε(v_)) - 0.1 * v_[1] - 0.05 * v_[0] * jno.np.sin(np.pi * x),
            u(xb, yb) - (0.0, 0.0),
            u(xl, yl) - u(xr, yr),
        ]
    )


def _slip_wedge(h=0.2):
    """Elasticity on a plate with a clamped edge and a slanted wall under the exact elimination of n.u = 0, whose P
    has real weights (not a selection, unlike a tie)."""
    d = jno.shape.polygon([(0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (0.0, 1.0)]).sized(h).domain()
    d.tag("clamp", lambda x, y: x < 1e-9)
    d.tag("wall", lambda x, y: (np.abs(y - (x - 2.0)) < 1e-9) & (y > 1e-9) & (y < 1 - 1e-9))
    x, y, _ = d.variable("interior", split=True)
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    xc, yc, _ = d.variable("clamp", split=True)
    c = d.variable("wall", normals=True, split=True)
    us = u(c[0], c[1])
    return jno.fem(
        [
            2.0 * ddot(ε(u_), ε(v_)) + jno.np.trace(ε(u_)) * jno.np.trace(ε(v_)) - 0.1 * v_[1],
            u(xc, yc) - (0.0, 0.0),
            c[-2] * us[0] + c[-1] * us[1] - 0.0,
        ]
    )


def _hanging_nodes():
    """Poisson on a quad grid refined in its middle: the hanging nodes make a P with weights 1/2."""
    from jno.utils.solver.fem_refine import refine_domain

    d = refine_domain(R(0, 0, 1, 1).quad().structured(n=4).domain(compute_mesh_connectivity=False), [5, 6, 9, 10])
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xb, yb) - 0.0])


@pytest.mark.parametrize(
    "make", [lambda: _periodic_layers(1e4, n=8), _slip_wedge, _hanging_nodes], ids=["tie", "slip", "hanging"]
)
def test_on_a_reduced_system_the_cells_are_carried_onto_it(make):
    from jno.precond import _representative_operator
    from jno.utils.solver.deflation import element_triplets, solved_map, triplet_map

    fem = make()
    A = _representative_operator(fem)
    n = int(A.shape[0])
    assert n < fem.dofs  # the reduction removed unknowns
    r, c, v, _cell = (np.asarray(a) for a in element_triplets(fem, reduced_ok=True))
    rr, cc, src, w = triplet_map(solved_map(fem, n), r, c)
    K = sp.coo_matrix((v[src] * w, (rr, cc)), shape=(n, n)).toarray()
    Ad = np.asarray(A.todense())
    free = np.abs(Ad).sum(axis=1) != np.abs(np.diag(Ad))  # rows with couplings: not eliminated
    np.testing.assert_allclose(K[np.ix_(free, free)], Ad[np.ix_(free, free)], rtol=1e-10, atol=1e-10 * np.abs(Ad).max())


def test_geneo_on_a_periodic_tie_holds_the_contrast():
    fem = _periodic_layers(1e4)
    ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    S = jno.precond.schwarz(parts=16, coarse=False)
    one = _iterations(fem, S, ref)
    S = jno.precond.schwarz(parts=16, coarse=False)
    Q = jno.precond.coarse(jno.space.geneo(), parts=S)
    two = _iterations(fem, Q >> S >> Q, ref)
    homogeneous = _periodic_layers(1)
    href = np.asarray(homogeneous.solve(linear=jno.solve.lu()))
    S = jno.precond.schwarz(parts=16, coarse=False)
    Q = jno.precond.coarse(jno.space.geneo(), parts=S)
    assert two < 0.5 * one and two <= 1.3 * _iterations(homogeneous, Q >> S >> Q, href), (one, two)


def test_a_diagonally_dominant_operator_splits_into_its_edges():
    from jno.utils.solver.deflation import operator_graph
    from jno.utils.solver.geneo import edge_split, edge_values

    rng = np.random.default_rng(0)
    n = 40
    B = sp.random(n, n, density=0.15, random_state=rng, format="csr")
    B = B + B.T
    B.data[rng.random(B.data.size) < 0.3] *= -1.0
    B = sp.csr_matrix(0.5 * (B + B.T))
    B.setdiag(0.0)
    B.eliminate_zeros()
    A = (B + sp.diags(np.abs(B).sum(axis=1).A1 + rng.random(n))).tocoo()
    Aj = jsparse.BCOO((jnp.asarray(A.data), jnp.asarray(np.stack([A.row, A.col], 1))), shape=(n, n))
    e = edge_split(operator_graph(Aj)[0])
    v = np.asarray(jax.jit(lambda M: edge_values(e, M))(Aj))
    K = sp.coo_matrix((v, (e.rows, e.cols)), shape=(n, n)).toarray()
    np.testing.assert_allclose(K, A.toarray(), rtol=1e-12, atol=1e-12)  # the terms sum to the operator
    sub = np.arange(n // 2)  # the terms inside a subset: 0 <= A_i <= A
    inside = np.isin(e.rows, sub) & np.isin(e.cols, sub)
    both = np.bincount(e.cells[inside], minlength=e.cells.max() + 1)[e.cells]
    full = np.bincount(e.cells, minlength=e.cells.max() + 1)[e.cells]
    keep = both == full
    Ai = sp.coo_matrix((v[keep], (e.rows[keep], e.cols[keep])), shape=(n, n)).toarray()
    assert np.linalg.eigvalsh(Ai).min() > -1e-10 and np.linalg.eigvalsh(A.toarray() - Ai).min() > -1e-10


def test_an_operator_that_is_not_diagonally_dominant_has_no_edges():
    from jno.utils.solver.deflation import operator_graph
    from jno.utils.solver.geneo import edge_split

    fem, _ = _layered(1, h=0.25)
    assert edge_split(operator_graph(fem._op[0])[0]) is None  # elasticity


def test_geneo_on_fdm_works_from_the_operators_edges(monkeypatch):
    import scipy.sparse.linalg as spla

    import jno.jnp_ops as jnn
    from jno.precond import _Schwarz

    captured = {}
    real = _Schwarz.materialize

    def capture(self, ctx):
        captured["A"] = ctx.A.bcoo
        return real(self, ctx)

    monkeypatch.setattr(_Schwarz, "materialize", capture)

    d = jno.domain(R(0.0, 0.0, 1.0, 1.0, size=0.02).structured())
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    prob = jno.fdm([-(ui.d2(x) + ui.d2(y)) - f, u(xb, yb) - 0.0])
    ref = np.asarray(prob.solve())
    S = jno.precond.schwarz(parts=16, coarse=False)
    G = jno.space.geneo()
    Q = jno.precond.coarse(G, parts=S)
    M = Q >> S >> Q
    sol = np.asarray(prob.solve(linear=jno.solve.cg(tol=1e-10, maxiter=2000), precond=M))
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())
    assert G.counts.max() >= 1

    A = captured["A"] if not isinstance(captured["A"].data, jax.core.Tracer) else prob._representative_operator()
    n = int(A.shape[0])
    ctx = PrecondContext(LinearOperator(A), prob)
    eye = jnp.eye(n)
    apply = lambda spec: np.asarray(jax.vmap(materialize_precond(spec, ctx), in_axes=1, out_axes=1)(eye))  # noqa: E731
    Ad, b = sp.csr_matrix(np.asarray(A.todense())), np.asarray(jax.random.normal(jax.random.PRNGKey(0), (n,)))

    def count(B):  # scipy's CG on the assembled operator: FDM records no iteration count
        its = []
        M_ = spla.LinearOperator((n, n), matvec=lambda r: B @ r)
        _x, info = spla.cg(Ad, b, M=M_, rtol=1e-8, maxiter=2000, callback=lambda _x: its.append(1))
        assert info == 0
        return len(its)

    assert count(apply(M)) < 0.7 * count(apply(S))


def test_without_cells_or_edges_geneo_says_so():
    fem, _ = _layered(1, h=0.25)  # elasticity: not diagonally dominant; handed over as a bare matrix
    S = jno.precond.schwarz(parts=4, coarse=False)
    with pytest.raises(ValueError, match="diagonally dominant"):
        jno.precond.coarse(jno.space.geneo(), parts=S).build(fem._op[0], None)


def _plate(space, contrast, h):
    """A plate in bending on four strips of rigidity 1 and 1/contrast, simply supported, under a uniform load."""
    soft = 1.0 / contrast
    d = (
        (
            R(0, 0.00, 1, 0.25).name("s0").attach(D=1.0)
            + R(0, 0.25, 1, 0.50).name("s1").attach(D=soft)
            + R(0, 0.50, 1, 0.75).name("s2").attach(D=1.0)
            + R(0, 0.75, 1, 1.00).name("s3").attach(D=soft)
        )
        .sized(h)
        .domain()
    )
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    w, phi = d.fem_symbols(space=space)
    wi, pi_ = w.bind(x=x, y=y), phi.bind(x=x, y=y)
    H = lambda f: jno.np.hessian(f, [x, y])  # noqa: E731
    return jno.fem([d.D * jno.np.inner(H(wi), H(pi_), 2) - pi_, w(xb, yb) - 0.0])


@pytest.mark.parametrize("space", ["Morley", "Argyris"])
def test_the_cells_of_a_non_nodal_element_sum_to_the_operator(space):
    from jno.utils.solver.deflation import element_triplets

    fem = _plate(space, 1e3, 0.25)
    A = fem._op[0]
    n = int(A.shape[0])
    r, c, v, _cell = (np.asarray(a) for a in element_triplets(fem))
    K = sp.coo_matrix((v, (r, c)), shape=(n, n)).toarray()
    Ad = np.asarray(A.todense())
    free = np.abs(Ad - np.diag(np.diag(Ad))).sum(axis=1) > 0
    np.testing.assert_allclose(K[np.ix_(free, free)], Ad[np.ix_(free, free)], rtol=1e-10, atol=1e-10 * np.abs(Ad).max())


def test_geneo_on_a_morley_plate_holds_the_contrast():
    fem = _plate("Morley", 1e4, 0.03)
    ref = np.asarray(fem.solve(linear=jno.solve.lu()))

    def count(make):
        M = make()
        np.testing.assert_allclose(
            np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-8, maxiter=4000), precond=M)),
            ref,
            rtol=0,
            atol=1e-4 * np.abs(ref).max(),
        )
        return fem.stats["linear_iterations"]["total"]

    def bnn():
        S = jno.precond.schwarz(parts=16, coarse=False)
        Q = jno.precond.coarse(jno.space.geneo(), parts=S)
        return Q >> S >> Q

    one, two = count(lambda: jno.precond.schwarz(parts=16, coarse=False)), count(bnn)
    assert two < 0.5 * one, (one, two)
