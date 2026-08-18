"""Interpolation-cover enrichment — ``space="cover"``.

p-refinement that adds DOFs at existing nodes instead of changing the mesh. The oracles here are
algebraic identities of the basis, not tolerances:

* a first-order cover spans exactly ``P2`` on a simplex, so it must reproduce **every global
  quadratic exactly** — strictly stronger than the usual linear patch test;
* the enrichment block is rank-deficient by exactly ``dim(dim+1)/2`` per scalar component
  (constants plus rotations), *independent of the mesh*, so the count is a real assertion;
* the enriched field must be **C⁰ across cells**, which is what fails if the covers are written in
  reference rather than physical coordinates.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jno.utils.solver.fem_cover import (
    cover_block,
    cover_count,
    cover_null_modes,
    expand_cover,
    nodal_scale,
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _simplex(dim, rng):
    """A deliberately non-degenerate, non-symmetric simplex."""
    v = np.eye(dim + 1, dim)
    v[1:] += 0.35 * rng.standard_normal((dim, dim))
    return v


def _p1_tables(verts, bary):
    """P1 hat values and PHYSICAL gradients on one simplex, plus the physical sample points."""
    dim = verts.shape[1]
    phi = bary  # (n_q, dim+1) — barycentric coordinates ARE the P1 hats
    jac = np.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim)
    inv = np.linalg.inv(jac)
    gref = np.vstack([-np.ones((1, dim)), np.eye(dim)])  # d(hat)/d(xi), (dim+1, dim)
    gphys = gref @ inv  # (dim+1, dim)
    dphi = np.broadcast_to(gphys[None, :, :], (bary.shape[0], dim + 1, dim))
    xq = bary @ verts
    return phi, np.array(dphi), xq


def _bary(dim, n, rng):
    b = rng.dirichlet(np.ones(dim + 1), size=n)
    return b


@pytest.mark.parametrize("dim", [2, 3])
def test_a_first_order_cover_reproduces_every_quadratic_exactly(dim):
    """The headline property: the enriched span is P2, so an arbitrary quadratic is recovered with
    the coefficients ``u_i = q(x_i)``, ``a_i = ½∇q(x_i)`` — no fitting, no least squares."""
    rng = np.random.default_rng(0)
    verts = _simplex(dim, rng)
    bary = _bary(dim, 60, rng)
    phi, dphi, xq = _p1_tables(verts, bary)
    scale = np.full(dim + 1, 0.7)  # deliberately not 1, so a missing 1/s would show

    c, b = rng.standard_normal(), rng.standard_normal(dim)
    a_mat = rng.standard_normal((dim, dim))
    a_mat = a_mat + a_mat.T  # the quadratic form must be symmetric
    q = lambda x: c + x @ b + 0.5 * np.einsum("...i,ij,...j->...", x, a_mat, x)  # noqa: E731
    gq = lambda x: b + x @ a_mat  # noqa: E731

    phi_e, dphi_e = expand_cover(phi, dphi, xq, verts, scale)
    blk = cover_block(dim)
    coef = np.zeros((dim + 1) * blk)
    for i in range(dim + 1):
        coef[i * blk] = q(verts[i])
        coef[i * blk + 1 : i * blk + 1 + dim] = 0.5 * gq(verts[i]) * scale[i]  # the 1/s is in the basis

    got = np.asarray(phi_e) @ coef
    assert np.abs(got - q(xq)).max() < 1e-12, f"value: max err {np.abs(got - q(xq)).max():.3e}"

    got_g = np.einsum("qnd,n->qd", np.asarray(dphi_e), coef)
    assert np.abs(got_g - gq(xq)).max() < 1e-11, f"gradient: max err {np.abs(got_g - gq(xq)).max():.3e}"


@pytest.mark.parametrize("dim", [2, 3])
def test_the_plain_hat_slots_still_reproduce_a_linear_field(dim):
    """With every cover coefficient zero the enriched basis must collapse to plain P1 — the
    guard that enrichment is an *addition*, not a replacement."""
    rng = np.random.default_rng(1)
    verts = _simplex(dim, rng)
    bary = _bary(dim, 40, rng)
    phi, dphi, xq = _p1_tables(verts, bary)
    scale = nodal_scale(verts, np.arange(dim + 1)[None, :])
    lin = lambda x: 2.0 - x @ np.arange(1.0, dim + 1)  # noqa: E731

    phi_e, _ = expand_cover(phi, dphi, xq, verts, scale)
    blk = cover_block(dim)
    coef = np.zeros((dim + 1) * blk)
    coef[::blk] = lin(verts)  # nodal values only
    assert np.abs(np.asarray(phi_e) @ coef - lin(xq)).max() < 1e-13


def test_the_cover_is_continuous_across_a_shared_face():
    """C⁰ conformity, and the reason the cover is written in PHYSICAL coordinates. Two triangles
    sharing an edge must agree on that edge for every basis function of a shared node. Reference
    covers ``h_i(ξ - ξ_i)`` would fail this — the two cells disagree about ξ."""
    a, b, c, d = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.3, 0.9]), np.array([1.2, 1.0])
    left, right = np.stack([a, b, c]), np.stack([b, d, c])  # share the edge b--c
    t = np.linspace(0.05, 0.95, 17)[:, None]
    edge = b[None, :] * (1 - t) + c[None, :] * t  # physical points on the shared edge

    def on_edge(verts, pts):
        jac = np.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)
        lam = np.linalg.solve(jac, (pts - verts[0]).T).T
        bary = np.column_stack([1 - lam.sum(1), lam])
        phi, dphi, xq = _p1_tables(verts, bary)
        return np.asarray(expand_cover(phi, dphi, xq, verts, np.ones(3))[0]), bary

    pl, _ = on_edge(left, edge)
    pr, _ = on_edge(right, edge)
    blk = cover_block(2)
    # node b is local 1 on the left, local 0 on the right; node c is local 2 / local 2
    for shared, il, ir in ((b, 1, 0), (c, 2, 2)):
        for m in range(blk):
            lv, rv = pl[:, il * blk + m], pr[:, ir * blk + m]
            assert np.abs(lv - rv).max() < 1e-12, f"node {shared} basis {m} jumps by {np.abs(lv - rv).max():.2e}"
    # and every basis function of the OPPOSITE vertex must vanish on the shared edge
    for m in range(blk):
        assert np.abs(pl[:, 0 * blk + m]).max() < 1e-12
        assert np.abs(pr[:, 1 * blk + m]).max() < 1e-12


@pytest.mark.parametrize("dim,n_comp", [(2, 1), (2, 2), (3, 1), (3, 3)])
def test_the_null_space_has_the_size_the_algebra_predicts(dim, n_comp):
    """``dim(dim+1)/2`` per scalar component — the ``dim`` constants plus ``dim(dim-1)/2``
    rotations. Independent of the mesh, which is what makes this a count and not a tolerance."""
    rng = np.random.default_rng(3)
    pts = rng.random((11, dim))
    modes = cover_null_modes(pts, dim, n_comp)
    assert modes.shape[0] == n_comp * dim * (dim + 1) // 2
    assert modes.shape[1] == 11 * cover_block(dim) * n_comp
    assert np.linalg.matrix_rank(modes) == modes.shape[0], "the modes must be independent"
    # the nodal-value slots carry none of the deficiency
    val_slots = (np.arange(11)[:, None] * cover_block(dim) * n_comp + np.arange(n_comp)[None, :]).ravel()
    assert np.abs(modes[:, val_slots]).max() == 0.0


@pytest.mark.parametrize("dim", [2, 3])
def test_the_predicted_modes_really_annihilate_the_enriched_field(dim):
    """Not just the right count — the right vectors. Feeding a predicted null mode as coefficients
    must give the identically-zero field on every cell of a real mesh."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(4)
    pts = rng.random((30 if dim == 2 else 40, dim))
    cells = Delaunay(pts).simplices
    scale = nodal_scale(pts, cells)
    modes = cover_null_modes(pts, dim, n_comp=1)
    blk = cover_block(dim)

    worst = 0.0
    for mode in modes:
        for t in cells[:25]:
            bary = _bary(dim, 12, rng)
            verts = pts[t]
            phi, dphi, xq = _p1_tables(verts, bary)
            phi_e, _ = expand_cover(phi, dphi, xq, verts, scale[t])
            # the mode's coefficients ARE stored against the scaled basis, so undo the scaling
            coef = np.zeros((dim + 1) * blk)
            for loc, i in enumerate(t):
                coef[loc * blk : (loc + 1) * blk] = mode[i * blk : (i + 1) * blk] * scale[i]
                coef[loc * blk] = 0.0
            worst = max(worst, float(np.abs(np.asarray(phi_e) @ coef).max()))
    assert worst < 1e-12, f"a predicted null mode produced a non-zero field: {worst:.3e}"


def test_a_higher_cover_order_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="first-order covers"):
        cover_count(2, cover_order=2)


def test_the_expansion_is_differentiable_in_the_node_positions():
    """A deformable-mesh problem needs ∂/∂X through the enrichment, not only through |det J|."""
    import jax.numpy as jnp

    rng = np.random.default_rng(5)
    verts0 = _simplex(2, rng)
    bary = _bary(2, 9, rng)

    def total(v):
        # the P1 tables in jnp, so the whole chain (geometry -> hats -> covers) is traced
        jac = jnp.stack([v[1] - v[0], v[2] - v[0]], axis=1)
        gphys = jnp.vstack([-jnp.ones((1, 2)), jnp.eye(2)]) @ jnp.linalg.inv(jac)
        phi = jnp.asarray(bary)
        dphi = jnp.broadcast_to(gphys[None], (phi.shape[0], 3, 2))
        xq = phi @ v
        pe, _ = expand_cover(phi, dphi, xq, v, jnp.ones(3))
        return jnp.sum(pe**2)

    g = jax.grad(total)(jnp.asarray(verts0))
    assert np.isfinite(np.asarray(g)).all() and float(jnp.abs(g).max()) > 0.0


# ---------------------------------------------------------------- assembled through jno.fem


def _poisson(space, size, dim=2, rhs=None, bc=0.0):
    """-Lap u = rhs with u = bc on the whole boundary, on the unit square/cube."""
    import jno

    grad, inner = jno.np.grad, jno.np.inner
    shp = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size) if dim == 2 else jno.Shape.box(0, 0, 0, 1, 1, 1, size=size)
    d = shp.domain()
    tol = 1e-9
    d.tag("walls", lambda *c: np.logical_or.reduce([(x < tol) | (x > 1 - tol) for x in c]))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = list(co[:dim])
    u, phi = d.fem_symbols(space=space)
    terms = [inner(grad(u, X), grad(phi, X), 1)]
    if rhs is not None:
        terms[0] = terms[0] - rhs(X) * phi
    terms.append(u(*cw[:dim]) - (bc(cw) if callable(bc) else bc))
    return d, jno.fem(terms), u, phi, X


def _dense_solve(fem):
    import jax.numpy as jnp

    dense = lambda a: jnp.asarray(a.todense() if hasattr(a, "todense") else a)  # noqa: E731
    return np.asarray(fem.solve(lambda a, b: jnp.linalg.solve(dense(a), jnp.asarray(b).reshape(-1)))).reshape(-1)


def _values(fem, d, sol):
    """The VALUE slot of each real node, and its coordinates."""
    nv = np.asarray(d.mesh.points).shape[0]
    blk = sol.size // nv
    pts = np.asarray(fem.field_points[0])
    return (sol[::blk], pts[::blk]) if blk > 1 else (sol, pts)


def test_the_assembled_stiffness_is_exact_on_a_harmonic_quadratic():
    """Assembly, isolated from boundary conditions. ``x²-y²`` is harmonic AND in the enriched span,
    so feeding its exact coefficients must make the FREE residual vanish at interior DOFs. This is
    the test that separates 'the element is right' from 'the boundary treatment is right'."""
    import jno

    grad, inner = jno.np.grad, jno.np.inner
    d, fem, u, phi, X = _poisson("cover", 0.4)
    stiff = inner(grad(u, X), grad(phi, X), 1)
    pts = np.asarray(d.mesh.points)[:, :2]
    s = nodal_scale(pts, np.asarray(d._cells_p1()))
    blk = cover_block(2)
    coef = np.zeros(len(pts) * blk)
    coef[0::blk] = pts[:, 0] ** 2 - pts[:, 1] ** 2
    coef[1::blk] = 0.5 * (2 * pts[:, 0]) * s
    coef[2::blk] = 0.5 * (-2 * pts[:, 1]) * s
    r = np.asarray(fem.eval(stiff, coef)).reshape(-1)
    onb = (pts[:, 0] < 1e-9) | (pts[:, 0] > 1 - 1e-9) | (pts[:, 1] < 1e-9) | (pts[:, 1] > 1 - 1e-9)
    interior = np.repeat(~onb, blk)
    assert np.abs(r[interior]).max() < 1e-12, f"interior residual {np.abs(r[interior]).max():.3e}"


def test_a_dirichlet_condition_pins_the_value_and_the_tangential_covers_only():
    """On an axis-aligned edge the tangential cover touches the trace and must be pinned; the
    NORMAL one is the ∂u/∂n freedom and must stay free -- pinning it is what capped the L2 rate at
    P1's. Corners have two tangents, so everything is pinned there. A handful of extra pins are the
    null-space gauge."""
    d, fem, *_ = _poisson("cover", 0.4)
    pts = np.asarray(d.mesh.points)[:, :2]
    tol = 1e-9
    onb = (pts[:, 0] < tol) | (pts[:, 0] > 1 - tol) | (pts[:, 1] < tol) | (pts[:, 1] > 1 - tol)
    corner = ((pts[:, 0] < tol) | (pts[:, 0] > 1 - tol)) & ((pts[:, 1] < tol) | (pts[:, 1] > 1 - tol))
    blk = cover_block(2)
    pinned = {int(i) for i, _ in d._fem_native_dirichlet_pairs}
    for n in np.flatnonzero(onb):
        assert n * blk in pinned, f"value slot of boundary node {n} not pinned"
        if corner[n]:
            assert {n * blk + 1, n * blk + 2} <= pinned, f"corner {n}: both covers must be pinned"
        else:
            on_x_edge = (pts[n, 1] < tol) | (pts[n, 1] > 1 - tol)  # tangent e_x
            t_slot = n * blk + 1 if on_x_edge else n * blk + 2
            f_slot = n * blk + 2 if on_x_edge else n * blk + 1
            assert t_slot in pinned, f"edge node {n}: tangential cover not pinned"
            assert f_slot not in pinned, f"edge node {n}: NORMAL cover pinned — the rate cap is back"


def test_the_cover_coefficients_are_pinned_to_zero_not_to_g():
    """A Dirichlet condition fixes the VALUE. Setting the covers to ``g`` as well would make the
    trace ``g + (x-x_i)·g/s``, which is not ``g``."""
    d, fem, *_ = _poisson("cover", 0.4, bc=lambda cw: cw[0] + 2.0)
    blk = cover_block(2)
    for dof, val in d._fem_native_dirichlet_pairs:
        if int(dof) % blk != 0:
            assert val == 0.0, f"cover slot {dof} pinned to {val}, must be 0"


@pytest.mark.parametrize("size", [0.4, 0.25])
def test_it_is_more_accurate_than_p1_on_the_same_mesh(size):
    """Homogeneous Dirichlet, where pinning the covers to zero is exact. Same mesh, more DOFs at the
    same nodes — the enriched answer must be better, or the enrichment is doing nothing."""
    import jno

    sin = jno.np.sin
    rhs = lambda X: 2 * np.pi**2 * sin(np.pi * X[0]) * sin(np.pi * X[1])  # noqa: E731
    errs = {}
    for space in ("Lagrange", "cover"):
        d, fem, *_ = _poisson(space, size, rhs=rhs)
        val, pts = _values(fem, d, _dense_solve(fem))
        ex = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        errs[space] = float(np.linalg.norm(val - ex) / np.linalg.norm(ex))
    assert errs["cover"] < errs["Lagrange"], f"cover {errs['cover']:.3e} vs P1 {errs['Lagrange']:.3e}"


def test_the_null_modes_are_gauged_away_without_changing_the_field():
    """The surviving zero modes are gauged by pinning a pivoting set of cover DOFs -- like a
    pressure pin, this selects one member of a solution family whose members are all the SAME
    field. Checked by the strongest available oracle: a one-edge (cantilever-style) Dirichlet
    problem, where the rotational mode survives the boundary pins, must still solve and must agree
    with plain P1 to discretisation accuracy."""
    import jno

    grad, inner = jno.np.grad, jno.np.inner
    vals = {}
    for space in ("Lagrange", "cover"):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
        d.tag("west", lambda x, y: x < 1e-9)
        co, cw = d.variable("interior", split=True), d.variable("west", split=True)
        X = [co[0], co[1]]
        u, phi = d.fem_symbols(space=space)
        fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(cw[0], cw[1]) - 0.0])
        val, pts = _values(fem, d, _dense_solve(fem))
        vals[space] = float(val[np.argmax(pts[:, 0])])  # u at the far edge
    assert np.isfinite(vals["cover"]), "the gauged system must solve"
    assert abs(vals["cover"] - vals["Lagrange"]) < 0.15 * abs(vals["Lagrange"]), (
        f"cover {vals['cover']:.4f} vs P1 {vals['Lagrange']:.4f} — a gauge must not move the field"
    )


def test_an_order_above_one_is_refused_by_name():
    import jno

    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols(space="cover", order=2)
    with pytest.raises(NotImplementedError, match="order=1"):
        jno.fem([inner(grad(u, X), grad(phi, X), 1), u(cw[0], cw[1]) - 0.0])


def _l2_error(d, sol, exact):
    """``‖u_h - u‖_L2`` by direct quadrature on the enriched basis.

    Not a nodal difference -- nodal FE values superconverge and would flatter the rate. And not the
    mass-matrix identity either: a cover space's mass matrix is genuinely SINGULAR (a null mode is
    the zero *function*, so it is in the mass kernel too), which the assembler now refuses to build.
    Evaluating the basis directly sidesteps both."""
    import basix
    from basix import CellType

    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d._cells_p1())
    qp, qw = basix.make_quadrature(CellType.triangle, 6)
    qp, qw = np.asarray(qp), np.asarray(qw)
    bary = np.column_stack([1.0 - qp.sum(1), qp])
    blk = sol.size // len(pts)
    sc = nodal_scale(pts, cells) if blk > 1 else None
    tot = 0.0
    for t in cells:
        v = pts[t]
        det = abs(np.linalg.det(np.stack([v[1] - v[0], v[2] - v[0]], axis=1)))
        phi, dphi, xq = _p1_tables(v, bary)
        if blk > 1:
            pe, _ = expand_cover(phi, dphi, xq, v, sc[t])
            uh = np.asarray(pe) @ np.concatenate([sol[i * blk : (i + 1) * blk] for i in t])
        else:
            uh = phi @ sol[t]
        tot += float(np.sum(qw * (uh - exact(xq)) ** 2) * det)
    return float(np.sqrt(tot))


def _study(space, sizes=(0.40, 0.30, 0.22, 0.16, 0.115)):
    """Measured L² error against ACTUAL mesh spacing ``h = sqrt(2|Omega|/n_cells)``.

    Not the nominal ``size=``: gmsh does not track it linearly, and fitting against it put even the
    known-good P1 control at order 1.59 instead of 2."""
    import jno

    sin = jno.np.sin
    PI = np.pi
    rhs = lambda X: 2 * PI**2 * sin(PI * X[0]) * sin(PI * X[1])  # noqa: E731
    ex = lambda p: np.sin(PI * p[:, 0]) * np.sin(PI * p[:, 1])  # noqa: E731
    hs, errs, dofs = [], [], []
    for size in sizes:
        d, fem, *_ = _poisson(space, size, rhs=rhs)
        hs.append(float(np.sqrt(2.0 / len(np.asarray(d._cells_p1())))))
        errs.append(_l2_error(d, _dense_solve(fem), ex))
        dofs.append(int(fem.dofs))
    return hs, errs, dofs, float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def test_p1_converges_at_its_own_rate_the_control():
    """The control. If this drifts, the measurement is broken, not the element."""
    _hs, errs, _nd, rate = _study("Lagrange")
    assert 1.75 <= rate <= 2.30, f"P1 L2 order {rate:.2f}; errs={errs}"


def test_the_cover_converges_one_order_faster_than_p1():
    """The p-refinement claim, measured. The enriched span is P2 element-wise, so the L2 rate is
    O(h^3) against P1's O(h^2). This took a fix to earn: pinning ALL covers at Dirichlet nodes
    capped the rate at 1.86 (the boundary layer is then effectively P1); pinning only the
    TANGENTIAL components -- the normal one is the ∂u/∂n freedom and never touches the trace on a
    straight facet -- released it. Measured 3.13 on sin·sin, err 37x below P1 at the finest mesh."""
    hs, e_cov, _d, r_cov = _study("cover")
    _h, e_p1, _d1, r_p1 = _study("Lagrange")
    assert all(b < a for a, b in zip(e_cov, e_cov[1:])), f"cover errors must decrease: {e_cov}"
    assert 2.6 <= r_cov <= 3.7, f"cover L2 order {r_cov:.2f}, expected ~3; errs={e_cov}"
    assert r_cov > r_p1 + 0.6, f"cover order {r_cov:.2f} must clearly beat P1's {r_p1:.2f}"
    assert e_cov[-1] < 0.1 * e_p1[-1], f"cover {e_cov[-1]:.2e} vs P1 {e_p1[-1]:.2e} at matched h"


def test_the_element_is_exact_on_a_distorted_mesh():
    """The decisive element check, run through ASSEMBLY rather than a solve.

    A harmonic quadratic is in the enriched span, so its free residual must vanish at interior DOFs
    however badly the cells are shaped. Done this way deliberately: the same statement *as a
    boundary-value problem* cannot pass, because pinning the covers to zero makes the Dirichlet
    trace the P1 interpolant of a quadratic ``g`` (see the module docstring's scope note). This
    isolates the element from that limitation."""
    import jno

    grad, inner = jno.np.grad, jno.np.inner
    rng = np.random.default_rng(7)
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    pts = np.asarray(d.mesh.points)
    tol = 1e-9
    inside = (pts[:, 0] > tol) & (pts[:, 0] < 1 - tol) & (pts[:, 1] > tol) & (pts[:, 1] < 1 - tol)
    pts[inside, :2] += 0.055 * rng.standard_normal((int(inside.sum()), 2))
    d.mesh.points = pts
    cells = np.asarray(d._cells_p1())
    v = pts[cells][:, :, :2]
    dets = np.abs(np.linalg.det(np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], -1)))
    assert dets.max() / dets.min() > 2.5, f"the distortion is not real: detJ spread {dets.max() / dets.min():.2f}"

    d.tag("walls", lambda x, y: (x < tol) | (x > 1 - tol) | (y < tol) | (y > 1 - tol))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols(space="cover")
    stiff = inner(grad(u, X), grad(phi, X), 1)
    fem = jno.fem([stiff, u(cw[0], cw[1]) - 0.0])
    xy = pts[:, :2]
    sc = nodal_scale(xy, cells)
    blk = cover_block(2)
    coef = np.zeros(len(xy) * blk)
    coef[0::blk] = xy[:, 0] ** 2 - xy[:, 1] ** 2
    coef[1::blk] = 0.5 * (2 * xy[:, 0]) * sc
    coef[2::blk] = 0.5 * (-2 * xy[:, 1]) * sc
    r = np.asarray(fem.eval(stiff, coef)).reshape(-1)
    onb = (xy[:, 0] < tol) | (xy[:, 0] > 1 - tol) | (xy[:, 1] < tol) | (xy[:, 1] > 1 - tol)
    err = float(np.abs(r[np.repeat(~onb, blk)]).max())
    assert err < 1e-11, f"distorted-mesh interior residual {err:.3e}"


def test_an_inhomogeneous_dirichlet_trace_is_only_piecewise_linear():
    """Pins the SCOPE LIMIT so it cannot regress silently into a claim of exactness.

    A Dirichlet condition pins the covers to zero, so the boundary trace is the P1 interpolant of
    ``g``. With a quadratic ``g`` the quadratic is therefore NOT recovered, even though it lies in
    the space. Documented in docs/fem.md; asserted here so the day someone fixes it, this test
    fails and tells them to update the docs."""
    d, fem, u, phi, X = _poisson("cover", 0.3, bc=lambda cw: cw[0] ** 2 - cw[1] ** 2)
    val, vpts = _values(fem, d, _dense_solve(fem))
    err = float(np.abs(val - (vpts[:, 0] ** 2 - vpts[:, 1] ** 2)).max())
    assert err > 1e-6, "the quadratic IS now recovered — the Dirichlet limitation is fixed, update the docs"
