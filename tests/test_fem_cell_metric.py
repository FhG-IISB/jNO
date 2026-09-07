"""Tests for the `dom.cell_metric` symbol (the element metric tensor `G = J^-T J^-1`).

`dom.cell_size` is `|det J|^(1/dim)` -- an isotropic SIZE, blind to stretch. A stabilization
parameter on a boundary-layer mesh needs the direction, which is what `G` carries: the SUPG/PSPG
tau of Tezduyar & Osawa, *CMAME* **190** (2000) Sec. 3 is built on it.

The oracle is `G` recomputed from the mesh in numpy -- `G` is constant on an affine simplex, so a
quadrature rule integrates it exactly and the assembled load vector must match to machine precision.
The test that matters most is `test_stretch_is_invisible_to_cell_size_and_not_to_cell_metric`: two
meshes with IDENTICAL cell sizes and different metrics.
"""

import jax
import numpy as np
import pytest

import jno

dense = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)  # noqa: E731
inner, trace = jno.np.inner, jno.np.trace


@pytest.fixture
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rect(n, lx=1.0, ly=1.0):
    return jno.Shape.rect(0.0, 0.0, lx, ly).structured(n=n).domain(compute_mesh_connectivity=False)


def _cell_metrics(dom):
    """`(G_c, area_c)` per cell, straight from the mesh -- the oracle these symbols must reproduce."""
    pts = np.asarray(dom.mesh.points)[:, :2]
    cells = np.asarray(dom._cells_p1())
    v = pts[cells]  # (n_cells, 3, 2)
    J = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=-1)  # columns are the edge vectors
    K = np.linalg.inv(J)
    return np.einsum("cki,ckj->cij", K, K), 0.5 * np.abs(np.linalg.det(J))


def _integrate(dom, expr_fn):
    """`∫_Ω f(G) dΩ`, read off the load vector of `M u = ∫ f(G) φ`."""
    u, v = dom.fem_symbols()
    xi, yi = dom.variable("interior", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui * vi - expr_fn(dom.cell_metric) * vi])
    return float(dense(fem.b).reshape(-1).sum())


def test_cell_metric_is_the_mesh_metric(_x64):
    """`∫ tr(G) dΩ` against `G` recomputed per cell in numpy. Exact: `G` is constant on an affine cell."""
    dom = _rect(6)
    G, area = _cell_metrics(dom)
    expect = float(np.sum(area * np.trace(G, axis1=1, axis2=2)))
    assert _integrate(dom, trace) == pytest.approx(expect, rel=1e-10)


def test_the_double_contraction_reads_as_written(_x64):
    """`inner(G, G, n_contract=2)` is `G:G` -- the second invariant the viscous leg of tau uses."""
    dom = _rect(5)
    G, area = _cell_metrics(dom)
    expect = float(np.sum(area * np.einsum("cij,cij->c", G, G)))
    got = _integrate(dom, lambda g: inner(g, g, n_contract=2))
    assert got == pytest.approx(expect, rel=1e-10)


def test_the_quadratic_form_reads_as_written(_x64):
    """`aᵀG a` for a constant direction `a` -- the advective leg of tau. Written as a nested `inner`."""
    dom = _rect(5)
    a = (0.6, -0.8)  # unit vector, so the result is a pure 1/length^2
    G, area = _cell_metrics(dom)
    expect = float(np.sum(area * np.einsum("i,cij,j->c", np.array(a), G, np.array(a))))
    got = _integrate(dom, lambda g: inner(a, inner(g, a, n_contract=1), n_contract=1))
    assert got == pytest.approx(expect, rel=1e-10)


def test_stretch_is_invisible_to_cell_size_and_not_to_cell_metric(_x64):
    """The whole reason this symbol exists.

    A 2x0.5 rectangle and the unit square, both meshed `n x n`, have the SAME cell areas and hence the
    same `|det J|^(1/dim)` -- `dom.cell_size` cannot tell them apart. Their cells are stretched 4:1
    relative to each other, and `G` says so.
    """
    n = 6
    square, stretched = _rect(n), _rect(n, lx=2.0, ly=0.5)

    def int_h(dom):
        u, v = dom.fem_symbols()
        xi, yi = dom.variable("interior", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        return float(dense(jno.fem([ui * vi - dom.cell_size * vi]).b).reshape(-1).sum())

    assert int_h(square) == pytest.approx(int_h(stretched), rel=1e-10), "cell_size must be blind to this"

    tr_sq, tr_st = _integrate(square, trace), _integrate(stretched, trace)
    # tr(G) = n^2 (1/s^2 + s^2) vs 2 n^2 per axis-aligned cell, so s=2 is a >2x separation.
    assert tr_st > 2.0 * tr_sq, f"cell_metric must see the stretch: {tr_st:.4g} vs {tr_sq:.4g}"


@pytest.mark.parametrize("build", ["1d", "nonnodal"])
def test_a_path_that_packs_no_jacobian_refuses_by_name(_x64, build):
    """Only the native 2-D/3-D volume kernel packs an element Jacobian. Everywhere else the symbol must
    SAY there is no metric.

    The placeholder exists only because the `Variable` constructor requires the tag in
    `domain.context`; it is deliberately zero-size so it can never be mistaken for a metric -- contrast
    an `np.ones` placeholder, which is what `dom.cell_size` had, and which read as a silent h = 1.0.
    """
    if build == "1d":
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        u, v = dom.fem_symbols()
        xi = dom.variable("interior", split=True)[0]
        ui, vi = u.bind(x=xi), v.bind(x=xi)
        terms = [ui.x * vi.x - trace(dom.cell_metric) * vi]
    else:
        pytest.importorskip("shapely", reason="shapely required for the box domain")
        from shapely.geometry import box

        dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4)
        u, v = dom.fem_symbols(space="Morley")
        xi, yi, _ = dom.variable("interior", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        H = jno.np.hessian
        terms = [inner(H(ui, [xi, yi]), H(vi, [xi, yi]), n_contract=2) - trace(dom.cell_metric) * vi]
    with pytest.raises(NotImplementedError, match="native 2-D/3-D assembler"):
        _ = jno.fem(terms).b


def test_a_boundary_term_is_refused_by_the_region_resolver(_x64):
    """A geometry symbol names no region, so pairing it with a boundary test function is refused there --
    the same way `dom.cell_size` already is. Pinned so the two symbols cannot drift apart."""
    dom = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    u, v = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _, nx, _ny = dom.variable("boundary", normals=True, split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    vb = v.bind(x=xb, y=yb)
    base = ui.x * vi.x + ui.y * vi.y
    for sym, name in ((dom.cell_metric, "cell_metric"), (dom.cell_size, "cell_size")):
        with pytest.raises(ValueError, match="spans multiple regions"):
            jno.fem([base, -nx * trace(sym) * vb if name == "cell_metric" else -nx * sym * vb])


def test_cell_metric_carries_a_mesh_gradient(_x64):
    """It is geometry, so an inverse problem that moves the mesh must see the metric move with it.

    Differentiated through the ASSEMBLY (a `.trainable()` coordinate scattered into the element
    Jacobian), not through a numpy restatement of the formula -- the latter would pass even if the
    assembler never used the moved points.
    """
    import jax.numpy as jnp

    dom = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain()
    ym = dom.variable("mov", where=lambda x, y: (x > 0.15) & (x < 0.85) & (y > 0.15) & (y < 0.85), split=True)[1]
    ym.trainable(name="Y0")
    ids = np.asarray(dom._trainable_coords[0]["ids"], dtype=int)
    y0 = jnp.asarray(np.asarray(dom.mesh.points)[ids, 1])

    u, v = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui * vi - trace(dom.cell_metric) * vi])
    total = lambda yv: jnp.sum(fem.operator.evaluate({"Y0": yv})[1])  # noqa: E731

    g = np.asarray(jax.grad(total)(y0))
    assert float(np.max(np.abs(g))) > 1e-6, "a metric frozen at the reference mesh would give exactly zero"
    k = int(np.argmax(np.abs(g)))
    eps, e = 1e-6, np.zeros(y0.shape)
    e[k] = 1.0
    fd = float((total(y0 + eps * e) - total(y0 - eps * e)) / (2 * eps))
    assert g[k] == pytest.approx(fd, rel=1e-5), f"AD {g[k]:.6e} vs FD {fd:.6e}"
