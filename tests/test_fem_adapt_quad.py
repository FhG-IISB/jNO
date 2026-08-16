"""The recovery error estimator on quadrilateral and hexahedral cells.

The adaptive loop is *solve → estimate → mark → size → remesh → transfer*, and only two of those
stages were ever simplicial. This file covers the **estimate** stage, which is the one that decides
whether adaptivity is worth anything: an indicator that merely returns non-negative numbers will
drive a loop that runs, looks like it works, and refines the wrong cells.

So the assertion here is the **effectivity index** ``eta / true error``, measured against a
manufactured solution on a sequence of meshes. It must stay bounded and approach a constant; a
falling effectivity is the failure mode this file exists to catch, and it is exactly what the
original centroid-rule indicator did on quads (0.81 → 0.53 → 0.35 over n = 8, 16, 32) while looking
entirely healthy.

Every measurement runs on a **triangle/tet control** built from the same geometry, so a quad-specific
bug cannot hide behind something that was wrong for both.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver.fem_adapt import _element_gradients, zz_error_indicators
from jno.utils.solver.fem_lagrange import lagrange_on
from jno.utils.solver.fem_native import _basix_ordered, mesh_cell_type

PI = np.pi
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _domain(cell, n, dim=2):
    s = (jno.Shape.rect(0, 0, 1, 1) if dim == 2 else jno.Shape.box(0, 0, 0, 1, 1, 1)).structured(n=n)
    return (s.quad() if cell == "tensor" else s).domain()


def _solve_poisson(cell, n):
    """-Delta u = 2 pi^2 sin(pi x) sin(pi y), u = 0 on the boundary. Exact: sin(pi x) sin(pi y)."""
    d = _domain(cell, n)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = 2 * PI**2 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])
    # A DIRECT solve on purpose: this file measures the ESTIMATOR, and the default Jacobi-BiCGStab makes
    # the measurement depend on iterative convergence -- which under x64 state leaked from a neighbouring
    # test file returned a NaN residual on the n=32 mesh. These systems are ~1k DOFs; LU is exact and free.
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    return d, sol


def _grad_exact(p):
    return np.stack(
        [PI * np.cos(PI * p[:, 0]) * np.sin(PI * p[:, 1]), PI * np.sin(PI * p[:, 0]) * np.cos(PI * p[:, 1])], axis=1
    )


def _true_energy_error(d, sol):
    """``||grad(u - u_h)||_L2``, by the element's OWN quadrature.

    Never a single centroid sample: on a quad the centre is the superconvergent point, so a centroid
    oracle reports an error converging at rate 2 where the true energy error converges at rate 1 —
    which makes a broken estimator look excellent. (That mistake was made first here.)
    """
    ct = mesh_cell_type(d, 2)
    spec = lagrange_on(ct, 1, quad_degree=4)
    cells = _basix_ordered(np.asarray(d.mesh.cells_dict[ct]), ct)
    X = np.asarray(d.mesh.points)[:, :2][cells]
    N = np.asarray(spec.ref_values)[:, :, 0]
    dN = np.asarray(spec.ref_grads)[:, :, 0, :]
    J = np.einsum("cai,qak->cqik", X, dN)
    gh = np.einsum("qak,cqki,ca->cqi", dN, np.linalg.inv(J), sol[cells])
    xq = np.einsum("qa,cai->cqi", N, X)
    err2 = np.sum((gh - _grad_exact(xq.reshape(-1, 2)).reshape(gh.shape)) ** 2, axis=2)
    return float(np.sqrt(np.einsum("cq,cq,q->", err2, np.abs(np.linalg.det(J)), np.asarray(spec.quad_weights))))


# ----------------------------------------------------------------- the gradient sampler is correct


@pytest.mark.parametrize("cell", ["simplex", "tensor"])
@pytest.mark.parametrize("dim", [2, 3])
def test_the_cell_measures_sum_to_the_domain_volume(cell, dim):
    """A quad's measure is not ``|det J|`` times the reference volume — det J varies over a bilinear
    cell, so it has to be integrated. The unit square/cube is the oracle."""
    _, measure, _ = _element_gradients(_domain(cell, 6 if dim == 2 else 3, dim))
    assert measure.sum() == pytest.approx(1.0, rel=1e-12)
    assert (measure > 0).all()


@pytest.mark.parametrize("cell", ["simplex", "tensor"])
@pytest.mark.parametrize("dim", [2, 3])
def test_a_linear_field_recovers_its_exact_constant_gradient(cell, dim):
    """Both families reproduce linears exactly, so this catches a wrong Jacobian or — the trap the
    quad/hex work hit repeatedly — cells fed to the tabulated basis in VTK rather than basix order,
    which evaluates a bow-tie."""
    from jno.utils.solver.fem_adapt import _recover_nodal_gradient

    d = _domain(cell, 6 if dim == 2 else 3, dim)
    p = np.asarray(d.mesh.points)[:, :dim]
    coef = np.array([2.0, -3.0, 0.5][:dim])
    g_star, _, _, _ = _recover_nodal_gradient(d, p @ coef)
    np.testing.assert_allclose(g_star, np.broadcast_to(coef, g_star.shape), atol=1e-12)


def test_the_quad_gradient_sample_is_superconvergent():
    """The design turns on this: the centroid is the Barlow point of a Q1 gradient, where it is
    ``O(h^2)`` accurate against ``O(h)`` elsewhere in the cell. Zienkiewicz–Zhu recovery averages its
    samples, so a non-superconvergent sample recovers nothing.

    Measured as a rate, not a tolerance: the raw per-cell gradient must converge at ~2 on quads
    (where it is Barlow-sampled) and at ~1 on triangles (where it is the exact constant gradient of a
    cell whose true gradient varies).
    """
    rates = {}
    for cell in ("simplex", "tensor"):
        errs = []
        for n in (8, 16, 32):
            d = _domain(cell, n)
            p = np.asarray(d.mesh.points)[:, :2]
            g, _, cells = _element_gradients(d)
            f = np.sin(1.7 * p[:, 0]) * np.exp(0.6 * p[:, 1])
            g_cell = np.einsum("cad,ca->cd", g, f[cells])
            cen = p[cells].mean(axis=1)
            exact = np.stack(
                [
                    1.7 * np.cos(1.7 * cen[:, 0]) * np.exp(0.6 * cen[:, 1]),
                    0.6 * np.sin(1.7 * cen[:, 0]) * np.exp(0.6 * cen[:, 1]),
                ],
                axis=1,
            )
            errs.append(np.abs(g_cell - exact).max())
        rates[cell] = np.log2(errs[0] / errs[-1]) / 2.0
    assert rates["tensor"] > 1.8, f"the quad centroid sample is not superconvergent (rate {rates['tensor']:.2f})"
    assert 0.8 < rates["simplex"] < 1.3, f"the simplex control moved (rate {rates['simplex']:.2f})"


# ------------------------------------------------------------------- the indicator is an ESTIMATOR


@pytest.mark.parametrize("cell", ["simplex", "tensor"])
def test_the_effectivity_index_is_bounded_and_converges(cell):
    """The assertion this file exists for. ``eta / true error`` must approach a constant — an
    indicator that shrinks faster than the error still runs and still marks cells, just the wrong
    ones. The centroid-rule version scored 0.81 / 0.53 / 0.35 here on quads.
    """
    eff = []
    for n in (8, 16, 32):
        d, sol = _solve_poisson(cell, n)
        _, est = zz_error_indicators(d, sol)
        eff.append(est / _true_energy_error(d, sol))
    assert all(0.5 < e < 2.0 for e in eff), f"{cell}: effectivity out of range: {eff}"
    assert abs(eff[-1] - 1.0) < abs(eff[0] - 1.0), f"{cell}: effectivity is not converging to 1: {eff}"


@pytest.mark.parametrize("cell", ["simplex", "tensor"])
def test_the_estimate_falls_at_the_rate_the_error_does(cell):
    """Same statement as a rate rather than a ratio, which is what a mis-sampled indicator breaks:
    the energy error of a P1/Q1 solve is ``O(h)``, so the estimate must be too."""
    ests = []
    for n in (8, 16, 32):
        d, sol = _solve_poisson(cell, n)
        ests.append(zz_error_indicators(d, sol)[1])
    rate = np.log2(ests[0] / ests[-1]) / 2.0
    assert 0.8 < rate < 1.3, f"{cell}: estimate converges at rate {rate:.2f}, expected ~1"


def test_the_indicator_finds_the_feature_it_should():
    """A localized bump: the largest indicators must sit on the cells that carry it, on both cells.
    A globally-correct estimate that marks uniformly would pass the rate tests above and be useless.
    """
    for cell in ("simplex", "tensor"):
        d = _domain(cell, 16)
        p = np.asarray(d.mesh.points)[:, :2]
        f = np.exp(-200.0 * ((p[:, 0] - 0.3) ** 2 + (p[:, 1] - 0.7) ** 2))
        eta, _ = zz_error_indicators(d, f)
        _, _, cells = _element_gradients(d)
        cen = p[cells].mean(axis=1)
        hot = cen[np.argsort(eta)[-10:]]  # the 10 worst cells
        assert np.abs(hot - [0.3, 0.7]).max() < 0.2, f"{cell}: indicator peaked away from the bump"


@pytest.mark.parametrize("cell", ["simplex", "tensor"])
def test_a_field_the_mesh_represents_exactly_has_a_vanishing_indicator(cell):
    """A linear field is in the space, so the true error is zero and the indicator must be too —
    the zero-extreme, and a check that the recovery is not manufacturing error out of geometry."""
    d = _domain(cell, 8)
    p = np.asarray(d.mesh.points)[:, :2]
    eta, est = zz_error_indicators(d, 2.0 * p[:, 0] - 3.0 * p[:, 1])
    assert est < 1e-10 and eta.max() < 1e-10


# ----------------------------------------------------------------------------- the loop still refuses


def _quad_fem():
    d = _domain("tensor", 6)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])


def test_a_structured_quad_plan_refuses_rather_than_silently_doing_nothing():
    """A `.structured()` lattice's resolution is its cell COUNTS, not a size field — so rebuilding it
    against a marked size field returns the very same mesh. That is a silent no-op: the loop runs,
    reports rounds, and refines nothing. Refuse it and say which knob actually controls resolution."""
    fem = _quad_fem()  # Shape.rect(...).structured(n=6).quad()
    with pytest.raises(NotImplementedError, match="cell COUNTS"):
        fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=2))


def test_a_single_round_estimates_a_quad_mesh_without_refusing():
    """One round is solve + estimate with no remesh, and that much now WORKS on quads — which is the
    point of the estimator change. `adapt_history` carries the estimate, so the indicator is usable
    as a diagnostic on a quad mesh even though the loop cannot refine one yet."""
    fem = _quad_fem()
    fem.solve(adapt=jno.solve.remesh(max_iters=1))
    assert len(fem.adapt_history) == 1
    assert fem.adapt_history[0]["estimate"] > 0.0
    assert {c.type for c in fem.domain.mesh.cells} == {"quad", "line"}  # untouched, as it must be


# ------------------------------------------------------------------- the loop, on quadrilaterals

L_SHAPE = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]


def _l_shape(cell, size=0.12):
    s = jno.Shape.polygon(L_SHAPE, size=size)
    d = (s.quad() if cell == "tensor" else s).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])


def _ritz_energy(d, sol):
    """``E = 1/2 int|grad u|^2 - int u`` by the element's OWN quadrature.

    ``E_h`` decreases to ``E_exact`` from above and ``E_h - E_exact = 1/2 ||u - u_h||_E^2``, so it
    ranks meshes without an exact solution. Not a centroid rule: the differences being ranked are
    ~1e-5 against energies of ~1e-2, inside a midpoint rule's own error — measured, the coarse
    version reversed this comparison and reported adaptivity as 0.70x WORSE than uniform.
    """
    spec = lagrange_on(mesh_cell_type(d, 2), 1, quad_degree=4)
    cells = _basix_ordered(np.asarray(d.mesh.cells_dict[mesh_cell_type(d, 2)]), mesh_cell_type(d, 2))
    X = np.asarray(d.mesh.points)[:, :2][cells]
    N, dN = np.asarray(spec.ref_values)[:, :, 0], np.asarray(spec.ref_grads)[:, :, 0, :]
    J = np.einsum("cai,qak->cqik", X, dN)
    gh = np.einsum("qak,cqki,ca->cqi", dN, np.linalg.inv(J), sol[cells])
    uh = np.einsum("qa,ca->cq", N, sol[cells])
    return float(np.einsum("cq,cq,q->", 0.5 * np.sum(gh**2, axis=2) - uh, np.abs(np.linalg.det(J)), spec.quad_weights))


@pytest.mark.slow
def test_the_quad_loop_refines_and_stays_all_quad():
    """The loop end to end. mmg cannot adapt a quad mesh, so the remesh stage rebuilds the `Shape`
    plan at the marked size field — which means the result has to be checked for purity, not assumed:
    a leftover triangle would be a mixed mesh the assembler refuses less clearly."""
    d, fem = _l_shape("tensor")
    n0 = len(d.mesh.points)
    sol = np.asarray(fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=3, refine_factor=1.6))).reshape(-1)
    blocks = {c.type: len(c.data) for c in fem.domain.mesh.cells}
    assert len(sol) > n0, "the loop did not add DOFs"
    assert "triangle" not in blocks, f"the rebuilt mesh is mixed: {blocks}"
    ests = [h["estimate"] for h in fem.adapt_history]
    assert all(b < a for a, b in zip(ests, ests[1:])), f"the estimate did not fall monotonically: {ests}"


@pytest.mark.slow
def test_refinement_concentrates_at_the_re_entrant_corner():
    """Where the DOFs go, not just how many. A globally-uniform 'refinement' would still pass the
    test above."""
    for cell in ("simplex", "tensor"):
        pytest.importorskip("mmgpy") if cell == "simplex" else None
        d, fem = _l_shape(cell)
        fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=3, refine_factor=1.6))
        p = np.asarray(fem.domain.mesh.points)[:, :2]
        _, meas, cells = _element_gradients(fem.domain)
        h = np.sqrt(meas)
        r = np.linalg.norm(p[cells].mean(axis=1) - [0.5, 0.5], axis=1)
        assert h[r < 0.15].mean() < h[r > 0.4].mean(), f"{cell}: cells are not finer at the corner"


@pytest.mark.slow
def test_adaptivity_beats_uniform_refinement_at_matched_dofs():
    """The measurement that decides whether any of this is worth running — and the one whose absence
    let the r-adaptivity regression (#109) sit on main through three releases. Adapting must beat
    simply making the mesh finer everywhere, at the same DOF count."""
    dref, fref = _l_shape("tensor", size=0.022)
    e_ref = _ritz_energy(dref, np.asarray(fref.solve()).reshape(-1))

    d0, f0 = _l_shape("tensor")
    sol = np.asarray(f0.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=4, refine_factor=1.6))).reshape(-1)
    err_adaptive = _ritz_energy(f0.domain, sol) - e_ref

    best = None
    for size in np.linspace(0.045, 0.10, 12):
        du, fu = _l_shape("tensor", size=size)
        n = len(du.mesh.points)
        if best is None or abs(n - len(sol)) < abs(best[0] - len(sol)):
            best = (n, _ritz_energy(du, np.asarray(fu.solve()).reshape(-1)) - e_ref)
    n_uniform, err_uniform = best

    assert abs(n_uniform - len(sol)) < 0.15 * len(sol), "the uniform comparison is not at matched DOFs"
    assert err_adaptive > 0 and err_uniform > 0, "the Ritz energy must exceed the reference from above"
    assert err_adaptive < err_uniform, (
        f"adaptivity did not pay: {err_adaptive:.3e} at {len(sol)} dofs vs uniform {err_uniform:.3e} at {n_uniform}"
    )


def test_a_quad_mesh_with_no_geometry_refuses_by_name(tmp_path):
    """The rebuild needs a plan to rebuild FROM. A mesh loaded from a file has none, and that is a
    real case — it must refuse rather than fall through to the simplex path."""
    meshio = pytest.importorskip("meshio")
    d0 = _domain("tensor", 5)
    path = str(tmp_path / "q.vtu")
    meshio.write(path, meshio.Mesh(points=d0.mesh.points, cells=[("quad", d0.mesh.cells_dict["quad"])]))
    d = jno.domain(path)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    with pytest.raises(NotImplementedError, match="no geometry to rebuild from"):
        fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=2))


def test_a_hex_mesh_refuses_with_the_real_reason():
    """Not 'not implemented': no general all-hex mesher exists, so there is nothing to remesh TO."""
    d = _domain("tensor", 3, dim=3)
    u, v = d.fem_symbols()
    ci = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=ci[0], y=ci[1], z=ci[2]), v.bind(x=ci[0], y=ci[1], z=ci[2])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(*cb[:3]) - 0.0])
    with pytest.raises(NotImplementedError, match="no general all-hex mesher exists"):
        fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=2))
