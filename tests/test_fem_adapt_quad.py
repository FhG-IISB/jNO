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
    sol = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0]).solve()).reshape(-1)
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


def test_h_adaptivity_refuses_at_the_MESHER_not_the_estimator():
    """The estimator is no longer the blocker, so the refusal must come from the stage that really
    cannot proceed — mmg, which adapts simplices by edge split/collapse/swap. Two rounds, because one
    round never reaches a remesh. If this ever passes instead of raising, the loop is quietly turning
    a quad mesh into a triangular one.

    The two stages between them used to raise their own bare ``KeyError: 'triangle'`` first, so the
    user was told nothing and told it by the wrong component.
    """
    with pytest.raises(NotImplementedError, match=r"h-adaptive remeshing \(mmg\)"):
        _quad_fem().solve(adapt=jno.solve.remesh(max_iters=2))


def test_a_single_round_estimates_a_quad_mesh_without_refusing():
    """One round is solve + estimate with no remesh, and that much now WORKS on quads — which is the
    point of the estimator change. `adapt_history` carries the estimate, so the indicator is usable
    as a diagnostic on a quad mesh even though the loop cannot refine one yet."""
    fem = _quad_fem()
    fem.solve(adapt=jno.solve.remesh(max_iters=1))
    assert len(fem.adapt_history) == 1
    assert fem.adapt_history[0]["estimate"] > 0.0
    assert {c.type for c in fem.domain.mesh.cells} == {"quad", "line"}  # untouched, as it must be
