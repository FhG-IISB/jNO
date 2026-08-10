"""Closed-form axisymmetric view factors, against cases with known exact answers.

Every reference here is analytic, not another numerical scheme: Howell's catalogue for coaxial discs,
the closure identity for a closed cavity, and the exact concentric-cylinder factors together with
their exact shadow boundary. Each of the numerical hazards documented in
``jno/domain/view_factor_closed`` has a test, because every one of them was found producing a
plausible wrong answer rather than a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from jno.domain.view_factor_closed import (
    segments_from_polygons,
)
from jno.domain.view_factor_closed import (
    view_factor_axisymmetric_closed as vf,
)


def seg(p0, p1, n, normal):
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    P = np.asarray(p0)[None, :] + t * (np.asarray(p1) - np.asarray(p0))[None, :]
    return P[:-1], P[1:], np.tile(np.asarray(normal, float), (n, 1))


def stack(*parts):
    return (np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]), np.vstack([p[2] for p in parts]))


def ring_areas(e0, e1):
    return 2 * np.pi * (0.5 * (e0[:, 0] + e1[:, 0])) * np.linalg.norm(e1 - e0, axis=1)


def band(F, A, ia, ib):
    return float((A[ia, None] * F[np.ix_(ia, ib)]).sum() / A[ia].sum())


def F_discs_analytic(r1, r2, L):
    """Howell C-41 / Modest App. D: coaxial parallel discs."""
    R1, R2 = r1 / L, r2 / L
    X = 1.0 + (1.0 + R2**2) / R1**2
    return 0.5 * (X - np.sqrt(X**2 - 4.0 * (R2 / R1) ** 2))


@pytest.mark.parametrize("rr", [0.25, 0.5, 1.0, 2.0])
def test_coaxial_discs_match_howell(rr):
    """The azimuthal integral is exact, so the only error is the meridional rule."""
    E0, E1, N = stack(seg((1e-9, 0.0), (rr, 0.0), 24, (0.0, 1.0)), seg((1e-9, 1.0), (rr, 1.0), 24, (0.0, -1.0)))
    F = vf(E0, E1, N, n_quad=6)
    got = band(F, ring_areas(E0, E1), np.arange(24), np.arange(24, 48))
    assert abs(got / F_discs_analytic(rr, rr, 1.0) - 1.0) < 1e-9


def test_closed_cavity_closure_without_enforcement():
    """A closed enclosure must have unit row sums with NO Sinkhorn normalisation applied."""
    E0, E1, N = stack(
        seg((1e-9, 0.0), (0.1, 0.0), 24, (0.0, 1.0)),
        seg((0.1, 0.0), (0.1, 0.1), 24, (-1.0, 0.0)),
        seg((0.1, 0.1), (1e-9, 0.1), 24, (0.0, -1.0)),
    )
    F = vf(E0, E1, N, n_quad=6)
    assert np.isfinite(F).all()
    assert np.abs(F.sum(1) - 1.0).max() < 2e-2


def test_reciprocity_is_exact():
    """A_i F_ij = A_j F_ji to round-off. Guaranteed by using one quadrature order on both sides and
    by mirroring the occlusion, not by any post-hoc symmetrisation."""
    E0, E1, N = stack(
        seg((1e-9, 0.0), (0.1, 0.0), 16, (0.0, 1.0)),
        seg((0.1, 0.0), (0.1, 0.1), 16, (-1.0, 0.0)),
        seg((0.1, 0.1), (1e-9, 0.1), 16, (0.0, -1.0)),
    )
    F = vf(E0, E1, N, n_quad=4)
    A = ring_areas(E0, E1)
    M = A[:, None] * F
    assert np.abs(M - M.T).max() / M.max() < 1e-12


def test_flat_annulus_cannot_see_itself():
    """A plane has zero self-view: a' + b' cos t vanishes identically. Exactly 0, not merely small."""
    E0, E1, N = seg((0.02, 0.0), (0.10, 0.0), 12, (0.0, 1.0))
    assert np.abs(vf(E0, E1, N, n_quad=3)).max() == 0.0


def test_coincident_points_are_finite_not_nan():
    """The diagonal's q == p term has a + b = 0, where every G diverges individually while their
    combination stays finite. A concave wall must give a positive self-view, not NaN."""
    E0, E1, N = seg((0.1, 0.0), (0.1, 0.1), 12, (-1.0, 0.0))
    F = vf(E0, E1, N, n_quad=3)
    assert np.isfinite(F).all()
    assert np.diag(F).min() > 0.0


@pytest.mark.parametrize("ratio", [0.4, 0.55, 0.8])
def test_concentric_cylinders_with_occlusion(ratio):
    """F21 = r1/r2 and F22 = 1 - r1/r2 for infinite coaxial cylinders; a long finite one approaches
    it, the residual being what escapes the open ends. Exercises the tangency and horizontal-side
    degeneracies of the occlusion test.

    Resolution is chosen so the MERIDIONAL rule is not the limiting error: at ratio 0.8 the annular
    gap is 0.2 while a coarser element would be 1.0 long, and 3 Gauss points across an element five
    times the gap width cannot resolve the near field (measured error +0.16, falling to +0.0001 at
    n=80/n_quad=6). That is the regime the source paper handles with 20-node rules and element
    splitting; here it is simply resolved.
    """
    r2, H, n = 1.0, 40.0, 80
    r1 = ratio * r2
    zs = np.linspace(0.0, H, n + 1)
    E0 = np.vstack([np.c_[np.full(n, r1), zs[:-1]], np.c_[np.full(n, r2), zs[:-1]]])
    E1 = np.vstack([np.c_[np.full(n, r1), zs[1:]], np.c_[np.full(n, r2), zs[1:]]])
    N = np.vstack([np.tile((1.0, 0.0), (n, 1)), np.tile((-1.0, 0.0), (n, 1))])
    F = vf(E0, E1, N, occluders=[(r1, -H, r1, 2 * H)], n_quad=6)
    A = ring_areas(E0, E1)
    mid = np.arange(n, 2 * n)[n // 2 - 2 : n // 2 + 2]
    assert abs(band(F, A, mid, np.arange(n)) - ratio) < 5e-3
    assert abs(band(F, A, mid, np.arange(n, 2 * n)) - (1.0 - ratio)) < 5e-3


def test_occlusion_is_symmetric_across_a_horizontal_side():
    """The chord i -> j at azimuth t IS the chord j -> i, so visibility must be symmetric. A chord
    crossing a HORIZONTAL side degenerates to a perfect square whose discriminant is exactly zero;
    rounding once split it into two roots that both missed the crossing, giving 'blocked' one way and
    'open' the other -- a full pi of asymmetry on the reference furnace."""
    from jno.domain.view_factor_closed import _visible_gaps

    mids = np.array([[0.0631, 0.0027], [0.03725, -0.016], [0.05, 0.02], [0.02, -0.03]])
    sides = np.array([[0.0, 0.0, 0.0631, 0.0], [0.0671, -0.012, 0.01, -0.012]])
    G0, G1 = _visible_gaps(mids, sides)
    vis = np.clip(G1 - G0, 0.0, None).sum(-1)
    assert np.abs(vis - vis.T).max() < 1e-12


def test_segments_from_polygons_includes_interior_rings():
    """A leftover-void region is scene minus solids -- a polygon WITH HOLES. Taking only exteriors
    silently drops most of the occluders."""
    shapely = pytest.importorskip("shapely.geometry")
    outer = shapely.box(0.0, 0.0, 1.0, 1.0)
    hole = shapely.box(0.3, 0.3, 0.6, 0.6)
    segs = segments_from_polygons([outer.difference(hole)])
    assert len(segs) == 8  # 4 exterior + 4 interior edges
    assert np.isclose(np.abs(segs[:, [0, 2]]).max(), 1.0)
    assert ((segs[:, 0] > 0.29) & (segs[:, 0] < 0.61)).any()
