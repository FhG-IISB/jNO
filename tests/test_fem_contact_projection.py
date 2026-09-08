"""Pairing two surfaces by proximity — :mod:`jno.utils.solver.contact_search`.

The interface machinery used to answer "where does this secondary point land on the main surface?" by
fitting ONE tangent plane to the whole main face, flattening everything onto it, and locating the
query in that coordinate. That is exact for a flat interface and degrades with curvature, because the
flattened coordinate is not arc length: the facet *under* a query stops being the facet *nearest* to
it. It over-reports separation, which is the silent direction — a contact pressure ``max(0, -g)`` then
never activates and free interpenetration is an exact root of the residual.

The oracle throughout is **brute force over every facet**, computing the very quantity under test
(``n . (closest point - x)``), not a Euclidean distance. Comparing against distance instead conflates
the search with the definition: the two differ by the cosine between the closest-point direction and
the normal, which is exactly the second-order effect being measured.

One trap worth naming, because it cost a debugging round: on a flat interface the *weights* may
legitimately differ from any other correct implementation while the *trace* does not. A query landing
on a facet edge is contained by two facets, either may be picked, and the weight on the non-shared
node is exactly zero. So these tests assert the trace and the gap, never the weight array.
"""

from __future__ import annotations

import numpy as np
import pytest

from jno.utils.solver.contact_search import OPEN_GAP, closest_point_on_triangle, project_points


# ----------------------------------------------------------------------------------------------
# Geometry and the brute-force oracle
# ----------------------------------------------------------------------------------------------
def _arc(radius, half_deg, n):
    t = np.linspace(-np.radians(half_deg), np.radians(half_deg), n)
    return np.stack([radius * np.cos(t), radius * np.sin(t)], 1)


def _polyline(n):
    return np.stack([np.arange(n - 1), np.arange(1, n)], 1)


def _cap(radius, half_deg, n):
    """A triangulated spherical cap: the 3-D counterpart of an arc."""
    from scipy.spatial import Delaunay

    r = np.sin(np.radians(half_deg))
    g = np.linspace(-r, r, n)
    X, Y = np.meshgrid(g, g)
    m = (X**2 + Y**2) <= r**2 + 1e-12
    x, y = X[m], Y[m]
    z = np.sqrt(np.maximum(radius**2 - x**2 - y**2, 0.0))
    return np.stack([x, y, z], 1), Delaunay(np.stack([x, y], 1)).simplices


def _brute_g0(q, P, facets, nrm):
    """``n . (closest point on the main surface - x)``, by exhaustive search. The definition itself."""
    V = P[facets]
    out = np.empty(len(q))
    for i, x in enumerate(q):
        xr = np.repeat(x[None, :], len(V), 0)
        if V.shape[1] == 2:
            a, b = V[:, 0], V[:, 1]
            ab = b - a
            t = np.clip(np.einsum("sd,sd->s", xr - a, ab) / np.einsum("sd,sd->s", ab, ab), 0, 1)
            p = a + t[:, None] * ab
        else:
            p, _ = closest_point_on_triangle(xr, V)
        out[i] = float(nrm[i] @ (p[int(np.argmin(np.linalg.norm(p - xr, axis=1)))] - x))
    return out


def _concentric_2d(half_deg, n_main=80, n_q=25, r_main=1.0, r_sec=1.3):
    """Two concentric arcs: the true separation is ``r_sec - r_main`` at every query, by construction."""
    P = _arc(r_main, half_deg, n_main)
    seg = _polyline(n_main)
    t = np.linspace(-np.radians(half_deg * 0.6), np.radians(half_deg * 0.6), n_q)
    q = r_sec * np.stack([np.cos(t), np.sin(t)], 1)
    nrm = -np.stack([np.cos(t), np.sin(t)], 1)  # secondary outward normal -> points at the main
    return P, seg, q, nrm


# ----------------------------------------------------------------------------------------------
# The defect: curvature
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("half_deg", [5, 20, 60, 100, 140])
def test_the_gap_on_a_curved_main_surface_is_exact(half_deg):
    """The measurement that motivated the module. Flattening reported, against a true 0.3:

        5 deg 0.300045 | 20 deg 0.300740 | 60 deg 0.308814 | 100 deg 0.358399 | 140 deg 0.432709

    i.e. 44 % high at 140 degrees, and monotone in curvature -- so no tolerance separates "curved
    enough to matter" from "flat enough to ignore"."""
    P, seg, q, nrm = _concentric_2d(half_deg)
    _ids, _w, g0, active = project_points(q, seg, P, nrm)
    assert active.all(), "an unbounded capture must pair every query"
    assert np.abs(g0 - _brute_g0(q, P, seg, nrm)).max() < 1e-12


@pytest.mark.parametrize("half_deg", [10, 40, 70])
def test_the_gap_on_a_curved_main_surface_is_exact_in_3d(half_deg):
    """Same statement for triangle facets, where flattening a spherical cap loses up to 6.3e-02."""
    P, tri = _cap(1.0, half_deg, 22)
    t = np.radians(half_deg * 0.5) * np.linspace(-1, 1, 7)
    a, b = np.meshgrid(t, t)
    d = np.stack([np.sin(a.ravel()), np.sin(b.ravel()), np.zeros(a.size)], 1)
    d[:, 2] = np.sqrt(np.maximum(1 - d[:, 0] ** 2 - d[:, 1] ** 2, 1e-9))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    _ids, _w, g0, active = project_points(1.3 * d, tri, P, -d)
    assert active.all()
    assert np.abs(g0 - _brute_g0(1.3 * d, P, tri, -d)).max() < 1e-12


# ----------------------------------------------------------------------------------------------
# The trace: what the weights are actually for
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_the_trace_reproduces_an_affine_main_field_exactly(dim):
    """``u_m . Phi`` must be exact for affine data -- partition of unity gives constants, and linear
    reproduction is what a first-order-accurate trace rests on. Asserted on the TRACE rather than on
    the weights, because two correct pairings may disagree on a facet edge and still agree here."""
    rng = np.random.default_rng(0)
    if dim == 2:
        P = np.stack([np.linspace(-1, 1, 40), np.zeros(40)], 1)
        F = _polyline(40)
        q = np.stack([np.linspace(-0.6, 0.6, 17), np.full(17, 0.25)], 1)
        nrm = np.tile([0.0, -1.0], (17, 1))
    else:
        from scipy.spatial import Delaunay

        g = np.linspace(-1, 1, 12)
        X, Y = np.meshgrid(g, g)
        P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        F = Delaunay(P[:, :2]).simplices
        qg = np.linspace(-0.5, 0.5, 5)
        QX, QY = np.meshgrid(qg, qg)
        q = np.stack([QX.ravel(), QY.ravel(), np.full(QX.size, 0.3)], 1)
        nrm = np.tile([0.0, 0.0, -1.0], (len(q), 1))

    ids, w, _g0, active = project_points(q, F, P, nrm)
    assert active.all()
    assert np.abs(w.sum(axis=1) - 1.0).max() < 1e-12, "partition of unity"
    # The trace of the COORDINATE field is the projected point itself, which is the only handle on
    # where the trace is evaluated -- `project_points` returns the gap, not the point.
    proj = np.einsum("qk,qkd->qd", w, P[ids])
    for _ in range(10):
        c, b = rng.normal(size=dim), float(rng.normal())
        f = P @ c + b  # an affine field, sampled at the main nodes
        got = np.einsum("qk,qk->q", w, f[ids])
        assert np.abs(got - (proj @ c + b)).max() < 1e-12, "affine data must be traced exactly"


def test_a_flat_interface_reads_the_separation_analytically():
    """The control the curved tests are measured against: on a flat pair the gap IS the height, so any
    error here would be in the definition rather than in the search."""
    P = np.stack([np.linspace(-1, 1, 40), np.zeros(40)], 1)
    for h in (0.05, 0.25, 1.0):
        q = np.stack([np.linspace(-0.6, 0.6, 17), np.full(17, h)], 1)
        nrm = np.tile([0.0, -1.0], (17, 1))
        _ids, _w, g0, _a = project_points(q, _polyline(40), P, nrm)
        assert np.abs(g0 - h).max() < 1e-12


def test_flipping_the_normal_flips_the_sign_and_nothing_else():
    """Orientation is the one thing the arrays themselves cannot tell you: handed the inward normal
    this returns ``-g``, and every downstream sign follows it into silent interpenetration."""
    P, seg, q, nrm = _concentric_2d(60)
    _i1, w1, g1, _a1 = project_points(q, seg, P, nrm)
    _i2, w2, g2, _a2 = project_points(q, seg, P, -nrm)
    assert np.abs(g1 + g2).max() < 1e-12
    assert np.abs(w1 - w2).max() < 1e-12, "only the sign of the gap depends on the normal"


# ----------------------------------------------------------------------------------------------
# Capture: what makes a re-search free
# ----------------------------------------------------------------------------------------------
def test_out_of_range_queries_keep_their_slot_and_contribute_nothing():
    """The mechanism a contact SEARCH rests on. A body drifting out of range must not change the table
    SHAPE -- shapes are what a retrace is keyed on -- so an unpaired query keeps its row with zero
    weights and a wide-open gap, and contributes exactly zero to the trace."""
    P = np.stack([np.linspace(-1, 1, 21), np.zeros(21)], 1)
    seg = _polyline(21)
    f = np.random.default_rng(1).normal(size=len(P))
    shapes = set()
    for h in (0.02, 0.10, 0.30, 0.80, 5.00):
        q = np.stack([np.linspace(-0.5, 0.5, 9), np.full(9, h)], 1)
        nrm = np.tile([0.0, -1.0], (9, 1))
        ids, w, g0, active = project_points(q, seg, P, nrm, capture=0.5)
        shapes.add((ids.shape, w.shape, g0.shape))
        if h <= 0.5:
            assert active.all() and np.abs(g0 - h).max() < 1e-12
        else:
            assert not active.any()
            assert (g0 == OPEN_GAP).all(), "out of range must read as open, not as touching"
            assert np.abs(np.einsum("qk,qk->q", w, f[ids])).max() == 0.0
    assert len(shapes) == 1, f"the table shape must not depend on where the bodies are: {shapes}"


def test_the_gap_is_continuous_as_the_nearest_facet_changes():
    """Sliding along the surface crosses facet after facet. If the pairing jumped at those crossings
    the tangent would be wrong exactly where a contact problem spends its time."""
    P = np.stack([np.linspace(-1, 1, 21), np.zeros(21)], 1)
    q = np.stack([np.linspace(-0.9, 0.9, 400), np.full(400, 0.05)], 1)
    _i, _w, g0, _a = project_points(q, _polyline(21), P, np.tile([0.0, -1.0], (400, 1)), capture=0.5)
    assert np.abs(np.diff(g0)).max() < 1e-12


def test_a_facet_can_be_excluded_from_its_own_neighbourhood():
    """What self-contact needs. A surface searching against ITSELF pairs every query with the facet it
    sits on -- gap zero, everywhere in contact with itself -- unless the query's own neighbourhood is
    excluded. Then the pairing finds the part of the surface that has folded back onto it.

    Two details this pins, both learned the hard way:

    * the fold must be CLOSER than the excluded neighbourhood is wide (0.1 against 0.3 here), or the
      nearest admissible facet is simply the next one along the same arm;
    * on a flat surface the exclusion is invisible in the gap however wide it is -- the remaining
      facets are laterally distant, and the separation is measured ALONG THE NORMAL, so it stays 0.
      Only a fold makes the pairing's choice observable.
    """
    n, fold, ring_r = 41, 0.1, 0.3
    lo = np.stack([np.linspace(-1.0, 1.0, n), np.zeros(n)], 1)
    up = np.stack([np.linspace(1.0, -1.0, n), np.full(n, fold)], 1)
    P = np.concatenate([lo, up])
    seg = np.stack([np.arange(2 * n - 1), np.arange(1, 2 * n)], 1)  # one chain, joined at x = 1

    qi = np.array([10, 15, 20, 25])  # queries ON the lower arm, away from the joint
    q, nrm = P[qi], np.tile([0.0, 1.0], (4, 1))  # outward from the lower arm -> toward the fold

    _ids, _w, g_all, _a = project_points(q, seg, P, nrm, capture=2.0)
    assert np.abs(g_all).max() < 1e-12, "unexcluded, a point on the surface pairs with itself"

    # the adjacency ring: every LOWER-arm node within `ring_r` of the query along the surface
    ring = [set(np.flatnonzero((P[:, 1] < fold / 2) & (np.abs(P[:, 0] - x) < ring_r)).tolist()) for x in q[:, 0]]
    ids_ex, _w2, g_ex, act = project_points(q, seg, P, nrm, capture=2.0, exclude_nodes=ring)
    assert act.all(), "the folded arm is well within capture"
    for row, ex in zip(ids_ex, ring):
        assert not (set(int(v) for v in row) & ex), "an excluded node must not be paired with"
    assert np.abs(g_ex - fold).max() < 1e-12, "the pairing must find the arm folded above, at 0.1"
