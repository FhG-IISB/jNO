"""``shape.quad().curved()``: the nodes of a Q_k field on the true boundary, so the spectral elements keep their
rate on a round domain. Straight-sided, the domain is a polygon and the error stalls at the chord-to-arc gap."""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _disk(curved, order, *, robin=False, size=0.3):
    """-Δu = 1 on the unit disk, u = (1 - r²)/4: Dirichlet u = 0, or Robin ∂u/∂n + u = -1/2 on the circle."""
    s = jno.shape.disk(0, 0, 1, size=size)
    d = (s.curved() if curved else s).quad().domain()
    u, v = d.fem_symbols(order=order)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    form = ui.x * vi.x + ui.y * vi.y - 1.0 * vi
    if robin:
        ub, vb = u.bind(x=xb, y=yb), v.bind(x=xb, y=yb)
        terms = [form, ub * vb + 0.5 * vb]  # the curved facets' length enters here
    else:
        terms = [form, u(xb, yb) - 0.0]
    p = jno.fem(terms, quadrature="gll")
    sol = np.asarray(p.solve(linear=jno.solve.lu())).reshape(-1)
    P = np.asarray(p.points)[:, :2]
    return np.abs(sol - (1.0 - (P**2).sum(axis=1)) / 4.0).max(), P


@pytest.mark.parametrize("robin", [False, True])
def test_curved_quadrilaterals_converge_spectrally_on_a_disk(robin):
    straight = [_disk(False, k, robin=robin)[0] for k in (2, 8)]
    curved = [_disk(True, k, robin=robin)[0] for k in (2, 4, 8)]
    assert straight[1] > 0.5 * straight[0] > 1e-4  # the polygon's error: no better at Q8 than at Q2
    assert curved[1] < 1e-2 * curved[0] and curved[2] < 1e-3 * curved[1] and curved[2] < 1e-10, curved


def test_the_boundary_nodes_lie_on_the_circle_and_the_vertices_do_not_move():
    _err, P = _disk(True, 6)
    _err0, P0 = _disk(False, 6)
    r, r0 = np.linalg.norm(P, axis=1), np.linalg.norm(P0, axis=1)
    assert np.sum(np.abs(r - 1.0) < 1e-12) > 4 * np.sum(np.abs(r0 - 1.0) < 1e-12)  # edge nodes joined the vertices
    assert r.max() < 1.0 + 1e-12  # nothing pushed outside
    nv = 0  # the first ids are the mesh vertices on both meshes
    while nv < P.shape[0] and np.allclose(P[nv], P0[nv]):
        nv += 1
    assert nv > 0 and np.allclose(P[:nv], P0[:nv])


def test_a_hole_is_curved_too():
    s = (jno.shape.rect(-1, -1, 1, 1) - jno.shape.disk(0, 0, 0.5)).sized(0.25)
    d = s.curved().quad().domain()
    u, v = d.fem_symbols(order=5)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    p = jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xb, yb) - 0.0], quadrature="gll")
    P = np.asarray(p.points)[:, :2]
    A = np.asarray(p.A)
    pinned = np.abs(A - np.diag(np.diag(A))).sum(axis=1) < 1e-14  # the Dirichlet rows: every boundary DOF
    on_hole = np.abs(np.linalg.norm(P, axis=1) - 0.5) < 1e-12
    assert pinned.sum() > 100 and on_hole.sum() > 40
    assert np.abs(np.asarray(s.sdf(P[pinned]))).max() < 1e-12  # square sides and hole alike


def test_a_cell_that_would_fold_says_why():
    with pytest.raises(ValueError, match="(?i)fold.*size="):
        for k in (3, 4, 8):  # at size 0.5 a cell has two boundary edges meeting on the circle
            _disk(True, k, size=0.5)
