"""Coupled / mixed multi-field FEM through ``jno.fem`` (Phase 2).

Each field is its own ``(trial, test)`` pair from a ``fem_symbols()`` call (they
share a ``field_key``); ``jno.fem`` detects several fields and assembles a block
(multi-variable) system via feax. Cross-field weak terms populate the off-diagonal
blocks, and feax autodiffs the universal kernel into the full block matrix. The
single-field path is unchanged (one field → existing assembly).
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import box  # noqa: E402


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def test_coupled_p1_recovers_manufactured_with_offdiagonal():
    # Two coupled scalar fields on the unit square:
    #   -lap u + p = f1 ,  -lap p + u = f2 ,  u = p = 0 on the boundary.
    # Manufactured u* = g, p* = 2g with g = x(1-x)y(1-y); f1 = -lap(u*) + p*,
    # f2 = -lap(p*) + u*. The cross terms (+p in u's eq, +u in p's eq) make the
    # off-diagonal blocks non-zero, so recovery genuinely exercises coupling.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    g = xi * (1 - xi) * yi * (1 - yi)
    lg = 2 * (xi * (1 - xi) + yi * (1 - yi))  # -lap(g)
    f1 = lg + 2 * g
    f2 = 2 * lg + g
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + p * vi - f1 * vi,  # u-equation (test v), couples to p
            pi.x * qi.x + pi.y * qi.y + u * qi - f2 * qi,  # p-equation (test q), couples to u
            u(xb, yb) - 0.0,
            p(xb, yb) - 0.0,
        ]
    )
    assert fem.is_linear
    assert fem.dofs == 2 * n  # block system over the two P1 fields

    A = _dense(fem.A)
    assert np.allclose(A, A.T, atol=1e-9)
    # the off-diagonal block (u-rows x p-cols) must be populated -> real coupling,
    # not two independent problems that happen to recover.
    assert np.any(np.abs(A[:n, n:]) > 1e-12)
    assert np.any(np.abs(A[n:, :n]) > 1e-12)

    sol = np.linalg.solve(A, np.asarray(fem.b).reshape(-1))
    uu, pp = sol[:n], sol[n:]
    c = np.asarray(d.mesh.points)[:, :2]
    gg = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    assert np.linalg.norm(uu - gg) / np.linalg.norm(gg) < 1e-2
    assert np.linalg.norm(pp - 2 * gg) / np.linalg.norm(2 * gg) < 1e-2


def test_coupled_neumann_not_yet_supported():
    # Coupled surface (Neumann/Robin) terms aren't supported yet -> clear error.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError):
        jno.fem(
            [
                ui.x * vi.x + ui.y * vi.y + p * vi,
                pi.x * qi.x + pi.y * qi.y + u * qi,
                -1.0 * v.bind(x=xr, y=yr),  # coupled Neumann -> not supported yet
                u(xb, yb) - 0.0,
                p(xb, yb) - 0.0,
            ]
        )
