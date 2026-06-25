"""Callable / bindable FEM trial & test symbols.

A weak form authored with the field idiom -- ``u.bind(x=xg, y=yg).x`` or the
positional-call sugar ``u(xg, yg).x`` -- must assemble to exactly the same FEM
operator as the classic ``u.d(xg)`` form, because ``.x`` lowers to the same
``Jacobian`` node. Mirrors the PINN authoring gesture ``net(x).scalar.bind(x=x)``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """The native assembly is compared entry-for-entry, so opt into x64 per-test (the session
    default may be x64-off when co-run with test_periodic). Save/restore keeps the flag local."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _poisson_systems():
    """The same Poisson assembled through three authoring idioms, each via ``jno.fem`` (the sole
    entry): the classic ``u.d(xg)``, the bound ``u.bind(x=xg).x`` and the positional-call
    ``u(xg, yg).x``. They must lower to the same ``Jacobian`` node and so assemble the same (A, b)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, phi = d.fem_symbols()
    xg, yg, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xg * (1.0 - xg) + yg * (1.0 - yg))

    fem_d = jno.fem([u.d(xg) * phi.d(xg) + u.d(yg) * phi.d(yg) - f * phi, u(xb, yb) - 0.0], quad_degree=3)

    ub, vb = u.bind(x=xg, y=yg), phi.bind(x=xg, y=yg)
    fem_bind = jno.fem([ub.x * vb.x + ub.y * vb.y - f * vb, u(xb, yb) - 0.0], quad_degree=3)

    uc, vc = u(xg, yg), phi(xg, yg)
    fem_call = jno.fem([uc.x * vc.x + uc.y * vc.y - f * vc, u(xb, yb) - 0.0], quad_degree=3)

    return fem_d, fem_bind, fem_call


def test_bind_authoring_matches_d_form():
    fem_d, fem_bind, _ = _poisson_systems()
    assert np.allclose(_dense(fem_d.A), _dense(fem_bind.A))
    assert np.allclose(np.asarray(fem_d.b).reshape(-1), np.asarray(fem_bind.b).reshape(-1))


def test_positional_call_authoring_matches_d_form():
    fem_d, _, fem_call = _poisson_systems()
    assert np.allclose(_dense(fem_d.A), _dense(fem_call.A))
    assert np.allclose(np.asarray(fem_d.b).reshape(-1), np.asarray(fem_call.b).reshape(-1))


def test_call_without_coords_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    u, phi = d.fem_symbols()
    with pytest.raises(TypeError):
        u()
    with pytest.raises(TypeError):
        phi()
