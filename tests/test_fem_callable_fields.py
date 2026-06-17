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

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import box  # noqa: E402


def _dense(J):
    return np.asarray(J.todense() if hasattr(J, "todense") else J)


def _poisson_operators():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    d.init_fem(
        element_type="TRI3",
        quad_degree=3,
        bcs=[d.dirichlet("boundary", 0.0)],
        fem_solver=True,
    )
    u, phi = d.fem_symbols()
    xg, yg, _ = d.variable("fem_gauss", split=True)
    f = 2.0 * (xg * (1.0 - xg) + yg * (1.0 - yg))

    op_d = (u.d(xg) * phi.d(xg) + u.d(yg) * phi.d(yg) - f * phi).assemble(d, target="fem_residual")

    ub, vb = u.bind(x=xg, y=yg), phi.bind(x=xg, y=yg)
    op_bind = (ub.x * vb.x + ub.y * vb.y - f * vb).assemble(d, target="fem_residual")

    uc, vc = u(xg, yg), phi(xg, yg)
    op_call = (uc.x * vc.x + uc.y * vc.y - f * vc).assemble(d, target="fem_residual")

    return op_d, op_bind, op_call


def test_bind_authoring_matches_d_form():
    op_d, op_bind, _ = _poisson_operators()
    assert op_bind.size == op_d.size
    rng = np.random.default_rng(0)
    u0 = rng.standard_normal(op_d.size)
    assert np.allclose(np.asarray(op_d.residual(u0)), np.asarray(op_bind.residual(u0)))
    assert np.allclose(_dense(op_d.jacobian(u0)), _dense(op_bind.jacobian(u0)))


def test_positional_call_authoring_matches_d_form():
    op_d, _, op_call = _poisson_operators()
    rng = np.random.default_rng(1)
    u0 = rng.standard_normal(op_d.size)
    assert np.allclose(np.asarray(op_d.residual(u0)), np.asarray(op_call.residual(u0)))
    assert np.allclose(_dense(op_d.jacobian(u0)), _dense(op_call.jacobian(u0)))


def test_call_without_coords_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    u, phi = d.fem_symbols()
    with pytest.raises(TypeError):
        u()
    with pytest.raises(TypeError):
        phi()
