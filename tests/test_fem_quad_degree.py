"""``jno.fem(quad_degree=...)``: the default integrates a field's mass term exactly; an explicit degree is
used as asked -- never silently raised -- and a degree below the default says so.

Oracle: the trace of the P1 mass matrix on a unit domain. Exactly integrated, every element contributes
``(dim+1) |K| / ((dim+1)(dim+2)/2)``, so the trace is ``2/(dim+2)``: 2/3, 1/2, 2/5 in 1-D, 2-D, 3-D. A
one-point rule (degree 0 or 1: the centroid / midpoint) sees every basis function as ``1/(dim+1)``, so
the trace is ``1/(dim+1)``: 1/2, 1/3, 1/4. Both are independent of the mesh.
"""

from __future__ import annotations

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


def _domain(dim):
    if dim == 1:
        return jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    if dim == 2:
        return jno.shape.rect(0, 0, 1, 1, size=0.3).domain()
    return jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()


def _mass_trace(dim, **kw):
    d = _domain(dim)
    co = d.variable("interior", split=True)[:dim]
    u, v = d.fem_symbols()
    names = ("x", "y", "z")[:dim]
    ub, vb = u.bind(**dict(zip(names, co))), v.bind(**dict(zip(names, co)))
    fem = jno.fem([ub * vb - 1.0 * vb], **kw)
    A = fem.A
    A = np.asarray(A.todense() if hasattr(A, "todense") else A)
    return float(np.trace(A)), fem


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_the_default_integrates_the_p1_mass_exactly(dim):
    # Break: lowering the default below 2 x order (the mass term's degree).
    tr, _ = _mass_trace(dim)
    assert tr == pytest.approx(2.0 / (dim + 2), rel=1e-12)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_an_explicit_lower_degree_is_used_as_asked(dim):
    # Break: raising the request to 2 x order behind the user's back (it used to be max(q, 2 x order)).
    tr, _ = _mass_trace(dim, quad_degree=1)
    assert tr == pytest.approx(1.0 / (dim + 1), rel=1e-12)


def test_a_degree_below_the_default_is_logged_with_what_it_under_integrates(monkeypatch):
    # Break: honouring the request silently. The user asked for it, so it is not an error -- but a mass
    # term integrated inexactly must not be a surprise.
    import jno.utils.solver.fem_utils as fu

    said = []
    monkeypatch.setattr(
        fu, "_log", type("L", (), {"warning": lambda self, m: said.append(m), "info": lambda self, m: None})()
    )
    _mass_trace(2, quad_degree=1)
    assert any("quad_degree=1" in m and "below" in m and "2" in m for m in said), said
    said.clear()
    _mass_trace(2)  # the default never warns
    _mass_trace(2, quad_degree=2)  # nor does an explicit degree at the default
    assert not said, said


def test_a_rebuild_keeps_the_default_rather_than_freezing_it_as_explicit():
    # The adaptive driver rebuilds with `jno.fem(constraints, **fem._fem_kwargs)`. Storing the RESOLVED
    # default there would turn it into an explicit request -- a rebuilt P2 problem would then run at
    # degree 2 instead of its default 4.
    _, fem = _mass_trace(2)
    assert fem._fem_kwargs.get("quad_degree") is None
    _, fem1 = _mass_trace(2, quad_degree=1)
    assert fem1._fem_kwargs.get("quad_degree") == 1
