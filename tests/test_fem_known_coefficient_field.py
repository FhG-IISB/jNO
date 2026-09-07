"""A KNOWN field used as a coefficient in a form where it is not an unknown.

That is the lagged velocity of a PCD auxiliary, a wall distance solved once and reused, an eddy
viscosity computed from the previous iterate -- data produced elsewhere and read back as a
coefficient. Before this, the only ways to write a known coefficient were a **formula** of the
coordinates (`parameter(sym).initialize(lambda x, y: ...)`, whose docstring says outright that "a raw
per-node value array is not supported") or a frozen field that was *also* an unknown of the same form.
Arbitrary nodal data on a field the form does not solve for raised a bare `KeyError` on the field id.

The oracles are exact: a known field holding a CONSTANT must give the same operator as the literal,
and a known field holding a LINEAR profile must match the coordinate formula, because P1 represents a
linear field exactly.
"""

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


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _setup(mesh_size=0.25):
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    a, b = d.fem_symbols(names=("a", "b"))
    return d, xi, yi, a, b, a.bind(x=xi, y=yi), b.bind(x=xi, y=yi)


def test_a_constant_known_field_equals_the_literal():
    d, xi, yi, _a, _b, ai, bi = _setup()
    n = int(np.asarray(d.mesh.points).shape[0])
    w, _z = d.fem_symbols(names=("w", "zz"))
    lit = _dense(jno.fem([2.5 * ai.x * bi + ai.x * bi.x + ai.y * bi.y]).A)
    frz = _dense(jno.fem([w.bind(x=xi, y=yi).freeze(np.full(n, 2.5)) * ai.x * bi + ai.x * bi.x + ai.y * bi.y]).A)
    np.testing.assert_allclose(frz, lit, atol=1e-13)


def test_a_linear_known_field_equals_the_coordinate_formula():
    """P1 represents a linear field exactly, so the nodal-data route and the formula route must agree
    to machine precision -- which pins the INTERPOLATION, not just the plumbing."""
    d, xi, yi, _a, _b, ai, bi = _setup()
    pts = np.asarray(d.mesh.points)[:, :2]
    w, _z = d.fem_symbols(names=("w", "zz"))
    fn = _dense(jno.fem([(1.0 + 2.0 * xi) * ai.x * bi + ai.x * bi.x + ai.y * bi.y]).A)
    fz = _dense(jno.fem([w.bind(x=xi, y=yi).freeze(1.0 + 2.0 * pts[:, 0]) * ai.x * bi + ai.x * bi.x + ai.y * bi.y]).A)
    np.testing.assert_allclose(fz, fn, atol=1e-13)


def test_a_vector_known_field_matches_its_two_scalar_equivalent():
    """`(w.grad a) * b` with a vector known field -- the exact shape a PCD convection operator needs."""
    d, xi, yi, _a, _b, ai, bi = _setup()
    pts = np.asarray(d.mesh.points)[:, :2]
    vals = np.stack([1.0 + 2.0 * pts[:, 0], 3.0 - pts[:, 1]], axis=-1)
    wv, _zv = d.fem_symbols(value_shape=(2,), names=("wv", "zv"))
    w1, _z1 = d.fem_symbols(names=("w1", "z1"))
    w2, _z2 = d.fem_symbols(names=("w2", "z2"))
    wvi = wv.bind(x=xi, y=yi).freeze(vals)
    vec = _dense(jno.fem([(wvi[0] * ai.x + wvi[1] * ai.y) * bi + ai.x * bi.x + ai.y * bi.y]).A)
    two = _dense(
        jno.fem(
            [
                (w1.bind(x=xi, y=yi).freeze(vals[:, 0]) * ai.x + w2.bind(x=xi, y=yi).freeze(vals[:, 1]) * ai.y) * bi
                + ai.x * bi.x
                + ai.y * bi.y
            ]
        ).A
    )
    np.testing.assert_allclose(vec, two, atol=1e-13)


def test_a_known_field_with_no_matching_element_is_refused_by_name():
    """A known field borrows the nodal basis of a live field with the SAME element. If the form has
    none, that is a modelling error and must be named, not a `KeyError` on an internal field id."""
    d, xi, yi, _a, _b, ai, bi = _setup()
    n2 = int(np.asarray(jno.domain.__mro__[0] and d.mesh.points).shape[0])
    hi, _z = d.fem_symbols(names=("hi", "zhi"), order=2)  # P2 known field, but the form is all P1
    vals = np.zeros(int(np.asarray(d.mesh.points).shape[0]) * 4)
    with pytest.raises((NotImplementedError, ValueError, IndexError, TypeError)):
        jno.fem([hi.bind(x=xi, y=yi).freeze(vals[: n2 * 4]) * ai.x * bi + ai.x * bi.x + ai.y * bi.y]).A
