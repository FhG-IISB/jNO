"""A frozen (known) field used as a COEFFICIENT inside a non-nodal form.

``u.bind(...).freeze(values)`` pins a field to a nodal vector. On the native nodal path the frozen
field is a pinned copy of one of the solved unknowns, so it gathers on that field's own cell DOFs.
On the non-nodal path the trial is N1E (edge DOFs) while a frozen coefficient lives on the P1
VERTEX space, so it gathers on the mesh cells and interpolates with the P1 basis. That is what lets
a source computed elsewhere -- ``J_s`` from an electrokinetic pre-solve, a measured conductivity --
enter an H(curl) form.

The reachability guard below is the point of the last test. Looking a field key up with ``.get()``
instead of ``[...]`` makes a MISSING key legal everywhere, not just for the frozen coefficient it
was widened for: a trial or test function absent from the field table would quietly receive the
top-level P1 shape data and assemble against the wrong basis. Only a ``FrozenField`` may be absent.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _curl_curl_with_coefficient(values_fn, n_extra=0, size=0.6):
    """Curl-curl + mass on N1E, loaded through a frozen P1 scalar coefficient."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    w, _wt = d.fem_symbols(names=("w", "wt"))  # Lagrange -- a coefficient, never a solved unknown
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    pts = np.asarray(d.mesh.points)

    vals = jno.np.asarray(values_fn(pts, len(pts) + n_extra))
    Wf = w.bind(x=x, y=y, z=z).freeze(vals)
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cA, cV = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    e1 = vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x)
    return d, jno.fem(
        [
            inner(cA, cV) + inner(A_, V_) - Wf * inner(e1, V_),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )


def _load(fem):
    return np.asarray(jno.np.asarray(fem.b)).reshape(-1)


def test_frozen_p1_coefficient_assembles_into_an_n1e_form():
    _d, fem = _curl_curl_with_coefficient(lambda p, n: 1.0 + 2.0 * p[:, 0])
    b = _load(fem)
    assert np.isfinite(b).all()
    assert np.abs(b).max() > 0.0, "the frozen coefficient contributed nothing to the load"


def test_the_coefficient_values_actually_reach_the_load():
    """A guard against the coefficient being accepted and then ignored: two different frozen value
    vectors must give two different right-hand sides, and a scaled one a scaled load."""
    _d, f1 = _curl_curl_with_coefficient(lambda p, n: 1.0 + 0.0 * p[:, 0])
    _d, f2 = _curl_curl_with_coefficient(lambda p, n: 2.0 + 0.0 * p[:, 0])
    b1, b2 = _load(f1), _load(f2)
    assert not np.allclose(b1, b2), "changing the frozen values did not change the load"
    assert np.allclose(b2, 2.0 * b1, rtol=1e-9), "the load is not linear in a constant coefficient"


def test_wrong_length_values_raise_naming_the_vertex_count():
    """On an N1E form the frozen field is a P1 coefficient, NOT a copy of the edge-DOF unknown --
    so a values array sized to the edges is the mistake to catch, loudly."""
    with pytest.raises(ValueError, match="one value per MESH VERTEX"):
        _curl_curl_with_coefficient(lambda p, n: np.ones(n), n_extra=5)


def test_a_missing_field_key_raises_instead_of_falling_back_to_p1():
    """Only a FrozenField may be absent from the field table. Anything else is an assembler bug,
    and answering it with the top-level P1 shape data would be a silently wrong basis."""
    from jno.utils.solver.fem_utils import _field_data, _field_space

    class _NotFrozen:  # a stand-in for a trial/test function that never reached the field table
        field_key = "u_missing"

    local = {
        "fields": [{"shape_vals": None, "shape_grads": None, "cell_sol": None, "space": "N1E"}],
        "field_index": {"u_present": 0},
        "shape_vals": "the P1 table",
    }
    for fn in (_field_data, _field_space):
        with pytest.raises(KeyError, match="not in this problem's field table"):
            fn(local, _NotFrozen())


def test_a_p0_cell_field_parameter_is_refused_rather_than_misread():
    """A P0 ``(n_cells,)`` coefficient has no cell-field branch on the non-nodal assembler.

    ``_field_param_names`` there is `_is_fem_field_parameter` only -- `_fem_field_kind` and the
    `"cell"` gather live in `fem_native`. So a per-cell array reaches
    ``_fv[name][cells_j[c]]``: gathered at VERTEX ids and interpolated with the P1 shape functions.
    Wrong values, out-of-range indices clamped by JAX, and nothing raised.

    ``RegionMask`` (``d.by_region``) is the mechanism that works here — one 0/1 per cell, threaded
    through ``_cell_masks`` — so the guard names it.
    """
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    _p0_trial, p0 = d.fem_symbols(names=("m", "mt"), space="P0")
    k = jno.np.parameter(p0, name="k")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    with pytest.raises(NotImplementedError, match="by_region"):
        jno.fem(
            [
                k * inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
                + inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))
                - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), v.bind(x=x, y=y, z=z)),
                u.vector.cross(d.variable("boundary", normals=True)),
            ]
        )
