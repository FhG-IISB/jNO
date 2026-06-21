"""``fem_symbols(space=...)`` — declaring a non-nodal element family.

The element zoo lets a field pick its space at authoring time; this is the API
seam (carried on the symbols, surfaced by ``_infer_fields``, detected by
``jno.fem``). Assembly of non-nodal spaces is wired incrementally — until then
``jno.fem`` must fail loudly rather than silently assemble a Lagrange system.
"""

from __future__ import annotations

import pytest

# aliased so pytest does not try to collect ``TestFunction`` as a test class
from jno.trace import TestFunction as _TestFunction
from jno.trace import TrialFunction


def test_symbol_space_defaults_to_lagrange_and_is_settable():
    assert TrialFunction().space == "Lagrange"
    assert _TestFunction().space == "Lagrange"
    assert TrialFunction(space="RT").space == "RT"
    assert _TestFunction(space="RT").space == "RT"


def test_infer_fields_carries_space():
    from jno.utils.solver.feax_utils import _infer_fields

    fields, _ = _infer_fields(TrialFunction(value_shape=(2,), space="RT"))
    assert len(fields) == 1
    assert fields[0]["space"] == "RT" and fields[0]["vec"] == 2
    # default stays Lagrange
    fields_l, _ = _infer_fields(TrialFunction())
    assert fields_l[0]["space"] == "Lagrange"


def test_trial_spaces_detects_nonnodal():
    from jno._fem import _trial_spaces

    assert _trial_spaces([TrialFunction()]) == {"Lagrange"}
    assert _trial_spaces([TrialFunction(space="RT")]) == {"RT"}


def test_fem_symbols_threads_space_and_rejects_not_yet_wired_family():
    # RT is wired through the DSL (see test_fem_nonnodal_dsl); a family not yet implemented must
    # still error clearly rather than silently assemble a Lagrange system.
    pytest.importorskip("feax", reason="feax required for jno.fem")
    pytest.importorskip("pygmsh", reason="pygmsh required for 2D meshing")
    from shapely.geometry import box

    import jno

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    assert u.space == "RT" and v.space == "RT" and u.field_key == v.field_key  # threaded through
    a, b = d.fem_symbols(names=("a", "b"), space="Argyris")  # C1, not yet wired
    xi, yi, _ = d.variable("interior", split=True)
    with pytest.raises(NotImplementedError):
        jno.fem([a.bind(x=xi, y=yi) * b.bind(x=xi, y=yi)])
