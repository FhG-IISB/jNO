"""Time-dependent ``PolygonDomain`` must follow jNO's ``(B, T, ..., C)`` pool
convention: spatial tag pools are broadcast over time to ``(T, N, D)`` and an
``"initial"`` (t=t0) slice ``(1, N, D)`` is registered -- reusing the same
``_add_time_dimension`` path the mesh-backed domain uses. Before this, the
Shapely-backed pools stayed 2-D, so ``variable("interior")`` raised on a
time-dependent domain and there was no IC region.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
pytest.importorskip("pygmsh", reason="pygmsh required for build_mesh")
from shapely.geometry import box  # noqa: E402


def _time_box(mesh_size=0.3, n_time=11):
    return jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, 1.0, n_time))


def test_interior_pool_is_time_broadcast():
    d = _time_box(n_time=11)
    interior = np.asarray(d._mesh_pool["interior"])
    assert interior.ndim == 3 and interior.shape[0] == 11  # (T, N, D)


def test_initial_tag_registered():
    d = _time_box(n_time=7)
    assert "initial" in d._mesh_pool
    initial = np.asarray(d._mesh_pool["initial"])
    assert initial.shape[0] == 1  # (1, N, D) -- the t=t0 slice
    # spatial coords match the interior slice at t0
    assert np.allclose(initial[0], np.asarray(d._mesh_pool["interior"])[0])


def test_variable_interior_and_initial_sample_without_error():
    d = _time_box()
    vi = d.variable("interior", split=True)
    assert len(vi) == 3  # x, y, t
    assert [getattr(v, "axis", None) for v in vi] == ["spatial", "spatial", "temporal"]
    init = d.variable("initial", split=True)
    assert len(init) == 3
    assert init[0].tag == "initial"


def test_rebuild_mesh_is_idempotent():
    d = _time_box(mesh_size=0.4)
    d.build_mesh(0.3)  # re-mesh on the same time-dependent domain
    interior = np.asarray(d._mesh_pool["interior"])
    initial = np.asarray(d._mesh_pool["initial"])
    assert interior.ndim == 3 and interior.shape[0] == 11
    assert initial.ndim == 3 and initial.shape[0] == 1


def test_stationary_box_has_no_time_axis_or_initial():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    assert np.asarray(d._mesh_pool["interior"]).ndim == 2  # (N, D)
    assert "initial" not in d._mesh_pool
