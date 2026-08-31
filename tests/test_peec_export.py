"""``sol.export_vtk(path)`` -- the solved currents, where a viewer can see them.

Same verb as :meth:`jno.domain.export_vtk`, because it is the same job: hand the geometry and what
was computed on it to meshio, and let ParaView do the looking. A partial-element network is a set of
line segments, so it exports as one -- each filament a cell, its current a cell datum.
"""

import pathlib

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7


def _solved(freq=0.0):
    bar = jno.Shape.box(0, 0, 0, 0.02, 0.004, 0.001, size=0.001).attach(sigma=SIG).name("bar")
    d = bar.domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.02 - 0.0011)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build().solve()


def test_it_writes_a_file_a_reader_can_read_back(tmp_path):
    meshio = pytest.importorskip("meshio")
    sol = _solved()
    out = tmp_path / "currents.vtk"
    sol.export_vtk(str(out))
    assert out.exists() and out.stat().st_size > 0
    m = meshio.read(str(out))
    assert m.points.shape[1] == 3
    assert any(c.type == "line" for c in m.cells)


def test_every_filament_is_a_cell_and_carries_its_own_current(tmp_path):
    meshio = pytest.importorskip("meshio")
    sol = _solved()
    out = tmp_path / "c.vtk"
    sol.export_vtk(str(out))
    m = meshio.read(str(out))
    ne = int(np.asarray(sol._fil.length).shape[0])
    assert sum(len(c.data) for c in m.cells if c.type == "line") == ne
    names = set(m.cell_data)
    assert {"current", "current_density"} <= names, sorted(names)
    got = np.concatenate(m.cell_data["current"])
    assert got.shape == (ne,)
    assert np.allclose(got, np.abs(np.asarray(sol.i)), rtol=1e-12)


def test_a_complex_solve_exports_magnitude_and_phase(tmp_path):
    """At a frequency the current is complex, and a viewer cannot plot a complex number."""
    meshio = pytest.importorskip("meshio")
    sol = _solved(freq=1e5)
    out = tmp_path / "ac.vtk"
    sol.export_vtk(str(out))
    m = meshio.read(str(out))
    assert {"current", "phase_deg"} <= set(m.cell_data), sorted(m.cell_data)
    ph = np.concatenate(m.cell_data["phase_deg"])
    assert np.all(np.isfinite(ph)) and np.all(np.abs(ph) <= 180.0 + 1e-9)


def test_a_swept_solution_says_which_frequency_rather_than_guessing(tmp_path):
    """With an array freq there is no single current field, so it has to be asked for."""
    bar = jno.Shape.box(0, 0, 0, 0.02, 0.004, 0.001, size=0.001).attach(sigma=SIG).name("bar")
    d = bar.domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.02 - 0.0011)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=[0.0, 1e5]).build().solve()
    with pytest.raises(ValueError, match="which frequency|freq="):
        sol.export_vtk(str(pathlib.Path("/tmp") / "never-written.vtk"))
