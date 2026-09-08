"""``fem.export`` writes the SOLUTION, not just the mesh.

``d.export_vtk()`` has always written geometry; there was no path from a solve to something ParaView
opens. The design question is what to do about a coupled problem, whose fields do not share a point
set — Taylor-Hood velocity is P2 (vertices + edge midpoints), its pressure P1 (vertices only). Writing
them into one file means interpolating one of them, so an exported field would no longer be the answer
that was computed. ``export`` therefore writes **one file per field**, each on its own points and its own
cells, taken from the assembler's own connectivity tables.

The discriminating test is the P2 round-trip: read the file back and require the velocity block to come
back bit-identical on the full P2 point count. A vertex-sampled export would pass a "looks right" eyeball
check and fail this.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from shapely.geometry import box

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(mesh_size=0.25):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    return d, fem, np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))


def _stokes(mesh_size=0.35):
    """Taylor-Hood: P2 velocity + P1 pressure, so the two fields have DIFFERENT point counts."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    d.point_region("ppin", (0.0, 0.0))
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    vb, wb = v.bind(x=xi, y=yi), psi.bind(x=xi, y=yi)
    pb, qb = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    mom = (
        vb.x[0] * wb.x[0]
        + vb.y[0] * wb.y[0]
        + vb.x[1] * wb.x[1]
        + vb.y[1] * wb.y[1]
        - pb * (wb.x[0] + wb.y[1])
        - 1.0 * wb[1]
    )
    fem = jno.fem([mom, qb * (vb.x[0] + vb.y[1]), v(xb, yb) - 0.0, p(xpn, ypn) - 0.0])
    return d, fem, np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))


def test_a_single_field_writes_the_given_path_and_round_trips(tmp_path):
    import meshio

    d, fem, sol = _poisson()
    out = str(tmp_path / "poisson.vtu")
    written = fem.export(sol, out)
    assert written == [out], "a single-field problem must not gratuitously rename the file"

    m = meshio.read(out)
    assert list(m.point_data) == ["u"], m.point_data.keys()
    got = np.asarray(m.point_data["u"]).reshape(-1)
    assert got.shape == sol.shape
    assert np.array_equal(got, sol), "the exported field must BE the solution, not a resampling of it"
    assert len(m.points) == len(np.asarray(fem.field_points[0]))


def test_taylor_hood_writes_one_file_per_field_on_its_own_points(tmp_path):
    """The discriminator. P2 velocity and P1 pressure have different point counts; each file must carry
    its own, and the P2 block must survive bit-identical rather than being sampled at vertices."""
    import meshio

    d, fem, sol = _stokes()
    out = str(tmp_path / "stokes.vtu")
    written = fem.export(sol, out)
    assert [p.rsplit("/", 1)[-1] for p in written] == ["stokes.v.vtu", "stokes.p.vtu"], written

    offs = list(fem.offsets)
    n_v, n_p = len(np.asarray(fem.field_points[0])), len(np.asarray(fem.field_points[1]))
    assert n_v > n_p, "Taylor-Hood: the P2 velocity must have more nodes than the P1 pressure"

    mv, mp = meshio.read(written[0]), meshio.read(written[1])
    assert len(mv.points) == n_v and len(mp.points) == n_p
    assert np.array_equal(np.asarray(mv.point_data["v"]), sol[offs[0] : offs[1]].reshape(-1, 2))
    assert np.array_equal(np.asarray(mp.point_data["p"]).reshape(-1), sol[offs[1] : offs[2]])
    # the P2 file carries quadratic cells, not a point cloud and not linear ones
    assert mv.cells[0].type == "triangle6", mv.cells[0].type
    assert mp.cells[0].type == "triangle"


def test_a_trajectory_and_a_complex_solution_are_refused_by_name(tmp_path):
    """Both are cases where picking for the user would silently export something that is not what they
    asked for -- a single step chosen at random, or one part of a complex field."""
    d, fem, sol = _poisson()
    out = str(tmp_path / "x.vtu")
    with pytest.raises(ValueError, match="one step"):
        fem.export(np.stack([sol, sol]), out)
    with pytest.raises(ValueError, match="complex"):
        fem.export(sol.astype(np.complex128), out)


def test_a_wrong_length_solution_is_refused(tmp_path):
    d, fem, sol = _poisson()
    with pytest.raises(ValueError, match="DOFs"):
        fem.export(sol[:-1], str(tmp_path / "x.vtu"))
