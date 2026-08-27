"""The coupling quantity: ohmic loss as a per-region source a weak form can consume.

``joule`` is a total, and a heat source is not a total — it is watts per cubic metre, per conductor.
``dissipation()`` returns exactly the ``{region: value}`` mapping ``domain.by_region`` takes, so the
thermal side needs no new concept:

    q = d.by_region(emag.dissipation(), default=0.0)
    heat = d.k * grad(T) . grad(s) - q * s
"""

import contextlib
import io

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIG, K_CU, RAD = 5.8e7, 400.0, 2e-4
HOST = 2.5 * RAD  # a thin solid needs a host within ~2.5 r; see Line.build


def bridged_traces():
    tr = lambda x0, x1, nm: jno.Shape.box(x0, 0, 0.001, x1, 0.004, 0.0015, size=HOST).attach(sigma=SIG, k=K_CU).name(nm)
    wire = (
        jno.Shape.line([(0.007, 0.002, 0.0015), (0.010, 0.002, 0.004), (0.013, 0.002, 0.0015)], r=RAD, size=HOST)
        .attach(sigma=SIG, k=K_CU)
        .name("W")
    )
    d = (tr(0, 0.008, "A") + tr(0.012, 0.020, "B") + wire).domain()
    d.tag("P", lambda x, y, z: x < 0.0011)
    d.tag("N", lambda x, y, z: x > 0.0189)
    return d


def solved(d):
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    return jno.peec([v(*at("P")) - v(*at("N")) - 1.0], freq=1e6).solve()


def test_dissipation_reconciles_with_the_total_it_came_from():
    """Every watt is accounted for: sum over regions of q_r * V_r is the total joule loss."""
    sol = solved(bridged_traces())
    q = sol.dissipation()
    vol = np.asarray(sol._vol)
    own = np.asarray(sol._owner)
    total = sum(float(q[n]) * float(vol[own == k].sum()) for k, n in enumerate(sol._names) if n in q)
    assert abs(total / float(sol.joule) - 1) < 1e-12


def test_a_thin_wire_dissipates_far_harder_than_the_trace_it_bridges():
    """Same current, much smaller section — which is why a total would hide it."""
    q = solved(bridged_traces()).dissipation()
    assert set(q) == {"A", "B", "W"}
    assert float(q["W"]) > 50 * max(float(q["A"]), float(q["B"]))


def test_the_dissipation_drives_a_thermal_solve_on_the_same_geometry():
    """One geometry, both solvers: PEEC never meshes it, the thermal solve does."""
    d = bridged_traces()
    emag = solved(d)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mesh = d.mesh
    assert len(np.asarray(mesh.points)) > 100

    T, phi = d.fem_symbols()
    xi, yi, zi = d.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    xn, yn, zn = d.variable("N", split=True)[:3]
    grad2 = lambda u, w: u.x * w.x + u.y * w.y + u.z * w.z
    Q = d.by_region({k: float(x) for k, x in emag.dissipation().items()}, default=0.0)
    Th = np.asarray(jno.fem([d.k * grad2(Ti, si) - Q * si, T(xn, yn, zn) - 300.0]).solve()).reshape(-1)

    assert Th.min() >= 300.0 - 1e-6  # nothing colder than the sink
    assert Th.max() > 300.0  # and the dissipation actually heats it
    assert np.isfinite(Th).all()


def test_a_solution_without_a_breakdown_says_so():
    sol = solved(bridged_traces())
    sol._owner = None
    with pytest.raises(ValueError, match="no per-conductor breakdown"):
        sol.dissipation()
