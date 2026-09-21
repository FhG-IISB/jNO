"""``jno.info`` on domain decomposition: regions, interface tags, ``jno.dd.couple`` and a coupling core.

Every count here is checked against the mesh coordinates directly, and the coupling method against
the geometry (overlapping regions -> Schwarz, a partition -> a Dirichlet-Neumann line).
"""

import jax
import numpy as np
import pytest

pytest.importorskip("shapely")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402
from jno.dd import couple  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rows(rep, section):
    return dict(next(rows for name, rows in rep.sections if name == section))


def _overlapping(mesh_size=0.1):
    """Two FDM subdomains on regions A = [0, 0.6] and B = [0.4, 1] of the unit square."""
    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=mesh_size)
    d.region("A", b1)
    d.region("B", b2)
    xa, ya, _ = d.variable("A", split=True)
    xb2, yb2, _ = d.variable("B", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    aa, ab = u.bind(x=xa, y=ya), u.bind(x=xb2, y=yb2)
    fa = 2 * np.pi**2 * jnn.sin(np.pi * xa) * jnn.sin(np.pi * ya)
    fb = 2 * np.pi**2 * jnn.sin(np.pi * xb2) * jnn.sin(np.pi * yb2)
    a = jno.fdm([-aa.d2(xa) - aa.d2(ya) - fa, u(xb, yb) - 0.0])
    b = jno.fdm([-ab.d2(xb2) - ab.d2(yb2) - fb, u(xb, yb) - 0.0])
    return d, a, b


def test_regions_added_after_the_mesh_are_listed():
    d, _, _ = _overlapping()
    tags = _rows(jno.info(d), "tags")
    # Both were missing: a region added after the mesh is sampled lazily and never enters the pool.
    assert "region" in tags["A"] and "[0, 0.6]" in tags["A"]
    assert "region" in tags["B"] and "[0.4, 1]" in tags["B"]
    # No count given, so A is ONE point redrawn every step (Monte-Carlo mode); the row must say so
    # rather than print a bare "1 points".
    assert tags["A"].startswith("1 points") and "resampled" in tags["A"]


def test_interface_tags_are_listed_once_with_their_nodes():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    d.region("L", box(0.0, 0.0, 0.5, 1.0))
    d.region("R", box(0.5, 0.0, 1.0, 1.0))
    d.build_mesh(mesh_size=0.2)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    on_line = int(np.sum(np.abs(p[:, 0] - 0.5) < 1e-9))

    tags = _rows(jno.info(d), "tags")
    assert tags["interface_L_R"].startswith(f"{on_line} points")
    assert "interface between regions L and R" in tags["interface_L_R"]
    assert tags["interface_R_L"] == "alias of interface_L_R"  # the same nodes, not counted twice
    assert tags["L"].startswith(f"{int(np.sum(p[:, 0] <= 0.5 + 1e-9))} points")


def test_a_coupling_core_says_what_it_couples_and_needs_no_optimizer():
    d, a, b = _overlapping()
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    rep = jno.info(jno.core([a, b]))
    subs = _rows(rep, "subdomain solves")
    assert f"owns {int(np.sum(p[:, 0] <= 0.6 + 1e-9))} of {len(p)} nodes" in subs["[0]"]
    assert f"owns {int(np.sum(p[:, 0] >= 0.4 - 1e-9))} of {len(p)} nodes" in subs["[1]"]
    assert subs["[0]"].startswith("fdm · on region A")
    assert _rows(rep, "coupling")["method"].startswith("overlap-Schwarz")
    # The generic core branch told this user to call .optimizer(...): false, nothing is trained.
    backend = _rows(rep, "training")["training backend"]
    assert backend.startswith("not needed") and "optimizer" not in backend


def test_couple_reports_its_last_solve():
    _, a, b = _overlapping()
    cp = couple([(a, jno.shape.rect(0.0, 0.0, 0.6, 1.0)), (b, jno.shape.rect(0.4, 0.0, 1.0, 1.0))])
    before = jno.info(cp)
    assert "(generic)" not in before.title
    assert _rows(before, "coupling")["method"].startswith("overlap-Schwarz")
    assert "last solve" not in [n for n, _ in before.sections]

    _, got = cp.solve(tol=1e-6, max_iter=60, return_info=True)
    last = _rows(jno.info(cp), "last solve")
    assert last["iterations"] == f"{got['iterations']} of max 60"
    assert last["overlap jump"].endswith("✓")


def test_a_partition_couples_on_a_line():
    """The method is decided by geometry alone, so it is reported before any solve."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    d.region("L", box(0.0, 0.0, 0.5, 1.0))
    d.region("R", box(0.5, 0.0, 1.0, 1.0))
    d.build_mesh(mesh_size=0.2)
    xl, yl, _ = d.variable("L", split=True)
    xr, yr, _ = d.variable("R", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ul, ur = u.bind(x=xl, y=yl), u.bind(x=xr, y=yr)
    left = jno.fdm([-ul.d2(xl) - ul.d2(yl) - 1.0, u(xb, yb) - 0.0])
    right = jno.fdm([-ur.d2(xr) - ur.d2(yr) - 1.0, u(xb, yb) - 0.0])
    cp = couple([(left, box(0.0, 0.0, 0.5, 1.0)), (right, box(0.5, 0.0, 1.0, 1.0))])
    assert _rows(jno.info(cp), "coupling")["method"].startswith("line-DN")
