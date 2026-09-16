"""Droplets meshed as separate bodies in a void merge when their gap closes: ``remesh(alpha=...)``.

Two disks are meshed as one domain with two components and driven toward each other by a geometry term.
At fixed connectivity they can only ever approach -- a mesh cannot change its topology by moving. With
``adapt=jno.solve.remesh(alpha=1.2, every=1)`` the driver re-triangulates the NODES between chunks and
keeps the triangles smaller than ``alpha`` x the starting mean edge length: once the gap falls below
about ``2 alpha h`` the bridging triangles survive the filter and the two bodies become one mesh.

Oracles: the same march without ``alpha`` stays two bodies; with it the final mesh is one body, on the
same nodes in the same order (so the P1 state carries by identity), and the field that was two flat
plateaus starts to diffuse across the new neck.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.reconnect import n_components

H = 0.08


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _two_drops(adapt=None, *, T=0.32, n=13, order=1, gap_speed=0.5):
    """Two disks, gap 0.35, closing at ``2 * gap_speed``; one interior law moves each body as a whole.

    The gap STARTS above the merge threshold (~2 alpha h = 0.18 at h = 0.075) and falls below it during
    the march, so a merge at the right moment is what the test sees -- not one at the first reconnection.
    """
    d = (jno.shape.disk(0.0, 0.0, 0.5, size=H) | jno.shape.disk(1.35, 0.0, 0.5, size=H)).domain(time=(0.0, T, n))
    u, v = d.fem_symbols(order=order)
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y),  # diffusive enough to cross the neck in a few steps
            u(ci[0], ci[1]) - jno.np.tanh(8.0 * (ci[0] - 0.675)),  # -1 in the left drop, +1 in the right
            # one law on the whole interior: each body drifts toward the other, the sign taken from where
            # it is. A region that re-derives after a reconnection (unlike a `where=` predicate).
            xi.d(ti) + gap_speed * jno.np.tanh(8.0 * (xi - 0.675)),
        ]
    )
    return fem, fem.solve() if adapt is None else fem.solve(adapt=adapt)


def _bodies(frame):
    return n_components(len(np.asarray(frame[0])), np.asarray(frame[1]))


def test_without_reconnection_two_drops_stay_two_bodies():
    _fem, traj = _two_drops()
    assert _bodies(traj.meshes[0]) == 2
    assert _bodies(traj.meshes[-1]) == 2, "moving a mesh cannot change its topology"


def test_two_drops_in_a_void_merge_into_one_body():
    fem, traj = _two_drops(jno.solve.remesh(alpha=1.2, every=1))
    bodies = [_bodies(m) for m in traj.meshes]
    assert bodies[0] == 2
    assert bodies[-1] == 1, "the drops never merged"
    # They must merge WHEN the gap closes, not at the first reconnection: the mesh is still two bodies
    # after the first step, and once one, it stays one.
    assert bodies[1] == 2, f"merged immediately -- the gap started below the threshold ({bodies})"
    assert bodies[bodies.index(1) :] == [1] * (len(bodies) - bodies.index(1))
    assert any(h["remeshed"] for h in fem.adapt_history), "nothing reconnected"
    n0 = len(np.asarray(traj.meshes[0][0]))
    assert all(len(np.asarray(m[0])) == n0 for m in traj.meshes), "reconnection kept every node"
    assert all(np.asarray(s).shape == np.asarray(traj.states[0]).shape for s in traj.states)


def test_the_field_diffuses_across_the_new_neck():
    """Before the merge the two plateaus cannot talk; after it, diffusion crosses the neck."""
    _f0, plain = _two_drops()
    _f1, merged = _two_drops(jno.solve.remesh(alpha=1.2, every=1))
    X = np.asarray(merged.meshes[-1][0])
    neck = np.abs(X[:, 0] - 0.675) < 2.0 * H
    u_merged = np.asarray(merged.states[-1]).reshape(-1)[neck]
    Xp = np.asarray(plain.meshes[-1][0])
    u_plain = np.asarray(plain.states[-1]).reshape(-1)[np.abs(Xp[:, 0] - 0.675) < 2.0 * H]
    assert np.abs(u_merged).min() < 0.5 * np.abs(u_plain).min(), "no diffusion across the neck"


def test_alpha_without_a_geometry_term_is_refused():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.2, 5))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(ci[0], ci[1]) - 1.0])
    with pytest.raises(NotImplementedError, match="nothing moves the nodes"):
        fem.solve(adapt=jno.solve.remesh(alpha=1.2))


def test_alpha_with_a_criterion_is_refused():
    with pytest.raises(NotImplementedError, match="different operations"):
        _two_drops(jno.solve.remesh(alpha=1.2, criterion=lambda d: jno.le(d.cell_aspect(), 3.0)), T=0.05, n=3)


def test_a_p2_field_is_refused():
    with pytest.raises(NotImplementedError, match="P1"):
        _two_drops(jno.solve.remesh(alpha=1.2), T=0.05, n=3, order=2)


def test_a_non_positive_alpha_is_refused():
    with pytest.raises(ValueError, match="must be > 0"):
        jno.solve.remesh(alpha=0.0)
