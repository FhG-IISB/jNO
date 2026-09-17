"""Beer-Lambert optical depth along a beam, against chords that are known exactly.

The path integral ``tau(x) = int alpha ds`` along the beam is non-local -- it depends on everything the
beam passed through -- but it is LINEAR in the absorption field, so it is a matrix built once per mesh
and applied in JAX. These tests pin it on geometry where the answer is a closed form:

* a uniform slab: ``tau`` grows linearly with depth, ``alpha * d``;
* a disk of radius R, beam offset by an impact parameter b: the chord is ``2 sqrt(R^2 - b^2)``, so
  ``tau = 2 alpha sqrt(R^2 - b^2)`` -- and it is zero for a ray that misses;
* a target in a VOID beyond two separated bodies sees both chords and neither gap, which is what makes
  the sampling construction worth its first-order edge error: shadowing needs no special case.

The last test is the one that matters for using this in a solve: ``d tau / d alpha`` must be exact, or
the attenuation could not take part in an inverse problem or an implicit coupling.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno
from jno.utils.optics import beam_paths, optical_depth


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _mesh(shape, size):
    d = shape.domain()
    return np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])


DOWN = (0.0, -1.0)  # a beam travelling downwards; the path to a point runs back up towards the source


def test_a_uniform_slab_attenuates_linearly_with_depth():
    """Through a slab of constant absorption the depth is exactly ``alpha * (top - y)``."""
    pts, cells = _mesh(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08), 0.08)
    alpha0 = 3.0
    ys = np.linspace(0.05, 0.95, 10)
    targets = np.c_[np.full_like(ys, 0.5), ys]
    nodes, w = beam_paths(pts, cells, DOWN, targets, n_samples=400)
    tau = np.asarray(optical_depth(np.full(len(pts), alpha0), nodes, w))
    exact = alpha0 * (1.0 - ys)
    assert np.abs(tau - exact).max() < 0.01 * alpha0, f"max error {np.abs(tau - exact).max():.4f}"


def test_a_disk_gives_its_chord_and_a_miss_gives_nothing():
    """``tau = 2 alpha sqrt(R^2 - b^2)`` below a disk of radius R, and 0 for a ray that misses it."""
    R, alpha0 = 0.5, 2.0
    pts, cells = _mesh(jno.shape.disk(0.0, 0.0, R, size=0.02), 0.02)
    bs = np.array([0.0, 0.15, 0.3, 0.45])
    targets = np.c_[bs, np.full_like(bs, -2.0 * R)]  # below the disk, in the void
    nodes, w = beam_paths(pts, cells, DOWN, targets, n_samples=600, span=4.0 * R)
    tau = np.asarray(optical_depth(np.full(len(pts), alpha0), nodes, w))
    exact = 2.0 * alpha0 * np.sqrt(R**2 - bs**2)
    assert np.abs(tau - exact).max() < 0.03 * exact.max(), f"chords {tau} against {exact}"

    miss = np.array([[1.5 * R, -2.0 * R]])
    nodes_m, w_m = beam_paths(pts, cells, DOWN, miss, n_samples=600, span=4.0 * R)
    assert float(optical_depth(np.full(len(pts), alpha0), nodes_m, w_m)[0]) == 0.0, "a miss must absorb nothing"


def test_two_bodies_shadow_each_other_and_the_gap_is_empty():
    """Sampling, rather than ray-boundary intersection, is what makes this need no special case."""
    R, alpha0, gap = 0.3, 1.5, 0.4
    shape = jno.shape.disk(0.0, 0.8, R, size=0.02) | jno.shape.disk(0.0, 0.8 - (2 * R + gap), R, size=0.02)
    pts, cells = _mesh(shape, 0.02)
    target = np.array([[0.0, -1.5]])  # below both drops
    nodes, w = beam_paths(pts, cells, DOWN, target, n_samples=1200, span=4.0)
    tau = float(optical_depth(np.full(len(pts), alpha0), nodes, w)[0])
    assert tau == pytest.approx(2.0 * alpha0 * 2.0 * R, rel=0.03), f"{tau} should be TWO chords, not one"


def test_a_beer_lambert_source_drives_a_solve_to_its_closed_form():
    """End to end: the depth becomes a volumetric heat source, and the temperature has a closed form.

    Steady conduction with ``Q(y) = alpha I0 exp(-alpha (1 - y))``, held at the bottom and insulated
    above, integrates twice to

        T(y) = (I0 / (alpha k)) [exp(-alpha) - exp(-alpha (1 - y))] + (I0 / k) y

    The source enters the term list as an ordinary KNOWN field -- ``.freeze(values)`` on the mesh nodes --
    which is the whole composition story: the attenuation is geometry plus a gather, and jNO needs no
    concept of a ray.
    """
    alpha0, I0, k = 2.5, 4.0, 0.7
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).domain()
    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    nodes, w = beam_paths(pts, cells, DOWN, pts, n_samples=600)  # the depth AT the mesh nodes
    tau = np.asarray(optical_depth(np.full(len(pts), alpha0), nodes, w))
    q_nodes = alpha0 * I0 * np.exp(-tau)  # Beer-Lambert absorption per unit volume

    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    q_field = u.bind(x=xi, y=yi).freeze(q_nodes)
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - q_field * vi, u(xb, yb) - 0.0])
    T = np.asarray(fem.solve()).reshape(-1)

    y = np.asarray(fem.points)[:, 1]
    exact = (I0 / (alpha0 * k)) * (np.exp(-alpha0) - np.exp(-alpha0 * (1.0 - y))) + (I0 / k) * y
    err = np.abs(T - exact).max() / np.abs(exact).max()
    assert err < 0.02, f"temperature is off its closed form by {100 * err:.2f} %"


def test_the_optical_depth_is_differentiable_in_the_absorption_field():
    """``d tau / d alpha_j`` is the path weight of node j -- exact, not a finite difference."""
    pts, cells = _mesh(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.12), 0.12)
    nodes, w = beam_paths(pts, cells, DOWN, np.array([[0.5, 0.2]]), n_samples=200)
    a0 = np.full(len(pts), 2.0)
    g = np.asarray(jax.grad(lambda a: optical_depth(a, nodes, w)[0])(jax.numpy.asarray(a0)))

    expect = np.zeros(len(pts))  # the same gather, accumulated by hand
    np.add.at(expect, nodes.reshape(-1), w.reshape(-1))
    assert np.abs(g - expect).max() < 1e-12
    eps = 1e-6  # and it agrees with a finite difference of the whole map
    j = int(np.argmax(expect))
    bumped = a0.copy()
    bumped[j] += eps
    fd = (float(optical_depth(bumped, nodes, w)[0]) - float(optical_depth(a0, nodes, w)[0])) / eps
    assert fd == pytest.approx(float(g[j]), rel=1e-6)
