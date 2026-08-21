"""A boundary condition anchored to nodes the optimiser may MOVE must not pass silently.

``domain.tag(name, predicate)`` resolves to node IDS once, when ``jno.fem`` is built, and those ids
carry the condition from then on. On a static mesh that is exactly right. Combined with
``Variable.trainable()`` on the coordinates it becomes a trap: the nodes drift, the ids stay bound,
and the condition quietly stops describing the geometry its predicate names. Every solve remains
well posed, so nothing raises -- the reported objective simply stops being the objective of the mesh
that gets saved, and only an independent re-solve of that mesh ever reveals it.

Measured on the 3-D bracket this guard was written for: the relocation region was a bounding box
over the interior nodes, which also promoted 175 boundary nodes; the bolt clamp went 146 nodes -> 106
still on the hole wall, and the reported compliance was 4.6412 against 5.2876 from an independent
solve of the same mesh and density.
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


class _CollectLog:
    """Stands in for the domain logger and keeps the warnings."""

    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _build(trainable_where, *, allow=False):
    """Poisson on the unit square, Dirichlet on the LEFT edge, some region promoted to trainable."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    xm, _ym, _t = d.variable("mv", where=trainable_where, split=True)
    xm.trainable(name="mesh_x")
    if allow:
        d._allow_moving_bc = True

    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", where=lambda x, y: x < 1e-9, split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    log = _CollectLog()
    d.log = log
    jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xl, yl) - 0.0], quad_degree=2)
    return [w for w in log.warnings if "anchored to nodes" in w]


def test_warns_when_the_dirichlet_edge_is_trainable():
    """The left half is promoted, so the clamped left edge is inside the movable set."""
    hits = _build(lambda x, y: x < 0.5)
    assert hits, "a Dirichlet region made of movable nodes must be reported"
    msg = hits[0]
    assert "'left'" in msg, msg
    assert "_allow_moving_bc" in msg, "the message must name the opt-out"


def test_silent_when_the_trainable_region_excludes_the_boundary_condition():
    """The strict interior box shares no node with the left edge -- nothing to report."""
    assert _build(lambda x, y: (x > 0.25) & (x < 0.75) & (y > 0.25) & (y < 0.75)) == []


def test_opt_out_silences_it():
    """Driving a supported surface on purpose is legitimate; it just has to be said out loud."""
    assert _build(lambda x, y: x < 0.5, allow=True) == []
