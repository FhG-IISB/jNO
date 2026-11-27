"""A Dirichlet pin on a `domain.tag` point set (a corner node, a stretch of an edge) works on a domain
built from named regions, as on a single-region one: the tag is not a boundary region, so it classified as a
volume term and was refused with "a residual with the trial but no test function must live on a boundary region"."""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("kind", ["corner", "edge_band"])
def test_a_point_set_pin_on_a_multi_region_domain_fixes_its_nodes(kind):
    W = 1.0
    lower = jno.shape.rect(0, 0, 1, W / 2).name("lower")
    upper = jno.shape.rect(0, W / 2, 1, W).name("upper")
    dom = (lower + upper).domain()
    if kind == "corner":
        dom.tag("pinned", lambda x, y: (x < 1e-9) & (y > W - 1e-9))
    else:  # the part of the top edge with |x - 0.5| < 0.3
        dom.tag("pinned", lambda x, y: (y > W - 1e-9) & (abs(x - 0.5) < 0.3))
    u, v = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xl, yl, _ = dom.variable("left", split=True)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - 1.0 * vi,
            u(xl, yl) - 0.0,
            u(*dom.variable("pinned", split=True)) - 2.0,
        ]
    )
    assert "dirichlet@pinned" in fem.classification
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points)
    sel = pts[:, 0] < 1e-9
    sel &= pts[:, 1] > W - 1e-9
    if kind == "edge_band":
        sel = (pts[:, 1] > W - 1e-9) & (np.abs(pts[:, 0] - 0.5) < 0.3)
    assert sel.any()
    np.testing.assert_allclose(sol[sel], 2.0, atol=1e-10)
