"""A geometry term may reuse the coordinates a weak form uses.

``jno.fem`` retags a weak term's coordinate Variables IN PLACE to the quadrature pool (``'fem_gauss'`` for a
volume term, ``'gauss_<region>'`` for a surface term). A geometry term holding the same objects then asked
the domain for region ``'fem_gauss'`` at solve time and raised "has no location function". A free surface
written the natural way reuses ``xs, ys`` in both the capillary term and the kinematic term, so it hit this.

Oracle: the march is identical -- mesh and state, frame by frame -- to the same problem with the geometry
term given its own coordinate objects (which always worked).
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


def _interior_law(share):
    """Heat on the unit square; the interior shears, x' = 0.3 (y - 0.5)."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.2, 5))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    gx, gy, gt = (xi, yi, ti) if share else d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y), gx.d(gt) - 0.3 * (gy - 0.5), u(x0, y0) - 1.0])
    return fem.solve()


def _boundary_law(share):
    """Heat with a flux through the top edge, which also rises at 0.05."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.2, 5))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xt, yt, tt = d.variable("top", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    gx, gy, gt = (xt, yt, tt) if share else d.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y),
            -0.1 * v.bind(x=xt, y=yt),  # a flux in through the top: a SURFACE term on the same coordinates
            gy.d(gt) - 0.05,
            u(x0, y0) - 0.0,
        ]
    )
    return fem.solve()


@pytest.mark.parametrize("march", [_interior_law, _boundary_law], ids=["interior", "boundary"])
def test_a_geometry_term_may_share_coordinates_with_the_weak_form(march):
    shared, own = march(True), march(False)
    for k in (1, len(own) - 1):
        assert np.allclose(np.asarray(shared.meshes[k][0]), np.asarray(own.meshes[k][0]), atol=1e-12), f"mesh, frame {k}"
        assert np.allclose(np.asarray(shared.states[k]), np.asarray(own.states[k]), atol=1e-12), f"state, frame {k}"
    moved = np.abs(np.asarray(own.meshes[-1][0]) - np.asarray(own.meshes[0][0])).max()
    assert moved > 1e-3, "the oracle itself did not move, so the comparison proves nothing"
