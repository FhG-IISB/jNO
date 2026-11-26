"""A boundary term and a volume term in ONE residual must raise, not integrate over the volume.

``ui.x * vi.x + ui.y * vi.y - 1.0 * v.bind(x=xr, y=yr)`` used to classify as a plain volume term: the
boundary operand's coordinates live on its view, which the arithmetic discarded, so the region
resolver saw only the interior and integrated the "flux" over the whole domain (max 0.5 instead of
1.0 for -Laplace(u) = 0, u = 0 left, unit flux right). With a coefficient that carries a coordinate
it already raised "spans multiple regions"; a constant coefficient must raise the same way.
"""

import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _symbols():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xl, yl, _ = d.variable("left", split=True)
    return d, u, v, (xi, yi), (xr, yr), (xl, yl)


@pytest.mark.parametrize("coef", ["constant", "coordinate"])
def test_boundary_and_volume_in_one_residual_raise(coef):
    d, u, v, (xi, yi), (xr, yr), (xl, yl) = _symbols()
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    c = 1.0 if coef == "constant" else yr
    with pytest.raises(ValueError, match="spans multiple regions.*separate entries"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - c * v.bind(x=xr, y=yr), u(xl, yl) - 0.0])


def test_separate_entries_give_the_flux_solution():
    d, u, v, (xi, yi), (xr, yr), (xl, yl) = _symbols()
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, -1.0 * v.bind(x=xr, y=yr), u(xl, yl) - 0.0])
    sol = np.asarray(fem.solve()).reshape(-1)
    assert abs(sol.max() - 1.0) < 1e-8  # u = x
