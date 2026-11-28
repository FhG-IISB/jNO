"""The cross-check cases, shared by ``generate_reference.py`` (reference libraries) and
``tests/test_rcwa_reference.py`` (jNO). Plain numpy only.

Units: lengths in micrometres-free normalised units, ``k_in`` in units of the vacuum wavenumber ``k0``,
``polarization`` the tangential incident field ``(E_x, E_y)``. Every case is solved with Laurent's rule and the
parallelogramic truncation ``|m|, |n| <= M`` for each ``M`` in ``truncations``.
"""

import numpy as np


def grid_of(spec):
    """The permittivity grid of a layer spec ``(shape, params)``, pixel-centred samples on ``n x n``."""
    shape, p = spec
    n = p["n"]
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    if shape == "disk":
        inside = (X - p.get("cx", 0.0)) ** 2 + (Y - p.get("cy", 0.0)) ** 2 < p["r"] ** 2
    elif shape == "rect":
        inside = (np.abs(X - p.get("cx", 0.0)) < p["wx"] / 2) & (np.abs(Y - p.get("cy", 0.0)) < p["wy"] / 2)
    elif shape == "bar":  # a 1-D grating along x
        inside = np.abs(X) < p["w"] / 2
    else:
        raise ValueError(shape)
    return np.where(inside, p["eps_in"], p.get("eps_out", 1.0)).astype(complex)


def pure(k_in, kind):
    """The tangential incident field of a pure s (TE) or p (TM) wave for ``k_in`` (units of k0): s along
    ``(-sin phi, cos phi)``, p along ``(cos phi, sin phi)``, ``phi`` the azimuth of the plane of incidence (x at
    normal incidence). Only the direction matters: efficiencies are normalised by the incident power."""
    phi = float(np.arctan2(k_in[1], k_in[0])) if np.hypot(*k_in) > 0 else 0.0
    return (-np.sin(phi), np.cos(phi)) if kind == "s" else (np.cos(phi), np.sin(phi))


_DISK = ("disk", {"n": 64, "r": 0.3, "eps_in": 6.0})
_POST = ("rect", {"n": 64, "wx": 0.5, "wy": 0.36, "cx": 0.07, "cy": -0.05, "eps_in": 8.0 + 0.2j})
_BAR = ("bar", {"n": 64, "w": 0.4, "eps_in": 4.0})

CASES = {
    "disk_normal_x": dict(
        period=(1.0, 1.0),
        wavelength=1.3,
        eps_top=1.0,
        eps_bottom=2.25,
        k_in=(0.0, 0.0),
        pol="p",
        layers=[(0.5, _DISK)],
        truncations=[3, 5, 7],
    ),
    "disk_oblique_te": dict(
        period=(1.0, 1.0),
        wavelength=1.3,
        eps_top=1.0,
        eps_bottom=2.25,
        k_in=(0.34, 0.0),
        pol="s",
        layers=[(0.5, _DISK)],
        truncations=[3, 5, 7],
    ),
    "disk_conical_te": dict(
        period=(1.0, 1.0),
        wavelength=1.3,
        eps_top=1.0,
        eps_bottom=2.25,
        k_in=(0.3, 0.17),
        pol="s",
        layers=[(0.5, _DISK)],
        truncations=[3, 5],
    ),
    "lossy_post_two_layers_tm": dict(
        period=(1.0, 1.0),
        wavelength=0.9,
        eps_top=1.0,
        eps_bottom=2.25,
        k_in=(0.2, 0.1),
        pol="p",
        layers=[(0.3, _POST), (0.25, _DISK)],
        truncations=[3, 5],
    ),
    "bar_tm": dict(
        period=(0.8, 0.8),
        wavelength=1.0,
        eps_top=1.0,
        eps_bottom=1.0,
        k_in=(0.22, 0.0),
        pol="p",
        layers=[(0.6, _BAR)],
        truncations=[4, 8],
    ),
}
for _c in CASES.values():
    _c["polarization"] = pure(_c["k_in"], _c["pol"])
