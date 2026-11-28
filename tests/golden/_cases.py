"""Canonical RCWA golden cases: geometry only, no solver imports.

The single description of the problems whose outputs ``tests/golden/regen_goldens.py`` freezes and
``tests/test_rcwa_golden.py`` re-checks. Each case gives the period, wavelength, truncation, layers, incidence,
formulation and dtype as plain NumPy data; the ``.npz`` files hold only the frozen outputs.

Layer materials: a ``float`` (uniform isotropic), a 2-D array (patterned isotropic), a 5-tuple ``(exx, exy, eyx, eyy,
ezz)`` (uniform anisotropic) or a 10-tuple adding ``(mxx, mxy, myx, myy, mzz)`` (electric and magnetic).
"""

from __future__ import annotations

import numpy as np

# Shared constants (keep in lockstep with the analytical formulas in the tests).
WL = 1.0
K0 = 2 * np.pi


def disk_grid(grid: int = 64, eps: float = 11.0, r: float = 0.3, period: float = 0.6) -> np.ndarray:
    """A crossed-grating disk: eps inside a circle of radius ``r`` in vacuum."""
    x = (np.arange(grid) + 0.5) / grid
    xx, yy = np.meshgrid(x, x, indexing="ij")
    inside = (xx - 0.5) ** 2 + (yy - 0.5) ** 2 < r**2
    return np.where(inside, eps, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Case specs.
# ---------------------------------------------------------------------------
# Each entry: (name, spec-dict). ``spec`` keys:
#   wavelength, period (Px, Py), orders, k_in, formulation, dtype ('complex64'/'complex128'),
#   amb_sup, amb_sub, layers: list of (thickness, material)
# A ``None`` formulation means the backend default (JONES_DIRECT_FOURIER).
# ``rel_tol``/``abs_tol`` are kept for reference only: the golden test compares at the regression tolerance
# of tests/test_rcwa_golden.py (same code, same precision).
CASES = {
    # --- uniform slab: analytic Fresnel/Airy power T/R (order-0 optics) ------
    "slab_airy": {
        "description": "uniform n=2 slab in vacuum — order-0 Fresnel/Airy T/R",
        "wavelength": WL,
        "period": (0.6, 0.6),
        "orders": 60,
        "k_in": (0.0, 0.0),
        "formulation": "FFT",
        "dtype": "complex128",
        "amb_sup": 1.0,
        "amb_sub": 1.0,
        "layers": [(0.4, 4.0)],
        # |E|^2 interior sample: mid-film / incident-superstrate ratio (convention-free)
        "e2_planes": 3,
        "e2_nx": 16,
        "rel_tol": 1e-8,
        "abs_tol": 1e-9,
    },
    # --- crossed-grating disk, coarse truncation ----------------------------
    "disk_40": {
        "description": "crossed-grating disk, orders 40",
        "wavelength": WL,
        "period": (0.6, 0.6),
        "orders": 40,
        "k_in": (0.0, 0.0),
        "formulation": "FFT",
        "dtype": "complex128",
        "amb_sup": 1.0,
        "amb_sub": 1.0,
        "layers": [(0.35, disk_grid())],
        "rel_tol": 1e-6,
        "abs_tol": 1e-7,
    },
    # --- crossed-grating disk, fine truncation ------------------------------
    "disk_160": {
        "description": "crossed-grating disk, orders 160 (high-order stress)",
        "wavelength": WL,
        "period": (0.6, 0.6),
        "orders": 160,
        "k_in": (0.0, 0.0),
        "formulation": "JONES_DIRECT_FOURIER",
        "dtype": "complex128",
        "amb_sup": 1.0,
        "amb_sub": 1.0,
        "layers": [(0.35, disk_grid())],
        "rel_tol": 1e-6,
        "abs_tol": 1e-7,
    },
    # --- a-Si pillar (high contrast) ----------------------------------------
    "asi_disk": {
        "description": "a-Si pillar slab (high-contrast eps) — convergence stress",
        "wavelength": WL,
        "period": (0.6, 0.6),
        "orders": 120,
        "k_in": (0.0, 0.0),
        "formulation": "JONES_DIRECT_FOURIER",
        "dtype": "complex128",
        "amb_sup": 1.0,
        "amb_sub": 1.0,
        "layers": [(0.35, disk_grid(eps=12.25))],
        "rel_tol": 1e-5,
        "abs_tol": 1e-6,
    },
    # --- uniform biaxial anisotropic slab (5-component electric) ------------
    "aniso5": {
        "description": "uniform biaxial 5-component anisotropic slab diag(8,4,6)",
        "wavelength": WL,
        "period": (0.6, 0.6),
        "orders": 41,
        "k_in": (0.0, 0.0),
        "formulation": "FFT",
        "dtype": "complex128",
        "amb_sup": 1.0,
        "amb_sub": 1.0,
        "layers": [(0.3, (8.0, 0.0, 0.0, 4.0, 6.0))],
        "rel_tol": 1e-6,
        "abs_tol": 1e-7,
    },
    # --- PML-frame material: uniaxial 10-component (diagonal eps AND mu) ----
    "pml10": {
        "description": "uniform uniaxial 10-component frame (diagonal eps & mu), the PML material",
        "wavelength": WL,
        "period": (0.6, 0.6),
        "orders": 41,
        "k_in": (0.0, 0.0),
        "formulation": "FFT",
        "dtype": "complex128",
        "amb_sup": 1.0,
        "amb_sub": 1.0,
        "layers": [(0.3, (6.0, 0.0, 0.0, 4.0, 5.0, 1.0, 0.0, 0.0, 2.0, 1.5))],
        "rel_tol": 1e-6,
        "abs_tol": 1e-7,
    },
    # --- internal emitter (delta x-dipole) in a two-ambient stack ------------
    "emitter": {
        "description": "internal delta x-dipole in a vacuum film over a higher-index substrate",
        "wavelength": WL,
        "period": (1.0, 1.0),
        "orders": 60,
        "k_in": (1e-3, 1e-3),
        "formulation": "FFT",
        "dtype": "complex128",
        "amb_sup": 1.0,
        "amb_sub": 2.25,
        "layers": [(1.0, 1.0)],
        # emission source (mirrors the test_rcwa_source engine geometry)
        "source": {
            "layer": 0,
            "t_upper": 0.5,
            "t_lower": 0.5,
            "kind": "delta",
            "loc": [[0.5, 0.5]],
            "orient": [1, 0, 0],
            "amp": 1.0,
            "fwhm": 0.2,
        },
        "rel_tol": 2e-4,
        "abs_tol": 1e-5,
    },
}
