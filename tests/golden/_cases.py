"""Canonical RCWA golden cases — geometry only, no solver imports.

This module is the single source of truth for the physical problems that the
golden oracle freezes. It defines each case's *geometry* (period, wavelength,
Fourier truncation, layer thicknesses and materials, incidence, formulation,
dtype) as plain NumPy data. It imports **neither** ``jno`` **nor** ``fmmax``:

* ``tests/golden/regen_goldens.py`` reads these specs and computes the fmmax
  1.7.1 reference T/R for each case (plus, for the uniform slab, a small
  interior |E|^2 sample; for the emitter, the radiated up/down powers), then
  writes ``tests/golden/<name>.npz`` and the ``manifest.json``. The ``.npz``
  files freeze T/R (and those per-case extras) — they do NOT store per-order
  flux arrays.
* ``tests/test_rcwa_golden.py`` reads the same specs, rebuilds the equivalent
  ``jno._rcwa.stack.StateStack`` problem, solves it with the self-owned eig-free
  backend and compares to the frozen ``.npz`` — passing whether fmmax is present
  or not.

Keeping the geometry here (rather than baking it into each ``.npz``) means the
regen script and the golden test can never drift on the problem they describe;
the ``.npz`` holds only the frozen *reference outputs*.

Layer material encodings (shared by the fmmax oracle and the jno backend):

* a Python ``float`` — a uniform isotropic permittivity (the fmmax oracle feeds
  it as a ``[[eps]]`` 1×1 grid; jno's uniform fast path handles the scalar);
* a 2-D ``np.ndarray`` — a patterned isotropic permittivity grid (both sides
  feed the same ``(g, g)`` array);
* a length-5 tuple — a uniform general-anisotropic ``(exx, exy, eyx, eyy, ezz)``
  layer (magnetic trivial);
* a length-10 tuple — a uniform ``(exx, exy, eyx, eyy, ezz, mxx, mxy, myx, myy,
  mzz)`` layer (both electric *and* magnetic response, the PML-frame material).
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
# ``abs_tol``/``rel_tol`` document the per-case agreement bound for T/R at the
# given dtype. NOTE: all committed cases are complex128 (the ``complex64``
# ~1e-5..1e-4 band is documented for a future sweep but NOT exercised by any
# golden today). Golden generation always runs the oracle in complex128; the
# *backend* is exercised at the case's stated dtype.
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
