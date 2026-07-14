"""Tests for jno.rcwa: physical correctness (vs analytic Fresnel/TMM) + the never-silent guards.

Skipped in full when the optional `fmmax` backend is not installed (mirrors tests/test_iree.py)."""

import numpy as np
import pytest

pytest.importorskip("fmmax", reason="fmmax (jno.rcwa backend) not installed")

import jno  # noqa: E402
from jno.rcwa import Rcwa, RcwaError  # noqa: E402

WL = 1.03
INF = np.inf


def tmm_slab(nn, h, lam):
    """Analytic air|slab|air transmittance via a 2x2 transfer matrix."""

    def iface(a, b):
        r = (a - b) / (a + b)
        t = 2 * a / (a + b)
        return np.array([[1, r], [r, 1]]) / t

    beta = 2 * np.pi * nn * h / lam
    M = iface(1, nn) @ np.array([[np.exp(-1j * beta), 0], [0, np.exp(1j * beta)]]) @ iface(nn, 1)
    return abs(1 / M[0, 0]) ** 2


@pytest.mark.parametrize("nn,h", [(1.45, 0.5), (2.0, 0.4), (3.317, 0.3)])
def test_fresnel_matches_analytic(nn, h):
    """A uniform slab reproduces the analytic Fresnel transmittance and conserves energy."""
    rc = Rcwa([(INF, 1.0), (h, nn**2), (INF, 1.0)], period=(1.0, 1.0), orders=5, wavelength=WL, assume_periodic=True)
    sol = rc.solve(inc=None)
    T, R = sol.efficiency("T"), sol.efficiency("R")
    assert abs(T - tmm_slab(nn, h, WL)) < 2e-3
    assert abs(T + R - 1) < 1e-3


def test_functional_entry_point_matches_class():
    """jno.rcwa(...) builds the same problem as the Rcwa class."""
    rc = jno.rcwa([(INF, 1.0), (0.4, 4.0), (INF, 1.0)], period=(1.0, 1.0), orders=5, wavelength=WL, assume_periodic=True)
    assert isinstance(rc, Rcwa)
    T = rc.solve().efficiency("T")
    assert abs(T - tmm_slab(2.0, 0.4, WL)) < 2e-3


def test_guard_non_periodic_raises():
    with pytest.raises(RcwaError, match="assume_periodic"):
        Rcwa([(INF, 1.0), (0.3, 11.0), (INF, 1.0)], period=(1.0, 1.0), orders=5, wavelength=WL)


def test_guard_bad_period_raises():
    with pytest.raises(RcwaError, match="period"):
        Rcwa([(INF, 1.0)], period=(-1.0, 1.0), orders=5, wavelength=WL, assume_periodic=True)


def test_guard_empty_layers_raises():
    with pytest.raises(RcwaError, match="layers"):
        Rcwa([], period=(1.0, 1.0), orders=5, wavelength=WL, assume_periodic=True)


def test_guard_missing_wavelength_raises():
    rc = Rcwa([(INF, 1.0), (0.3, 11.0), (INF, 1.0)], period=(1.0, 1.0), orders=5, assume_periodic=True)
    with pytest.raises(RcwaError, match="wavelength"):
        rc.solve(None)


def test_guard_coarse_grid_vs_orders_raises():
    big = np.ones((6, 6)) * 11.0
    with pytest.raises(RcwaError, match="under-resolve"):
        Rcwa([(INF, 1.0), (0.3, big), (INF, 1.0)], period=(1.0, 1.0), orders=400, wavelength=WL, assume_periodic=True)


def test_bad_efficiency_kind_raises():
    rc = Rcwa([(INF, 1.0), (0.4, 4.0), (INF, 1.0)], period=(1.0, 1.0), orders=5, wavelength=WL, assume_periodic=True)
    with pytest.raises(RcwaError, match="'T' or 'R'"):
        rc.solve().efficiency("Q")


def test_as_precond_without_transfer_raises():
    """as_precond must never hand back a silent no-op preconditioner."""
    rc = Rcwa([(INF, 1.0), (0.4, 4.0), (INF, 1.0)], period=(1.0, 1.0), orders=5, wavelength=WL, assume_periodic=True)
    spec = rc.as_precond()
    with pytest.raises(RcwaError, match="transfer"):
        spec.materialize(ctx=None)
