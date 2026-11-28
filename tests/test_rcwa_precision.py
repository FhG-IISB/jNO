"""The ``dtype=`` choice of ``jno.rcwa``: complex64 or complex128, independent of the process-wide x64 flag where JAX
allows it."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jno.rcwa import Rcwa, RcwaError  # noqa: E402

N = 32
_X = (np.arange(N) + 0.5) / N - 0.5
GX, GY = np.meshgrid(_X, _X, indexing="ij")
DISK = np.where(GX**2 + GY**2 < 0.3**2, 6.0, 1.0)


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def x32():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rc(dtype, formulation="JONES_DIRECT_FOURIER"):
    return Rcwa(
        [(np.inf, 1.0), (0.4, DISK), (np.inf, 2.25)],
        period=(0.8, 0.8),
        orders=49,
        wavelength=1.0,
        k_in=(0.5, 0.2),
        formulation=formulation,
        assume_periodic=True,
        dtype=dtype,
    )


@pytest.mark.parametrize("formulation", ["FFT", "JONES_DIRECT_FOURIER"])
def test_complex64_under_x64_agrees_with_complex128(x64, formulation):
    sol64 = _rc("complex64", formulation).solve()
    sol128 = _rc("complex128", formulation).solve()
    t64, t128 = sol64.efficiency("T"), sol128.efficiency("T")
    assert jnp.result_type(t64) == jnp.float32 and jnp.result_type(t128) == jnp.float64
    assert float(t64) == pytest.approx(float(t128), abs=1e-5)
    assert float(sol64.efficiency("R")) == pytest.approx(float(sol128.efficiency("R")), abs=1e-5)


def test_the_default_follows_the_x64_flag(x64):
    assert _rc(None)._dtype == jnp.complex128


def test_complex128_without_x64_raises(x32):
    with pytest.raises(RcwaError, match="jax_enable_x64"):
        _rc("complex128")


def test_an_unknown_dtype_raises(x64):
    with pytest.raises(RcwaError, match="dtype must be"):
        _rc("float32")


def test_a_complex64_gradient_is_finite_and_close_to_complex128(x64):
    def dT(dtype):
        rc = _rc(dtype)
        f = lambda s: rc.solve(layers=[(np.inf, 1.0), (0.4, 1.0 + (jnp.asarray(DISK) - 1.0) * s), (np.inf, 2.25)])  # noqa: E731
        return float(jax.grad(lambda s: jnp.real(f(s).efficiency("T")))(1.0))

    g64, g128 = dT("complex64"), dT("complex128")
    assert np.isfinite(g64)
    assert g64 == pytest.approx(g128, rel=1e-3)
