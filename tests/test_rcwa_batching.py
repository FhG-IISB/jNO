"""``jax.vmap`` over wavelength and incidence goes through ``Rcwa.solve`` and equals the loop of single solves (the
design's Fourier matrices do not depend on the batched inputs, so vmap builds them once)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jno.rcwa import Rcwa  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def rc():
    n = 32
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    disk = np.where(X**2 + Y**2 < 0.3**2, 6.0, 1.0)
    return Rcwa(
        [(np.inf, 1.0), (0.4, disk), (np.inf, 2.25)], period=(0.8, 0.8), orders=49, wavelength=1.0, assume_periodic=True
    )


def test_vmap_over_wavelength_equals_the_loop(rc):
    wls = jnp.linspace(0.95, 1.1, 5)
    batched = np.asarray(jax.jit(jax.vmap(lambda w: rc.solve(wavelength=w).efficiency("T")))(wls))
    loop = np.asarray([float(rc.solve(wavelength=w).efficiency("T")) for w in wls])
    np.testing.assert_allclose(batched, loop, atol=1e-13)


def test_vmap_over_wavelength_and_incidence_together(rc):
    wls = jnp.linspace(0.95, 1.1, 3)
    kins = jnp.stack([jnp.linspace(0.0, 1.5, 4), jnp.zeros(4)], -1)
    f = jax.vmap(jax.vmap(lambda w, k: rc.solve(wavelength=w, k_in=k).efficiency("T"), (None, 0)), (0, None))
    grid = np.asarray(jax.jit(f)(wls, kins))
    assert grid.shape == (3, 4)
    assert grid[1, 2] == pytest.approx(float(rc.solve(wavelength=wls[1], k_in=kins[2]).efficiency("T")), abs=1e-13)
