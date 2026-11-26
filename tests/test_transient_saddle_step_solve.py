"""The transient step solve on a SADDLE problem: the default picks a direct solve, and the Krylov default
fails loud instead of marching a NaN.

Unsteady Stokes (P2/P1): the pressure has no time derivative and its row of ``M + dt A`` has a zero
diagonal. Jacobi has nothing to scale there and the Schur complement shrinks with dt, so the Krylov
default broke down -- measured fine at dt = 1e-3 and NaN, in silence, at dt = 1e-4 and below.
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402

import jno  # noqa: E402
from tests.test_fem_adaptive_timestep import _heat, _stokes  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _small_dt_stokes():
    return _stokes(6, t1=5e-4)  # dt = 1e-4


def test_saddle_is_detected_and_dirichlet_rows_are_not_saddle_rows():
    from jno.utils.solver.solver_api import _is_saddle_block

    assert _is_saddle_block(_small_dt_stokes().operator)
    _, heat = _heat(mesh_size=0.3)  # zero-mass Dirichlet rows, but each fixes its own unknown
    assert not _is_saddle_block(heat.operator)


def test_saddle_default_step_solve_is_direct_and_right():
    out = np.asarray(_small_dt_stokes().solve().fn())
    ref = np.asarray(_small_dt_stokes().solve(linear=jno.solve.lu()).fn())
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_krylov_step_solve_failure_raises_by_name(monkeypatch):
    """With the saddle default switched off, the Krylov step solve breaks down; that must raise, naming the
    solve, rather than return a NaN trajectory."""
    from jno.utils.solver import solver_api

    monkeypatch.setattr(solver_api, "_is_saddle_block", lambda *a, **k: False)
    with pytest.raises(Exception, match="default time-step solve"):
        np.asarray(_small_dt_stokes().solve().fn())
