"""The cuDSS backend refuses a cuDSS that nvmath-python cannot bind at SELECTION, naming versions and the pin."""

from __future__ import annotations

import pytest

from jno.utils.solver import linear as lin


@pytest.fixture
def mismatch(monkeypatch):
    """nvmath-python's own check failing, as it does with a cuDSS 1.0 against its 0.8."""
    pytest.importorskip("nvmath.sparse.advanced")
    pytest.importorskip("cupy")
    import importlib

    ds = importlib.import_module("nvmath.sparse.advanced.direct_solver")

    def refuse():
        raise RuntimeError("nvmath-python requires cuDSS version 0.8, while you have cuDSS version 1.0.")

    monkeypatch.setattr(ds, "_check_cudss_version", refuse)
    lin._cudss_version_problem.cache_clear()
    yield
    lin._cudss_version_problem.cache_clear()


def test_a_cudss_version_mismatch_makes_the_stack_unavailable_with_a_warning(mismatch):
    with pytest.warns(RuntimeWarning, match=r"cuDSS version 1\.0.*nvidia-cudss-cu12>=0\.8,<0\.9|cuDSS version 1\.0"):
        assert lin._cudss_available() is False


def test_the_message_names_the_versions_and_the_working_pin(mismatch):
    msg = lin._cudss_version_problem()
    assert "requires cuDSS version 0.8" in msg and "cuDSS version 1.0" in msg
    assert "nvidia-cudss-cu12>=0.8,<0.9" in msg and "Installed:" in msg


def test_selecting_the_cudss_backend_raises_the_version_message_not_a_generic_one(mismatch):
    import jax.experimental.sparse as jsparse
    import jax.numpy as jnp

    A = jsparse.BCOO.fromdense(jnp.eye(3))
    with pytest.raises(ImportError, match=r"nvidia-cudss-cu12>=0\.8,<0\.9"):
        lin.cudss_lu_solve(A, jnp.ones(3))


def test_the_default_backend_falls_back_to_device_on_a_mismatch(mismatch, monkeypatch):
    from types import SimpleNamespace

    import jax

    from jno import solve

    monkeypatch.setattr("jno.utils.solver.placement.solve_device", lambda: SimpleNamespace(platform="gpu"))
    with jax.enable_x64(True), pytest.warns(RuntimeWarning):
        assert solve._preferred_direct_backend() == "device"


def test_a_matching_stack_reports_no_problem():
    pytest.importorskip("nvmath.sparse.advanced")
    lin._cudss_version_problem.cache_clear()
    assert lin._cudss_version_problem() is None
