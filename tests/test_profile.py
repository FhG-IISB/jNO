"""jno.fem / jno.fdm ``.solve(profile=True)`` — JAX performance profiling, mirroring jno.core / jno.rcwa.

A concrete forward solve is run inside a JAX Perfetto trace; a one-line size/time summary is printed and the
trace is written to ``./jno_traces``. Shared machinery (``jno.utils.profiling``) across fem / fdm / rcwa.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")


def test_fem_solve_profile(tmp_path, monkeypatch, capsys):
    """fem.solve(profile=True) profiles the linear solve: returns the field, prints a summary, writes a trace."""
    monkeypatch.chdir(tmp_path)
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.12).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    sol = np.asarray(fem.solve(profile=True))
    assert sol.shape[0] == fem.dofs and np.all(np.isfinite(sol))
    assert "fem profile" in capsys.readouterr().out
    assert (tmp_path / "jno_traces").is_dir() and any((tmp_path / "jno_traces").iterdir())


def test_fdm_solve_profile(tmp_path, monkeypatch, capsys):
    """fdm.solve(profile=True) profiles the strong-form Newton solve — same summary + trace contract."""
    monkeypatch.chdir(tmp_path)
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    sol = np.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve(profile=True)).reshape(-1)
    assert np.all(np.isfinite(sol)) and sol.size > 0
    assert "fdm profile" in capsys.readouterr().out
    assert (tmp_path / "jno_traces").is_dir() and any((tmp_path / "jno_traces").iterdir())
