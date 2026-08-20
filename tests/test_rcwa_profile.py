"""rc.solve(orders=…, profile=…) — re-solve at a different Fourier truncation, and JAX performance profiling.

``orders=N`` builds a fresh engine at truncation N (the construction ``orders`` is untouched) — for a
convergence sweep or profiling at scale. ``profile=True`` runs the solve eagerly inside a JAX Perfetto trace
(per-stage annotated), prints the problem size + wall time, and writes the trace — mirroring
``jno.core.solve(profile=True)``.
"""

import importlib.util
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the fmmax solve OOMs on a small GPU at these orders

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


import jno  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

K0 = 2 * jnp.pi
P, LZ = 1.1, 3.2


def _slab_cons():
    """A uniform a-Si slab (no lateral structure -> converges at tiny truncation, fast)."""
    d = jno.domain(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2))
    e = 1e-6
    for nm, f in [
        ("left", lambda x, y, z: x < e),
        ("right", lambda x, y, z: x > P - e),
        ("front", lambda x, y, z: y < e),
        ("back", lambda x, y, z: y > P - e),
        ("bottom", lambda x, y, z: z < e),
        ("top", lambda x, y, z: z > LZ - e),
    ]:
        d.tag(nm, f)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def face(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = face("bottom")
    utp, vtp = face("top")
    ul, _ = face("left")
    ur, _ = face("right")
    uf, _ = face("front")
    ub, _ = face("back")
    eps = jno.fn(lambda x, y, z: jnp.where((z >= 0.8) & (z < 1.15), 6.0, 1.0), [xi, yi, zi])
    return [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]


@needs_fmmax
def test_solve_orders_override_matches_construction():
    """rc.solve(orders=N) re-solves at truncation N and matches an engine built at N — the construction
    orders is just a default, overridable per solve (for a convergence sweep or profiling)."""
    cons = _slab_cons()
    rc = jno.rcwa(cons, orders=20, grid=24)
    t_over = float(rc.solve(orders=40).efficiency("T"))
    t_con = float(jno.rcwa(_slab_cons(), orders=40, grid=24).solve().efficiency("T"))
    assert t_over == pytest.approx(t_con, abs=1e-9)
    # the construction default is untouched: a plain solve still uses orders=20
    assert float(rc.solve().efficiency("T")) == pytest.approx(float(rc.solve(orders=20).efficiency("T")), abs=1e-9)


@needs_fmmax
def test_solve_orders_sweep_converges():
    """A uniform slab is truncation-insensitive: T at orders 8, 16, 32 agree — the sweep the caller uses to
    check convergence (Richardson: compare N vs ~1.5N)."""
    rc = jno.rcwa(_slab_cons(), orders=8, grid=24)
    ts = [float(rc.solve(orders=o).efficiency("T")) for o in (8, 16, 32)]
    assert max(ts) - min(ts) < 1e-3


@needs_fmmax
def test_solve_profile_runs_and_writes_trace(tmp_path, monkeypatch, capsys):
    """profile=True runs an eager solve inside a JAX Perfetto trace, returns a valid solution, prints a
    size/time summary, and writes the trace to ./rcwa_traces."""
    monkeypatch.chdir(tmp_path)  # keep the trace out of the repo
    rc = jno.rcwa(_slab_cons(), orders=20, grid=24)
    sol = rc.solve(profile=True)
    assert 0.0 <= float(sol.efficiency("T")) <= 1.0  # a real, usable solution
    assert "rcwa profile" in capsys.readouterr().out
    assert (tmp_path / "jno_traces").is_dir() and any((tmp_path / "jno_traces").iterdir())
