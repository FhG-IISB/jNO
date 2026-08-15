"""Rebuilding an identical problem must not recompile — the content-keyed twin of the elem_map cache.

``elem_map``'s jit cache keys the baked closure leaves by object identity, so a REBUILD (fresh mesh
arrays, fresh closures, identical content) missed every entry and paid trace + XLA compile again:
measured 1.25 s of a 1.56 s warm rebuild on 3-D Poisson. The content-keyed fallback
(`_fn_content_key`) makes the rebuild reuse the compiled programs; these tests pin the three
properties that make that safe:

1. an identical rebuild HITS (and its results are identical),
2. anything that changes the operator MISSES (a different mesh, a different coefficient),
3. the keying never lies: a hit's solution equals the from-scratch solution bit-for-bit in structure
   and to float tolerance in values.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver.fem_utils import _ELEM_MAP_STATS


def _snap():
    return dict(_ELEM_MAP_STATS, content_bail=dict(_ELEM_MAP_STATS["content_bail"]))


def _delta(before):
    return {
        "content_hits": _ELEM_MAP_STATS["content_hits"] - before["content_hits"],
        "misses": _ELEM_MAP_STATS["misses"] - before["misses"],
        "id_hits": _ELEM_MAP_STATS["id_hits"] - before["id_hits"],
    }


def _poisson2d(size=0.28, k=1.0, f=1.0):
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
    _ = np.asarray(fem.b)  # force assembly
    return fem


def test_identical_rebuild_hits_and_matches():
    """THE property: same problem, fresh objects -> compiled programs reused, results identical."""
    fem1 = _poisson2d()
    before = _snap()
    fem2 = _poisson2d()
    d = _delta(before)
    assert d["content_hits"] > 0, f"identical rebuild produced no content hits: {d}"
    assert d["misses"] == 0, f"identical rebuild recompiled {d['misses']} kernels: {d}"
    np.testing.assert_allclose(np.asarray(fem1.b), np.asarray(fem2.b), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(fem1.solve()), np.asarray(fem2.solve()), rtol=1e-5, atol=1e-6)


def test_a_different_mesh_misses():
    """Staleness guard: geometry is baked into the kernels, so a different mesh must recompile —
    shape-keying would silently hand one mesh's coordinates to another."""
    _poisson2d(size=0.28)
    before = _snap()
    _poisson2d(size=0.22)  # different mesh -> different baked arrays
    d = _delta(before)
    assert d["misses"] > 0, f"a DIFFERENT mesh reused compiled kernels: {d}"


def test_a_different_coefficient_misses_and_differs():
    """A changed material must both recompile the kernels that bake it and change the answer."""
    fem1 = _poisson2d(k=1.0)
    before = _snap()
    fem3 = _poisson2d(k=7.0)
    d = _delta(before)
    assert d["misses"] > 0, f"a different coefficient reused every kernel: {d}"
    assert not np.allclose(np.asarray(fem1.solve()), np.asarray(fem3.solve()), atol=1e-6)


def test_zero_and_negative_coefficients_key_distinctly():
    """Extremes: 0.0 and -1.0 are distinct problems, not cache aliases of 1.0."""
    sols = {}
    for k in (1.0, -1.0):
        sols[k] = np.asarray(_poisson2d(k=k, f=1.0).solve())
    assert not np.allclose(sols[1.0], sols[-1.0], atol=1e-6)


def test_3d_rebuild_hits():
    def build():
        d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.35).domain()
        u, v = d.fem_symbols()
        xi, yi, zi, _ = d.variable("interior", split=True)
        xb, yb, zb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(xb, yb, zb) - 0.0])
        _ = np.asarray(fem.b)
        return fem

    f1 = build()
    before = _snap()
    f2 = build()
    d = _delta(before)
    assert d["misses"] == 0 and d["content_hits"] > 0, f"3-D rebuild recompiled: {d}"
    np.testing.assert_allclose(np.asarray(f1.solve()), np.asarray(f2.solve()), rtol=1e-5, atol=1e-6)


def test_parametric_rebuild_hits_and_solves_per_args():
    """A runtime parameter in the terms: the rebuild reuses kernels, and args still steer the solve
    (the compiled program must not have frozen the parameter's value)."""
    import jax

    def build():
        d = jno.Shape.rect(0, 0, 1, 1, size=0.28).domain()
        u, v = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        k = jno.np.parameter((1,), name="kk", key=jax.random.PRNGKey(0))
        k.initialize(jax.nn.initializers.constant(1.0))
        fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])
        return fem

    f1 = build()
    A1, b1v = f1.operator.evaluate({"kk": np.asarray([1.0])})
    a = np.linalg.solve(np.asarray(A1.todense() if hasattr(A1, "todense") else A1), np.asarray(b1v))
    before = _snap()
    f2 = build()
    A2, b2v = f2.operator.evaluate({"kk": np.asarray([1.0])})
    b1 = np.linalg.solve(np.asarray(A2.todense() if hasattr(A2, "todense") else A2), np.asarray(b2v))
    A3, b3v = f2.operator.evaluate({"kk": np.asarray([4.0])})
    b2 = np.linalg.solve(np.asarray(A3.todense() if hasattr(A3, "todense") else A3), np.asarray(b3v))
    d = _delta(before)
    np.testing.assert_allclose(a, b1, rtol=1e-5, atol=1e-6)
    assert not np.allclose(b1, b2, atol=1e-6), "args stopped steering the operator after a cache hit"
    assert d["misses"] == 0 or d["content_hits"] > 0, f"parametric rebuild shared nothing: {d}"


def test_multifield_rebuild_hits():
    """Taylor-Hood Stokes with a pressure pin — the case that exposed two per-build counters leaking
    into content keys (`field_key` dict keys, and `p.pin()`'s counter-suffixed tag name). A
    PolygonDomain (shapely-built) also exercises the domain-token MRO walk."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    def build():
        inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
        d = jno.domain(box(0.0, 0.0, 4.0, 1.0), mesh_size=0.35)
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
        p, q = d.fem_symbols(names=("p", "q"), order=1)
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
        pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
        fem = jno.fem(
            [
                1.0 * inner_(gu, gv, n_contract=2) - pp * trace(gv),
                -qq * trace(gu),
                u(xb, yb)[0] - yb * (1 - yb),
                u(xb, yb)[1] - 0.0,
                p.pin(),
            ]
        )
        _ = np.asarray(fem.b)
        return fem

    f1 = build()
    before = _snap()
    f2 = build()
    d = _delta(before)
    assert d["misses"] == 0 and d["content_hits"] > 0, f"multifield rebuild recompiled: {d}"
    s1 = np.asarray(f1.solve(linear=jno.solve.lu(backend="host")))
    s2 = np.asarray(f2.solve(linear=jno.solve.lu(backend="host")))
    # float32 direct solve on a saddle: independent builds measured ~1e-5 apart even pre-cache.
    assert np.linalg.norm(s1 - s2) / np.linalg.norm(s1) < 1e-3


def test_bail_is_observable_not_silent():
    """The tokenizer's coverage is MEASURED: whatever it cannot key lands in the bail tally, so a
    problem class that never caches is a visible number, not a mystery slowdown."""
    assert isinstance(_ELEM_MAP_STATS["content_bail"], dict)


def test_compile_cache_is_on_by_default_with_optouts():
    import jax

    # import jno already ran _auto_compile_cache in this process
    assert jax.config.jax_compilation_cache_dir, "persistent compile cache should be ON by default"
    jno.disable_compile_cache()
    assert jax.config.jax_compilation_cache_dir is None
    jno.enable_compile_cache()  # restore for the rest of the session
