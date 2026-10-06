"""A conforming periodic tie is a 0/1 node map, built without the mortar machinery.

``build_periodic_prolongation`` computed the integrated (dual-mortar) rows of every face pair -- pure-Python
polygon clipping per facet -- and only afterwards found that every node had an exact partner, and threw the
rows away. On a 32^3 periodic box that was 14 of a 34 s build (most of a 94 s build at 48^3). The match is
now decided first. Its nearest-node search was a dense (n_s, n_m) distance table, quadratic in the face
size; it is a k-d tree now.

Oracles: on a conforming periodic box the mortar row builders are never called and the tie reports
"conforming"; the k-d tree nearest node equals the dense argmin, including its lowest-index tie break.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

import jno
import jno.utils.solver.fem_utils as fu


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_a_conforming_periodic_box_never_builds_mortar_rows(monkeypatch):
    calls = []
    for name in ("_mortar_rows_3d", "_mortar_rows_2d"):
        real = getattr(fu, name)
        monkeypatch.setattr(fu, name, lambda *a, _r=real, _n=name, **k: calls.append(_n) or _r(*a, **k))
    L, e = 1.0, 1e-9
    d = jno.shape.box(0, 0, 0, L, L, L).structured(n=4).domain()
    faces = {"x0": lambda x, y, z: x < e, "x1": lambda x, y, z: x > L - e,
             "y0": lambda x, y, z: y < e, "y1": lambda x, y, z: y > L - e,
             "z0": lambda x, y, z: z < e, "z1": lambda x, y, z: z > L - e}  # fmt: skip
    for nm, f in faces.items():
        d.tag(nm, f)
    u, v = d.fem_symbols(names=("u", "v"))
    x, y, z = d.variable("interior", split=True)[:3]
    ub, vb = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    at = lambda r: d.variable(r, split=True)[:3]  # noqa: E731
    terms = [ub.x * vb.x + ub.y * vb.y + ub.z * vb.z + ub * vb - jno.np.sin(2 * np.pi * x) * vb]
    for lo, hi in (("x0", "x1"), ("y0", "y1"), ("z0", "z1")):
        terms.append(u(*at(lo)) - u(*at(hi)))
    fem = jno.fem(terms)
    sol = np.asarray(fem.solve())
    assert calls == [], f"a conforming tie built mortar rows: {calls}"
    assert np.isfinite(sol).all() and np.abs(sol).max() > 1e-3
    pts = np.asarray(fem.points)
    # The tie holds: x = 0 and x = 1 carry the same value at every matching (y, z).
    lo, hi = pts[:, 0] < e, pts[:, 0] > L - e
    key = lambda p: (round(p[1], 9), round(p[2], 9))  # noqa: E731
    at_hi = {key(p): sol[i] for i, p in zip(np.flatnonzero(hi), pts[hi])}
    assert max(abs(sol[i] - at_hi[key(p)]) for i, p in zip(np.flatnonzero(lo), pts[lo])) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_kd_tree_nearest_node_equals_the_dense_argmin(seed):
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 6, size=(40, 2)).astype(float)  # a lattice with repeated points: distance ties
    s = rng.integers(0, 6, size=(25, 2)).astype(float) + rng.choice([0.0, 0.5], size=(25, 2))
    nn, dist = fu._nearest_in_interface(s, m)
    d2 = np.sum((s[:, None, :] - m[None, :, :]) ** 2, axis=-1)
    want = np.argmin(d2, axis=1)
    assert np.array_equal(nn, want)
    assert np.allclose(dist, np.sqrt(d2[np.arange(len(s)), want]))
