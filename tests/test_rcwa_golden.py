"""Golden-oracle parity tests: the eig-free RCWA backend vs frozen fmmax refs.

Every canonical case in :mod:`tests.golden._cases` was solved in complex128 by
**fmmax 1.7.1** (the reference RCWA) and its T/R frozen into
``tests/golden/<name>.npz`` by ``tests/golden/regen_goldens.py`` (the uniform
slab also freezes an interior |E|^2 sample, the emitter the radiated up/down
powers). These tests re-solve the SAME physical problem with the self-owned,
eig-free ``jno._rcwa.stack.StateStack`` backend and check it reproduces the
frozen references.

This file does **not** import fmmax: it runs and passes whether or not fmmax is
installed. fmmax is only ever the regen-time oracle, never a runtime or test
dependency. The goldens freeze the parity contract so a regression in the
backend is caught even in a clean env without fmmax.

Tolerances are documented per case in ``_cases.py``. All committed goldens are
complex128 (agreement essentially exact, ~1e-8); the complex64 ~1e-5..1e-4 band
is documented but NOT exercised by any committed case today (future work).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

from jno._rcwa.stack import StateStack  # noqa: E402
from tests.golden import _cases  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64 (the golden dtype). Save/restore the flag."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _build_stack(spec):
    """A StateStack for a plane-wave case spec (materials decoded like _cases)."""
    dtype = jnp.complex128 if spec["dtype"] == "complex128" else jnp.complex64
    layers = []
    for thick, mat in spec["layers"]:
        if isinstance(mat, np.ndarray):
            layers.append((thick, jnp.asarray(mat)))
        else:
            layers.append((thick, mat))
    return StateStack(
        wavelength=spec["wavelength"],
        period=spec["period"],
        amb_sup=spec["amb_sup"],
        amb_sub=spec["amb_sub"],
        layers=layers,
        orders=spec["orders"],
        k_in=spec["k_in"],
        formulation=spec.get("formulation"),
        dtype=dtype,
    )


def _ref(name, key):
    return np.load(os.path.join(HERE, "golden", f"{name}.npz"))[key]


ALL_PLANE_WAVE = [
    "slab_airy",
    "disk_40",
    "disk_160",
    "asi_disk",
    "aniso5",
    "pml10",
]


@pytest.mark.parametrize("name", ALL_PLANE_WAVE)
def test_plane_wave_matches_golden(name):
    """The eig-free backend reproduces the fmmax-frozen T/R for a plane-wave stack."""
    spec = _cases.CASES[name]
    st = _build_stack(spec)
    res = st.solve(validate=False)
    tol = spec["rel_tol"]
    atol = spec["abs_tol"]
    assert float(res.T) == pytest.approx(float(_ref(name, "t_ref")), rel=tol, abs=atol), name
    assert float(res.R) == pytest.approx(float(_ref(name, "r_ref")), rel=tol, abs=atol), name


def test_slab_e2_matches_golden():
    """The interior |E|^2 of the uniform slab (mid-film) matches the frozen analytic Airy value.

    A uniform slab at normal incidence has a spatially-constant transverse field; the backend
    launches a unit incident field, so the mid-film |E|^2 is the closed-form Airy value (0.4910
    here). (Normal incidence only — the documented oblique-|E|^2 caveat does not apply.)"""
    name = "slab_airy"
    spec = _cases.CASES[name]
    st = _build_stack(spec)
    e2_ref = float(_ref(name, "e2_ref"))

    # interior field amplitudes at the mid-film plane, then |E|^2 on the same grid
    from jno._rcwa import fields as jf

    nx = 16
    thicknesses = [1.0, float(spec["layers"][0][0]), 1.0]  # sup / film / sub
    roles = ["sup", 0, "sub"]
    chunk_counts = st._chunks
    # 3 film planes (z=0, mid, full) + 2 each in sup/sub -> 7 planes
    planes = jf.field_planes(thicknesses, [2, 3, 2], roles, chunk_counts)
    grids = [jnp.asarray([[spec["layers"][0][1]]])]  # uniform [[4.0]]
    ex, ey, ez, _, _, _ = jf.interior_field_amplitudes(
        st,
        planes,
        grids,
        st.wavelength,
        jnp.asarray(st.k_in),
        (1.0, 0.0),
        st.amb_sup,
        st.amb_sub,
    )
    inten, _, _ = jf.volume_intensity(ex, ey, ez, st.expansion, jnp.asarray(st.k_in), st.period, nx)
    # planes order: sup(2), film(z=0, mid, full) 3, sub(2)  => film mid = global index 3
    mid_inten = float(np.asarray(inten[..., 3]).mean())
    assert mid_inten == pytest.approx(e2_ref, rel=1e-4), (mid_inten, e2_ref)


def test_emitter_matches_golden():
    """The jump-condition emission solve reproduces the frozen fmmax up/down powers."""
    name = "emitter"
    spec = _cases.CASES[name]
    st = _build_stack(spec)
    from jno._rcwa import sources as jno_sources

    up, down = jno_sources.emission_solve(st, dict(spec["source"]))
    tol = spec["rel_tol"]
    assert float(up) == pytest.approx(float(_ref(name, "up_ref")), rel=tol), name
    assert float(down) == pytest.approx(float(_ref(name, "down_ref")), rel=tol), name


def test_manifest_sha256_matches_committed():
    """The committed .npz hashes match the manifest — a tampered golden is caught."""
    import hashlib
    import json

    with open(os.path.join(HERE, "golden", "manifest.json")) as fh:
        manifest = json.load(fh)
    for name, entry in manifest.items():
        path = os.path.join(HERE, "golden", f"{name}.npz")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == entry["sha256"], f"sha256 drift in {name}.npz"
