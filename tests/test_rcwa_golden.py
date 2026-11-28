"""Regression goldens: the backend reproduces its own frozen outputs.

``tests/golden/regen_goldens.py`` froze, for every case in ``tests/golden/_cases.py``, what jNO's solver computed
(T, R, per-order efficiencies, the slab's mid-film |E|^2, an emitter's powers, two design gradients). A change that
moves any of them fails here; if the move is intended, regenerate and say why in the commit. Correctness itself is
tested against analytic results (``test_rcwa_analytic.py``) and independent libraries (``test_rcwa_reference.py``).
"""

from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.golden import _cases, _jno  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
#: Same code, same precision: only BLAS/LAPACK rounding can differ between machines.
RTOL, ATOL = 1e-9, 1e-11


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _golden(name):
    return dict(np.load(os.path.join(HERE, f"{name}.npz")))


PLANE_WAVE = [name for name, spec in _cases.CASES.items() if "source" not in spec]


@pytest.mark.parametrize("name", PLANE_WAVE)
def test_plane_wave_matches_golden(name):
    ref, out = _golden(name), _jno.plane_wave(_cases.CASES[name])
    np.testing.assert_array_equal(out["orders"], ref["orders"])
    for key in ("T", "R", "T_orders", "R_orders"):
        np.testing.assert_allclose(out[key], ref[key], rtol=RTOL, atol=ATOL, err_msg=f"{name}.{key}")


def test_slab_e2_matches_golden():
    spec = _cases.CASES["slab_airy"]
    assert _jno.slab_e2(spec) == pytest.approx(float(_golden("slab_airy")["e2"]), rel=RTOL)


def test_emitter_matches_golden():
    ref, out = _golden("emitter"), _jno.emitter(_cases.CASES["emitter"])
    for key in ("up", "down"):
        assert out[key] == pytest.approx(float(ref[key]), rel=RTOL), key


def test_design_gradients_match_golden():
    ref, out = _golden("disk_40"), _jno.gradients(_cases.CASES["disk_40"])
    for key in ("dT_dcontrast", "dT_dthickness"):
        assert out[key] == pytest.approx(float(ref[key]), rel=1e-8), key


def test_manifest_sha256_matches_committed():
    """The committed .npz hashes match the manifest: an edited golden is caught."""
    with open(os.path.join(HERE, "manifest.json")) as fh:
        manifest = json.load(fh)
    assert set(manifest) == set(_cases.CASES)
    for name, entry in manifest.items():
        with open(os.path.join(HERE, f"{name}.npz"), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == entry["sha256"], f"sha256 drift in {name}.npz"
