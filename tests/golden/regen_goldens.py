#!/usr/bin/env python
"""Regenerate the RCWA golden files with jNO's own backend.

For every case in ``tests/golden/_cases.py`` this freezes the outputs of the current solver into
``tests/golden/<name>.npz`` (T, R, per-order efficiencies; the slab's mid-film |E|^2; the emitter's up/down powers;
for ``disk_40`` two design gradients) and writes ``manifest.json`` with each file's sha256 and the git commit it was
made from. The goldens guard against regressions only. Correctness is checked elsewhere: against analytic results
(``tests/test_rcwa_analytic.py``) and against independent libraries (``tests/test_rcwa_reference.py``).

Regenerate only on purpose, after a change that is meant to move the numbers, and say why in the commit:

    jnorun python tests/golden/regen_goldens.py          # every case
    jnorun python tests/golden/regen_goldens.py disk_40  # one case
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tests.golden import _cases, _jno  # noqa: E402

GRADIENT_CASES = ("disk_40",)


def regen_one(name: str) -> None:
    spec = _cases.CASES[name]
    if "source" in spec:
        out = _jno.emitter(spec)
    else:
        out = _jno.plane_wave(spec)
        if name == "slab_airy":
            out["e2"] = _jno.slab_e2(spec)
        if name in GRADIENT_CASES:
            out.update(_jno.gradients(spec))
    np.savez(os.path.join(HERE, f"{name}.npz"), **out)
    print(f"{name}: " + ", ".join(f"{k}={v:.12g}" for k, v in out.items() if np.ndim(v) == 0), flush=True)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="cases to regenerate (default: all)")
    names = parser.parse_args().names or list(_cases.CASES)
    for name in names:
        regen_one(name)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    manifest = {
        name: {
            "description": spec["description"],
            "dtype": spec["dtype"],
            "sha256": _sha256(os.path.join(HERE, f"{name}.npz")),
            "made_at_commit": commit,
        }
        for name, spec in _cases.CASES.items()
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
