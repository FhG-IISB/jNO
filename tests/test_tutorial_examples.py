"""Subprocess-based runner for every tutorial under ``docs/tutorial_examples/``.

Each tutorial script is the **single source of truth** for both the user-facing
documentation (rendered through ``--8<--`` includes in mkdocs-material) and for
CI. There are no parallel "smoke-test" duplicates: the same minimal script that
ships in the docs runs unmodified here.

Each tutorial ends with at least one tolerance ``assert``. A failing assert
exits the script with a non-zero return code, which fails the pytest case.

Marked ``@pytest.mark.slow`` so a ``pytest -m "not slow"`` run skips the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TUT_DIR = ROOT / "docs" / "tutorial_examples"
SCRIPTS = sorted(p for p in TUT_DIR.rglob("*.py") if not p.name.startswith("_"))

# Default per-tutorial timeout (seconds). Tutorials should be quick by design;
# anything that exceeds this is a problem to fix at the tutorial level, not by
# bumping the bound here.
_TIMEOUT = 180


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(TUT_DIR)))
def test_tutorial(script: Path) -> None:
    """Run ``script`` end-to-end in a subprocess and require exit code 0."""
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if result.returncode != 0:
        pytest.fail(
            f"Tutorial {script.relative_to(TUT_DIR)} failed (exit {result.returncode})\n"
            f"--- stderr (last 4 KiB) ---\n{result.stderr[-4000:]}\n"
            f"--- stdout (last 2 KiB) ---\n{result.stdout[-2000:]}\n"
        )
