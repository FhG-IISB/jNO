"""Test: 05 — HyCo Poisson 1D

Runs the hyco_poisson_1d tutorial script as a smoke test.
Asserts:
  - Physical model relative L2 error < 5%
  - Synthetic model relative L2 error < 10%
"""

import subprocess
import sys
from pathlib import Path


def test_hyco_poisson_1d():
    script = Path(__file__).parent.parent.parent.parent / "docs/tutorial_examples/05_coupled_and_inverse/hyco_poisson_1d.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"hyco_poisson_1d.py failed:\n{result.stdout}\n{result.stderr}"
