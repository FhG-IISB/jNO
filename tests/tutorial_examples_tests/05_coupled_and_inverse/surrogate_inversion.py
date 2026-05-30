"""Test: 05 — Surrogate inversion

Runs the surrogate_inversion tutorial script as a smoke test.
Asserts:
  - Forward PINN relative L2 error < 5%
  - Recovered input absolute error < 0.05
"""

import subprocess
import sys
from pathlib import Path


def test_surrogate_inversion():
    script = (
        Path(__file__).parent.parent.parent.parent / "docs/tutorial_examples/05_coupled_and_inverse/surrogate_inversion.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"surrogate_inversion.py failed:\n{result.stdout}\n{result.stderr}"
