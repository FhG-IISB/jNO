"""Smoke tests for tutorial example scripts.

These tests execute scripts under tests/tutorial_examples_tests as subprocesses and
fail if a script exits non-zero or exceeds the configured timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_ROOT = REPO_ROOT / "tests" / "tutorial_examples_tests"
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("JNO_TUTORIAL_SMOKE_TIMEOUT", "120"))
PATTERN = os.environ.get("JNO_TUTORIAL_SMOKE_PATTERN", "**/*.py")


def _discover_scripts() -> list[Path]:
    scripts = sorted(TUTORIAL_ROOT.glob(PATTERN))
    return [p for p in scripts if p.name != "__init__.py"]


SCRIPTS = _discover_scripts()
SCRIPT_IDS = [str(p.relative_to(REPO_ROOT)) for p in SCRIPTS]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("script_path", SCRIPTS, ids=SCRIPT_IDS)
def test_tutorial_script_smoke(script_path: Path):
    env = os.environ.copy()

    # Keep bytecode/cache writes local and avoid polluting user environment.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    cmd = [sys.executable, str(script_path)]
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_tail = (exc.stdout or "")[-3000:]
        stderr_tail = (exc.stderr or "")[-3000:]
        pytest.fail(f"Timeout after {DEFAULT_TIMEOUT_SECONDS}s for {script_path.relative_to(REPO_ROOT)}\n" f"--- stdout (tail) ---\n{stdout_tail}\n" f"--- stderr (tail) ---\n{stderr_tail}")

    if result.returncode != 0:
        stdout_tail = (result.stdout or "")[-4000:]
        stderr_tail = (result.stderr or "")[-4000:]
        pytest.fail(f"Script failed: {script_path.relative_to(REPO_ROOT)}\n" f"Command: {' '.join(cmd)}\n" f"Exit code: {result.returncode}\n" f"--- stdout (tail) ---\n{stdout_tail}\n" f"--- stderr (tail) ---\n{stderr_tail}")
