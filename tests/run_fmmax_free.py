#!/usr/bin/env python
"""Run a pytest target with fmmax forcibly ABSENT.

Blocks any fmmax import (sets ``sys.modules['fmmax'] = None``) before pytest
starts, then runs ``pytest.main(argv)``. Use to prove the rcwa suite is green
without the oracle dependency — golden tests must pass, oracle tests must skip.

Usage:
    python tests/run_fmmax_free.py tests/test_rcwa_golden.py -q
    python tests/run_fmmax_free.py tests/test_rcwa.py -q      # oracle tests skip
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Block fmmax so the non-test code path and the golden tests are exercised with
# no oracle installed. A module blocked this way still `import fmmax`? No: any
# ``import fmmax`` or ``from fmmax import ...`` raises ImportError because the
# sys.modules entry is None (Python treats a None entry as "not importable").
sys.modules["fmmax"] = None

import pytest  # noqa: E402

argv = sys.argv[1:] or ["tests/test_rcwa_golden.py", "-q"]
raise SystemExit(pytest.main(argv))
