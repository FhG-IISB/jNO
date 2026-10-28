"""Every public object carries its full docstring, in the source -- what an editor hover shows.

Two guards, both over the whole package:

* ``tools/audit_docstrings.py`` -- no public function, class, method or property is undocumented, only
  points elsewhere, leaves a parameter undescribed, is NumPy-style, or is built by a factory whose
  docstring a static hover cannot see;
* ``tools/sync_docstrings.py --check`` -- every proxy marked ``# docstring-from: <target>`` carries its
  target's current docstring, so a wrapper never shows a stale or partial copy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_public_object_is_documented():
    rows = _tool("audit_docstrings").collect()
    bad = [r for r in rows if r["status"] != "ok"]
    report = "\n".join(
        f"  jno/{r['file']}:{r['line']}  {r['name']}  [{r['status']}]"
        + (f"  undocumented: {r['missing_params']}" if r["missing_params"] else "")
        for r in bad
    )
    assert len(rows) > 1000, f"the audit found only {len(rows)} public objects; is it walking the package?"
    assert not bad, f"{len(bad)} public object(s) need docstring work (python tools/audit_docstrings.py --show):\n{report}"


def test_proxy_docstrings_match_their_targets():
    sync = _tool("sync_docstrings")
    stale = [s for f in sorted((sync.ROOT / "jno").rglob("*.py")) for s in sync.sync_file(f, write=False)]
    assert not stale, "out-of-date proxy docstrings (run python tools/sync_docstrings.py):\n" + "\n".join(stale)
