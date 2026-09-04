"""``jax_enable_x64`` is process-wide, so nothing may set it where the setting outlives one test.

The flag decides the float width of every subsequent computation. Set at module scope it fires
during COLLECTION -- before a single test runs, for every module pytest imported -- and there is no
scope for it to be restored at. The result is a suite whose outcome depends on which files were
selected and in what order.

That is not hypothetical. Before this was fixed::

    pytest tests/test_fdm.py                          ->  41 passed
    pytest tests/test_node_eval.py tests/test_fdm.py  ->  16 failed, 33 passed

One unrelated 8-test file in front, and `newton_krylov` in test_fdm stopped converging, because its
solves had dropped to float32. Cross-file effects of exactly this shape were repeatedly mistaken for
GPU contention and for real regressions, and cost hours of re-running to rule out.

In-test mutations are caught at runtime by the ``_restore_x64`` fixture in conftest. Module-scope
ones cannot be, which is what this test is for.
"""

import ast
import pathlib

import pytest

TESTS = pathlib.Path(__file__).parent
# Not collected as a test module -- it is executed as a subprocess, where module scope is the only
# scope there is and nothing can leak out of the process.
EXEMPT = {"_sharding_inner.py"}


def _module_scope_x64_writes(path: pathlib.Path) -> list[int]:
    """Line numbers where this module sets ``jax_enable_x64`` at module scope."""
    tree = ast.parse(path.read_text())
    hits = []
    for node in tree.body:  # top level ONLY: a write inside a def/class is scoped and fine
        for sub in ast.walk(node) if isinstance(node, (ast.Expr, ast.If, ast.Try, ast.With)) else []:
            if not isinstance(sub, ast.Call):
                continue
            if not (isinstance(sub.func, ast.Attribute) and sub.func.attr == "update"):
                continue
            if sub.args and isinstance(sub.args[0], ast.Constant) and sub.args[0].value == "jax_enable_x64":
                hits.append(sub.lineno)
    return hits


@pytest.mark.parametrize("path", sorted(p for p in TESTS.glob("*.py") if p.name not in EXEMPT), ids=lambda p: p.name)
def test_no_module_scope_x64_mutation(path):
    """Set it in an autouse fixture that restores it, the way ``tests/test_fem_1d.py`` does."""
    lines = _module_scope_x64_writes(path)
    assert not lines, (
        f"{path.name} sets jax_enable_x64 at module scope (line{'s' if len(lines) > 1 else ''} "
        f"{', '.join(map(str, lines))}). That runs at import, for every module in the selection, and "
        f"cannot be undone -- put it in an autouse fixture that saves and restores the previous value."
    )
