"""Guards against the docs' code rotting away from the library.

Two of them, and they catch different things:

* :func:`test_fem_index_intro_snippet_runs` EXECUTES the self-contained intro on the FEM landing
  page. (The per-tutorial *full scripts* are already run by the subprocess tutorial tests via their
  ``--8<--`` includes; the inline excerpts elsewhere are fragments that reference setup variables and
  are intentionally not executed.)
* :func:`test_every_jno_reference_in_the_docs_resolves` cannot execute a fragment, but it can still
  check the thing that actually rots: whether the ``jno.*`` names a snippet uses still EXIST. It
  parses every Python block in ``docs/`` and resolves each dotted chain rooted at ``jno``.

The second one was written after an audit found three dead references sitting in the published docs:
``jno.core.load`` (the module-level function is ``jno.load``), ``jno.diff(u, x, order=2)`` (no such
function -- it is ``u.dd(x)``), and ``jno.domain.poseidon`` (a ``Geometries`` staticmethod that the
domain class does not re-export -- jNO's own error message pointed at the same non-existent
spelling). None of them could be caught by executing anything, because none of those blocks is
runnable on its own.
"""

import ast
import re
from pathlib import Path

import jax
import pytest

import jno

DOCS = Path(__file__).parent.parent / "docs"


def _python_blocks(md_path: Path):
    """Every ```python block, minus the ``--8<--`` transclusions (those are files, not source)."""
    for block in re.findall(r"```python\n(.*?)```", md_path.read_text(errors="ignore"), re.DOTALL):
        if "--8<--" not in block:
            yield block


def _attribute_chains(src: str):
    """Dotted attribute chains rooted at a bare name: ``jno.solve.gmres`` -> ``('jno','solve','gmres')``."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return  # a deliberately partial fragment; the other guard covers runnable code
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts, cur = [], node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            yield (cur.id, *reversed(parts))


def test_fem_index_intro_snippet_runs():
    """The FEM landing page's intro is a complete, self-contained ``jno.fem`` solve; it must execute.

    The headline API hands back ready-to-use arrays (``fem.A`` dense, ``fem.b`` flat) for
    ``jnp.linalg.solve``; this guards that intro and the flat-accessor contract. The path is
    ``fem/index.md`` since the guide was split -- this test pointed at the old ``fem.md`` and had
    been failing with ``FileNotFoundError`` ever since.
    """
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # FEM assembly is float64
    try:
        block = next(_python_blocks(DOCS / "fem" / "index.md"), None)
        assert block is not None, "no ```python block on the FEM landing page"
        ns: dict = {}
        exec(compile(block, "<fem/index.md intro>", "exec"), ns)
        u_h = ns["u_h"]
        assert u_h.ndim == 1 and u_h.shape[0] > 0  # solved a real linear system
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_every_jno_reference_in_the_docs_resolves():
    """No snippet may name a ``jno.*`` attribute that does not exist.

    Deliberately structural rather than executable: most inline snippets are fragments that cannot
    run on their own, but a name that has been renamed or removed is still a broken promise to the
    reader -- and it is exactly what a refactor leaves behind.
    """
    missing: dict[str, set[str]] = {}
    checked = 0
    for md in sorted(DOCS.rglob("*.md")):
        for block in _python_blocks(md):
            for chain in _attribute_chains(block):
                if chain[0] != "jno":
                    continue
                checked += 1
                obj = jno
                for part in chain[1:]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        missing.setdefault(".".join(chain), set()).add(md.relative_to(DOCS).as_posix())
                        break
    assert checked > 200, f"the walker found only {checked} jno.* references — it has stopped working"
    assert not missing, "docs reference jno attributes that do not exist: " + "; ".join(
        f"{name} ({', '.join(sorted(where))})" for name, where in sorted(missing.items())
    )
