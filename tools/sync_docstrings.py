"""Copy the docstring of a forwarded-to object into the source of its proxy.

A proxy, alias or thin wrapper should show the same help as the object it forwards to, and editor
hovers (pyright, ty, jedi) read the docstring from the SOURCE, so ``functools.wraps`` or an assigned
``__doc__`` does not reach them. Mark the proxy with a comment naming its target, directly above the
``def`` / ``class`` (or above its decorators):

    # docstring-from: jno.trace.Model.initialize
    def initialize(self, weights, *, key=None):
        return self.model.initialize(weights, key=key)

and run ``python tools/sync_docstrings.py``: the target's docstring is written into the proxy, so the
target stays the one source of truth. ``--check`` changes nothing and exits 1 when a proxy is out of
date (``tests/test_public_docstrings.py`` runs it).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path

MARKER = re.compile(r"^\s*#\s*docstring-from:\s*([\w.]+)\s*$")
ROOT = Path(__file__).resolve().parents[1]


def _module_file(dotted: str) -> tuple[Path, list[str]]:
    """Split ``dotted`` into the source file of its longest importable module prefix and the rest."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            spec = importlib.util.find_spec(".".join(parts[:i]))
        except (ImportError, ValueError):
            continue
        if spec is not None and spec.origin and spec.origin.endswith(".py"):
            return Path(spec.origin), parts[i:]
    raise ImportError(f"docstring-from: no module found for {dotted}")


def _definition(tree: ast.AST, path: list[str]):
    """The def/class node at ``path`` (``["Class", "method"]``) inside a module tree; a property's
    getter, never its setter; never an ``@overload`` stub."""
    node = tree
    for name in path:
        found = None
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or child.name != name:
                continue
            decorators = {getattr(d, "attr", getattr(d, "id", None)) for d in child.decorator_list}
            if "overload" in decorators or "setter" in decorators:
                continue
            found = child
            break
        if found is None:
            raise LookupError(f"docstring-from: {name!r} not found")
        node = found
    return node


def target_doc(dotted: str) -> str:
    """The cleaned docstring of the object ``dotted`` names, read from its SOURCE.

    Reading the source rather than the imported object means a proxy of a proxy picks up the text
    the first pass just wrote, and nothing about the target has to be importable but its module.
    """
    path, rest = _module_file(dotted)
    node = _definition(ast.parse(path.read_text()), rest)
    doc = ast.get_docstring(node, clean=False)
    if not doc or not doc.strip():
        raise ValueError(f"docstring-from: {dotted} has no docstring to copy")
    return inspect.cleandoc(doc)


def render(doc: str, indent: str) -> list[str]:
    """``doc`` as docstring source lines at ``indent``, raw when it holds a backslash."""
    quote = 'r"""' if "\\" in doc else '"""'
    lines = doc.splitlines()
    if len(lines) == 1:
        return [f'{indent}{quote}{lines[0]}"""\n']
    body = [f"{indent}{line}\n" if line.strip() else "\n" for line in lines[1:]]
    return [f"{indent}{quote}{lines[0]}\n", *body, f'{indent}"""\n']


def marked_defs(tree: ast.AST, lines: list[str]):
    """Yield ``(node, target)`` for every def/class whose marker comment sits right above it."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        m = MARKER.match(lines[first - 2]) if first >= 2 else None
        if m:
            yield node, m.group(1)


def sync_file(path: Path, write: bool = True) -> list[str]:
    """Rewrite the marked docstrings in ``path`` (unless ``write`` is false); return the out-of-date ones."""
    text = path.read_text()
    if "docstring-from:" not in text:
        return []
    lines = text.splitlines(keepends=True)
    edits, stale = [], []
    for node, dotted in marked_defs(ast.parse(text), lines):
        first_stmt = node.body[0]
        indent = " " * first_stmt.col_offset
        new = render(target_doc(dotted), indent)
        has_doc = isinstance(first_stmt, ast.Expr) and isinstance(getattr(first_stmt, "value", None), ast.Constant)
        has_doc = has_doc and isinstance(first_stmt.value.value, str)
        start = first_stmt.lineno - 1
        end = first_stmt.end_lineno if has_doc else start
        if not has_doc:  # a new docstring goes right under the signature, above the body's leading comments
            start = min([first_stmt.lineno] + [d.lineno for d in getattr(first_stmt, "decorator_list", [])]) - 1
            while start > node.lineno and lines[start - 1].strip().startswith("#"):
                start -= 1
            end = start
        if lines[start:end] != new:
            edits.append((start, end, new))
            stale.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name} <- {dotted}")
    for start, end, new in sorted(edits, reverse=True):
        lines[start:end] = new
    if edits and write:
        path.write_text("".join(lines))
    return stale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report out-of-date proxies, change nothing")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    files = sorted((ROOT / "jno").rglob("*.py"))
    if args.check:
        stale = [s for f in files for s in sync_file(f, write=False)]
        if stale:
            print("out-of-date proxy docstrings (run python tools/sync_docstrings.py):")
            print("\n".join(f"  {s}" for s in stale))
        return 1 if stale else 0
    stale: list[str] = []
    for _ in range(3):  # a proxy of a proxy settles on the second pass
        round_stale = [s for f in files for s in sync_file(f)]
        if not round_stale:
            break
        stale += round_stale
    print(f"updated {len(dict.fromkeys(stale))} proxy docstring(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
