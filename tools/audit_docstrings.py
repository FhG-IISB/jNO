"""Audit the docstrings of jNO's public API.

Walks every public module, function, class, method and property reachable from ``jno`` (plus every public
class defined in a public module, since many are met only as return values) and reports, per object:

* ``missing``   -- no docstring at all;
* ``pointer``   -- a short docstring that only points elsewhere ("Proxy for ...", "see ...");
* ``params``    -- a parameter of the signature that the docstring never names;
* ``indent``    -- prose indented past its section, which renders badly in editor hovers;
* ``numpy``     -- a NumPy-style ``Parameters``/``----`` section (the package is Google style);
* ``generated`` -- made by a factory at import time, so editors (which read the source) show no docstring.

Usage::

    python tools/audit_docstrings.py                     # summary
    python tools/audit_docstrings.py --show jno/fn.py    # every finding in the named files
    python tools/audit_docstrings.py --csv out.csv       # the full table

``tests/test_public_docstrings.py`` runs :func:`collect` and fails on any finding.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import inspect
import os
import pkgutil
import re
import textwrap
import types
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jno  # noqa: E402

ROOT = os.path.dirname(jno.__file__)
POINTER = re.compile(
    r"\b(proxy|alias|see\b|same as|forward|shortcut|delegat|equivalent to|thin wrapper|wrapper (for|around))", re.I
)
NUMPY_SECTION = re.compile(r"^\s*(Parameters|Returns|Raises)\s*\n\s*-{3,}", re.M)
KEYWORD_ARGS = re.compile(r"^\s*Keyword Arg(ument)?s:", re.M)
#: private by name, but the home of public API (``jno.fem`` and ``jno.FEM`` live here)
PUBLIC_PRIVATE_MODULES = {"_fem.py"}


def _source(obj):
    """``(file, first line)`` of ``obj``'s source, or ``(None, None)``."""
    try:
        return inspect.getsourcefile(obj), inspect.getsourcelines(obj)[1]
    except (TypeError, OSError):
        return None, None


def _ours(obj) -> bool:
    """Whether ``obj`` is defined in a public jNO module."""
    path, _ = _source(obj)
    if path is None or not os.path.abspath(path).startswith(ROOT):
        return False
    parts = os.path.relpath(path, ROOT).split(os.sep)
    return not any(p.startswith("_") and p != "__init__.py" and p not in PUBLIC_PRIVATE_MODULES for p in parts)


def _odd_indent(doc: str) -> bool:
    """Prose indented past its section (a paragraph under nothing, or 12+ spaces outside a code block)."""
    lines = inspect.cleandoc(doc).splitlines()
    in_code, section = False, None
    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        indent = len(line) - len(line.lstrip())
        if text.startswith((".. code", "```", ">>>")) or text.endswith("::"):
            in_code = True
            continue
        if indent == 0:
            in_code = False
            section = text.rstrip(":") if text.endswith(":") else None
            continue
        if in_code:
            continue
        if section is None and indent >= 4 and not text.startswith(("-", "*", "|")) and not lines[i - 1].strip():
            return True
        if indent >= 12:
            return True
    return False


def _missing_params(fn, doc: str) -> list[str]:
    """Parameters of ``fn`` (public, not ``self``/``cls``) that ``doc`` never names."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    missing = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls") or name.startswith("_"):
            continue
        if p.kind == p.VAR_KEYWORD and KEYWORD_ARGS.search(doc):
            continue
        if not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", doc):
            star = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "")
            missing.append(star + name)
    return missing


def _doc_in_source(obj) -> bool:
    """Whether ``obj``'s docstring is written in its own source (what an editor reads)."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0]
    except (TypeError, OSError, SyntaxError, IndexError):
        return False
    return bool(ast.get_docstring(node))


class _Audit:
    def __init__(self):
        self.rows: list[dict] = []
        self._seen: set[int] = set()

    def record(self, qual, obj, kind, sig_fn=None, extra_doc=""):
        """Classify one object's docstring and append a row."""
        holder = obj.fget if kind == "property" else obj
        doc = getattr(holder, "__doc__", None)  # its own text: inspect.getdoc would inherit a base's
        path, line = _source(sig_fn or holder)
        status = []
        if "<locals>" in getattr(sig_fn or holder, "__qualname__", "") and not _doc_in_source(holder):
            status.append("generated")  # built by a factory, its __doc__ assigned at runtime
        if not doc or not doc.strip():
            status.append("missing")
        else:
            if len(doc.strip()) < 200 and POINTER.search(doc):
                status.append("pointer")
            if _odd_indent(doc):
                status.append("indent")
            if NUMPY_SECTION.search(doc):
                status.append("numpy")
        # a class whose constructor is not jNO's own (inherited, a Protocol placeholder) has no parameters of its own
        checked = kind in ("function", "method") or (kind == "class" and sig_fn is not None)
        missing = _missing_params(sig_fn or holder, (doc or "") + extra_doc) if checked else []
        if missing and doc:
            status.append("params")
        self.rows.append(
            dict(
                name=qual,
                kind=kind,
                file=os.path.relpath(path, ROOT) if path else "",
                line=line or 0,
                status=",".join(status) or "ok",
                missing_params=" ".join(missing),
            )
        )

    def walk_class(self, qual, cls):
        if id(cls) in self._seen:
            return
        self._seen.add(id(cls))
        init = cls.__dict__.get("__init__")
        generated = getattr(init, "__qualname__", "").startswith(cls.__qualname__ + ".")  # e.g. a dataclass's
        if inspect.isfunction(init) and not _ours(init) and not generated:
            init = None  # e.g. the placeholder ``typing.Protocol`` installs
        new = cls.__dict__.get("__new__")
        new = getattr(new, "__func__", new)  # stored as a staticmethod
        if not inspect.isfunction(init) and inspect.isfunction(new):
            init = new  # a class constructed through __new__ (``jno.nn``)
        ctor = init if inspect.isfunction(init) else None
        self.record(qual, cls, "class", sig_fn=ctor, extra_doc=(ctor.__doc__ or "") if ctor else "")
        for name, member in cls.__dict__.items():
            if name.startswith("_"):
                continue
            q = f"{qual}.{name}"
            if isinstance(member, property):
                self.record(q, member, "property", sig_fn=member.fget)
            elif isinstance(member, (staticmethod, classmethod)):
                self.record(q, member.__func__, "method")
            elif inspect.isfunction(member):
                self.record(q, member, "method")
            elif inspect.isclass(member) and _ours(member):
                self.walk_class(q, member)

    def walk(self, qual, obj, depth=0):
        if inspect.ismodule(obj):
            if id(obj) in self._seen or not obj.__name__.startswith("jno") or depth > 4:
                return
            self._seen.add(id(obj))
            names = getattr(obj, "__all__", None) or [n for n in dir(obj) if not n.startswith("_")]
            for name in names:
                if name.startswith("_"):
                    continue
                try:
                    attr = getattr(obj, name)
                except Exception:  # noqa: BLE001 -- a lazy attribute may fail to import an optional backend
                    continue
                if inspect.ismodule(attr) and not attr.__name__.startswith("jno"):
                    continue
                self.walk(f"{qual}.{name}", attr, depth + 1)
        elif inspect.isclass(obj):
            if _ours(obj):
                self.walk_class(qual, obj)
        elif inspect.isfunction(obj) or isinstance(obj, types.BuiltinFunctionType):
            if id(obj) not in self._seen and _ours(obj):
                self._seen.add(id(obj))
                self.record(qual, obj, "function")
        elif callable(obj) and _ours(type(obj)):
            self.walk_class(f"{qual} (instance type {type(obj).__name__})", type(obj))


def collect() -> list[dict]:
    """Every public object's row: ``name``, ``kind``, ``file``, ``line``, ``status``, ``missing_params``."""
    audit = _Audit()
    audit.walk("jno", jno)
    # classes users meet only as return values: every public class defined in a public module
    for info in pkgutil.walk_packages(jno.__path__, "jno."):
        if any(p.startswith("_") for p in info.name.split(".")[1:]) and info.name != "jno._fem":
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:  # noqa: BLE001 -- an optional backend that is not installed
            continue
        for name, cls in vars(module).items():
            if inspect.isclass(cls) and not name.startswith("_") and cls.__module__ == module.__name__ and _ours(cls):
                audit.walk_class(f"{module.__name__}.{name}", cls)
    return audit.rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", help="write the full table to this file")
    ap.add_argument("--show", nargs="*", default=None, help="list the findings (in the named files only, if given)")
    args = ap.parse_args(argv)
    rows = collect()
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.show is not None:
        want = {f[4:] if f.startswith("jno/") else f for f in args.show}
        for r in sorted(rows, key=lambda r: (r["file"], int(r["line"]))):
            if r["status"] != "ok" and (not want or r["file"] in want):
                extra = f"  undocumented: {r['missing_params']}" if "params" in r["status"] else ""
                print(f"jno/{r['file']}:{r['line']}  {r['name']}  [{r['status']}]{extra}")
    counts = Counter(s for r in rows for s in r["status"].split(","))
    print(len(rows), "public objects;", dict(counts))
    return 0 if set(counts) <= {"ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
