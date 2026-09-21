"""``jno.info(obj)`` — one front door for "what is this, and is it what I meant?".

Every jNO object already knows a great deal about itself, and almost none of it was reachable
without knowing the attribute name to ask for. What *was* reachable lived behind three different
spellings on three different classes (``domain.summary()``, ``core.print_tree()``,
``core.print_shapes()``), so there was no single thing to reach for and no way to learn there was
anything to reach for at all.

``info`` dispatches on what it is handed and returns a printable :class:`Info`::

    print(jno.info(d))                       # a domain: mesh, quality, tags, attachments
    print(jno.info(fem))                     # a form: mode, blocks, classification
    print(jno.info(fem, deep=True))          # + operator sparsity, symmetry, empty rows
    print(jno.info(jno.solve.newton(direct=True)))   # what that spec actually does

Two rules it keeps:

* **Cheap by default.** Anything that costs an assembly or a factorisation sits behind
  ``deep=True``; a default ``info`` is reads and arithmetic on what is already built.
* **Never force a lazy result.** A transient solve is a trace node on purpose. ``info`` reports
  that it is one rather than evaluating it, exactly as the post-solve log line does.

``Info.as_dict()`` returns the same content as data, so a test can assert on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Info", "info"]

_RULE = "─"


class Info:
    """A printable report. ``str(...)`` renders it; :meth:`as_dict` returns the same content."""

    def __init__(self, title: str, sections: list[tuple[str, list[tuple[str, Any]]]]):
        self.title = title
        self.sections = sections

    def as_dict(self) -> dict:
        return {"title": self.title, **{s: dict(rows) for s, rows in self.sections}}

    def __str__(self) -> str:
        width = 78
        out = [f"{_RULE*3} {self.title} ".ljust(width, _RULE)]
        for name, rows in self.sections:
            if not rows:
                continue
            if name:
                out.append(f"  {name}")
            pad = max((len(str(k)) for k, _ in rows), default=0)
            for k, v in rows:
                out.append(f"    {str(k).ljust(pad)}  {v}")
        return "\n".join(out)

    __repr__ = __str__


def _fmt_n(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:  # noqa: BLE001
        return str(x)


def _bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------------------------
# domain
# ---------------------------------------------------------------------------------------------
def _info_domain(d, deep: bool) -> Info:
    from .domain.mesh_utils import _mesh_quality

    geo: list = [("dimension", f"{d.dimension}D ({', '.join(getattr(d, 'spatial', []) or [])})")]
    if getattr(d, "_is_time_dependent", False) and getattr(d, "time", None) is not None:
        t0, t1, nt = d.time
        geo.append(("time", f"[{t0}, {t1}] · {nt} steps"))
    geo.append(("samples", _fmt_n(getattr(d, "total_samples", "?"))))

    mesh: list = []
    built = d.__dict__.get("_mesh") is not None
    mesh.append(("built", "yes" if built else "no — still a lazy shape plan"))
    if built:
        m = d.mesh
        pts = np.asarray(m.points)
        mesh.append(("points", _fmt_n(len(pts))))
        for ctype, cells in (m.cells_dict or {}).items():
            q = _mesh_quality(pts, np.asarray(cells), str(ctype)).lstrip(" ·").strip()
            mesh.append((ctype, f"{_fmt_n(len(cells))}" + (f"  ·  {q}" if q else "")))
        # meshio always stores 3 columns; a 2-D mesh's third is identically zero and reporting
        # `[0, 0]` as an extent invites the reader to wonder what it means.
        dim = int(d.dimension)
        lo, hi = pts[:, :dim].min(axis=0), pts[:, :dim].max(axis=0)
        mesh.append(("extent", " × ".join(f"[{a:.4g}, {b:.4g}]" for a, b in zip(lo, hi))))

    tags: list = []
    dim = int(d.dimension)
    breg = getattr(d, "_boundary_registry", None) or {}
    for tag, pts in (getattr(d, "_mesh_pool", None) or {}).items():
        row = f"{_fmt_n(np.asarray(pts).shape[0])} points"
        # Extents for a boundary tag -- the check that `lambda x, y: y > 1-1e-9` actually caught the
        # edge you meant. `domain.summary()` showed these and it was the most useful thing it did.
        bp = (breg.get(tag) or {}).get("points") if isinstance(breg.get(tag), dict) else None
        if bp is not None and len(bp):
            q = np.asarray(bp)[:, :dim]
            row += "   " + " × ".join(f"[{q[:, a].min():.4g}, {q[:, a].max():.4g}]" for a in range(dim))
        tags.append((tag, row))

    attached: list = []
    att = d.__dict__.get("_region_attachments") or {}
    kinds = d.__dict__.get("_attachment_kind") or {}
    default_key = getattr(type(d), "_DEFAULT_TARGET", "\x00default")
    for prop, values in att.items():
        bits = []
        for target, v in values.items():
            label = "(default)" if target == default_key else target
            bits.append(f"{label}={v!r}" if not hasattr(v, "shape") else f"{label}=<{type(v).__name__}>")
        kind = {kinds.get(t) for t in values if t != default_key} - {None}
        attached.append((f"d.{prop}", f"{', '.join(bits)}   [{'/'.join(sorted(kind)) or 'volume'}]"))

    return Info(
        f"domain · {d.dimension}D",
        [("geometry", geo), ("mesh", mesh), ("tags", tags), ("attached (d.<prop>)", attached)],
    )


# ---------------------------------------------------------------------------------------------
# fem
# ---------------------------------------------------------------------------------------------
def _info_fem(f, deep: bool) -> Info:
    form: list = [
        ("mode", f.mode + ("" if f.mode != "transient" else ("-linear" if f.is_linear else "-nonlinear"))),
        ("dofs", _fmt_n(f.dofs)),
    ]
    if getattr(f, "is_complex", False):
        form.append(("complex", "yes — solved as a real-equivalent 2n block"))
    if f.mode == "transient":
        form.append(("time window", f"[{getattr(f, 't0', '?')}, {getattr(f, 't1', '?')}]"))
    if getattr(f, "_periodic", None) is not None:
        form.append(("periodic", "yes — solved in the reduced space, then prolonged"))
    if getattr(f, "_saddle_blocks", None):
        form.append(("saddle blocks", ", ".join(map(str, f._saddle_blocks))))

    terms: list = [(f"[{i}]", c) for i, c in enumerate(f.classification or [])]

    blocks: list = []
    offs = list(getattr(f, "offsets", None) or [])
    keys = list(getattr(f, "_block_field_keys", None) or [])
    shapes = list(getattr(f, "_block_value_shapes", None) or [])
    dom = getattr(f, "domain", None)
    orders = list(getattr(dom, "_fem_native_field_orders", None) or [])
    for i in range(max(0, len(offs) - 1)):
        key = keys[i] if i < len(keys) else None
        name = f"field {key}" if key is not None else f"block {i}"
        vs = f" · value_shape {tuple(shapes[i])}" if i < len(shapes) and shapes[i] else ""
        od = f" · P{orders[i]}" if i < len(orders) else ""
        blocks.append((name, f"dofs {offs[i]}:{offs[i+1]}  ({_fmt_n(offs[i+1]-offs[i])}){vs}{od}"))

    op: list = []
    A, b = getattr(f, "_A", None), getattr(f, "_b", None)
    if A is not None:
        try:
            nnz = int(getattr(A, "nse", None) or np.asarray(A.data).size)
            n = int(f.dofs)
            op.append(("nnz", f"{_fmt_n(nnz)}  ·  fill {nnz / max(n * n, 1):.2e}  ·  ~{nnz / max(n,1):.1f}/row"))
            op.append(("dense equivalent", _bytes(n * n * 8)))
            dt = str(np.asarray(A.data).dtype)
            op.append(("dtype", dt + ("" if "64" in dt else "   ← float32: is jax_enable_x64 set?")))
        except Exception:  # noqa: BLE001
            pass
    if deep and A is not None:
        try:
            import jax.numpy as jnp

            dense = np.asarray(jnp.asarray(A.todense()))
            asym = float(np.abs(dense - dense.T).max())
            scale = float(np.abs(dense).max()) or 1.0
            op.append(("symmetry", f"max|A - Aᵀ| = {asym:.2e}  ({'symmetric' if asym / scale < 1e-12 else 'NON-symmetric'})"))
            empty = int((np.abs(dense).max(axis=1) == 0).sum())
            op.append(("empty rows", f"{empty}" + ("  ← singular" if empty else "")))
        except Exception as e:  # noqa: BLE001
            op.append(("deep", f"unavailable ({type(e).__name__})"))
    if b is not None:
        bb = np.asarray(b).reshape(-1)
        op.append(("load ‖b‖", f"{np.linalg.norm(bb):.4g}" + ("   ← ALL ZERO" if not np.any(bb) else "")))

    return Info(
        f"fem · {f.mode}",
        [("form", form), ("terms (as classified)", terms), ("field blocks", blocks), ("operator", op)],
    )


# ---------------------------------------------------------------------------------------------
# rcwa / fdm / core / solver specs
# ---------------------------------------------------------------------------------------------
def _info_rcwa(r, deep: bool) -> Info:
    setup: list = []
    for label, attr in (("orders", "orders"), ("period", "period"), ("wavelength", "wavelength")):
        v = getattr(r, attr, None)
        if v is not None:
            setup.append((label, str(v)))
    layers: list = []
    for i, lay in enumerate(getattr(r, "layers", None) or []):
        layers.append((f"[{i}]", str(lay)[:70]))
    result: list = []
    try:
        up, down = float(r.power("up")), float(r.power("down"))
        result.append(("power up / down", f"{up:.6f} / {down:.6f}"))
        result.append(("balance", f"{up + down:.6f}" + ("   ✓ energy conserved" if abs(up + down - 1) < 1e-6
                                                        else "   ← does NOT sum to 1 (absorbing, or wrong)")))
    except Exception:  # noqa: BLE001 - not solved yet, or a lossy stack with no such readout
        pass
    return Info("rcwa", [("setup", setup), ("layers", layers), ("result", result)])


def _info_fdm(o, deep: bool) -> Info:
    d = getattr(o, "domain", None)
    rows: list = [("route", "structured stencil + GMG" if getattr(d, "_structured_grid", None) else "cotangent / P1 operator")]
    for label, attr in (("dofs", "dofs"), ("mode", "mode")):
        v = getattr(o, attr, None)
        if v is not None:
            rows.append((label, str(v)))
    if d is not None and getattr(d, "time", None):
        rows.append(("time", str(d.time)))
    return Info("fdm", [("solver", rows)])


def _info_core(c, deep: bool) -> Info:
    import jax

    models: list = []
    for i, m in enumerate(getattr(c, "models", None) or []):
        try:
            n = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(m))
            models.append((f"[{i}] {type(m).__name__}", f"{_fmt_n(n)} parameters"))
        except Exception:  # noqa: BLE001
            models.append((f"[{i}]", type(m).__name__))
    cons: list = []
    try:
        for i, t in enumerate(c.get_constraint_tags(c.constraints)):
            cons.append((f"[{i}]", str(t)))
    except Exception:  # noqa: BLE001
        cons = [("count", _fmt_n(len(getattr(c, "constraints", []) or [])))]
    training: list = []
    for label, attr in (("optimizer", "_optimizer"), ("step", "_step"), ("best loss", "_best_loss")):
        v = getattr(c, attr, None)
        if v is not None:
            training.append((label, str(v)[:60]))
    sections = [("models", models), ("constraints", cons), ("training", training)]
    if deep:
        # What `core.print_tree()` and `core.print_shapes()` used to print. They were two more
        # spellings to know about; as text builders they become sections of the one report.
        for label, fn in (("computation tree", "_tree_text"), ("tensor shapes", "_shapes_text")):
            try:
                sections.append((label, [("", line) for line in str(getattr(c, fn)()).splitlines() if line.strip()]))
            except Exception as e:  # noqa: BLE001
                sections.append((label, [("", f"unavailable ({type(e).__name__}: {e})")]))
    return Info("core", sections)


def _info_spec(s, deep: bool) -> Info:
    name = getattr(s, "name", None) or type(s).__name__.lstrip("_")
    # Skip the closures: a spec carries the driver it built as `_fn`, and printing
    # `<function _root_driver.<locals>._fn at 0x...>` tells a reader nothing they can act on.
    rows = []
    for k, v in sorted(vars(s).items()):
        if k.startswith("__") or callable(v):
            continue
        rows.append((k.lstrip("_"), ", ".join(f"{a}={b!r}" for a, b in v.items()) if isinstance(v, dict) else repr(v)))
    doc = (type(s).__doc__ or getattr(s, "__doc__", "") or "").strip().split("\n")[0]
    return Info(f"spec · {name}", [("what it does", [("", doc)] if doc else []), ("settings", rows)])



# ---------------------------------------------------------------------------------------------
# the environment — `jno.info()` with nothing to inspect
# ---------------------------------------------------------------------------------------------
def _info_env(deep: bool) -> Info:
    """What the run is actually sitting on. The two that silently ruin an answer are first.

    **x64.** jNO assembles in float64 and says so; with it off, a stiffness matrix is built in
    float32 and the solve is quietly less accurate than every convergence table in the docs. It is
    a process-wide flag set before the first array, so discovering it late is discovering it too
    late. (Measured elsewhere in this library: `jno.litho` returns an all-NaN volume in float32.)

    **Device.** "Is this actually on the GPU" is not answerable from any jNO output today, and the
    answer changes wall-clock by an order of magnitude.
    """
    import jax

    build: list = []
    try:
        x64 = bool(jax.config.jax_enable_x64)
    except Exception:  # noqa: BLE001
        x64 = None
    build.append((
        "float64 (x64)",
        "ON" if x64 else "**OFF** — jNO assembles in float64; set jax.config.update('jax_enable_x64', True) "
        "before the first array",
    ))
    build.append(("default dtype", str(np.zeros(1).dtype) + " (numpy) / " + str(jax.numpy.zeros(1).dtype) + " (jax)"))
    try:
        build.append(("backend", jax.default_backend()))
    except Exception:  # noqa: BLE001
        pass

    devices: list = []
    try:
        for i, dev in enumerate(jax.devices()):
            row = f"{dev.device_kind} ({dev.platform})"
            try:
                st = dev.memory_stats() or {}
                used, lim = st.get("bytes_in_use"), st.get("bytes_limit")
                if used is not None and lim:
                    row += f"   ·   {_bytes(used)} / {_bytes(lim)} used ({100*used/lim:.0f} %)"
            except Exception:  # noqa: BLE001 - CPU devices have no memory_stats
                pass
            devices.append((f"[{i}] {dev}", row))
    except Exception:  # noqa: BLE001
        pass

    versions: list = []
    for mod in ("jax", "jaxlib", "numpy"):
        try:
            versions.append((mod, __import__(mod).__version__))
        except Exception:  # noqa: BLE001
            pass
    import os as _os

    envvars = [
        (k, _os.environ[k])
        for k in ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "JNO_COMPILE_CACHE")
        if k in _os.environ
    ]
    return Info("environment", [("build", build), ("devices", devices), ("versions", versions), ("env", envvars)])


# ---------------------------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------------------------
#: Extra handlers, keyed by class name. A module that lives on another branch (``jno.peec``) can
#: register itself here at import time rather than this file having to import it.
REGISTRY: dict = {}


def info(obj: Any = None, *, deep: bool = False) -> Info:
    """Report what ``obj`` is and whether it is what you meant. See the module docstring.

    With no argument, reports the **environment** instead: x64, device, memory, versions.
    """
    if obj is None:
        return _info_env(deep)
    cls = type(obj).__name__
    if cls in REGISTRY:
        return REGISTRY[cls](obj, deep)
    if hasattr(obj, "classification") and hasattr(obj, "offsets"):
        return _info_fem(obj, deep)
    if cls == "domain" or (hasattr(obj, "variable") and hasattr(obj, "dimension")):
        return _info_domain(obj, deep)
    if cls.lower().startswith("rcwa") or hasattr(obj, "efficiency"):
        return _info_rcwa(obj, deep)
    if hasattr(obj, "solve_pinned") or cls.lower().startswith("fdm"):
        return _info_fdm(obj, deep)
    if hasattr(obj, "get_constraint_tags"):
        return _info_core(obj, deep)
    # A solver / preconditioner / time-scheme spec, identified by WHERE IT COMES FROM rather than
    # by its class name: they are variously `*Spec`, `_ThetaScheme`, or a plain closure holder, and
    # a name test missed the ones that matter.
    mod = str(getattr(type(obj), "__module__", ""))
    if hasattr(obj, "__dict__") and (mod.startswith(("jno.solve", "jno.precond", "jno.utils.solver")) or cls.endswith("Spec")):
        return _info_spec(obj, deep)
    raise TypeError(
        f"jno.info: nothing to report for {cls!r}. Handled: a domain, a jno.fem form, a jno.fdm or "
        f"jno.rcwa solver, a jno.core, and the jno.solve / jno.precond specs. Register another with "
        f"jno.info.REGISTRY[<class name>] = <handler>."
    )


# `from .info import info` binds `jno.info` to the FUNCTION, which shadows this module -- so
# `jno.info.REGISTRY` would not resolve for a user following the docs. Hang them off the function
# so the documented path is the real one.
info.REGISTRY = REGISTRY
info.Info = Info
