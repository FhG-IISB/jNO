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
        cn = getattr(f, "_complex_n", None)
        form.append((
            "complex",
            "yes — solved as a real-equivalent 2n block"
            + (f"; the field blocks below index the REAL half (n = {_fmt_n(cn)}), the imaginary half follows at +n"
               if cn else ""),
        ))
    if f.mode == "transient":
        form.append(("time window", f"[{getattr(f, 't0', '?')}, {getattr(f, 't1', '?')}]"))
    if getattr(f, "_periodic", None) is not None:
        form.append(("periodic", "yes — solved in the reduced space, then prolonged"))
    rpe = getattr(getattr(f, "_op", None), "runtime_parameter_exprs", None)
    if rpe:
        form.append(("runtime parameters", ", ".join(sorted(map(str, rpe)))))
    if getattr(f, "_saddle_blocks", None):
        form.append(("saddle blocks", ", ".join(map(str, f._saddle_blocks))))

    terms: list = [(f"[{i}]", c) for i, c in enumerate(f.classification or [])]
    given = len(getattr(f, "_constraints", None) or [])
    if given and given != len(f.classification or []):
        terms.append(("", f"— {len(f.classification or [])} of {given} terms appear here; the rest "
                          f"(a periodic tie, a gauge) carry no classification entry"))

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
    """`jno.rcwa` has TWO objects and neither matched what this first reported.

    An UNSOLVED `_RcwaProblem` keeps everything on `.spec` (period, wavelength, layers, source),
    and a SOLVED `_Sol` keeps it privately (`_period`, `_wl`, `_layers`). The first version read
    `orders` / `period` / `wavelength` straight off the object and called `r.power(...)`, which
    `_Sol` does not have -- written from attribute names, never run.
    """
    spec = getattr(r, "spec", None)
    if spec is not None:                                            # the problem, before solving
        setup: list = [("orders", str(getattr(r, "orders", "?")))]
        if getattr(r, "formulation", None) is not None:
            setup.append(("formulation", str(r.formulation)))
        for label, attr in (("period", "period"), ("wavelength", "wavelength"),
                            ("periodic axes", "periodic_axes"), ("source face", "source_face"),
                            ("k_in", "k_in")):
            v = getattr(spec, attr, None)
            if v is not None:
                setup.append((label, str(v)[:64]))
        layers: list = []
        for i, lay in enumerate(getattr(spec, "layers", None) or []):
            # A layer is (thickness, permittivity...) and the permittivity is a full grid. Dumping
            # it prints a few thousand numbers to say "this layer is glass"; report its range.
            try:
                thick = lay[0]
                eps = np.asarray(lay[1][0] if isinstance(lay[1], (tuple, list)) else lay[1])
                tl = "semi-infinite ambient" if not np.isfinite(thick) else f"thickness {float(thick):.4g}"
                lo, hi = float(np.real(eps).min()), float(np.real(eps).max())
                rng = f"eps {lo:.4g}" if abs(hi - lo) < 1e-12 else f"eps {lo:.4g}–{hi:.4g}  (patterned, {eps.shape} grid)"
                layers.append((f"[{i}]", f"{tl}  ·  {rng}"))
            except Exception:  # noqa: BLE001
                layers.append((f"[{i}]", str(lay)[:70]))
        return Info("rcwa (problem, unsolved)", [("setup", setup), ("layers", layers),
                                                 ("result", [("", "not solved — call .solve()")])])

    setup = []
    for label, attr in (("period", "_period"), ("wavelength", "_wl")):
        v = getattr(r, attr, None)
        if v is not None:
            setup.append((label, str(v)[:64]))
    th = getattr(r, "_thick", None)
    if th is not None:
        setup.append(("layers", f"{len(np.atleast_1d(th))}  ·  thicknesses {np.round(np.atleast_1d(th), 4).tolist()}"))

    result: list = []
    try:
        T, R = float(r.efficiency("T")), float(r.efficiency("R"))
        result.append(("efficiency T / R", f"{T:.6f} / {R:.6f}"))
        tot = T + R
        # For a LOSSLESS stack T + R = 1 exactly. That is the energy-conservation oracle, and it is
        # the one number that says whether the truncation order was enough.
        result.append((
            "T + R",
            f"{tot:.6f}" + ("   ✓ energy conserved" if abs(tot - 1.0) < 1e-6
                            else "   ← not 1: absorbing stack, or too few orders"),
        ))
    except Exception as e:  # noqa: BLE001
        result.append(("efficiency", f"unavailable ({type(e).__name__})"))
    return Info("rcwa (solved)", [("setup", setup), ("result", result)])


def _info_fdm(o, deep: bool) -> Info:
    """A collocation (strong-form) solve. Its state is the PDE list, the unknown(s) and the points --
    the first report here said only "route", which is true and useless."""
    d = getattr(o, "domain", None)
    structured = bool(getattr(d, "_structured_grid", None))
    rows: list = [
        ("route", "structured stencil + GMG preconditioner" if structured else "cotangent / P1 Laplace-Beltrami operator"),
        ("regime", "transient (method of lines)" if getattr(o, "_transient", False) else "steady"),
    ]
    pts = getattr(o, "_pts", None)
    if pts is not None:
        q = np.asarray(pts)
        rows.append(("collocation points", _fmt_n(q.shape[0])))
        if q.ndim >= 2:
            rows.append(("extent", " × ".join(f"[{q[:, a].min():.4g}, {q[:, a].max():.4g}]" for a in range(q.shape[1]))))
    unk = list(getattr(o, "unknowns", None) or [])
    if unk:
        rows.append(("unknowns", f"{len(unk)}  ({', '.join(str(getattr(m, 'name', None) or type(m).__name__) for m in unk)})"))
        if pts is not None and len(unk):
            rows.append(("dofs", _fmt_n(len(np.asarray(pts)) * len(unk))))
    terms = list(getattr(o, "_pde", None) or [])
    if terms:
        rows.append(("residual terms", _fmt_n(len(terms))))
    if getattr(o, "_periodic_axes", None):
        rows.append(("periodic axes", str(o._periodic_axes)))
    if getattr(o, "region", None) is not None:
        rows.append(("region", str(o.region)))
    if d is not None and getattr(d, "time", None):
        t0, t1, nt = d.time
        rows.append(("time", f"[{t0}, {t1}] · {nt} steps"))
    return Info("fdm (collocation)", [("solver", rows)])


def _info_core(c, deep: bool) -> Info:
    import jax

    models: list = []
    # `core.models` is a DICT keyed by model id, not a list -- enumerating it walks the KEYS and
    # reports every model as `int, 1 parameters`. Measured, and the reason this is spelled out.
    raw = getattr(c, "models", None) or {}
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    total = 0
    for key, m in items:
        try:
            n = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(m) if hasattr(x, "shape") or np.ndim(x))
            total += n
            extra = []
            for attr in ("in_features", "output_dim"):
                v = getattr(m, attr, None)
                if v is not None:
                    extra.append(f"{attr}={v}")
            models.append((f"[{key}] {type(m).__name__}", f"{_fmt_n(n)} parameters" + (f"  ·  {', '.join(extra)}" if extra else "")))
        except Exception:  # noqa: BLE001
            models.append((f"[{key}]", type(m).__name__))
    if len(models) > 1:
        models.append(("total", f"{_fmt_n(total)} parameters"))
    cons: list = []
    try:
        for i, t in enumerate(c.get_constraint_tags(c.constraints)):
            cons.append((f"[{i}]", str(t)))
    except Exception:  # noqa: BLE001
        cons = [("count", _fmt_n(len(getattr(c, "constraints", []) or [])))]
    training: list = []
    dom = getattr(c, "domain", None)
    if dom is not None:
        training.append(("domain", f"{getattr(dom, 'dimension', '?')}D · {_fmt_n(getattr(dom, 'total_samples', '?'))} samples"))
    names = list(getattr(c, "_tracker_names", None) or [])
    if names:
        training.append(("trackers", ", ".join(map(str, names))))
    # `core.models` holds the UNWRAPPED modules; the optimizer lives on the `Model` WRAPPER, which
    # the core reaches through `_collect_flax_modules()`. Checking `core.models` found nothing and
    # told a user to call `.optimizer(...)` they had already called.
    opt_on_models = []
    try:
        for _lid, fm in (c._collect_flax_modules() or {}).items():
            if getattr(fm, "_opt_fn", None) is not None or getattr(fm, "_bayesian_cfg", None) is not None:
                nm = getattr(fm, "name", None)
                opt_on_models.append(str(nm) if isinstance(nm, str) and nm else type(getattr(fm, "module", None)).__name__)
    except Exception:  # noqa: BLE001
        pass
    if getattr(c, "_opt_states", None):
        training.append(("optimizer", "set on the core"))
    elif opt_on_models:
        training.append(("optimizer", f"set per-model on {', '.join(opt_on_models)}  (net.optimizer(...))"))
    else:
        training.append(("optimizer", "not set — call .optimizer(...) on the core or on each net before .solve()"))
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
# the small parts: an expression, a variable, a network, a shape, a result
#
# These matter MORE than the assembled ones. The assembled object is where you find out something
# was wrong; the small ones are where it went wrong. They also know less about themselves, so each
# report says plainly what is not built yet rather than building it to have more to say.
# ---------------------------------------------------------------------------------------------
def _walk(node):
    """Every traced node in the tree, once. Uses jNO's own child iterator so a new node type is
    picked up here the moment `iter_children` learns about it."""
    from .utils.solver.solver_helper import iter_children

    seen, stack, out = set(), [node], []
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        out.append(n)
        try:
            stack.extend(iter_children(n))
        except Exception:  # noqa: BLE001
            pass
    return out


def _node_label(n) -> str:
    for attr in ("name", "_name"):
        v = getattr(n, attr, None)
        if isinstance(v, str) and v:          # `name` is a METHOD on a Placeholder, and truthy
            return v
    return type(n).__name__


def _info_expr(e, deep: bool, outer=None) -> Info:
    from .trace import Model, TestFunction, TrialFunction, Variable

    nodes = _walk(e)
    # A bound-but-underived trial function keeps its coordinates on the VIEW (`_coord_vars`), not in
    # the IR -- `u.bind(x=xb, y=yb) * v.bind(...)` has no Variable anywhere in its tree. Measured:
    # without this, the region row is simply absent for exactly the terms a weak form is made of.
    for holder in (outer, e):
        for val in (vars(holder) if hasattr(holder, "__dict__") else {}).values():
            if isinstance(val, dict):
                nodes.extend(v for v in val.values() if isinstance(v, Variable))
            elif isinstance(val, Variable):
                nodes.append(val)
    nm = getattr(e, "_name", None) or ""
    if isinstance(nm, str) and "solve" in nm:
        what: list = [("type", f"a DEFERRED {nm} — the solve runs when you evaluate it through jno.core")]
    else:
        what = [("type", type(e).__name__)]
    op = getattr(e, "op", None)
    if isinstance(op, str):
        what.append(("operator", op))
    sh = getattr(e, "shape", None)
    if sh is not None and not callable(sh):
        what.append(("shape", str(sh)))

    # WHICH REGION it samples. A PDE residual accidentally bound to `boundary` instead of
    # `interior` is a classic mistake and is invisible everywhere else.
    spatial, temporal = set(), set()
    for n in nodes:
        if isinstance(n, Variable):
            (temporal if getattr(n, "axis", None) == "temporal" else spatial).add(str(getattr(n, "tag", "?")))
    reads: list = []
    if spatial:
        reads.append(("regions", ", ".join(sorted(map(str, spatial)))))
    if temporal:
        reads.append(("temporal", "yes"))
    nets, params = set(), set()
    for n in nodes:
        m = n.model if type(n).__name__ == "ModelCall" and hasattr(n, "model") else (n if isinstance(n, Model) else None)
        if m is None:
            continue
        nm = getattr(m, "name", None)
        arch = type(getattr(m, "module", None)).__name__
        label = arch + (f" ({nm})" if isinstance(nm, str) and nm else "")
        (params if arch == "_Parameter" else nets).add(label)
    if nets:
        reads.append(("networks", ", ".join(sorted(nets))))
    if params:
        reads.append(("trainable parameters", ", ".join(sorted(params)).replace("_Parameter ", "")))
    tri = [n for n in nodes if isinstance(n, TrialFunction)]
    tst = [n for n in nodes if isinstance(n, TestFunction)]
    if tri or tst:
        reads.append(("weak form", f"trial {'yes' if tri else 'no'} · test {'yes' if tst else 'no'}"
                                   + ("   ← a jno.fem term" if tri and tst else "")))

    struct: list = []
    try:
        from .utils.solver.solver_helper import max_temporal_derivative_order

        struct.append(("d/dt order", str(max_temporal_derivative_order(e))))
    except Exception:  # noqa: BLE001
        pass
    jac = sum(1 for n in nodes if type(n).__name__ in ("Jacobian", "Hessian"))
    struct.append(("derivative nodes", str(jac)))
    struct.append(("tree size", f"{len(nodes)} nodes"))

    tree: list = []
    if deep:
        def _render(n, depth=0, out=None):
            from .utils.solver.solver_helper import iter_children

            out = [] if out is None else out
            out.append(("", "  " * depth + _node_label(n)))
            if depth < 6:
                try:
                    for c in iter_children(n):
                        _render(c, depth + 1, out)
                except Exception:  # noqa: BLE001
                    pass
            return out

        tree = _render(e)[:200]
    return Info(f"expression · {type(e).__name__}", [("what", what), ("reads", reads),
                                                     ("structure", struct), ("tree", tree)])


def _info_tensor_tag(t, deep: bool) -> Info:
    """`dom.variable("k", values)` returns a TensorTag -- a PARAMETER, not a coordinate. It has no
    mesh pool, so the coordinate report said nothing about it at all."""
    d = getattr(t, "_domain", None)
    tag = str(getattr(t, "tag", "?"))
    rows: list = [("tag", tag), ("kind", "parameter tag (a per-sample value, not a coordinate)")]
    if getattr(t, "dim_index", None) is not None:
        rows.append(("component", str(t.dim_index)))
    vals = (getattr(d, "context", None) or {}).get(tag)
    if vals is not None:
        q = np.asarray(vals)
        rows.append(("values", f"shape {q.shape} · [{q.min():.6g}, {q.max():.6g}]"))
        rows.append(("samples", _fmt_n(q.shape[0])))
    return Info(f"parameter · {tag}", [("", rows)])


def _info_variable(v, deep: bool) -> Info:
    rows: list = [("tag", str(getattr(v, "tag", "?"))), ("axis", str(getattr(v, "axis", "spatial")))]
    d = getattr(v, "_domain", None)
    if d is not None:
        pool = (getattr(d, "_mesh_pool", None) or {}).get(getattr(v, "tag", None))
        if pool is not None:
            q = np.asarray(pool)
            rows.append(("points", _fmt_n(q.shape[0])))
            if q.ndim >= 2:
                dim = int(getattr(d, "dimension", q.shape[-1]))
                rows.append(("extent", " × ".join(f"[{q[..., a].min():.4g}, {q[..., a].max():.4g}]" for a in range(min(dim, q.shape[-1])))))
        if getattr(d, "normals_by_tag", None) and getattr(v, "tag", None) in d.normals_by_tag:
            rows.append(("normals", "available — domain.variable(tag, normals=True)"))
    return Info(f"variable · {getattr(v, 'tag', '?')}", [("", rows)])


def _info_model(m, deep: bool) -> Info:
    import jax

    mod = getattr(m, "module", None)
    nm = getattr(m, "name", None)
    rows: list = [("name", str(nm) if isinstance(nm, str) and nm else "(unnamed)"),
                  ("architecture", type(mod).__name__ if mod is not None else "?")]
    try:
        n = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(mod) if np.ndim(x))
        rows.append(("parameters", _fmt_n(n)))
        leaves = [x for x in jax.tree_util.tree_leaves(mod) if np.ndim(x)]
        if leaves:
            rows.append(("dtype", str(np.asarray(leaves[0]).dtype)))
    except Exception:  # noqa: BLE001
        pass
    for label, attr in (("input dim", "input_dim"), ("layer id", "layer_id"), ("weight path", "weight_path")):
        val = getattr(m, attr, None)
        if isinstance(val, (str, int, float, bool)) and str(val):
            rows.append((label, str(val)))
    if getattr(m, "_frozen", False):
        rows.append(("frozen", "yes — excluded from the optimizer"))
    return Info(f"model · {nm if isinstance(nm, str) and nm else type(mod).__name__}", [("", rows)])


def _info_shape(sh, deep: bool) -> Info:
    """A shape BEFORE `.domain()`. Nothing here is meshed, so nothing here reports mesh quality --
    what a CSG tree produced is checkable without paying gmsh for it."""
    geom: list = [("dim", str(getattr(sh, "dim", "?")))]
    try:
        lo, hi = sh.bounds()
        dim = int(getattr(sh, "dim", 2) or 2)
        geom.append(("bounds", " × ".join(f"[{a:.4g}, {b:.4g}]" for a, b in list(zip(lo, hi))[:dim])))
    except Exception:  # noqa: BLE001
        pass
    if getattr(sh, "_size", None) is not None:
        geom.append(("mesh size", str(sh._size)))
    if getattr(sh, "_mesh_order", 1) != 1:
        geom.append(("geometry order", str(sh._mesh_order) + "  (curved)"))
    if getattr(sh, "_structured", None):
        geom.append(("structured", "yes"))
    geom.append(("meshed", "no — call .domain() (jno.info on the domain then reports quality)"))

    regions: list = []
    try:
        for name, sub in sh._region_items():
            att = getattr(sub, "_attach", None) or {}
            regions.append((str(name), ", ".join(f"{k}={v!r}" for k, v in att.items()) or "(no attached properties)"))
    except Exception:  # noqa: BLE001
        pass

    tree: list = []

    def _csg(node, depth=0, label=None):
        if not isinstance(node, tuple) or not node:
            return
        kind = str(node[0])
        tree.append(("", "  " * depth + (f"{label}: {kind}" if label else kind)))
        if kind == "regions":                       # ('regions', ((name, shape), ...), conforming)
            for name, sub in node[1]:
                _csg(getattr(sub, "_node", None), depth + 1, label=str(name))
            return
        if kind == "leaf":                          # ('leaf', primitive, id)
            tree.append(("", "  " * (depth + 1) + type(node[1]).__name__))
            return
        for child in node[1:]:                      # cut / fuse / inter
            _csg(getattr(child, "_node", None), depth + 1)

    _csg(getattr(sh, "_node", None))
    return Info("shape (unmeshed)", [("geometry", geom), ("regions", regions), ("CSG tree", tree[:40])])


def _info_result(r, deep: bool, context=None) -> Info:
    """A solved array, or the per-frame trajectory an adaptive transient returns."""
    if hasattr(r, "times") and hasattr(r, "states"):
        rows = [("frames", _fmt_n(len(r.times))),
                ("time", f"[{float(np.min(r.times)):.4g}, {float(np.max(r.times)):.4g}]"),
                ("meshes", "one per frame — call .resample() for a uniform array")]
        try:
            rows.append(("dofs per frame", f"{min(len(np.asarray(s)) for s in r.states)}–{max(len(np.asarray(s)) for s in r.states)}"))
        except Exception:  # noqa: BLE001
            pass
        return Info("trajectory (adaptive)", [("", rows)])

    a = np.asarray(r)
    rows = [("shape", str(a.shape)), ("dtype", str(a.dtype))]
    if a.size == 0:
        rows.append(("range", "— the array is EMPTY"))
        return Info("result", [("array", rows), ("by field block", [])])
    finite = np.isfinite(a)
    if not finite.all():
        rows.append(("finite", f"**{int((~finite).sum())} non-finite of {a.size}**"))
        good = a[finite]
        if good.size:
            rows.append(("range (finite part)", f"[{good.min():.6g}, {good.max():.6g}]"))
    else:
        rows.append(("range", f"[{a.min():.6g}, {a.max():.6g}]"))
        if not np.any(a):
            rows.append(("note", "ALL ZERO"))
    rows.append(("norm", f"{np.linalg.norm(a.reshape(-1)):.6g}"))

    blocks: list = []
    offs = list(getattr(context, "offsets", None) or [])
    keys = list(getattr(context, "_block_field_keys", None) or [])
    flat = a.reshape(-1)
    if len(offs) > 1 and a.ndim == 2 and a.shape[-1] == offs[-1]:
        blocks.append(("", f"{a.shape[0]} time steps x {a.shape[1]} dofs"))
        for i in range(len(offs) - 1):
            seg = a[:, offs[i]:offs[i + 1]]
            nm2 = f"field {keys[i]}" if i < len(keys) else f"block {i}"
            rng = f"[{seg.min():.6g}, {seg.max():.6g}]" if seg.size else "(empty)"
            blocks.append((nm2, f"{_fmt_n(seg.shape[1])} dofs · {rng}  (over all steps)"))
        return Info("result", [("array", rows), ("by field block", blocks)])
    if len(offs) > 1 and offs[-1] == flat.size:
        for i in range(len(offs) - 1):
            seg = flat[offs[i]:offs[i + 1]]
            nm = f"field {keys[i]}" if i < len(keys) else f"block {i}"
            rng = f"[{seg.min():.6g}, {seg.max():.6g}]" if seg.size else "(empty)"
            blocks.append((nm, f"{_fmt_n(seg.size)} dofs · {rng}"))
    elif context is not None:
        blocks.append(("", "context given but its offsets do not span this array"))
    return Info("result", [("array", rows), ("by field block", blocks)])


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


def info(obj: Any = None, *, deep: bool = False, context: Any = None) -> Info:
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
    # The small parts. Dispatch on STRUCTURE, not class names: a Variable and a Model are both
    # trace nodes, so they are tested before the generic expression handler, and everything else
    # that walks like a trace node reaches `_info_expr` whatever it is called.
    from .trace import Model, Placeholder, Variable

    if isinstance(obj, (tuple, list)) and obj and all(isinstance(o, Variable) for o in obj):
        # `domain.variable(tag)` hands back a tuple (x, y[, z], t) even without split=True.
        # The components of one tag differ only in which column they read, so reporting each in
        # full says the same thing three times. Group by tag.
        seen, secs = set(), []
        for v in obj:
            tag = str(getattr(v, "tag", "?"))
            if tag in seen:
                continue
            seen.add(tag)
            rows = _info_variable(v, deep).sections[0][1]
            n = sum(1 for o in obj if str(getattr(o, "tag", "?")) == tag)
            secs.append((f"{tag}  ({n} component{'s' if n > 1 else ''})", rows))
        return Info(f"variables · {len(obj)} returned by domain.variable(...)", secs)
    if type(obj).__name__ == "TensorTag":
        return _info_tensor_tag(obj, deep)
    if isinstance(obj, Variable):
        return _info_variable(obj, deep)
    if isinstance(obj, Model):
        return _info_model(obj, deep)
    if hasattr(obj, "_region_items") and hasattr(obj, "_node"):
        return _info_shape(obj, deep)
    inner = getattr(obj, "expr", None)
    if isinstance(inner, Placeholder):
        return _info_expr(inner, deep, outer=obj)
    if isinstance(obj, Placeholder):
        return _info_expr(obj, deep)
    if hasattr(obj, "times") and hasattr(obj, "states"):
        return _info_result(obj, deep, context)
    if isinstance(obj, np.ndarray) or (hasattr(obj, "shape") and hasattr(obj, "dtype")):
        return _info_result(obj, deep, context)
    if cls == "domain" or (hasattr(obj, "variable") and hasattr(obj, "dimension")):
        return _info_domain(obj, deep)
    if str(getattr(type(obj), "__module__", "")) == "jno.rcwa" or hasattr(obj, "efficiency"):
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
