"""Parametric geometry: a shape coordinate that is an expression in ``jno.np.parameter``\\ s.

``jno.shape.box(0, 0, 0, w, 20e-3, 3e-4)`` with ``w`` a parameter is a box whose upper x face is a
design variable. The shape meshes, samples and discretises at the parameters' INITIAL values (the
reference), which is all a build plan can do -- gmsh and every structural decision need numbers --
and keeps the expression beside the reference, so a solver can put the design back in under a trace:
the domain stretches its mesh, a PEEC network its grid lines and wire vertices.

This module holds what every consumer needs and no solver does on its own: splitting a coordinate
into (reference, expression), evaluating an expression at given parameter values, and finding the
parameters an expression reads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def is_expr(v) -> bool:
    """True for a trace expression (a ``Placeholder``), False for a number or an array."""
    from jno.trace import Placeholder

    return isinstance(v, Placeholder)


def model_calls(node):
    """The ``jno.np.parameter`` nodes inside an expression, or ``[]`` for a plain value."""
    if not is_expr(node):
        return []
    from jno._fem import _walk

    return [n for n in _walk(node) if type(n).__name__ == "ModelCall"]


def parameters_in(values):
    """``{name: ModelCall}`` for every ``jno.np.parameter`` reachable from ``values``."""
    out = {}
    for v in values:
        for mc in model_calls(v):
            name = getattr(getattr(mc, "model", None), "_parameter_name", None)
            if name is not None:
                out[name] = mc
    return out


def initial_value(mc):
    """A parameter's INITIALISED value, not the zeros it is born with (None without an initializer).

    ``.initialize(...)`` only records the initializer -- the training loop applies it -- so reading
    ``module.value`` before training gives zeros. A reference geometry, like a reference
    conductivity, has to be the point the optimisation starts from, so the initializer is applied here.
    """
    m = getattr(mc, "model", None)
    fn = getattr(m, "_initializer_fn", None)
    if fn is None:
        return None
    v = jnp.asarray(m.module.value)
    key = getattr(m, "_initializer_key", None)
    if key is None:
        key = jax.random.PRNGKey(0)
    try:
        return jnp.asarray(fn(key, v.shape, v.dtype))
    except TypeError:
        return jnp.asarray(fn(v.shape))


def evaluate(expr, values=None):
    """``expr`` with each parameter set to ``values[name]`` (its initialised value where absent).

    A plain value comes back unchanged. Differentiable in ``values``; position-dependent expressions
    (a coordinate ``Variable`` inside) are refused, since a coordinate has no position to be read at.
    """
    if not is_expr(expr):
        return expr
    import equinox as eqx

    from jno._fem import _walk
    from jno.trace import Variable
    from jno.trace_evaluator import TraceEvaluator

    if any(isinstance(n, Variable) for n in _walk(expr)):
        raise ValueError(
            "jno.shape: a coordinate depends on POSITION (it contains a domain variable). A shape "
            "coordinate is one number per design -- write it in `jno.np.parameter`s only."
        )
    values = values or {}
    table = {}
    for mc in model_calls(expr):
        mod = mc.model.module
        name = getattr(mc.model, "_parameter_name", None)
        val = values.get(name) if name is not None else None
        if val is None:
            val = initial_value(mc)
        if val is not None:
            mod = eqx.tree_at(lambda m: m.value, mod, jnp.asarray(val))
        table[mc.model.layer_id] = mod
    return TraceEvaluator(table).evaluate(expr, context={})


def reference(v, what="coordinate") -> float:
    """The number a coordinate stands for at the reference: itself, or its expression at the
    parameters' initial values. A parameter with no initializer is refused, since its zeros are
    almost never the intended geometry and would build a degenerate shape without a word."""
    if not is_expr(v):
        return float(v)
    for mc in model_calls(v):
        if getattr(getattr(mc, "model", None), "_initializer_fn", None) is None:
            name = getattr(mc.model, "_parameter_name", None) or "a parameter"
            raise ValueError(
                f"jno.shape: the {what} reads {name!r}, which has no initial value, so the shape has no "
                "reference to mesh at (a parameter is ZEROS until initialised). Give it the design's "
                "starting point: `.initialize(jax.nn.initializers.constant(x0))`."
            )
    a = np.asarray(jax.lax.stop_gradient(jnp.asarray(evaluate(v)))).reshape(-1)
    if a.size != 1:
        raise ValueError(f"jno.shape: a {what} must be one number; this expression has {a.size} values.")
    return float(a[0])


def split(values, what="coordinate"):
    """``(references, exprs)`` for a sequence of coordinates: the numbers to build with, and
    ``((index, expr), ...)`` for the ones that are expressions (empty when none is)."""
    vals = list(values)
    refs = tuple(reference(v, what) for v in vals)
    exprs = tuple((i, v) for i, v in enumerate(vals) if is_expr(v))
    return refs, exprs


# ---- a shape's coordinates in world space, with the expressions put back --------------------------


def _leaf_exprs(node):
    return dict(node[3]) if len(node) > 3 else {}


def _add(a, b):
    """``a + b`` that keeps ``a`` itself when ``b`` is a plain zero, so an expression stays the SAME
    object -- two faces written with one expression are recognised as one plane."""
    if not is_expr(b) and float(b) == 0.0:
        return a
    if not is_expr(a) and float(a) == 0.0:
        return b
    return a + b


def _offset(node):
    """``(inner node, [dx, dy, dz])`` through any chain of translations (entries may be expressions)."""
    off = [0.0, 0.0, 0.0]
    while node[0] == "translate":
        ex = dict(node[3]) if len(node) > 3 else {}
        off = [_add(o, ex.get(a, node[2][a])) for a, o in enumerate(off)]
        node = node[1]._node
    return node, off


def has_params(node) -> bool:
    """True if any coordinate anywhere in this plan is an expression."""
    kind = node[0]
    if kind == "leaf":
        return len(node) > 3
    if kind == "regions":
        return any(has_params(sub._node) for _n, sub in node[1])
    if kind in ("translate", "extrude") and len(node) > 3:
        return True
    return any(has_params(c._node) for c in node[1:] if hasattr(c, "_node"))


def world_box(sh):
    """The six faces ``[x0, y0, z0, x1, y1, z1]`` of an axis-aligned box in world coordinates, each a
    number or an expression, or ``None`` if ``sh`` is not a (translated) box, extruded rect or -- in
    2-D, with ``z0 = z1 = 0`` -- rect."""
    from .primitives import Box, Rect

    node, off = _offset(sh._node)
    if node[0] == "leaf" and isinstance(node[1], Box):
        b, ex = node[1], _leaf_exprs(node)
        f = [ex.get(i, v) for i, v in enumerate((b.x0, b.y0, b.z0, b.x1, b.y1, b.z1))]
    elif node[0] == "leaf" and isinstance(node[1], Rect):  # 2-D: a box with no z extent
        r, ex = node[1], _leaf_exprs(node)
        x0, y0, x1, y1 = (ex.get(i, v) for i, v in enumerate((r.x0, r.y0, r.x1, r.y1)))
        f = [x0, y0, 0.0, x1, y1, 0.0]
    elif node[0] == "extrude":
        inner, off2 = _offset(node[1]._node)
        if not (inner[0] == "leaf" and isinstance(inner[1], Rect)):
            return None
        r, ex = inner[1], _leaf_exprs(inner)
        h = node[3] if len(node) > 3 else node[2]
        x0, y0, x1, y1 = (ex.get(i, v) for i, v in enumerate((r.x0, r.y0, r.x1, r.y1)))
        f = [_add(x0, off2[0]), _add(y0, off2[1]), 0.0, _add(x1, off2[0]), _add(y1, off2[1]), h]
    else:
        return None
    return [_add(f[i], off[i % 3]) for i in range(6)]


def world_line(sh):
    """``(points, radius)`` of a (translated) line in world coordinates, every entry a number or an
    expression, or ``None`` if ``sh`` is not one."""
    from .primitives import Line

    node, off = _offset(sh._node)
    if not (node[0] == "leaf" and isinstance(node[1], Line)):
        return None
    ln, ex = node[1], _leaf_exprs(node)
    n = len(ln.points)
    pts = [[_add(ex.get(3 * k + a, ln.points[k][a]), off[a]) for a in range(3)] for k in range(n)]
    return pts, ex.get(3 * n, ln.r)


def stack(entries, values=None):
    """A nested list of numbers / expressions as one array, the expressions evaluated at ``values``."""
    if isinstance(entries, (list, tuple)):
        return jnp.stack([stack(e, values) for e in entries])
    return jnp.reshape(jnp.asarray(evaluate(entries, values), dtype=float), ())


def any_expr(entries) -> bool:
    if isinstance(entries, (list, tuple)):
        return any(any_expr(e) for e in entries)
    return is_expr(entries)


def flat(entries):
    if isinstance(entries, (list, tuple)):
        return [x for e in entries for x in flat(e)]
    return [entries]


# ---- the stretch a design applies to a domain ------------------------------------------------------


def _walk_planes(node, off, out, where):
    """Every axis-aligned face plane in the plan: ``out[axis]`` gets ``(reference, entry, where)``."""
    sh_like = type("_S", (), {"_node": node})
    wb = world_box(sh_like)
    if wb is not None:
        for a in range(3):
            for side in (0, 1):
                e = wb[3 * side + a]
                e = _add(e, off[a])
                out[a].append((reference(e), e, where))
        return
    kind = node[0]
    if kind == "regions":
        for name, sub in node[1]:
            _walk_planes(sub._node, off, out, name)
    elif kind in ("cut", "fuse", "inter"):
        _walk_planes(node[1]._node, off, out, where)
        _walk_planes(node[2]._node, off, out, where)
    elif kind == "translate":
        inner, o = _offset(node)
        _walk_planes(inner, [_add(a, b) for a, b in zip(off, o)], out, where)
    elif kind == "leaf" and len(node) > 3 and type(node[1]).__name__ != "Line":
        raise NotImplementedError(f"jno.shape: a parametric {type(node[1]).__name__} cannot move with the domain.")
    # a rotation, a revolve, a fixed curved primitive, a line: no axis plane, it is stretched along


def stretch_planes(plan, tol=1e-12):
    """Per axis, the face planes of a plan that a design moves: ``(refs, entries)`` sorted by
    reference, with only the axes where at least one plane is an expression (``None`` elsewhere).

    Every axis-aligned face of every box in the plan is a plane, fixed or not: a FIXED face is an
    anchor that stays put, so a design moves only what lies between the faces it moves and the
    nearest fixed ones. Two faces at one reference plane are one plane of the mesh, so they must move
    together: both fixed, or the same expression -- a fixed face and a moving one, or two different
    expressions, there are refused by name, since a conforming mesh cannot split along that plane.
    """
    out = [[], [], []]
    _walk_planes(plan._node, [0.0, 0.0, 0.0], out, getattr(plan, "_region_name", None))
    axes = []
    for a in range(3):
        if not any(is_expr(e) for _r, e, _w in out[a]):
            axes.append(None)
            continue
        pl = sorted(out[a], key=lambda t: t[0])
        span = max(1.0, abs(pl[-1][0] - pl[0][0])) if pl else 1.0
        refs, ents, whos = [], [], []
        for r, e, w in pl:
            if refs and abs(r - refs[-1]) <= tol * span:
                if e is ents[-1] or (not is_expr(e) and not is_expr(ents[-1])):
                    continue
                raise ValueError(
                    f"jno.shape: the faces of {whos[-1]!r} and {w!r} share the plane {'xyz'[a]} = {r:.9g} "
                    "at the reference, and one moves differently from the other "
                    f"({'an expression' if is_expr(e) else 'fixed'} against "
                    f"{'an expression' if is_expr(ents[-1]) else 'fixed'}). A conforming mesh cannot split "
                    "along that plane: give both faces the SAME expression, or start them apart."
                )
            refs.append(r)
            ents.append(e)
            whos.append(w)
        axes.append((np.asarray(refs), ents))
    return axes


def stretch_column(ref_col, refs, entries, values=None):
    """The moved coordinates of points at ``ref_col`` along one axis: each plane where its entry puts
    it, every point between two planes at its reference fraction of the gap (``jnp.interp``, so
    differentiable in the plane positions). Points outside the outermost planes shift with them."""
    pos = stack(list(entries), values)
    ref_col = jnp.asarray(ref_col)
    inside = jnp.interp(ref_col, jnp.asarray(refs), pos)
    below = ref_col + (pos[0] - refs[0])
    above = ref_col + (pos[-1] - refs[-1])
    return jnp.where(ref_col < refs[0], below, jnp.where(ref_col > refs[-1], above, inside))


def moving_ids(ref_col, refs, entries, tol=1e-12):
    """The points a design can move along this axis: those strictly inside an interval next to a
    moving plane, and on a moving plane itself. Points on fixed planes, and between two fixed ones,
    stay where they are for every design."""
    span = max(1.0, float(np.ptp(refs)))
    moving = np.array([is_expr(e) for e in entries])
    x = np.asarray(ref_col)
    k = np.clip(np.searchsorted(refs, x, side="right") - 1, 0, len(refs) - 1)
    on = np.abs(x - refs[k]) <= tol * span
    nxt = np.clip(k + 1, 0, len(refs) - 1)
    mv = np.where(on, moving[k], moving[k] | moving[nxt])
    mv |= (x < refs[0]) & moving[0]
    mv |= (x > refs[-1]) & moving[-1]
    return np.flatnonzero(mv)


def unfollowed(plan):
    """Region names (or ``None`` for an unnamed plan) whose parametric coordinates a stretched mesh
    does NOT follow: a line's vertices and radius. Its PEEC filaments move with them; its mesh would
    stay at the reference, so a mesh-based solver refuses it rather than answer for the wrong wire."""
    out = []

    def walk(node, where):
        kind = node[0]
        if kind == "leaf":
            if len(node) > 3 and type(node[1]).__name__ == "Line":
                out.append(where)
        elif kind == "regions":
            for name, sub in node[1]:
                walk(sub._node, name)
        else:
            for c in node[1:]:
                if hasattr(c, "_node"):
                    walk(c._node, where)

    walk(plan._node, getattr(plan, "_region_name", None))
    return out


def refuse_if_parametric(domain, solver, why):
    """Raise for a solver that does not follow a parametric plan yet (it would answer for the reference)."""
    plan = getattr(domain, "_shape_plan", None) or getattr(domain, "_lazy_plan", None)
    if plan is not None and getattr(plan, "_node", None) is not None and has_params(plan._node):
        raise NotImplementedError(
            f"{solver}: this domain's shape has `jno.np.parameter`s in its coordinates, and {solver} does "
            f"not move with them yet ({why}), so it would solve the reference geometry whatever the "
            "design. Use jno.fem or jno.peec for a parametric geometry, or build the shape with numbers."
        )


def plan_parameters(plan):
    """``{name: ModelCall}`` for every parameter in a plan's coordinates."""
    exprs = []

    def walk(node):
        kind = node[0]
        if kind == "leaf":
            exprs.extend(e for _i, e in (node[3] if len(node) > 3 else ()))
        elif kind == "regions":
            for _n, sub in node[1]:
                walk(sub._node)
        else:
            if kind in ("translate", "extrude") and len(node) > 3:
                tail = node[3]
                exprs.extend(e for _i, e in tail) if isinstance(tail, tuple) else exprs.append(tail)
            for c in node[1:]:
                if hasattr(c, "_node"):
                    walk(c._node)

    walk(plan._node)
    return parameters_in(exprs)
