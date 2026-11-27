"""How a domain hands out spectral unknowns and the points of a spectral problem, without a mesh.

``dom.unknown(space="spectral", order=(64, 65))`` reads the box from the domain's shape (a ``jno.shape.rect`` /
``box``, or a 1-D ``jno.domain.line``) and returns a valued nodal field whose values are the nodal values on
its :class:`~jno._spectral.space.SpectralSpace`. ``dom.variable(tag)`` then returns Variables whose tag names a
REGION of a node set: the problem resolves it late (as ``jno.fdm`` resolves its tags to mesh nodes), because
the positions wait for the families the problem's periodic ties fix. The context gets provisional points of the
right count, so everything that only reads shapes keeps working.
"""

from __future__ import annotations

import itertools

import numpy as np

from .space import FAMILIES, SpectralSpace, _split_rule, parse_orders, parse_rules, parse_space

__all__ = [
    "FACE_NAMES",
    "NodeSpec",
    "is_spectral_space",
    "make_spectral_unknown",
    "region_points",
    "spectral_box",
    "spectral_variable",
]

#: face names per dimension, (low, high) per axis -- the names ``jno.shape.rect``/``box`` and the lattices use
FACE_NAMES = {
    1: (("left", "right"),),
    2: (("left", "right"), ("bottom", "top")),
    3: (("left", "right"), ("front", "back"), ("bottom", "top")),
}
_WHOLE = ("interior", "initial")

_TAG_IDS = itertools.count()


def is_spectral_space(space) -> bool:
    names = (space,) if isinstance(space, str) else tuple(space) if isinstance(space, (tuple, list)) else ()
    return bool(names) and all(isinstance(s, str) and _split_rule(s)[0] in ("spectral",) + FAMILIES for s in names)


def spectral_box(domain):
    """``((a0, b0), ...)`` of a box domain: an axis-aligned ``rect``/``box`` plan, or a 1-D line mesh."""
    from ..geometry.primitives import Box, Rect

    plan = domain.__dict__.get("_lazy_plan") or domain.__dict__.get("_shape_plan")
    node = getattr(plan, "_node", None)
    prim = node[1] if (isinstance(node, tuple) and node and node[0] == "leaf") else None
    if isinstance(prim, Rect):
        return ((prim.x0, prim.x1), (prim.y0, prim.y1))
    if isinstance(prim, Box):
        return ((prim.x0, prim.x1), (prim.y0, prim.y1), (prim.z0, prim.z1))
    if int(getattr(domain, "dimension", 0)) == 1 and domain.__dict__.get("_mesh") is not None:
        x = np.asarray(domain.mesh.points)[:, 0]
        return ((float(x.min()), float(x.max())),)
    what = f"a {type(prim).__name__.lower()}" if prim is not None else ("a " + node[0] + " plan" if node else "this domain")
    raise NotImplementedError(
        f"domain.unknown(space='spectral'): a spectral space lives on a box -- jno.shape.rect(...) / box(...) or a "
        f"1-D jno.domain.line(...) -- and this is {what}. Couple a spectral box to an fem/fdm problem on the rest "
        "of the geometry through jno.core([...])."
    )


def make_spectral_unknown(domain, value_shape, name, order, space, *, complex, symmetric, constant, region):
    """The valued nodal field of a spectral unknown: ``(size,) + value_shape`` values, its space on ``.basis``."""
    from ..architectures.models import parameter
    from ..trace import TrialFunction

    for flag, what in ((complex, "complex=True"), (symmetric, "symmetric=True"), (constant, "constant=True")):
        if flag:
            raise NotImplementedError(f"domain.unknown(space='spectral', {what}): not supported for a spectral unknown.")
    if region is not None:
        raise NotImplementedError(
            "domain.unknown(space='spectral', region=...): a spectral unknown lives on the whole box; put the region "
            "on its own domain and couple the two through jno.core([...])."
        )
    box = spectral_box(domain)
    ndim = len(box)
    families = parse_space(space, ndim)
    sp = SpectralSpace(
        box, parse_orders(order, ndim), families=families, value_shape=value_shape, rules=parse_rules(space, ndim)
    )
    trial = TrialFunction(name=name, value_shape=value_shape, order=1, space="spectral")
    trial._domain = domain
    trial._spectral_space = sp
    u = parameter((sp.size,) + tuple(value_shape))
    m = u.model
    m._fem_field = "node"
    m._fem_field_key = trial.field_key
    m._fem_field_domain = domain
    m._unknown_symbol = trial
    m._spectral_space = sp
    domain.__dict__.setdefault("_spectral_unknowns", []).append(u)
    return u


class NodeSpec:
    """A node set derived from a spectral space: other node counts or quadrature rules on some axes.

    ``jno.sp.gauss(y=31)`` -- the Gauss nodes, 31 of them, on axis ``y``, the other axes as the base space has
    them. Axis names are ``x, y, z`` (or integers). Resolved against a base space (the first spectral unknown
    of the domain, or ``of=U``).
    """

    def __init__(self, rule, of=None, **orders):
        self.rule = rule
        self.of = of
        self.orders = orders

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in self.orders.items())
        return f"jno.sp.{self.rule}({args})"

    def axes(self, ndim):
        names = {"x": 0, "y": 1, "z": 2}
        out = {}
        for k, n in self.orders.items():
            i = names.get(k, k)
            if not isinstance(i, int) or not 0 <= i < ndim:
                raise ValueError(f"{self!r}: unknown axis {k!r} for a {ndim}-D space (use x, y, z).")
            out[i] = int(n)
        return out


class DerivedSpace:
    """A node set on the families of ``base``, with :class:`NodeSpec`'s rule and counts on its axes."""

    def __init__(self, base: SpectralSpace, spec: NodeSpec):
        self.base, self.spec = base, spec
        self.overrides = spec.axes(base.ndim)
        self.ndim = base.ndim

    @property
    def key(self):
        return ("derived", id(self.base), self.spec.rule, tuple(sorted(self.overrides.items())))

    def _grid(self, families):
        from .bases import axis_from_spec
        from .grid import Grid

        axes = []
        for i, (f, n, (a, b)) in enumerate(zip(families, self.base.orders, self.base.box)):
            if i in self.overrides:
                if f == "fourier":
                    if self.spec.rule != "uniform":
                        raise ValueError(f"{self.spec!r}: axis {i} is Fourier; its nodes are equispaced (jno.sp.uniform).")
                    axes.append(axis_from_spec(f, self.overrides[i], a, b))
                else:
                    rule = "lobatto" if self.spec.rule == "uniform" else self.spec.rule
                    axes.append(axis_from_spec(f, self.overrides[i], a, b, rule=rule))
            else:
                axes.append(self.base.grid_for(families).axes[i])
        return Grid(axes)

    @property
    def grid(self):
        return self._grid(self.base.families)

    @property
    def provisional_grid(self):
        return (
            self.base.provisional_grid
            if not self.overrides
            else self._grid(
                self.base.families if self.base.resolved else tuple(f or "chebyshev" for f in self.base.explicit)
            )
        )


def _space_of(nodes):
    from ..trace import ModelCall

    if isinstance(nodes, ModelCall):
        sp = nodes.model.__dict__.get("_spectral_space")
        if sp is None:
            raise TypeError("domain.variable(..., nodes=U): U is not a spectral unknown (dom.unknown(space='spectral')).")
        return sp
    if isinstance(nodes, SpectralSpace):
        return nodes
    return None


def resolve_nodes(domain, nodes):
    """The space a ``nodes=`` argument names (``None`` -> the domain's first spectral unknown's)."""
    sp = _space_of(nodes)
    if sp is not None:
        return sp
    if isinstance(nodes, NodeSpec):
        base = _space_of(nodes.of) if nodes.of is not None else None
        if base is None:
            unknowns = domain.__dict__.get("_spectral_unknowns") or []
            if not unknowns:
                raise ValueError(
                    f"domain.variable(..., nodes={nodes!r}): the domain has no spectral unknown to derive from."
                )
            base = unknowns[0].model._spectral_space
        return DerivedSpace(base, nodes)
    if nodes is None:
        unknowns = domain.__dict__.get("_spectral_unknowns") or []
        return unknowns[0].model._spectral_space if unknowns else None
    raise TypeError(f"domain.variable(..., nodes={nodes!r}): give a spectral unknown or jno.sp.gauss/lobatto/uniform(...).")


def region_points(grid, region):
    """``(points (m, ndim), node_indices or None, normals or None)`` of a region of a grid.

    ``interior`` / ``initial``: every node. A face: its nodes, with the outward normal; on a Fourier axis the high
    face is the image of the low one: its points are the low face's shifted by the period, its node indices the
    low face's. ``boundary``: every node on a face of a non-periodic
    axis.
    """
    P = grid.points
    nd = grid.ndim
    if region in _WHOLE:
        return P, np.arange(grid.size), None
    if region == "boundary":
        idx = grid.boundary
        nrm = np.zeros((idx.size, nd))
        for i, a in enumerate(grid.axes):
            if a.periodic:
                continue
            lo, hi = np.isclose(P[idx, i], a.a), np.isclose(P[idx, i], a.b)
            nrm[lo, i] -= 1.0
            nrm[hi, i] += 1.0
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
        return P[idx], idx, nrm
    for i, (lo_name, hi_name) in enumerate(FACE_NAMES.get(nd, ())):
        if region not in (lo_name, hi_name):
            continue
        side = 0 if region == lo_name else 1
        a = grid.axes[i]
        nrm = np.zeros((1, nd))
        nrm[0, i] = 1.0 if side else -1.0
        if a.periodic and side == 1:
            # the image of the low face: its points at x = b, its rows those of the low face's nodes (the same
            # nodal values, by periodicity)
            low = grid.face(i, 0)
            pts = P[low].copy()
            pts[:, i] = a.b
            return pts, low, np.repeat(nrm, len(low), axis=0)
        idx = grid.face(i, side)
        return P[idx], idx, np.repeat(nrm, idx.size, axis=0)
    return None


def spectral_variable(domain, tag, nodes, *, normals=False, split=False, time_value=None):
    """Register a spectral region tag and its provisional points in the context; return the tag the Variables
    carry, or ``None`` when ``tag`` is not a spectral region (a tensor tag, a mesh tag: the normal path)."""
    space = resolve_nodes(domain, nodes)
    if space is None:
        return None
    grid = space.provisional_grid
    found = region_points(grid, tag)
    if found is None:
        if nodes is not None:
            names = sorted(set(_WHOLE) | {"boundary"} | {n for pair in FACE_NAMES.get(grid.ndim, ()) for n in pair})
            raise ValueError(f"domain.variable({tag!r}, nodes=...): a spectral region is one of {names}.")
        return None
    registry = domain.__dict__.setdefault("_spectral_tags", {})
    default = nodes is None or (_space_of(nodes) is not None and _space_of(nodes) is resolve_nodes(domain, None))
    name = tag if default else f"{tag}@sp{next(_TAG_IDS)}"
    registry[name] = (tag, space)
    pts, _idx, nrm = found
    dtype = np.asarray(0.0).dtype
    domain.context[name] = np.asarray(pts, dtype=dtype)[None]
    if nrm is not None:
        domain.context[f"n_{name}"] = np.asarray(nrm, dtype=dtype)[None]
    if tag == "initial" and getattr(domain, "_is_time_dependent", False) and domain.time is not None:
        t0 = float(domain.time[0]) if time_value is None else float(time_value)
        domain.context[f"__time_{name}__"] = np.asarray([[t0]], dtype=dtype)
    return name
