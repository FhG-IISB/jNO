"""Mapped regions: ``.attach(map=f)`` makes a region of the mesh stand for another region of space.

``f(x1', x2'[, x3']) -> (x1, x2[, x3])`` maps the region's mesh coordinates onto physical coordinates; it is
written with ``jno.np`` (or plain arithmetic) like any coordinate function. Every weak term integrated over the
region is then evaluated by the change of variables (J = df/dx', from automatic differentiation of f):

    a coordinate x_i          ->  f_i(x')                     (the physical position)
    a derivative du/dx_i      ->  sum_j (J^-T)_ij du/dx'_j    (the physical gradient)
    the integration measure   ->  det J  x  (mesh measure)

So the equations are written once, for the whole domain, and the mapped region carries them to where f says --
to infinity (an inversion: rho = R^2 / (2R - rho') sends the region's outer edge to infinity), along a stretched
axis, onto a shape. A term written over the whole interior is split into the part outside the mapped regions
(unchanged) and the part inside them (transformed), with the region masks; a term written on the region's own
coordinates is transformed whole.

Requirements, checked: f is the identity where the region meets an unmapped region (otherwise the field would
jump there); det J != 0 inside the region (f may blow up on its outer edge only -- that is how infinity is reached).
Complex maps (a perfectly matched layer, rho~ = rho' + (i/k) int sigma) make the form complex; it then goes through
jNO's real-equivalent complex route. Scope, first version: first derivatives (a second derivative in a mapped region raises); volume terms
(a surface term on a mapped region's own faces raises, Dirichlet values are unchanged); fields whose components
transform as numbers (scalar fields, the azimuthal potential, Cartesian components) -- not edge / face elements,
which need a Piola map.
"""

from __future__ import annotations

import copy
import operator

import jax
import jax.numpy as jnp
import numpy as np

from ..trace import BinaryOp, FunctionCall, Jacobian, Literal, Placeholder, TestFunction, Variable

_BINARY = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "**": operator.pow}


def _maps(domain):
    """{region: f} of the domain's mapped regions."""
    att = (domain.__dict__.get("_region_attachments") or {}).get("map") or {}
    return {r: f for r, f in att.items() if callable(f) and r != getattr(domain, "_DEFAULT_TARGET", None)}


def _coord_eval(e, env):
    """Evaluate a coordinate expression (Literal, BinaryOp, FunctionCall, Variable) on arrays."""
    if not isinstance(e, Placeholder):
        return e
    if isinstance(e, Variable):
        return env[id(e)]
    if isinstance(e, Literal):
        return e.value
    if isinstance(e, BinaryOp):
        f = _BINARY.get(e.op)
        if f is None:
            raise NotImplementedError(f"map=: the operation {e.op!r} is not supported in a coordinate map.")
        return f(_coord_eval(e.left, env), _coord_eval(e.right, env))
    if isinstance(e, FunctionCall):
        return e.fn(*[_coord_eval(a, env) for a in e.args], **(e.kwargs or {}))
    raise NotImplementedError(f"map=: a coordinate map may use arithmetic and jno.np functions, not {type(e).__name__}.")


def _unwrap(e):
    from ..trace.views import _VIEW_TYPES

    while isinstance(e, _VIEW_TYPES):
        e = e._expr
    return e


def _walk(e, visit):
    stack, seen = [_unwrap(e)], set()
    while stack:
        x = stack.pop()
        if id(x) in seen or not isinstance(x, Placeholder):
            continue
        seen.add(id(x))
        visit(x)
        if _is_leaf(x):
            continue  # field symbols carry each other as attributes (a trial's own test function)
        for v in x.__dict__.values():
            v = _unwrap(v)
            if isinstance(v, Placeholder):
                stack.append(v)
            elif isinstance(v, (list, tuple)):
                stack.extend(_unwrap(i) for i in v if isinstance(_unwrap(i), Placeholder))


class _Map:
    """A region's map for one set of coordinate Variables: the physical coordinates X_i as expressions, and the
    entries of J^-T and det J as coordinate functions (FunctionCall nodes the assembler evaluates pointwise)."""

    def __init__(self, f, coords, region):
        self.coords = list(coords)
        X = f(*self.coords)
        X = list(X) if isinstance(X, (list, tuple)) else [X]
        if len(X) != len(self.coords):
            raise ValueError(
                f"map= on {region!r}: f takes {len(self.coords)} coordinates and must return as many, got {len(X)}."
            )
        self.X = [x if isinstance(x, Placeholder) else Literal(x) for x in X]
        ids = [id(c) for c in self.coords]
        Xs = self.X

        def g(p):  # (dim,) -> (dim,), real or complex (a complex map: an absorbing layer)
            env = dict(zip(ids, list(p)))
            vals = [jnp.asarray(_coord_eval(x, env)) for x in Xs]
            dt = jnp.result_type(*vals, p.dtype)
            return jnp.stack([jnp.broadcast_to(v.astype(dt), ()) for v in vals])

        # complex if f produces complex values; the form must then go through jNO's complex route
        self.complex = bool(jnp.iscomplexobj(g(jnp.full(len(ids), 0.5, dtype=jnp.float64))))

        jac = jax.vmap(jax.jacfwd(g, holomorphic=False))
        dim = len(ids)

        def geom(*xs):
            shape = jnp.shape(xs[0])
            p = jnp.stack([jnp.broadcast_to(jnp.asarray(x, jnp.float64), shape).reshape(-1) for x in xs], axis=1)
            J = jac(p)  # (N, dim, dim): J[n, i, j] = d f_i / d x'_j
            return J, shape

        def ginv(i, j):
            def fn(*xs):
                J, shape = geom(*xs)
                return jnp.linalg.inv(jnp.swapaxes(J, 1, 2))[:, i, j].reshape(shape)  # (J^-T)_ij

            return fn

        def det(*xs):
            J, shape = geom(*xs)
            return jnp.linalg.det(J).reshape(shape)

        self.ginv = [
            [FunctionCall(ginv(i, j), self.coords, name=f"map_JinvT_{i}{j}") for j in range(dim)] for i in range(dim)
        ]
        self.det = FunctionCall(det, self.coords, name="map_detJ")
        self.g = g


def _is_leaf(e):
    """Nodes the rewrite must keep as they are: the field symbols (trial and test functions, their components),
    region and tag masks, constants and attached values."""
    from ..trace import AttachedLiteral, RegionMask, TagMask, TestFunction, TrialFunction

    try:
        from ..trace import _FieldComponentIndex
    except ImportError:  # pragma: no cover
        _FieldComponentIndex = ()
    return isinstance(e, (TrialFunction, TestFunction, RegionMask, TagMask, Literal, AttachedLiteral)) or (
        bool(_FieldComponentIndex) and isinstance(e, _FieldComponentIndex)
    )


def _rewrite(e, mp, memo):
    """The expression with coordinates replaced by f(x') and derivatives by the chain rule."""
    e = _unwrap(e)
    if not isinstance(e, Placeholder):
        return e
    if id(e) in memo:
        return memo[id(e)]
    if _is_leaf(e):
        memo[id(e)] = e  # field symbols, masks, constants: never copied (a copied trial function is a new field)
        return e
    if isinstance(e, Variable):
        out = mp.X[mp.coords.index(e)] if e in mp.coords else e
    elif isinstance(e, Jacobian):
        target = _unwrap(e.target)
        nested = []
        _walk(target, lambda x: nested.append(x) if isinstance(x, Jacobian) else None)
        if nested or len(e.variables) != 1:
            raise NotImplementedError(
                "map=: a second derivative in a mapped region needs the map's Hessian; not supported yet."
            )
        v = e.variables[0]
        if v not in mp.coords:
            out = e
        else:
            i = mp.coords.index(v)
            t = _rewrite(target, mp, memo)
            out = None
            for j, c in enumerate(mp.coords):
                part = mp.ginv[i][j] * Jacobian(t, [c], scheme=e.scheme)
                out = part if out is None else out + part
    else:
        out = copy.copy(e)
        changed = False
        for k, v in list(e.__dict__.items()):
            if k.startswith("_auto_op"):
                continue
            uv = _unwrap(v)
            if isinstance(uv, Placeholder):
                nv = _rewrite(uv, mp, memo)
                if nv is not uv:
                    out.__dict__[k] = nv
                    changed = True
            elif isinstance(v, (list, tuple)) and any(isinstance(_unwrap(i), Placeholder) for i in v):
                nl = [_rewrite(i, mp, memo) if isinstance(_unwrap(i), Placeholder) else i for i in v]
                if any(a is not _unwrap(b) for a, b in zip(nl, v)):
                    out.__dict__[k] = type(v)(nl)
                    changed = True
        if not changed:
            out = e
        else:
            out.__dict__.pop("_auto_op", None)
    memo[id(e)] = out
    return out


def _term_coords(term):
    """The term's spatial coordinate Variables, by tag: {tag: [Variable sorted by dimension]} -- from the expression
    and from the coordinates its bound views (``T.bind(r=r, z=z)``) carry."""
    from ..trace.views import _VIEW_TYPES

    found = {}

    def note(x):
        if isinstance(x, Variable) and getattr(x, "axis", "spatial") == "spatial":
            # a jno.fem build retags a term's coordinates in place ("interior" -> "fem_gauss") and keeps the
            # original in `_jno_region_tag`; a Variable reused in a later build must still count as its region
            tag = x.tag
            if isinstance(tag, str) and (tag == "fem_gauss" or tag.startswith("gauss_")):
                tag = getattr(x, "_jno_region_tag", tag)
            found.setdefault(tag, {}).setdefault(x.dim[0], x)

    stack, seen = [term], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, _VIEW_TYPES):
            for c in (getattr(x, "_coord_vars", None) or {}).values():
                note(c)
            stack.append(x._expr)
            continue
        if not isinstance(x, Placeholder):
            continue
        note(x)
        if _is_leaf(x):
            continue
        for v in x.__dict__.values():
            if isinstance(v, (list, tuple)):
                stack.extend(v)
            else:
                stack.append(v)
    return {tag: [vs[k] for k in sorted(vs)] for tag, vs in found.items()}


def _check_identity_on_interfaces(domain, region, f):
    """f must be the identity on the faces the region shares with unmapped regions."""
    from .mesh_utils import mesh_cell_region_membership, p1_cells_dict

    mesh = domain.mesh
    cells = np.asarray(p1_cells_dict(mesh)["triangle" if int(domain.dimension) == 2 else "tetra"])
    regions = mesh_cell_region_membership(mesh, int(domain.dimension))
    maps = _maps(domain)
    inside = np.zeros(len(cells), bool)
    inside |= np.asarray(regions[region], bool)
    others = np.zeros(len(cells), bool)
    for r, m in regions.items():
        if r not in maps:
            others |= np.asarray(m, bool)
    shared = np.intersect1d(np.unique(cells[inside]), np.unique(cells[others]))
    if shared.size == 0:
        return
    pts = np.asarray(mesh.points)[shared, : int(domain.dimension)]
    # f evaluated on plain arrays (jno.np nodes over arrays evaluate directly) -- no domain.variable(...) call,
    # which would register a new coordinate set on the domain each time
    X = f(*[jnp.asarray(pts[:, i]) for i in range(pts.shape[1])])
    X = list(X) if isinstance(X, (list, tuple)) else [X]
    got = np.stack([np.broadcast_to(np.real(np.asarray(_coord_eval(x, {}))), (len(pts),)) for x in X], axis=1)
    err = np.abs(got - pts).max()
    if err > 1e-9 * max(1.0, float(np.abs(pts).max())):
        raise ValueError(
            f"map= on {region!r}: f is not the identity where {region!r} meets the unmapped regions (moves a shared "
            f"node by {err:.3g}), so the field would jump across that interface. A map must leave those faces in place."
        )


def lower_mapped_regions(constraints):
    """Apply every mapped region's change of variables to the weak terms (see the module docstring)."""
    if not isinstance(constraints, (list, tuple)):
        return constraints
    out = []
    checked = set()
    for term in constraints:
        t = _unwrap(term)
        if not isinstance(t, Placeholder):
            out.append(term)
            continue
        tests = []
        _walk(t, lambda x: tests.append(x) if isinstance(x, TestFunction) else None)
        if not tests:
            out.append(term)  # an essential condition: values, not integrals
            continue
        by_tag = _term_coords(term)
        domain = next((v[0]._domain for v in by_tag.values() if v and getattr(v[0], "_domain", None) is not None), None)
        maps = _maps(domain) if domain is not None else {}
        if not maps:
            out.append(term)
            continue
        for region, f in maps.items():
            if (id(domain), region) not in checked:
                _check_identity_on_interfaces(domain, region, f)
                checked.add((id(domain), region))
        dim = int(domain.dimension)
        vol = [tag for tag, vs in by_tag.items() if tag == "interior" or tag in maps]
        if not vol:
            out.append(term)  # a term on another region or a boundary tag
            continue
        if len(vol) > 1:
            raise NotImplementedError(
                f"map=: a term mixing coordinates of {vol} cannot be transformed; write it on one set."
            )
        tag = vol[0]
        coords = by_tag[tag]
        if len(coords) != dim:
            # a term that does not mention every coordinate still needs all of them for the map
            coords = list(domain.variable(tag, split=True))[:dim]
            coords = [c for c in coords]
        from ..trace import RegionMask

        if tag in maps:
            mp = _Map(maps[tag], coords, tag)
            new = _rewrite(t, mp, {}) * mp.det
            out.append(new * Literal(jnp.asarray(1.0 + 0.0j)) if mp.complex else new)
            continue
        outside = t
        mask_sum = None
        mapped = []
        for region, f in maps.items():
            mp = _Map(f, coords, region)
            mask = RegionMask(region)
            part = _rewrite(t, mp, {}) * mp.det * mask
            mapped.append(part * Literal(jnp.asarray(1.0 + 0.0j)) if mp.complex else part)  # flags the form complex
            mask_sum = mask if mask_sum is None else mask_sum + mask
        new = outside * (1.0 - mask_sum)
        for m in mapped:
            new = new + m
        out.append(new)
    return out
