"""``jno.fvm``: a finite-volume discretisation of the strong-form constraint list ``jno.fdm`` reads.

Based on:
  Moukalled, F., Mangani, L., & Darwish, M. (2016). The Finite Volume Method in Computational Fluid
  Dynamics. Springer -- the cell-centred discretisation of diffusion (ch. 8-9) and convection (ch. 11).
  LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems. Cambridge University Press --
  conservation form and numerical fluxes (ch. 4, 12).

How a PDE term is read. Every term is integrated over a control volume. An outermost spatial derivative
``(F).x`` is a divergence: by the divergence theorem it becomes the face sum ``Σ_f F_f S_f,x``, so
``ui.xx + ui.yy`` is ``Σ_f ∇u_f·S_f`` and ``(κ*ui.x).x + (κ*ui.y).y`` is ``Σ_f κ_f ∇u_f·S_f``. A
coefficient that multiplies a whole divergence and does not contain an unknown stays at the volume.
Inside a flux ``F``, a sub-term that holds a derivative of an unknown is evaluated centrally at the face
(face gradient, interpolated values); a sub-term in the unknowns' values alone is convective, and all
such sub-terms of all equations are gathered into one normal flux ``F·n`` that a numerical flux (Rusanov
by default) evaluates on the left and right states. Terms without an outer derivative (``u.t``, a
source, a reaction) are evaluated at the volume centroid (the midpoint rule). A derivative of an
unknown that is not the outermost operator of its term (``b*ui.x``) is evaluated with the least-squares
volume gradient: consistent, but not conservative, which a warning says.

Boundary conditions. A boundary face carries a value ``q_f``. With the owner value carried to the foot of
the face normal, ``q_P'``, the face's normal derivative is ``(q_f - q_P')/d⊥``. Every condition is one
equation in ``q_f`` per face (Dirichlet ``u - g``, Neumann ``u.d(n) - h``, Robin, any nonlinear flux
law), solved by a per-face Newton step; a face without a condition has zero normal derivative.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .. import precond as jno_precond
from .. import solve as _solve
from .._strong_form import (
    StrongFormProblem,
    _contains_unknown,
    _context_leaf,
    _find_unknowns,
    _iter,
    _rtol,
    _set_temporal,
    _spatial_tags,
    _unwrap,
    _zero_temporal,
)
from .geometry import cell_graph, dual_graph
from .ops import FVOps
from .scheme import FVScheme

__all__ = ["fvm"]

_NEWTON_STEPS = 12  # per-face Newton for a nonlinear boundary condition (one step when it is affine)
_BOUNDARY_PASSES = 6  # boundary values and volume gradients, alternated to their fixed point on skewed cells


# -- the trace ------------------------------------------------------------------------------------------


def _concrete_literal(node):
    """A ``Literal`` with a concrete value (not a value traced by ``jax.grad``, which stays a coefficient)."""
    from ..trace import Literal

    return isinstance(node, Literal) and not isinstance(node.value, jax.core.Tracer)


def _mentions(node, model):
    """Does ``node`` contain the unknown ``model``, inside a time derivative ``u.t`` too (the generic walk does
    not enter a ``TemporalDerivative``'s target)?"""
    from ..trace import TemporalDerivative

    n = _unwrap(node)
    if isinstance(n, TemporalDerivative):
        return _mentions(n.target, model)
    return _contains_unknown(n, model) or any(_mentions(c, model) for c in _iter(n))


def _is_spatial(v):
    tag = str(getattr(v, "tag", ""))
    return getattr(v, "axis", "spatial") == "spatial" and not tag.startswith("n_") and tag not in ("cell_size",)


def _is_normal(v):
    return str(getattr(v, "tag", "")).startswith("n_")


def _signed_terms(node, sign=1.0, distribute=True):
    """``node`` as a signed sum ``[(±1, term), ...]`` over ``+``, ``-`` and a literal factor ``-1``."""
    from ..trace import BinaryOp

    n = _unwrap(node)
    if isinstance(n, BinaryOp) and n.op in ("+", "-"):
        right = sign if n.op == "+" else -sign
        return _signed_terms(n.left, sign, distribute) + _signed_terms(n.right, right, distribute)
    if isinstance(n, BinaryOp) and n.op == "*" and _concrete_literal(n.left) and np.ndim(n.left.value) == 0:
        if float(n.left.value) == -1.0:
            return _signed_terms(n.right, -sign, distribute)
    if not distribute:
        return [(sign, n)]
    # a product or quotient distributes over a sum, so `nu*(ui.xx + ui.yy)` is two divergence terms
    if isinstance(n, BinaryOp) and n.op == "*":
        left, right = _signed_terms(n.left), _signed_terms(n.right)
        if len(left) > 1 or len(right) > 1:
            return [(sign * sl * sr, BinaryOp("*", tl, tr)) for sl, tl in left for sr, tr in right]
    if isinstance(n, BinaryOp) and n.op == "/":
        top = _signed_terms(n.left)
        if len(top) > 1:
            return [(sign * st, BinaryOp("/", tt, n.right)) for st, tt in top]
    return [(sign, n)]


def _factors(node):
    """The factors of a product, ``a*b/c`` -> ``[a, b, 1/c]``."""
    from ..trace import BinaryOp, Literal

    n = _unwrap(node)
    if isinstance(n, BinaryOp) and n.op == "*":
        return _factors(n.left) + _factors(n.right)
    if isinstance(n, BinaryOp) and n.op == "/":
        return _factors(n.left) + [BinaryOp("/", Literal(1.0), n.right)]
    return [n]


def _product(factors):
    from ..trace import BinaryOp, Literal

    if not factors:
        return Literal(1.0)
    out = factors[0]
    for f in factors[1:]:
        out = BinaryOp("*", out, f)
    return out


def _signed_sum(terms):
    from ..trace import BinaryOp, Literal

    if not terms:
        return None
    sign0, out = terms[0]
    out = out if sign0 > 0 else BinaryOp("*", Literal(-1.0), out)
    for sign, t in terms[1:]:
        out = BinaryOp("+" if sign > 0 else "-", out, t)
    return out


def _is_divergence(node):
    """An outer spatial derivative: ``Jacobian(T, [x_k])`` or a ``Hessian`` over spatial variables."""
    from ..trace import Hessian, Jacobian

    n = _unwrap(node)
    if isinstance(n, Jacobian):
        return len(n.variables) == 1 and _is_spatial(n.variables[0])
    if isinstance(n, Hessian):
        return all(_is_spatial(v) for v in n.variables) and (n.trace or len(n.variables) == 2)
    return False


def _divergence_parts(node):
    """``[(axis, F)]`` with ``node = Σ ∂F/∂x_axis``: a first derivative is one part, a Laplacian
    ``Hessian(T, vars, trace)`` one part ``(v, ∂T/∂v)`` per variable, a mixed ``∂²T/∂a∂b`` the part
    ``(a, ∂T/∂b)``."""
    from ..trace import Jacobian

    n = _unwrap(node)
    if isinstance(n, Jacobian):
        return [(int(n.variables[0].dim[0]), n.target)]
    if n.trace:
        return [(int(v.dim[0]), Jacobian(n.target, [v], n.scheme)) for v in n.variables]
    a, b = n.variables
    return [(int(a.dim[0]), Jacobian(n.target, [b], n.scheme))]


def _expand_hessians(node):
    """Every ``Hessian`` as nested first derivatives, so a second derivative inside a term is a gradient of
    a gradient (the volume and face gradients are first-order operators)."""
    from .._strong_form import _map_children
    from ..trace import Hessian, Jacobian

    n = _unwrap(node)
    if isinstance(n, Hessian):
        target = _expand_hessians(n.target)
        if n.trace:
            parts = [Jacobian(Jacobian(target, [v], n.scheme), [v], n.scheme) for v in n.variables]
            return _signed_sum([(1.0, p) for p in parts])
        if len(n.variables) == 2:
            a, b = n.variables
            return Jacobian(Jacobian(target, [b], n.scheme), [a], n.scheme)
        raise NotImplementedError("jno.fvm([...]): a full Hessian matrix is not supported; write its entries.")
    return _map_children(n, _expand_hessians)


class _FaceMean:
    """Marker: ``expr`` taken at an interior face as the arithmetic or harmonic mean of its two cell values."""

    def __init__(self, expr, mode):
        self.expr, self.mode = expr, mode

    def __mul__(self, other):
        from ..trace import BinaryOp

        return BinaryOp("*", _AsPlaceholder(self), other)


def _AsPlaceholder(marker):
    """A ``FunctionCall`` node that carries the marker (never evaluated: the evaluator replaces it)."""
    from ..trace import FunctionCall

    node = FunctionCall(lambda: None, [], "fvm_face_mean")
    node._fvm_face_mean = marker
    return node


def _face_mean(sub, mode):
    """``coef * grad`` with the coefficient factors (those without a derivative) under a face mean."""
    from ..trace import BinaryOp

    factors = _factors(sub)
    coef = [f for f in factors if not _has_derivative(f)]
    rest = [f for f in factors if _has_derivative(f)]
    if not coef or not rest:
        return sub
    return BinaryOp("*", _AsPlaceholder(_FaceMean(_product(coef), mode)), _product(rest))


def _has_derivative(node):
    from ..trace import Hessian, Jacobian

    n = _unwrap(node)
    if isinstance(n, (Jacobian, Hessian)):
        return True
    return any(_has_derivative(c) for c in _iter(n))


# -- the problem ----------------------------------------------------------------------------------------


class _TraceFVM(StrongFormProblem):
    """A finite-volume problem authored as a strong-form constraint list (see :func:`fvm`)."""

    _who = "jno.fvm"

    def __init__(self, constraints, centering="cell"):
        if centering not in ("cell", "vertex"):
            raise ValueError(f"jno.fvm([...], centering={centering!r}): use 'cell' or 'vertex'.")
        self.centering = centering
        self._constraints = list(constraints)
        self.unknowns = _find_unknowns(constraints, "jno.fvm")
        self.unknown = self.unknowns[0]
        self.domain = self.unknown._fem_field_domain
        from ..geometry.parametric import refuse_if_parametric

        refuse_if_parametric(self.domain, "jno.fvm", "its control volumes are built from the mesh as constants")
        dim = int(self.domain.dimension)
        cells, kind = self.domain._cells_topo()
        self._mesh_points = np.asarray(self.domain.mesh_connectivity["points"])[:, :dim]
        build = cell_graph if centering == "cell" else dual_graph
        self.graph = build(self._mesh_points, cells, kind)
        self._N = self.graph.n
        self._ncomp = [int(np.prod(np.shape(w.module.value)[1:], dtype=int)) for w in self.unknowns]
        self._block0 = [int(b) for b in np.cumsum([0] + self._ncomp)[:-1]]
        self._nf = int(sum(self._ncomp))
        self._Ntot = self._nf * self._N
        self._classify(constraints)
        joined = set()
        for tie in self._periodic_ties:  # one pairing per pair of regions, whichever unknowns the ties name
            key = frozenset(tie)
            if key in joined:
                continue
            joined.add(key)
            fa, fb = (self.graph.boundary_faces_on(self._region_mesh_nodes(t)) for t in tie)
            shift = self.graph.b_centroid[fa].mean(axis=0) - self.graph.b_centroid[fb].mean(axis=0)
            self.graph = self.graph.with_periodic(fa, fb, shift)
        self.ops = FVOps(self.graph)
        self._pts = self.ops.centroid
        self.region, self.region_geometry = self._pde_region()
        if self.region is not None:
            raise NotImplementedError("jno.fvm([...]): a PDE on a sub-region is not supported yet.")
        self._saddle = self._saddle_pairs()  # {constraint equation k: the unknown j its fluxes carry}
        self._equations = [self._split_equation(e) for e in self._pde_exprs()]
        for k, j in self._saddle.items():
            eq = self._equations[k]
            eq["mass"] = {a: F for g in eq["groups"] for a, F in g["convective"].items()}
            eq["partner"] = j
            for g in eq["groups"]:
                g["convective"] = {}
            if eq["scheme"] is not None:
                eq["scheme"] = None
        self._conditions = self._boundary_conditions()
        self._pins = self._pin_rows()

    # -- regions --------------------------------------------------------------------------------------
    def _region_mesh_nodes(self, tag):
        """Mesh node ids of a region tag (the nodes its boundary faces or pinned cells are found from)."""
        if tag in (None, "boundary"):
            return np.asarray(self.domain.mesh_connectivity["boundary_indices"], dtype=int)
        reg = getattr(self.domain, "_boundary_registry", {}).get(tag)
        if reg is not None and len(reg.get("point_indices", [])) > 0:
            return np.asarray(reg["point_indices"], dtype=int)
        facets = self.domain.tag_facets(tag)
        if facets is not None:
            return np.unique(np.asarray(facets).ravel())
        pool = getattr(self.domain, "_mesh_pool", {}).get(tag)
        if pool is not None:
            from scipy.spatial import cKDTree

            pts = np.unique(np.asarray(pool).reshape(-1, np.asarray(pool).shape[-1])[:, : self.graph.dim], axis=0)
            dist, nodes = cKDTree(self._mesh_points).query(pts)
            if len(pts) and np.all(dist <= _rtol() * float(np.ptp(self._mesh_points, axis=0).max())):
                return np.unique(nodes).astype(int)
        raise ValueError(
            f"jno.fvm([...]): no mesh nodes found for region {tag!r}. Use a boundary tag of the domain, "
            "'boundary', or a `domain.point_region(name, xy)` node."
        )

    def _region_faces(self, tag):
        """Boundary face indices of a region: exact facet matches where the tag stores facets, else the faces
        whose nodes all lie on the region."""
        if tag in (None, "boundary"):
            return np.arange(self.graph.b_owner.shape[0])
        facets = self.domain.tag_facets(tag)
        if facets is not None and np.shape(facets)[1] == self.graph.b_nodes.shape[1]:
            keys = {tuple(r) for r in np.sort(np.asarray(facets), axis=1)}
            mine = np.sort(self.graph.b_nodes, axis=1)
            return np.flatnonzero([tuple(r) in keys for r in mine])
        return self.graph.boundary_faces_on(self._region_mesh_nodes(tag))

    # -- the PDE as volume terms and fluxes ----------------------------------------------------------
    def _saddle_pairs(self):
        """``{k: j}`` for every constraint equation: equation ``k`` holds no term in its own unknown ``k`` but
        carries a flux of unknown ``j`` (``div U = 0`` for the pressure ``p``). Unknown ``k`` is then a Lagrange
        multiplier of that constraint. Its equation gets a central flux with a momentum-weighted (Rhie-Chow)
        correction instead of a Riemann flux, and the gradient of ``k`` elsewhere is read in Green-Gauss form."""
        if len(self.unknowns) < 2:
            return {}
        out = {}
        for k, e in enumerate(self._pde_exprs()):
            if _mentions(e, self.unknowns[k]):
                continue
            others = [j for j, w in enumerate(self.unknowns) if j != k and _mentions(e, w)]
            if len(others) != 1:
                raise ValueError(
                    f"jno.fvm([...]): equation {k} has no term in its own unknown, so it is a constraint, but it "
                    f"involves {len(others)} other unknowns; a constraint equation must carry the flux of one."
                )
            out[k] = others[0]
        return out

    def _is_multiplier(self, node):
        """Is ``node`` a bare constraint unknown (a pressure)?"""
        from ..trace import ModelCall

        n = _unwrap(node)
        return isinstance(n, ModelCall) and any(n.model is self.unknowns[k] for k in getattr(self, "_saddle", {}))

    def _has_noncons_derivative(self, node):
        """A derivative of an unknown inside a volume term, other than the gradient of a multiplier (that one is
        read in Green-Gauss form, which is the divergence of ``p I`` and so conservative)."""
        from ..trace import Hessian, Jacobian

        n = _unwrap(node)
        if isinstance(n, (Jacobian, Hessian)) and self._has_unknown(n.target):
            return not (isinstance(n, Jacobian) and self._is_multiplier(n.target))
        return any(self._has_noncons_derivative(c) for c in _iter(n))

    def _pde_exprs(self):
        if len(self.unknowns) == 1:
            expr = self._pde[0]
            for c in self._pde[1:]:
                expr = expr + c
            return [_unwrap(expr)]
        return [_unwrap(self._pde[k]) for k in range(len(self.unknowns))]

    def _has_unknown(self, node):
        return any(_mentions(node, u) for u in self.unknowns)

    def _has_unknown_derivative(self, node):
        from ..trace import Hessian, Jacobian

        n = _unwrap(node)
        if isinstance(n, (Jacobian, Hessian)) and self._has_unknown(n.target):
            return True
        return any(self._has_unknown_derivative(c) for c in _iter(n))

    def _split_equation(self, expr):
        """``{"volume": expr | None, "groups": [...], "nonconservative": bool}`` of one equation.

        A group is the fluxes under one volume coefficient: ``{"coef": node | None, "central": {axis:
        expr}, "convective": {axis: expr}}``."""
        from ..trace import BinaryOp, Literal

        volume, groups, noncons, schemes = [], {}, False, set()
        for sign, term in _signed_terms(expr):
            factors = _factors(term)
            divs = [f for f in factors if _is_divergence(f)]
            rest = [f for f in factors if not _is_divergence(f)]
            if len(divs) == 1 and not any(self._has_unknown(f) for f in rest):
                literal = [f for f in rest if _concrete_literal(f)]  # a traced number is a coefficient
                coef = [f for f in rest if not _concrete_literal(f)]
                scale = sign * float(np.prod([float(np.asarray(f.value)) for f in literal])) if literal else sign
                key = id(coef[0]) if len(coef) == 1 else (None if not coef else tuple(id(c) for c in coef))
                grp = groups.setdefault(key, {"coef": _product(coef) if coef else None, "central": {}, "convective": {}})
                scheme = getattr(_unwrap(divs[0]), "scheme", None)
                for axis, F in _divergence_parts(divs[0]):
                    pieces = []
                    for s1, whole in _signed_terms(F, distribute=False):
                        whole = _expand_hessians(whole)
                        if (
                            isinstance(scheme, FVScheme)
                            and scheme.average in ("arithmetic", "harmonic")
                            and self._has_unknown_derivative(whole)
                        ):
                            whole = _face_mean(whole, scheme.average)  # the whole coefficient, before distributing
                        pieces += [(s1 * s2, sub) for s2, sub in _signed_terms(whole)]
                    for s2, sub in pieces:
                        kind = "convective" if self._has_unknown(sub) and not _has_derivative(sub) else "central"
                        if kind == "convective":
                            schemes.add(scheme if isinstance(scheme, FVScheme) else None)
                        mag = abs(scale)
                        if mag != 1.0:
                            sub = BinaryOp("*", Literal(mag), sub)
                        grp[kind].setdefault(axis, []).append((np.sign(scale) * s2, sub))
                continue
            if self._has_noncons_derivative(term):
                noncons = True
            volume.append((sign, _expand_hessians(term)))
        if noncons:
            warnings.warn(
                "jno.fvm([...]): a derivative of an unknown is not the outer operator of its term (e.g. `b*ui.x`), so "
                "it is evaluated with the volume gradient: consistent, but not conservative -- a shock or a "
                "discontinuity moves at the wrong speed. Write it in divergence form, e.g. `(b*ui).x` or "
                "`(ui**2/2).x`.",
                UserWarning,
                stacklevel=4,
            )
        out = []
        for grp in groups.values():
            out.append(
                {
                    "coef": grp["coef"],
                    "central": {a: _signed_sum(t) for a, t in grp["central"].items()},
                    "convective": {a: _signed_sum(t) for a, t in grp["convective"].items()},
                }
            )
        if any(g["convective"] and g["coef"] is not None for g in out):
            raise NotImplementedError(
                "jno.fvm([...]): a convective flux under a variable coefficient (`c(x)*(b*u).x`) is not "
                "supported; move the coefficient inside the derivative."
            )
        if len(schemes) > 1:
            raise ValueError(
                "jno.fvm([...]): the convective terms of one equation carry different schemes. They form one normal "
                "flux F·n, so give every axis the same `scheme=jno.fv(...)` (or none)."
            )
        return {"volume": _signed_sum(volume), "groups": out, "scheme": next(iter(schemes)) if schemes else None}

    def _convective_scheme(self):
        """The one scheme of the system's convective fluxes (they are solved together): an equation's
        ``jno.fv(...)``, the default where none is given."""
        given = {e["scheme"] for e in self._equations if e["scheme"] is not None}
        if len(given) > 1:
            raise ValueError(
                "jno.fvm([...]): the equations of a system carry different `jno.fv(...)` schemes; their convective "
                "fluxes go through one numerical flux, so give them the same one."
            )
        return next(iter(given)) if given else FVScheme()

    def _field_of(self, constraint):
        """The unknown a condition constrains: the one it mentions."""
        ks = [k for k, u in enumerate(self.unknowns) if _mentions(constraint, u)]
        if len(ks) != 1:
            raise ValueError(
                f"jno.fvm([...]): a boundary condition or pin must mention exactly one unknown; this one mentions "
                f"{len(ks)}."
            )
        return ks[0]

    def _condition_region(self, c):
        normals = {str(v.tag)[2:] for v in _iter_vars(c) if _is_normal(v)}
        tags = normals or {t for t in _spatial_tags(c)}
        tags.discard(None)
        if len(tags) != 1:
            raise ValueError(
                f"jno.fvm([...]): a condition must live on one region; this one uses {sorted(map(str, tags))}."
            )
        return next(iter(tags))

    def _boundary_conditions(self):
        """``[(field, face_ids, expr)]``: every condition that lives on boundary faces."""
        out, claimed = [], {}
        vertex = self.centering == "vertex"
        for c in self._neumann if vertex else self._dirichlet + self._neumann:  # vertex: values are strong rows
            tag = self._condition_region(c)
            faces = self._region_faces(tag) if self._has_boundary_faces(tag) else None
            if faces is None:
                continue  # a pin on cells (see _pin_rows)
            k = self._field_of(c)
            seen = claimed.setdefault(k, np.zeros(self.graph.b_owner.shape[0], dtype=bool))
            if seen[faces].any():
                raise ValueError(
                    f"jno.fvm([...]): two conditions on the same boundary faces of unknown {k} (region {tag!r}). "
                    "Give each face one condition per unknown."
                )
            seen[faces] = True
            out.append((k, jnp.asarray(faces, dtype=jnp.int32), _expand_hessians(_unwrap(c))))
        return out

    def _has_boundary_faces(self, tag):
        try:
            return self._region_faces(tag).size > 0
        except ValueError:
            return False

    def _pin_rows(self):
        """``[(field, cells, expr)]``: value conditions on regions without boundary faces (a point region,
        a gauge): they replace the equations of the cells nearest to the region's nodes."""
        from scipy.spatial import cKDTree

        out = []
        for c in self._dirichlet:
            tag = self._condition_region(c)
            if self.centering == "vertex":  # a value on a boundary region holds at its nodes (as in jno.fdm)
                out.append((self._field_of(c), jnp.asarray(self._region_mesh_nodes(tag), dtype=jnp.int32), _unwrap(c)))
                continue
            if self._has_boundary_faces(tag):
                continue
            nodes = self._region_mesh_nodes(tag)
            _, cells = cKDTree(self.graph.centroid).query(self._mesh_points[nodes])
            out.append((self._field_of(c), jnp.asarray(np.unique(cells), dtype=jnp.int32), _unwrap(c)))
        return out

    # -- evaluation -----------------------------------------------------------------------------------
    def _split_dofs(self, dofs):
        q = jnp.asarray(dofs).reshape(self._nf, self._N).T  # (n, nf)
        return [q[:, b : b + c] for b, c in zip(self._block0, self._ncomp)]

    def _evaluator(self, q, qb, t, scope):
        return _Evaluator(self, q, qb, t, scope)

    def _residual_fn(self, *, spatial=True, extra_params=None, first_order=False):
        """The finite-volume residual ``R(dofs)`` (blocked DOF layout), volume terms integrated (``V·term``)
        and fluxes summed over faces. ``spatial=True`` drops the ``u.t`` terms."""
        eqs = self._equations
        scope = self._params_scope(extra_params)
        nB = self.graph.b_owner.shape[0]
        self._decide_condition_affinity(self._params_scope(self._current_params()))  # decided on concrete values

        def residual(dofs, t=None):
            q = self._split_dofs(dofs)
            qb = [qk[self.ops.b_owner] if nB else jnp.zeros((0, qk.shape[1]), qk.dtype) for qk in q]
            ev = self._evaluator(q, qb, t, scope)
            ev.first_order = first_order
            for i in range(self.ops.boundary_passes):
                ev.solve_boundary(first=i == 0)
            out = []
            conv = ev.convective_divergence(eqs)  # (n, nf) or None, all equations' convective fluxes
            for k, eq in enumerate(eqs):
                c = self._ncomp[k] if len(eqs) > 1 else self._nf
                cols = slice(self._block0[k], self._block0[k] + c) if len(eqs) > 1 else slice(0, self._nf)
                r = jnp.zeros((self._N, c), dtype=jnp.asarray(dofs).dtype)
                if eq["volume"] is not None:
                    r = r + self.ops.volume[:, None] * ev.rows(ev.values(_zero_temporal(eq["volume"]), "cells"), c, "cells")
                for grp in eq["groups"]:
                    if grp["central"]:
                        fi, fb = ev.central_flux(grp["central"], c)
                        div = self.ops.divergence(fi, fb)
                        if grp["coef"] is not None:
                            div = div * ev.rows(ev.values(grp["coef"], "cells"), 1, "cells")
                        r = r + div
                if conv is not None:
                    r = r + conv[:, cols]
                if eq.get("mass"):
                    r = r + self.ops.divergence(*ev.mass_flux(eq["mass"], eq["partner"], k, c))
                out.append(r)
            R = jnp.concatenate(out, axis=1)  # (n, nf)
            for k, cells, expr in self._pins:
                cols = slice(self._block0[k], self._block0[k] + self._ncomp[k])
                R = R.at[cells, cols].set(ev.rows(ev.values(expr, ("cells", cells)), self._ncomp[k], ("cells", cells)))
            R = self._require_real(R, "the finite-volume residual (a complex coefficient or source)")
            return R.T.reshape(-1)

        eq_expr = self._equilibrium()
        if eq_expr is None:
            return residual
        keep = np.ones(self._Ntot, dtype=bool)
        for k, cells, _ in self._pins:
            for b in range(self._block0[k], self._block0[k] + self._ncomp[k]):
                keep[b * self._N + np.asarray(cells)] = False
        keep = jnp.asarray(keep)

        def well_balanced(dofs, t=None):
            """``R(q) − R(q_eq)``: the discrete residual of the equilibrium is removed, so it is a steady state of
            the scheme to round-off (Berberich, Chandrashekar & Klingenberg 2021); pinned rows are left alone."""
            q = [jnp.zeros((self._N, c), jnp.asarray(dofs).dtype) for c in self._ncomp]
            qb = [jnp.zeros((nB, c), jnp.asarray(dofs).dtype) for c in self._ncomp]
            ev = self._evaluator(q, qb, t, scope)
            q_eq = ev.rows(ev.values(eq_expr, "cells"), self._nf, "cells").T.reshape(-1)
            return residual(dofs, t) - jnp.where(keep, residual(q_eq, t), 0.0)

        return well_balanced

    def _equilibrium(self):
        """The ``equilibrium=`` state of the convective scheme, if one was given."""
        if not any(g["convective"] for eq in self._equations for g in eq["groups"]):
            return None
        return self._convective_scheme().equilibrium

    def _decide_condition_affinity(self, scope):
        """Which boundary conditions are affine in their face value (one Newton step solves them exactly),
        decided once on concrete values: the jvp of the condition at two face states must agree."""
        cache = self.__dict__.setdefault("_condition_affine", {})
        self.__dict__.setdefault("_condition_partial", {})
        if "_mirror" not in self.__dict__:
            self._mirror = jnp.zeros((self.graph.b_owner.shape[0], self._nf), dtype=bool)
        todo = [(k, f, e) for k, f, e in self._conditions if id(e) not in cache]
        if not todo:
            return
        with jax.ensure_compile_time_eval():
            q = [jnp.full((self._N, c), 0.3) for c in self._ncomp]
            qb = [jnp.full((self.graph.b_owner.shape[0], c), 0.3) for c in self._ncomp]
            ev = self._evaluator(q, qb, None, scope)
            for k, faces, expr in todo:
                q0 = jnp.zeros((int(faces.shape[0]), self._ncomp[k]))
                tangent = jnp.ones_like(q0)
                j0 = jax.jvp(lambda z: ev._condition(k, faces, expr, z), (q0,), (tangent,))[1]
                j1 = jax.jvp(lambda z: ev._condition(k, faces, expr, z), (q0 + 0.71,), (tangent,))[1]
                cache[id(expr)] = bool(jnp.all(jnp.abs(j1 - j0) <= _rtol(j0.dtype) * (float(jnp.abs(j0).max()) + 1.0)))
                partial = int(j0.shape[1]) < self._ncomp[k]
                self._condition_partial[id(expr)] = partial
                if partial:  # the outer Riemann state of these faces is the mirror image of the inner one
                    cols = slice(self._block0[k], self._block0[k] + self._ncomp[k])
                    self._mirror = self._mirror.at[faces, cols].set(True)

    # -- solving --------------------------------------------------------------------------------------
    def _pattern(self, key, fun, u):
        """``(pattern, colour, n_colours)`` of ``fun``'s Jacobian on the face graph, verified like fdm's."""
        cache = self.__dict__.setdefault("_sparsity_cache", {})
        if key in cache:
            return cache[key]
        from ..fdm import _assemble_sparse, _color_columns, _stencil_pattern

        pairs = np.stack([self.graph.owner, self.graph.neigh], axis=1)
        if isinstance(u, jax.core.Tracer):  # decided on concrete values: a traced probe point becomes zeros
            with jax.ensure_compile_time_eval():
                u = jnp.zeros(u.shape, u.dtype)
        n = int(np.prod(jnp.shape(u)))
        rng = np.random.default_rng(0)
        v = jnp.asarray(rng.standard_normal(n), dtype=jnp.asarray(u).dtype)
        # verified at a GENERIC state near `u`: at a piecewise-constant one a limiter's dependence on the second
        # ring vanishes, a too-narrow pattern passes, and the tangents assembled later are wrong
        u = jnp.asarray(u)
        u = u + 0.01 * (jnp.abs(u) + 1.0) * jnp.asarray(rng.standard_normal(n), dtype=u.dtype)
        ref = jax.jvp(fun, (u,), (v,))[1]
        for radius in (1, 2, 3, 4):
            pattern = _stencil_pattern(pairs, self._N, radius, n_fields=n // self._N)
            color, n_colors = _color_columns(pattern)
            A = _assemble_sparse(fun, u, pattern, color, n_colors)
            if float(jnp.linalg.norm(A @ v - ref)) <= _rtol(ref.dtype) * (float(jnp.linalg.norm(ref)) or 1.0):
                cache[key] = (pattern, color, n_colors)
                return cache[key]
        raise ValueError("jno.fvm: could not assemble the residual's Jacobian on the face graph.")

    def _operator(self, key, fun, u):
        from ..fdm import _assemble_sparse

        pattern, color, n_colors = self._pattern(key, fun, u)
        return _assemble_sparse(fun, u, pattern, color, n_colors)

    def _defaults(self, A, linear, precond):
        """``cg`` + ``amg`` for a symmetric operator, ``gmres`` + ``ilu`` otherwise."""
        if linear is not None:
            return linear, precond
        a, b = self._symmetry_probe(A)
        symmetric = abs(a - b) <= _rtol() * max(abs(a), abs(b), np.finfo(float).tiny)
        if symmetric:
            return _solve.cg(tol=1e-12), (precond if precond is not None else jno_precond.amg())
        return _solve.gmres(tol=1e-12), (precond if precond is not None else jno_precond.ilu())

    def solve(self, nonlinear=None, x0=None, *, linear=None, precond=None, time=None, save_ts=None):
        """Solve the problem: the cell values, ``(N,)`` for one scalar unknown, ``(n_fields, N)`` otherwise.

        Args:
            nonlinear: a ``jno.solve`` nonlinear driver; default Newton on the assembled tangent.
            x0: initial guess (blocked DOF vector).
            linear: a ``jno.solve`` linear solver; default ``cg`` (symmetric) or ``gmres``.
            precond: a ``jno.precond`` spec; default ``amg`` (symmetric) or ``ilu``.
        """
        trainable = self._trainable_params()
        if trainable:
            return self._parametric_node(
                trainable, nonlinear=nonlinear, x0=x0, linear=linear, precond=precond, time=time, save_ts=save_ts
            )
        self._record_fixed_networks()
        return self._solve_now(nonlinear=nonlinear, x0=x0, linear=linear, precond=precond, time=time, save_ts=save_ts)

    def _solve_now(self, *, nonlinear=None, x0=None, linear=None, precond=None, time=None, save_ts=None, extra_params=None):
        if self._transient:
            traj = self._march(nonlinear=nonlinear, linear=linear, precond=precond, time=time, save_ts=save_ts)
            return traj if self._nf == 1 else traj.reshape(traj.shape[0], self._nf, self._N)
        u = self._steady_solve(nonlinear=nonlinear, x0=x0, linear=linear, precond=precond, extra_params=extra_params)
        return self._shape(u)

    def _parametric_node(self, trainable, **slots):
        """With a trainable ``jno.np.parameter`` or network in the constraints, the solve is a trace node, as
        ``fem.solve()`` and ``fdm(...).solve()`` are, so it composes into ``jno.core``: at every step the
        parameters resolve to their current values and the solve runs again, differentiated through its
        Newton (``custom_root``) or its march."""
        from ..trace import FunctionCall

        lids = list(trainable)
        args = [self._trainable_arg(trainable[lid]) for lid in lids]

        def _solve(*values):
            extra = {lid: self._trainable_module(trainable[lid], v) for lid, v in zip(lids, values)}
            if not self._transient:
                return self._solve_now(**slots, extra_params=extra)
            self._override = extra
            try:
                return self._solve_now(**slots)
            finally:
                self._override = None

        node = FunctionCall(_solve, args, name="fvm_solve")
        node._domain = self.domain  # jno.core infers the domain from the graph
        return node

    # -- transient ------------------------------------------------------------------------------------
    def _time_coefficients(self, scope):
        """Per-cell coefficients of ``u.t`` and ``u.tt`` in every equation, as functions of time
        ``(c(t), m(t))``, each ``(n, nf)``; probed as ``F(u.t=1) − F(0)``, and refused when they depend on u."""
        exprs = [e["volume"] for e in self._equations]

        def coef(t_val, tt_val, t, state):
            q = [jnp.full((self._N, c), state) for c in self._ncomp]
            qb = [jnp.full((self.graph.b_owner.shape[0], c), state) for c in self._ncomp]
            ev = self._evaluator(q, qb, t, scope)
            cols = []
            for k, e in enumerate(exprs):
                c = self._ncomp[k] if len(exprs) > 1 else self._nf
                if e is None:
                    cols.append(jnp.zeros((self._N, c)))
                    continue
                on = ev.rows(ev.values(_set_temporal(e, t_val, tt_val), "cells"), c, "cells")
                off = ev.rows(ev.values(_set_temporal(e, 0.0, 0.0), "cells"), c, "cells")
                cols.append(on - off)
            return jnp.concatenate(cols, axis=1)

        out = []
        for t_val, tt_val, what in ((1.0, 0.0, "u.t"), (0.0, 1.0, "u.tt")):
            with jax.ensure_compile_time_eval():
                if not self._same(coef(t_val, tt_val, None, 0.0), coef(t_val, tt_val, None, 0.83)):
                    raise ValueError(
                        f"jno.fvm([...]): the coefficient of {what} depends on u; only coordinates and time are supported."
                    )
            out.append(lambda t=None, a=t_val, b=tt_val: coef(a, b, t, 0.0))
        return out

    def _initial_values(self, scope, velocity=False):
        """Initial cell values ``(n, nf)`` from the ``u(xi, yi) - u0`` conditions (or the velocity from
        ``ui0.t - v0``), each solved per cell for its unknown (one Newton step when affine). A field without
        one starts at 0."""
        q0 = [jnp.zeros((self._N, c)) for c in self._ncomp]
        conds = self._vel_ic if velocity else self._ic
        for c in conds:
            k = self._field_of(c)
            expr = _unwrap(c)
            qb = [jnp.zeros((self.graph.b_owner.shape[0], cc)) for cc in self._ncomp]

            def res(z, k=k, expr=expr):
                q = [z if j == k else q0[j] for j in range(len(q0))]
                ev = self._evaluator(q, qb, None, scope)
                if velocity:  # the condition is affine in u.t: c·v + rest
                    e1, e0 = _set_temporal(expr, 1.0), _set_temporal(expr, 0.0)
                    on = ev.rows(ev.values(e1, "cells"), self._ncomp[k], "cells")
                    off = ev.rows(ev.values(e0, "cells"), self._ncomp[k], "cells")
                    return off, on - off
                return ev.rows(ev.values(expr, "cells", {k: z}), self._ncomp[k], "cells"), None

            if velocity:
                off, slope = res(q0[k])
                q0[k] = -off / jnp.where(slope != 0, slope, 1.0)
                continue
            z = q0[k]
            for _ in range(_NEWTON_STEPS):
                r = res(z)[0]
                d = jax.jvp(lambda w: res(w)[0], (z,), (jnp.ones_like(z),))[1]
                z = z - r / jnp.where(d != 0, d, 1.0)
            q0[k] = z
        return jnp.concatenate(q0, axis=1).T.reshape(-1)

    def _cfl_dt(self, residual_first_order, mass_diag):
        """``(state, cfl) -> Δt``: ``2·cfl / ρ`` with ``ρ`` the spectral radius of the scaled first-order
        tangent ``M⁻¹ ∂R/∂u``, by Gelfand's formula on 40 matrix-free products (it converges for complex
        eigenvalue pairs too). For upwinded convection ``ρ = 2a/Δx``, so ``cfl`` is the classical Courant
        number; diffusion and reaction enter the same way. The first-order operator is read because a
        limiter's derivatives would inflate the bound."""

        m = mass_diag(None)
        dyn = m > 0
        f = lambda y: residual_first_order(y, self._start_time(), {})  # noqa: E731
        v0 = jnp.where(dyn, jnp.asarray(np.random.default_rng(3).standard_normal(m.shape[0])), 0.0)

        @jax.jit
        def rho_of(w):
            _, lin = jax.linearize(f, w)

            def body(i, carry):
                v, acc = carry
                v = jnp.where(dyn, lin(v) / jnp.where(dyn, m, 1.0), 0.0)
                nrm = jnp.linalg.norm(v)
                acc = acc + jnp.where(i >= 20, jnp.log(jnp.maximum(nrm, jnp.finfo(v.dtype).tiny)), 0.0)
                return v / jnp.maximum(nrm, jnp.finfo(v.dtype).tiny), acc

            _, acc = jax.lax.fori_loop(0, 40, body, (v0 / jnp.linalg.norm(v0), 0.0))
            return jnp.exp(acc / 20.0)

        def stable(w, cfl):
            rho = float(rho_of(jnp.asarray(w)))
            return 2.0 * float(cfl) / max(rho, np.finfo(float).tiny)

        return stable

    def _march(self, *, nonlinear=None, linear=None, precond=None, time=None, save_ts=None):
        """Method of lines: ``M u̇ + R(u, t) = 0`` with ``M = diag(c·V)`` (zero on pinned cells), marched by
        ``time=`` (backward Euler by default, any ``jno.solve`` scheme, ``jno.solve.erk`` explicitly).
        A ``u.tt`` problem marches ``[u; v]``: ``u̇ = v``, ``m v̇ + c v + R(u) = 0``."""
        import jax.experimental.sparse as jsparse

        from ..utils.solver.backend_blocks import SemidiscreteTimeBlock, _block_time_grid
        from ..utils.solver.solver_api import compose_transient_step_solvers
        from ..utils.solver.time_route import _infer_time_window
        from ..utils.solver.timeschemes import _ERKScheme

        t0, t1, dt = _infer_time_window(self.domain)
        if dt is None:
            raise ValueError("jno.fvm([...]): domain.time must give n_steps >= 2 for a transient march.")
        scope = self._params_scope()
        spatial = self._residual_fn(extra_params=getattr(self, "_override", None))
        spatial_first = self._residual_fn(extra_params=getattr(self, "_override", None), first_order=True)
        c_of, m_of = self._time_coefficients(scope)
        V = jnp.tile(self.ops.volume, self._nf)
        pinned = np.zeros(self._Ntot, dtype=bool)
        for k, cells, _ in self._pins:
            for b in range(self._block0[k], self._block0[k] + self._ncomp[k]):
                pinned[b * self._N + np.asarray(cells)] = True
        pinned = jnp.asarray(pinned)

        def blocked(c):
            return c.T.reshape(-1)

        u0 = self._initial_values(scope)
        second = self._time_order == 2
        n = self._Ntot
        if second:
            v0 = self._initial_values(scope, velocity=True) if self._vel_ic else jnp.zeros(n)
            state0 = jnp.concatenate([u0, v0])

            def mass_diag(t=None):
                return jnp.concatenate([jnp.ones(n), jnp.where(pinned, 0.0, blocked(m_of(t)) * V)])

            def residual(y, t, args):
                u, v = y[:n], y[n:]
                r = spatial(u, t) + jnp.where(pinned, 0.0, blocked(c_of(t)) * V * v)
                return jnp.concatenate([-v, r])

        else:
            state0 = u0

            def mass_diag(t=None):
                return jnp.where(pinned, 0.0, blocked(c_of(t)) * V)

            def residual(y, t, args):
                return spatial(y, t)

        N2 = int(state0.size)
        diag = jnp.stack([jnp.arange(N2), jnp.arange(N2)], axis=1)
        common = dict(state0=state0, t0=float(t0), t1=float(t1), dt=float(dt), metadata={})
        jac = lambda w, t, args: self._operator("march", lambda y: residual(y, t, args), w)  # noqa: E731
        with jax.ensure_compile_time_eval():
            self._pattern("march", lambda y: residual(y, float(t0), {}), state0)  # where the flux is defined
        if second:
            first = lambda y, t, args: jnp.concatenate([-y[n:], spatial_first(y[:n], t)])  # noqa: E731
        else:
            first = lambda y, t, args: spatial_first(y, t)  # noqa: E731
        common["metadata"]["cfl_dt"] = self._cfl_dt(first, mass_diag)
        block = SemidiscreteTimeBlock(
            mass=lambda t, args: jsparse.BCOO((mass_diag(t), diag), shape=(N2, N2)),
            residual=residual,
            jacobian=jac,
            **common,
        )
        from ..utils.solver.segregated import _Segregated

        if isinstance(nonlinear, _Segregated):
            with jax.ensure_compile_time_eval():
                pattern = np.asarray(self._operator("march", lambda y: residual(y, float(t0), {}), state0).indices)
            # the step tangent M/dt + dR/du has the same pattern (M is diagonal, its entries already in it)
            nonlinear = nonlinear.with_constraint(self._constraint_rows(N2), pattern)
        if isinstance(time, _ERKScheme):
            steppers = (None, None)
        else:
            nonlinear = nonlinear if nonlinear is not None else _solve.newton(direct=True)
            linear = linear if linear is not None else _solve.lu()
            steppers = compose_transient_step_solvers(
                nonlinear, linear, precond, self._host(), block, scheme=time, state=state0
            )
        from ..fdm import _integrate_transient

        ts = _block_time_grid(block) if save_ts is None else jnp.asarray(save_ts)
        traj = _integrate_transient(block, ts, time, *steppers)
        return jnp.asarray(traj)[:, :n] if second else jnp.asarray(traj)

    def _shape(self, u):
        u = jnp.asarray(u)
        if self._nf == 1:
            return u.reshape(self._N)
        return u.reshape(self._nf, self._N)

    def _steady_solve(self, *, nonlinear=None, x0=None, extra_params=None, linear=None, precond=None):
        from ..utils.solver.solver_api import compose_linear_solve_fn, compose_nonlinear_solve_fn

        residual = jax.jit(self._residual_fn(extra_params=extra_params))
        zeros = jnp.zeros(self._Ntot)
        make_probe = lambda: jax.jit(self._residual_fn(extra_params=self._current_params()))  # noqa: E731
        with jax.ensure_compile_time_eval():  # structure is decided on concrete values, also inside jno.core
            zc = jnp.zeros(self._Ntot)
            probe_at = zc if x0 is None or isinstance(x0, jax.core.Tracer) else jnp.asarray(x0)  # m/rho is undefined at 0
            linear_problem = self._is_affine("steady", make_probe, self._Ntot)
            self._pattern("steady", make_probe(), probe_at)
        if linear_problem:
            if nonlinear is not None:
                raise ValueError("jno.fvm: nonlinear= given, but this problem is linear.")
            with jax.ensure_compile_time_eval():  # the defaults are chosen on the concrete operator
                linear, precond = self._defaults(self._operator("steady", make_probe(), zc), linear, precond)
                precond = self._frozen(precond, lambda: self._operator("steady", make_probe(), zc))
            A = self._operator("steady", residual, zeros)
            return compose_linear_solve_fn(linear, precond, x0, fem=self._host())(A, -residual(zeros))
        u0 = zeros if x0 is None else jnp.asarray(x0)
        tangent = lambda w: self._operator("steady", residual, w)  # noqa: E731
        u0c = probe_at  # the concrete start, for setups that need values
        from ..utils.solver.segregated import _Segregated

        if isinstance(nonlinear, _Segregated):  # SIMPLE: the problem says which rows are constraints
            with jax.ensure_compile_time_eval():
                pattern = np.asarray(self._operator("steady", make_probe(), u0c).indices)
            return nonlinear.with_constraint(self._constraint_rows(), pattern)(residual, u0, jacobian=tangent)
        nonlinear = nonlinear if nonlinear is not None else _solve.newton(direct=True, line_search=True)
        # no linear= given: Newton's direct variant factorises the assembled tangent (sparse LU), as jno.fdm does
        with jax.ensure_compile_time_eval():
            precond = self._frozen(precond, lambda: self._operator("steady", make_probe(), u0c))
        return compose_nonlinear_solve_fn(nonlinear, linear, precond, fem=self._host())(residual, u0, jacobian=tangent)

    def _constraint_rows(self, n=None):
        """Boolean mask of the DOFs that are multipliers of a constraint equation (the pressure), over a state
        of length ``n`` (the DOF vector, or ``[u; v]`` of a second-order march, whose first half it marks)."""
        rows = np.zeros(self._Ntot, dtype=bool)
        for k in self._saddle:
            for b in range(self._block0[k], self._block0[k] + self._ncomp[k]):
                rows[b * self._N : (b + 1) * self._N] = True
        if not rows.any():
            raise ValueError("jno.solve.simple / piso: this problem has no constraint equation (no saddle pair).")
        return rows if n is None or n == rows.size else np.concatenate([rows, np.zeros(n - rows.size, dtype=bool)])

    def _host(self):
        """What the solver composers see as the owning problem: this one, with the structured-grid descriptor
        of the CONTROL VOLUMES (``jno.precond.gmg`` reads it): the cell lattice of a quad / hex structured
        mesh, whose cells come in lattice order, or the node lattice for the vertex-centred dual."""
        grid = (getattr(self.domain, "mesh_connectivity", {}) or {}).get("grid")
        if grid is not None and self.centering == "cell":
            if self.graph.cell_type in ("quad", "hexahedron") and self._N == int(np.prod(np.asarray(grid["shape"]) - 1)):
                h = np.asarray(grid["spacing"], dtype=float)
                grid = {
                    "shape": tuple(int(n) - 1 for n in grid["shape"]),
                    "spacing": tuple(grid["spacing"]),
                    "origin": tuple(np.asarray(grid["origin"], dtype=float) + h / 2),
                }
            else:
                grid = None
        return _PrecondHost(self, grid)

    def _frozen(self, precond, representative):
        """A preconditioner that needs a concrete matrix (amg, ilu) is set up once on the tangent at the
        initial guess, as fdm does for its Newton."""
        if precond is None:
            return None
        from ..utils.solver.solver_api import (
            LinearOperator,
            PrecondContext,
            _FrozenMarchPrecond,
            _untraceable_leaves,
            materialize_precond,
            prepare_precond,
        )

        if not _untraceable_leaves(precond)[1]:
            return precond
        host = self._host()
        prepare_precond(precond, host)
        return _FrozenMarchPrecond(
            materialize_precond(precond, PrecondContext(LinearOperator(representative()), host)), precond
        )

    # -- what the user reads --------------------------------------------------------------------------
    @property
    def centroids(self):
        """``(N, dim)`` control-volume centroids (where the returned values live)."""
        return np.asarray(self.graph.centroid)

    @property
    def volumes(self):
        """``(N,)`` control-volume measures."""
        return np.asarray(self.graph.volume)

    def to_nodes(self, sol):
        """Volume-weighted average of cell values at the mesh nodes, ``(..., n_nodes)``."""
        sol = np.asarray(sol)
        if self.centering == "vertex":
            return sol
        cells, V = self.graph.cells, self.graph.volume
        n_nodes = self._mesh_points.shape[0]
        w = np.zeros(n_nodes)
        np.add.at(w, cells.ravel(), np.repeat(V, cells.shape[1]))
        flat = sol.reshape(-1, self._N)
        out = np.zeros((flat.shape[0], n_nodes))
        for i, row in enumerate(flat):
            np.add.at(out[i], cells.ravel(), np.repeat(V * row, cells.shape[1]))
        return (out / w).reshape(sol.shape[:-1] + (n_nodes,))


def _iter_vars(node):
    from ..trace import Variable

    n = _unwrap(node)
    if isinstance(n, Variable):
        return [n]
    return [v for c in _iter(n) for v in _iter_vars(c)]


class _PrecondHost:
    """The problem as the solver composers read it, with the control volumes' grid descriptor."""

    def __init__(self, problem, grid):
        import types

        self._problem = problem
        self.domain = types.SimpleNamespace(mesh_connectivity={"grid": grid} if grid is not None else {})

    def __getattr__(self, name):
        return getattr(self._problem, name)


class _Evaluator:
    """One evaluation of the residual: the unknowns' volume and boundary values, and the trace evaluated at
    volumes, interior faces or boundary faces with derivatives replaced by finite-volume gradients."""

    def __init__(self, prob, q, qb, t, scope):
        self.p, self.ops, self.q, self.qb, self.t = prob, prob.ops, list(q), list(qb), t
        self.scope = scope
        self.first_order = False  # first-order face states (the CFL bound reads that operator)
        self.gn = [jnp.zeros_like(b) for b in self.qb]  # vertex-centred: normal derivatives of the boundary pieces
        self._grad_cache = {}

    # sites: "cells", "faces", ("cells", idx), ("bfaces", idx), "bfaces"
    def _site(self, site):
        kind, idx = (site, None) if isinstance(site, str) else site
        return kind, idx

    def _coords(self, kind, idx):
        o = self.ops
        x = {"cells": o.centroid, "faces": o.fcentroid, "bfaces": o.bcentroid}[kind]
        return x if idx is None else x[idx]

    def _unknown_at(self, k, kind, idx, override):
        if override is not None and k in override:
            return override[k]
        if kind == "cells":
            v = self.q[k]
        elif kind == "faces":
            v = self.ops.face_value(self.q[k])
        else:
            v = self.qb[k]
        return v if idx is None else v[idx]

    def count(self, site):
        """Number of points of a site."""
        kind, idx = self._site(site)
        if idx is not None:
            return int(idx.shape[0])
        return {"cells": self.ops.n, "faces": int(self.ops.owner.shape[0]), "bfaces": int(self.ops.b_owner.shape[0])}[kind]

    def rows(self, value, c, site):
        """An evaluated term as ``(m, c)`` at a site of ``m`` points (a constant is broadcast)."""
        m = self.count(site)
        v = jnp.asarray(value)
        if v.ndim == 0 or v.size == 1:
            return jnp.broadcast_to(v.reshape(1, 1), (m, c))
        v = v.reshape(m, -1)
        return jnp.broadcast_to(v, (m, c)) if v.shape[1] == 1 and c > 1 else v

    def values(self, expr, site, override=None):
        """``expr`` evaluated at a site, ``(m, ...)``; ``override`` maps an unknown's index to its values there."""
        from .._strong_form import _map_children
        from ..trace import Jacobian, ModelCall
        from ..trace_evaluator import TraceEvaluator

        kind, idx = self._site(site)
        x = self._coords(kind, idx)
        m = x.shape[0]
        context = {"__time__": jnp.full((m, 1), self.p._start_time() if self.t is None else self.t, dtype=x.dtype)}
        leaves = {}

        def leaf(name, value):
            tag = f"fvm_{name}_{len(leaves)}"
            leaves[tag] = jnp.asarray(value).reshape(m, -1)
            return _context_leaf(tag, None, self.p.domain)

        def rewrite(node):
            n = _unwrap(node)
            marker = getattr(n, "_fvm_face_mean", None)
            if marker is not None:
                if kind == "faces":
                    Tc = self.rows_of(self.values(marker.expr, "cells"), "cells")
                    a, b = Tc[self.ops.owner], Tc[self.ops.neigh]
                    mean = 0.5 * (a + b) if marker.mode == "arithmetic" else 2.0 * a * b / (a + b)
                    return leaf("mean", mean if idx is None else mean[idx])
                return leaf("mean", self.values(marker.expr, site, override))
            if isinstance(n, ModelCall):
                for k, w in enumerate(self.p.unknowns):
                    if n.model is w:
                        return leaf(f"q{k}", self._unknown_at(k, kind, idx, override))
                data = self._nodal_data(n.model, kind, idx)
                if data is not None:
                    return leaf("data", data)
                return n
            if isinstance(n, Jacobian):
                if all(_is_normal(v) for v in n.variables):
                    if kind != "bfaces":
                        raise ValueError("jno.fvm([...]): a normal derivative `u.d(n)` belongs on a boundary.")
                    g = self._gradient(n.target, kind, idx, override)  # (m, c, dim)
                    nrm = self.ops.bn if idx is None else self.ops.bn[idx]
                    return leaf("dn", jnp.einsum("mcd,md->mc", g, nrm))
                if not all(_is_spatial(v) for v in n.variables):
                    raise NotImplementedError("jno.fvm([...]): only spatial and normal derivatives are supported.")
                g = self._gradient(n.target, kind, idx, override)
                return leaf("grad", jnp.concatenate([g[:, :, int(v.dim[0])] for v in n.variables], axis=1))
            return _map_children(n, rewrite)

        expr = rewrite(expr)
        for v in _iter_vars(expr):
            tag = v.tag
            if tag in leaves or tag in context:
                continue
            if _is_normal(v):
                if kind != "bfaces":
                    raise ValueError("jno.fvm([...]): a boundary normal is used away from the boundary.")
                context[tag] = self.ops.bn if idx is None else self.ops.bn[idx]
            elif tag == "cell_size":
                h = self.ops.volume ** (1.0 / self.ops.dim)
                h = {"cells": h, "faces": 0.5 * (h[self.ops.owner] + h[self.ops.neigh]), "bfaces": h[self.ops.b_owner]}[
                    kind
                ]
                context[tag] = (h if idx is None else h[idx])[:, None]
            elif getattr(v, "axis", "spatial") == "spatial":
                context[tag] = x
        context.update(leaves)
        ev = TraceEvaluator(params={**self.scope, **self.p._live_params()})
        return ev.evaluate(expr, context=context, var_bindings={})

    def _nodal_data(self, model, kind, idx):
        """A data field given at the mesh nodes, averaged to the site (a volume's or a face's nodes)."""
        value = getattr(getattr(model, "module", None), "value", None)
        g, o = self.p.graph, self.ops
        if value is None or np.ndim(value) == 0:
            return None
        if np.shape(value)[0] == g.points.shape[0]:  # one value per mesh node: the mean over the site's nodes
            v = jnp.asarray(value).reshape(g.points.shape[0], -1)
            nodes = {"cells": g.cells, "faces": g.f_nodes, "bfaces": g.b_nodes}[kind]
            out = v[jnp.asarray(nodes)].mean(axis=1)
        elif np.shape(value)[0] == g.n:  # one value per volume: interpolated to faces, the owner's at the boundary
            v = jnp.asarray(value).reshape(g.n, -1)
            out = {"cells": v, "faces": o.face_value(v), "bfaces": v[o.b_owner]}[kind]
        else:
            return None
        return out if idx is None else out[idx]

    def _gradient(self, target, kind, idx, override):
        """Gradient ``(m, c, dim)`` of an expression at a site: least squares at volumes, the corrected face
        gradient at interior faces, and at boundary faces the owner's gradient with its normal component
        from the face value."""
        key = id(target)
        if key not in self._grad_cache:
            Tc = self.rows_of(self.values(target, "cells"), "cells")
            Tb = self.rows_of(self.values(target, "bfaces"), "bfaces") if self.ops.b_owner.shape[0] else None
            self._grad_cache[key] = (Tc, Tb, self.ops.cell_gradient(Tc, Tb))
        Tc, Tb, G = self._grad_cache[key]
        if kind == "cells":
            out = self.ops.green_gauss(Tc, Tb) if self.p._is_multiplier(target) else G
        elif kind == "faces":
            out = self.ops.face_gradient(Tc, G)
        elif self.ops.vertex:  # the node's gradient, its normal part the boundary piece's normal derivative
            sel = self.ops.b_owner if idx is None else self.ops.b_owner[idx]
            Gp = G[sel]
            k = next(
                (
                    kk
                    for kk, w in enumerate(self.p.unknowns)
                    if _unwrap(target) is not None and getattr(_unwrap(target), "model", None) is w
                ),
                None,
            )
            if k is None:
                return Gp
            trial = getattr(self, "_gn_trial", None)
            if trial is not None and trial[0] == k and trial[1] is idx:
                gn = trial[2]
            else:
                gn = self.gn[k] if idx is None else self.gn[k][idx]
            nrm = self.ops.bn if idx is None else self.ops.bn[idx]
            return Gp + (gn - jnp.einsum("fcd,fd->fc", Gp, nrm))[:, :, None] * nrm[:, None, :]
        else:
            Tsite = (
                Tb if override is None else self.rows_of(self.values(target, ("bfaces", idx), override), ("bfaces", idx))
            )
            if override is None:
                out = self.ops.boundary_gradient(Tc, G, Tb)
                return out if idx is None else out[idx]
            Gp = G[self.ops.b_owner[idx]]
            foot = Tc[self.ops.b_owner[idx]] + jnp.einsum("fcd,fd->fc", Gp, self.ops.br_t[idx])
            dn = (Tsite - foot) / self.ops.bdperp[idx][:, None]
            nrm = self.ops.bn[idx]
            return Gp + (dn - jnp.einsum("fcd,fd->fc", Gp, nrm))[:, :, None] * nrm[:, None, :]
        return out if idx is None else out[idx]

    def rows_of(self, value, site):
        """An evaluated expression as ``(m, c)``, its own number of components."""
        m = self.count(site)
        v = jnp.asarray(value)
        if v.size == 1:
            return jnp.broadcast_to(v.reshape(1, 1), (m, 1))
        return v.reshape(m, -1)

    # -- boundary values ------------------------------------------------------------------------------
    def solve_boundary(self, first=True):
        """Boundary face values for this pass: each condition solved on its faces by Newton (one step when
        affine), every other face at zero normal derivative. Uses the volume gradients of the previous
        pass for the owner value at the foot of the normal."""
        o = self.ops
        if not o.b_owner.shape[0]:
            return
        if o.vertex:  # boundary values are the nodes' own; flux conditions give the normal derivative
            self.qb = [qk[o.b_owner] for qk in self.q]
            self.gn = [jnp.zeros_like(qb) for qb in self.qb]
            self._grad_cache = {}
            for k, faces, expr in self.p._conditions:
                z = jnp.zeros((int(faces.shape[0]), self._width(k)), self.q[k].dtype)
                affine = self.p._condition_affine.get(id(expr), False)
                for _ in range(1 if affine else _NEWTON_STEPS):
                    z = self._newton_step(k, faces, expr, z)
                self.gn[k] = self.gn[k].at[faces].set(z)
                self._grad_cache = {}
            return
        G = [o.cell_gradient(self.q[k], self.qb[k]) for k in range(len(self.q))]
        new = [o.owner_at_face(self.q[k], G[k]) for k in range(len(self.q))]
        self._grad_cache = {}
        for k, faces, expr in self.p._conditions:
            partial = self.p._condition_partial.get(id(expr), False)  # projected from the interior state each pass
            qf = new[k][faces] if first or partial else self.qb[k][faces]  # later passes start from the last values
            affine = self.p._condition_affine.get(id(expr), False)
            for _ in range(1 if affine else (_NEWTON_STEPS if first else 4)):
                qf = self._newton_step(k, faces, expr, qf)
            new[k] = new[k].at[faces].set(qf)
        self.qb = new
        self._grad_cache = {}

    def _width(self, k):
        return self.q[k].shape[1]

    def _condition(self, k, faces, expr, qf):
        if self.ops.vertex:  # qf is the normal derivative of the boundary pieces; their value is the node's
            self._gn_trial = (k, faces, qf)
            self._grad_cache = {}
            try:
                r = self.rows_of(self.values(expr, ("bfaces", faces)), ("bfaces", faces))
            finally:
                self._gn_trial = None
                self._grad_cache = {}
            if r.shape[1] != qf.shape[1]:
                raise NotImplementedError(
                    "jno.fvm([...], centering='vertex'): a partial boundary condition is not supported."
                )
            return r
        r = self.rows_of(self.values(expr, ("bfaces", faces), {k: qf}), ("bfaces", faces))
        if r.shape[1] > qf.shape[1]:
            raise ValueError(
                f"jno.fvm([...]): a boundary condition gives {r.shape[1]} equations per face for an unknown with "
                f"{qf.shape[1]} components."
            )
        return r

    def _newton_step(self, k, faces, expr, qf):
        """One Newton step on a face condition ``r(q_f) = 0``. With as many equations as components it is
        the plain step; with fewer (``U·n = 0`` on a vector unknown) the face state moves only along the
        condition's gradients, ``q_f ← q_f − Jᵀ(JJᵀ)⁻¹ r``: the projection of the interior state onto the
        condition set, with the unconstrained directions extrapolated (for an affine condition, the face
        value of the mirror state)."""
        c = qf.shape[1]
        r = self._condition(k, faces, expr, qf)
        cols = [
            jax.jvp(lambda z: self._condition(k, faces, expr, z), (qf,), (jnp.zeros_like(qf).at[:, j].set(1.0),))[1]
            for j in range(c)
        ]
        J = jnp.stack(cols, axis=-1)  # (m, p, c): d r_i / d q_j, face by face
        if J.shape[1] == c:
            return qf - jnp.linalg.solve(J, r[..., None])[..., 0]
        lam = jnp.linalg.solve(J @ jnp.swapaxes(J, 1, 2), r[..., None])
        return qf - (jnp.swapaxes(J, 1, 2) @ lam)[..., 0]

    # -- fluxes ---------------------------------------------------------------------------------------
    def mass_flux(self, by_axis, j, k, c):
        """Interior and boundary fluxes of a constraint equation (``div U = 0``): the central flux of the
        interpolated state, with the momentum-weighted (Rhie-Chow) correction ``-D_f [∂p/∂d − ∇p̄·d̂](S·d̂)`` at
        interior faces, which couples the multiplier ``p`` (unknown ``k``) to its neighbours; boundary faces carry
        the boundary state alone.

        Based on: Rhie, C. M., & Chow, W. L. (1983). Numerical study of the turbulent flow past an airfoil with
        trailing edge separation. AIAA Journal 21(11), 1525-1532."""
        o = self.ops
        fi = sum(self.rows(self.values(F, "faces"), c, "faces") * o.S[:, a : a + 1] for a, F in by_axis.items())
        fb = None
        if o.b_owner.shape[0]:
            fb = sum(self.rows(self.values(F, "bfaces"), c, "bfaces") * o.bS[:, a : a + 1] for a, F in by_axis.items())
        P = self.q[k]
        G = o.green_gauss(P, self.qb[k])
        Gf = G[o.owner] + o.gw[:, None, None] * (G[o.neigh] - G[o.owner])
        jump = (P[o.neigh] - P[o.owner]) / o.dmag[:, None] - jnp.einsum("fcd,fd->fc", Gf, o.dhat)
        D = self.momentum_weight(j, k)
        sd = jnp.einsum("fd,fd->f", o.S, o.dhat)
        return fi - (D * sd)[:, None] * jump, fb

    def momentum_weight(self, j, k):
        """``D_f = β V/a_P`` at the faces for the momentum of unknown ``j`` and the multiplier ``k``, all probed
        from the momentum equation itself (nothing names a viscosity, a velocity or a permeability):
        ``a_P = Σ_f |S_f| (λ_f + κ_f/|d_f|) + V |∂(volume term)/∂U|`` -- ``λ`` the spectral radius of its
        convective normal flux, ``κ`` its diffusivity (the sensitivity of the central normal flux to the
        gradient), the last a drag or reaction (Darcy's ``U/K``) -- and ``β`` the coefficient of ``∇p`` in it.
        Differentiable."""
        o, p = self.ops, self.p
        eq = p._equations[j]
        c = p._ncomp[j]
        lam = jnp.zeros(o.owner.shape[0], o.S.dtype)
        conv = {a: F for g in eq["groups"] for a, F in g["convective"].items()}
        if conv:

            def Fn(U):
                states = {
                    jj: (U if jj == j else self._unknown_at(jj, "faces", None, None)) for jj in range(len(p.unknowns))
                }
                return sum(
                    self.rows(self.values(F, "faces", states), c, "faces") * o.nhat[:, a : a + 1] for a, F in conv.items()
                )

            lam = _spectral_radius(_flux_jacobian(Fn, o.face_value(self.q[j])))
        kappa = jnp.zeros_like(lam)
        central = [g["central"] for g in eq["groups"] if g["central"] and g["coef"] is None]
        if central:
            q0 = self.q[j]

            def flux_of(qj):
                saved, self.q[j] = self.q[j], qj
                self._grad_cache = {}
                try:
                    return sum(self.central_flux(by, c)[0] for by in central)  # (n_if, c)
                finally:
                    self.q[j] = saved
                    self._grad_cache = {}

            acc = jnp.zeros_like(lam)
            for comp in range(c):
                for a in range(o.dim):
                    t = jnp.zeros_like(q0).at[:, comp].set(o.centroid[:, a])  # a unit gradient along axis a
                    dF = jax.jvp(flux_of, (q0,), (t,))[1][:, comp]
                    acc = acc + o.nhat[:, a] * dF
            kappa = jnp.abs(acc) / (c * o.Smag)
        aface = o.Smag * (lam + kappa / o.dmag)
        aP = jax.ops.segment_sum(aface, o.owner, num_segments=o.n) + jax.ops.segment_sum(aface, o.neigh, num_segments=o.n)
        if o.b_owner.shape[0]:
            aP = aP + jax.ops.segment_sum(o.bSmag * kappa.mean() / o.bdperp, o.b_owner, num_segments=o.n)
        beta = jnp.ones(o.n, o.S.dtype)
        if eq["volume"] is not None:
            vol = _zero_temporal(eq["volume"])

            def vol_of(i, qi, qbi=None):
                saved, self.q[i] = self.q[i], qi
                saved_b = self.qb[i]
                if qbi is not None:
                    self.qb[i] = qbi
                self._grad_cache = {}
                try:
                    return self.rows(self.values(vol, "cells"), c, "cells")
                finally:
                    self.q[i], self.qb[i] = saved, saved_b
                    self._grad_cache = {}

            drag = jnp.zeros(o.n, o.S.dtype)
            for comp in range(c):  # the volume term's own diagonal in U
                t = jnp.zeros_like(self.q[j]).at[:, comp].set(1.0)
                drag = drag + jnp.abs(jax.jvp(lambda z: vol_of(j, z), (self.q[j],), (t,))[1][:, comp]) / c
            aP = aP + o.volume * drag
            grads = []
            for a in range(min(c, o.dim)):  # the coefficient of ∂p/∂x_a in component a
                t = jnp.zeros_like(self.q[k]).at[:, 0].set(o.centroid[:, a])
                tb = jnp.zeros_like(self.qb[k]).at[:, 0].set(o.bcentroid[:, a])  # the boundary values move too
                f = lambda z, zb: vol_of(k, z, zb)  # noqa: E731
                grads.append(jax.jvp(f, (self.q[k], self.qb[k]), (t, tb))[1][:, a])
            if grads:
                beta = jnp.abs(sum(grads) / len(grads))
        DP = beta * o.volume / jnp.maximum(aP, jnp.finfo(aP.dtype).tiny)
        return 0.5 * (DP[o.owner] + DP[o.neigh])

    def central_flux(self, by_axis, c):
        """Interior and boundary normal fluxes ``Σ_axis F_axis S_axis`` of the central part of a group."""
        o = self.ops
        fi = sum(self.rows(self.values(F, "faces"), c, "faces") * o.S[:, a : a + 1] for a, F in by_axis.items())
        fb = None
        if o.b_owner.shape[0]:
            fb = sum(self.rows(self.values(F, "bfaces"), c, "bfaces") * o.bS[:, a : a + 1] for a, F in by_axis.items())
        return fi, fb

    def convective_divergence(self, eqs):
        """``Σ_faces F*·S`` of every equation's convective fluxes, ``(n, nf)``: the gathered normal flux
        ``F·n`` of the whole system on reconstructed face states, through the scheme's numerical flux;
        ``None`` when there are none."""
        if not any(g["convective"] for eq in eqs for g in eq["groups"]):
            return None
        o, p = self.ops, self.p
        scheme = p._convective_scheme()

        def normal_flux(states, site, normals):
            cols = []
            for k, eq in enumerate(eqs):
                c = p._ncomp[k] if len(eqs) > 1 else p._nf
                acc = jnp.zeros((normals.shape[0], c), dtype=normals.dtype)
                for g in eq["groups"]:
                    for a, F in g["convective"].items():
                        acc = acc + self.rows(self.values(F, site, states), c, site) * normals[:, a : a + 1]
                cols.append(acc)
            return jnp.concatenate(cols, axis=1)

        def split(Q):
            return {k: Q[:, b : b + c] for k, (b, c) in enumerate(zip(p._block0, p._ncomp))}

        rows = []  # the components whose equations carry a convective flux: only they are damped
        for k, eq in enumerate(eqs):
            c = p._ncomp[k] if len(eqs) > 1 else p._nf
            rows += [any(g["convective"] for g in eq["groups"])] * c
        live = jnp.asarray(rows, dtype=jnp.concatenate(self.q, axis=1).dtype)

        def numerical(QL, QR, site, normals):
            Fn = lambda Q: normal_flux(split(Q), site, normals)  # noqa: E731
            if callable(scheme.flux):
                return scheme.flux(QL, QR, normals, Fn)
            QR = QL + live * (QR - QL)  # a row without a convective flux sees no jump to damp
            fL, fR = Fn(QL), Fn(QR)
            if scheme.flux == "rusanov":
                sL = self._speed(QL, site, normals, Fn, split, scheme)
                sR = self._speed(QR, site, normals, Fn, split, scheme)
                # a smooth max: where the two speeds tie (a stationary shock) max() has a kink that stalls Newton.
                # Exact at a tie (so a linear flux stays exact upwinding), within δ/2 of the max elsewhere.
                gap, delta = sL - sR, 1e-3 * (sL + sR)
                a = 0.5 * (sL + sR + jnp.sqrt(gap * gap + delta * delta + jnp.finfo(sL.dtype).tiny) - delta)

                return 0.5 * (fL + fR) - 0.5 * a[:, None] * (QR - QL)
            if scheme.flux == "hll":
                lo_L, hi_L = _eig_extremes(_flux_jacobian(Fn, QL))
                lo_R, hi_R = _eig_extremes(_flux_jacobian(Fn, QR))
                sl = jnp.minimum(jnp.minimum(lo_L, lo_R), 0.0)[:, None]
                sr = jnp.maximum(jnp.maximum(hi_L, hi_R), 0.0)[:, None]
                width = jnp.where(sr - sl > 0, sr - sl, 1.0)
                hll = (sr * fL - sl * fR + sl * sr * (QR - QL)) / width
                return jnp.where(sr - sl > 0, hll, 0.5 * (fL + fR))
            # "roe": the jump damped by |A| at the mean state, |A| = sqrt(A^2 + d^2 I) (entropy fix built in)
            A = _flux_jacobian(Fn, 0.5 * (QL + QR))
            rho = _spectral_radius(A)
            absA = _abs_matrix(A, scheme.entropy_fix * rho)
            return 0.5 * (fL + fR) - 0.5 * jnp.einsum("mij,mj->mi", absA, QR - QL)

        QL, QR, Qin, Qb = self._face_states(scheme)
        fi = o.Smag[:, None] * numerical(QL, QR, "faces", o.nhat)
        fb = None
        if o.b_owner.shape[0]:
            # where a condition fixes only part of the state (a slip wall), the outer state is the mirror image
            # 2 q_f - q_P, so the normal flux of what the condition removes vanishes (no mass through a wall)
            Qout = Qb
            for k, faces, expr in p._conditions:
                if not p._condition_partial.get(id(expr), False):
                    continue
                cols = slice(p._block0[k], p._block0[k] + p._ncomp[k])
                inner = Qin[faces][:, cols]
                z = inner  # the inner state projected onto the condition set, then reflected through it
                for _ in range(1 if p._condition_affine.get(id(expr), False) else 4):
                    z = self._newton_step(k, faces, expr, z)
                Qout = Qout.at[faces, cols].set(2.0 * z - inner)
            fb = o.bSmag[:, None] * numerical(Qin, Qout, "bfaces", o.bn)
        return o.divergence(fi, fb)

    def _face_states(self, scheme):
        """``(Q_L, Q_R, Q_inner, Q_boundary)``: the reconstructed states on both sides of every interior face
        and on the inner side of every boundary face (the outer one is the boundary value)."""
        o = self.ops
        Q = jnp.concatenate(self.q, axis=1)
        Qb = jnp.concatenate(self.qb, axis=1) if o.b_owner.shape[0] else jnp.zeros((0, Q.shape[1]), Q.dtype)
        if scheme.recon == "none" or self.first_order:
            return Q[o.owner], Q[o.neigh], Q[o.b_owner], Qb
        G = o.cell_gradient(Q, Qb) if scheme.recon == "lsq" else o.green_gauss(Q, Qb)
        if scheme.multidimensional:
            phi = o.multidimensional_limiter(Q, Qb, G, scheme.limiter, scheme.venkat_k)
            QL, QR, Qin = o.reconstruct(Q, phi[:, :, None] * G)
            return QL, QR, Qin, Qb
        if scheme.limiter is None:
            QL, QR, Qin = o.reconstruct(Q, G)
            return QL, QR, Qin, Qb
        # a ψ(r) limiter per face, on the upwind difference of Darwish & Moukalled (2003), written as a·ψ(b/a)
        eps = (scheme.venkat_k * o.dmag) ** 3
        dQ = Q[o.neigh] - Q[o.owner]
        upL = 2.0 * jnp.einsum("fcd,fd->fc", G[o.owner], o.delta) - dQ
        upR = 2.0 * jnp.einsum("fcd,fd->fc", G[o.neigh], o.delta) - dQ
        QL = Q[o.owner] + 0.5 * scheme.increment(dQ, upL, eps[:, None])
        QR = Q[o.neigh] - 0.5 * scheme.increment(dQ, upR, eps[:, None])
        dB = 2.0 * (Qb - Q[o.b_owner])  # against a mirror cell beyond the face
        upB = 2.0 * jnp.einsum("fcd,fd->fc", G[o.b_owner], 2.0 * o.br) - dB
        epsB = (scheme.venkat_k * 2.0 * o.bdperp) ** 3
        Qin = Q[o.b_owner] + 0.5 * scheme.increment(dB, upB, epsB[:, None])
        return QL, QR, Qin, Qb

    def _speed(self, Q, site, normals, Fn, split, scheme):
        """Wave speed per face: the scheme's ``speed=`` expression, else the spectral radius of
        ``∂(F·n)/∂Q``."""
        if scheme.speed is not None:
            return self.rows(self.values(scheme.speed, site, split(Q)), 1, site)[:, 0]
        return _spectral_radius(_flux_jacobian(Fn, Q))


def _flux_jacobian(Fn, Q):
    """``∂Fn/∂Q`` face by face, ``(m, nf, nf)``: faces are independent, so one jvp per component."""
    nf = Q.shape[1]
    cols = [jax.jvp(Fn, (Q,), (jnp.zeros_like(Q).at[:, j].set(1.0),))[1] for j in range(nf)]
    return jnp.stack(cols, axis=-1)


def _charpoly(A):
    """Coefficients ``c_0..c_n`` of ``det(xI − A) = Σ c_k x^k`` for each ``A`` (Faddeev-LeVerrier)."""
    n = A.shape[1]
    eye = jnp.eye(n, dtype=A.dtype)
    coef = [jnp.ones(A.shape[0], A.dtype)]  # c_n
    Mk = jnp.zeros_like(A)
    for k in range(1, n + 1):
        Mk = A @ Mk + coef[-1][:, None, None] * eye
        coef.append(-jnp.trace(A @ Mk, axis1=1, axis2=2) / k)
    return jnp.stack(coef[::-1], axis=1)  # (m, n+1): c_0 .. c_n


def _extreme_root(coef, x0, iters=60):
    """The real root of a real-rooted polynomial nearest beyond ``x0`` (Newton from outside the roots moves
    monotonically onto the extreme one)."""
    n = coef.shape[1] - 1
    dcoef = coef[:, 1:] * jnp.arange(1, n + 1, dtype=coef.dtype)

    def poly(c, x):
        out = jnp.zeros_like(x)
        for k in range(c.shape[1] - 1, -1, -1):
            out = out * x + c[:, k]
        return out

    def body(_, x):
        d = poly(dcoef, x)
        return x - poly(coef, x) / jnp.where(d != 0, d, 1.0)

    return jax.lax.fori_loop(0, iters, body, x0)


def _eig_extremes(A):
    """``(λ_min, λ_max)`` of each ``A``, whose eigenvalues are real (a hyperbolic system): the extreme roots of
    the characteristic polynomial, by Newton from the bounds ``∓‖A‖_∞``. Exact, differentiable, and the same
    on every device (no eigendecomposition)."""
    if A.shape[1] == 1:
        return A[:, 0, 0], A[:, 0, 0]
    # on A/β, β = ‖A‖_∞ held constant: eigenvalues scale with A, so this is exact, and it stays finite at A = 0
    beta = jax.lax.stop_gradient(jnp.abs(A).sum(axis=2).max(axis=1))
    beta = jnp.where(beta > 0, beta, 1.0)
    coef = _charpoly(A / beta[:, None, None])
    edge = jnp.full_like(beta, 1.0 + 1e-8)
    return beta * _extreme_root(coef, -edge), beta * _extreme_root(coef, edge)


def _smooth_abs(a, delta=1e-8):
    """``|a|``, rounded to a parabola within ``δ`` of zero (C¹, so Newton does not stall at a sonic point) and
    exact elsewhere. ``δ`` is a fixed constant: one that followed the state but was held out of the tangent
    made the tangent wrong exactly at the sonic face, and Newton stalled there."""
    return jnp.where(jnp.abs(a) >= delta, jnp.abs(a), 0.5 * (a * a / delta + delta))


def _spectral_radius(A):
    """``max |λ|`` of each ``A`` (real eigenvalues, as for a hyperbolic system)."""
    if A.shape[1] == 1:
        return _smooth_abs(A[:, 0, 0])
    lo, hi = _eig_extremes(A)
    return jnp.maximum(jnp.abs(lo), jnp.abs(hi))


def _abs_matrix(A, delta, iters=20):
    """``|A|`` as ``sqrt(A² + δ²I)`` by the Denman-Beavers iteration (``δ`` per matrix); ``|a|`` smoothed to
    ``sqrt(a² + δ²)`` for a scalar."""
    if A.shape[1] == 1:
        return jnp.sqrt(A * A + (delta * delta)[:, None, None])
    eye = jnp.eye(A.shape[1], dtype=A.dtype)
    Y = A @ A + (delta * delta)[:, None, None] * eye + jnp.finfo(A.dtype).tiny * eye
    Z = jnp.broadcast_to(eye, Y.shape)
    for _ in range(iters):
        Yi, Zi = jnp.linalg.inv(Y), jnp.linalg.inv(Z)
        Y, Z = 0.5 * (Y + Zi), 0.5 * (Z + Yi)
    return Y


def fvm(constraints, centering: str = "cell"):
    """A finite-volume problem from a strong-form constraint list, written as for :func:`jno.fdm`::

        u = d.unknown(); ui = u.bind(x=x, y=y)
        sol = jno.fvm([-(ui.xx + ui.yy) - f, u(xb, yb) - g]).solve()   # (n_cells,)

    Each PDE term is integrated over a control volume: an outer derivative ``(F).x`` becomes a flux
    through the faces, other terms are evaluated at the centroid. Boundary conditions are written as in
    ``jno.fdm`` (``u(xb, yb) - g``, ``ur.d(nr) - h``, ``ur.d(nr) + a*(ur - u_inf)``) and hold on the
    boundary faces of their region.

    Args:
        constraints: the PDE residual(s), boundary conditions and pins.
        centering: ``"cell"`` -- one control volume per mesh cell.

    Returns:
        The problem; ``.solve()`` returns the volume values, ``.centroids`` says where they live.
    """
    return _TraceFVM(constraints, centering=centering)


def _info_fvm(o, deep: bool):
    """``jno.info`` for a finite-volume problem: the mesh it partitions, the unknowns and how each equation
    was read (fluxes, volume terms, constraints), the scheme, and the regime."""
    from ..info import Info, _equation_rows, _fmt_n

    g = o.graph
    rows = [
        ("centering", f"{o.centering} · {g.cell_type}"),
        (
            "control volumes",
            f"{_fmt_n(g.n)} · interior faces {_fmt_n(g.owner.size)} · boundary faces {_fmt_n(g.b_owner.size)}",
        ),
        ("regime", "transient (method of lines)" if o._transient else "steady"),
        ("unknowns", f"{len(o.unknowns)} · {_fmt_n(o._nf)} components · {_fmt_n(o._Ntot)} dofs"),
    ]
    for k, eq in enumerate(o._equations):
        kinds = []
        if eq["volume"] is not None:
            kinds.append("volume terms")
        if any(gr["central"] for gr in eq["groups"]):
            kinds.append("diffusive fluxes")
        if any(gr["convective"] for gr in eq["groups"]):
            kinds.append("convective fluxes")
        if eq.get("mass"):
            kinds.append(f"constraint on unknown {eq['partner']} (Rhie-Chow)")
        rows.append((f"equation {k}", ", ".join(kinds) or "-"))
    if any(gr["convective"] for eq in o._equations for gr in eq["groups"]):
        rows.append(("scheme", str(o._convective_scheme())[len("finite_difference:") :]))
    rows.append(("boundary conditions", f"{len(o._conditions)} on faces · {len(o._pins)} pinned cell sets"))
    if o.domain is not None and getattr(o.domain, "time", None):
        t0, t1, nt = o.domain.time
        rows.append(("time", f"[{t0}, {t1}] · {nt} steps"))
    return Info("fvm (finite volume)", [("equations", _equation_rows(list(o._constraints))), ("discretisation", rows)])


def _register_info():
    from .. import info as _info

    _info.REGISTRY["_TraceFVM"] = _info_fvm


_register_info()
