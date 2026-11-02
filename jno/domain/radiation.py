"""Surface-to-surface radiation written as the equation: ``d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v * dS``.

``d.irradiation(T)`` is the incident radiation G on every radiating face: the faces between a transparent region
(attached ``kappa == 0``) and the regions next to it, one diffuse-grey enclosure per connected transparent region
(Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 5)::

    J = eps sigma T^4 + (1 - eps) G,   G = F J   ->   J = (I - diag(1 - eps) F)^-1 eps sigma T^4,  G = F J

The emissivity is the attached property ``eps`` of the face's region (a number or a law of T, evaluated at the
face); ``kappa`` marks the media. These two names are the operator's documented convention. Axisymmetry comes from
the domain (``.domain(axisymmetric=True)``).

A term that contains ``d.irradiation`` is a FACE term: ``jno.fem`` lowers it here. Every factor is evaluated on each
radiating face -- the field at the face's mean temperature (the radiosity's own discretisation), a coordinate at two
Gauss points along it (so ``dS = 2 pi r`` is integrated exactly against each node's hat function, the enclosure's
ring-consistent weights), a region mask by the face's region -- and the result, which must be linear in the test
function, is scattered to the face's two nodes. The term becomes
an internal ``Coupling`` with its support declared, so Newton's tangent carries the whole term exactly (dense on the
radiating nodes), including dG/dT and d eps/dT.

Scope: 2-D and axisymmetric meshes (the enclosure's own scope); the term must use the field and the test function
unbound (``T``, ``v``), not their coordinate-bound views, and no derivatives.
"""

from __future__ import annotations

import operator

import jax
import jax.numpy as jnp
import numpy as np

from ..trace import Placeholder
from .enclosure import STEFAN_BOLTZMANN, radiosity_irradiation, radiosity_resolvent


class Irradiation(Placeholder):
    """``d.irradiation(T)``: the incident radiation on the radiating faces (lowered by ``jno.fem``)."""

    def __init__(self, domain, field):
        self._domain = domain
        self._of = field

    def __repr__(self):
        return f"Irradiation({getattr(self._of, 'name', self._of)})"


_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "@": operator.matmul,
}


def _children(e):
    from ..trace import BinaryOp, FunctionCall
    from ..trace.views import _VIEW_TYPES

    if isinstance(e, _VIEW_TYPES):
        return [e._expr]
    if isinstance(e, BinaryOp):
        return [e.left, e.right]
    if isinstance(e, FunctionCall):
        return [a for a in e.args if isinstance(a, Placeholder)]
    return []


def _find(e, cls):
    out, stack, seen = [], [e], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, cls):
            out.append(x)
        stack.extend(_children(x))
    return out


def _face_eval(e, env):
    """Evaluate a term on the radiating faces. ``env``: field key -> per-face values, 'test' -> the test value,
    'coords' -> (m, dim) midpoints, 'tags' -> per-face region names, 'G' -> per-face irradiation (or None)."""
    from ..trace import AttachedLiteral, BinaryOp, FunctionCall, Literal, RegionMask, TestFunction, TrialFunction, Variable
    from ..trace.views import _VIEW_TYPES

    if isinstance(e, _VIEW_TYPES):
        raise NotImplementedError(
            "d.irradiation: write the radiation term with the field and the test function themselves (T, v), "
            "not with their coordinate-bound views (Ti, vi): it is evaluated on the radiating faces, not at "
            "volume quadrature points."
        )
    if isinstance(e, (int, float, complex, np.ndarray)) or (hasattr(e, "shape") and not isinstance(e, Placeholder)):
        return e
    if isinstance(e, Irradiation):
        if env["G"] is None:
            raise ValueError("d.irradiation inside the emissivity: the emissivity cannot depend on the irradiation.")
        return env["G"]
    if isinstance(e, AttachedLiteral):
        return jnp.asarray(e.value)
    if isinstance(e, Literal):
        return e.value
    if isinstance(e, BinaryOp):
        f = _BINARY.get(e.op)
        if f is None:
            raise NotImplementedError(f"d.irradiation: the operation {e.op!r} is not supported in a radiation term.")
        return f(_face_eval(e.left, env), _face_eval(e.right, env))
    if isinstance(e, FunctionCall):
        args = [_face_eval(a, env) if isinstance(a, Placeholder) else a for a in e.args]
        return e.fn(*args, **(e.kwargs or {}))
    if isinstance(e, RegionMask):
        return jnp.asarray(env["tags"] == e.region, dtype=jnp.float64)
    if isinstance(e, Variable):
        lo, hi = e.dim[0], e.dim[1]
        c = env["coords"][:, lo:hi]
        return c[:, 0] if c.shape[1] == 1 else c
    if isinstance(e, TestFunction):
        return env["test"]
    if isinstance(e, TrialFunction):
        key = getattr(e, "field_key", None)
        if key not in env["fields"]:
            raise ValueError(
                f"d.irradiation: the radiation term reads the field {e!r}, which is not the field the irradiation is of."
            )
        return env["fields"][key]
    from ..trace import ModelCall

    if isinstance(e, ModelCall):
        raise NotImplementedError(
            "d.irradiation: a trainable parameter or network (jno.np.parameter, jno.nn) cannot be in a radiation term "
            "yet; it is evaluated on the radiating faces, outside the weak-form trace. Use a number or a law of T."
        )
    raise NotImplementedError(
        f"d.irradiation: {type(e).__name__} cannot be evaluated on a radiating face (no derivatives, networks or "
        "boundary bindings in a radiation term)."
    )


def radiating_faces(domain):
    """The domain's radiating faces: an EnclosureSet over every connected region with attached ``kappa == 0``,
    built once per domain (view factors cached on disk)."""
    cached = domain.__dict__.get("_radiation_faces")
    if cached is not None:
        return cached
    from .enclosure import enclosures_of_medium
    from .mesh_utils import mesh_cell_region_membership

    kappa = (domain.__dict__.get("_region_attachments") or {}).get("kappa")
    if not kappa:
        raise ValueError(
            "d.irradiation: no region has an attached 'kappa'. Mark the regions radiation crosses with .attach(kappa=0.0)."
        )
    regions = mesh_cell_region_membership(domain.mesh, int(domain.dimension))
    media = [r for r in regions if isinstance(kappa.get(r), (int, float)) and float(kappa[r]) == 0.0]
    if not media:
        raise ValueError("d.irradiation: no region has kappa == 0, so no medium carries the radiation.")
    rad = enclosures_of_medium(
        domain, media, axisymmetric=bool(domain.__dict__.get("axisymmetric", False)), enforce_closure=True
    )
    domain.__dict__["_radiation_faces"] = rad
    return rad


def lower_radiation_terms(constraints):
    """Replace every term that contains ``d.irradiation`` by the internal ``Coupling`` that integrates it over the
    radiating faces (see the module docstring); other terms pass through."""
    if not isinstance(constraints, (list, tuple)):
        return constraints
    out = []
    for term in constraints:
        nodes = _find(term, Irradiation) if isinstance(term, Placeholder) or hasattr(term, "_expr") else []
        out.append(_lower(term, nodes) if nodes else term)
    return out


def _lower(term, nodes):
    from .._fem import Coupling, _bare, _unknown_symbol_of
    from ..trace import TestFunction, TrialFunction

    domain = nodes[0]._domain

    def _field(n):  # `T = d.unknown()` stands for its trial symbol, as everywhere in jno.fem
        f = _bare(n._of)
        return _unknown_symbol_of(f) or f

    field = _field(nodes[0])
    if any(n._domain is not domain or _field(n) is not field for n in nodes):
        raise ValueError("d.irradiation: one radiation term must use one domain's irradiation of one field.")
    if not isinstance(field, TrialFunction):
        raise TypeError("d.irradiation(T): pass the field itself (the trial function), e.g. d.irradiation(T).")
    tests = _find(term, TestFunction)
    if len({getattr(t, "field_key", id(t)) for t in tests}) != 1:
        raise ValueError("d.irradiation: a radiation term needs exactly one test function, e.g. ... * v * dS.")

    rad = radiating_faces(domain)
    pts = np.asarray(domain.mesh.points)[:, : int(domain.dimension)]
    elements = np.concatenate([np.asarray(p.elements)[:, :2] for p in rad.parts])
    # Two Gauss points along each face: the fields (and G) are per face, as the radiosity is, but a coordinate
    # factor such as dS = 2 pi r is integrated exactly against each node's hat function -- the same nodal weights
    # as the enclosure's ring-consistent load.
    s_g = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
    p0, p1 = pts[elements[:, 0]], pts[elements[:, 1]]
    gauss = jnp.asarray(np.concatenate([p0 + s_g[0] * (p1 - p0), p0 + s_g[1] * (p1 - p0)]))  # (2m, dim)
    mids = jnp.asarray(0.5 * (p0 + p1))
    length = jnp.asarray(np.linalg.norm(p1 - p0, axis=1))
    phi_a = jnp.asarray(np.concatenate([np.full(len(p0), 1 - s_g[0]), np.full(len(p0), 1 - s_g[1])]))
    tags = np.asarray(rad.element_tags).astype(str)
    eps_expr = getattr(domain, "eps")(field)  # d.eps(T): numbers or laws, per region
    key = getattr(field, "field_key", None)
    a, b = jnp.asarray(elements[:, 0]), jnp.asarray(elements[:, 1])
    cuts = np.cumsum([0, *[p.size for p in rad.parts]])
    n_nodes = int(pts.shape[0])

    def eps_faces(t):
        env = {"fields": {key: t}, "coords": mids, "tags": tags, "G": None, "test": 1.0}
        return jnp.broadcast_to(jnp.asarray(_face_eval(eps_expr, env), dtype=t.dtype), t.shape)

    def faces_of(u):
        """Per-face field t (mean of the two nodes), emissivity e and irradiation G, from the nodal vector."""
        u = jnp.asarray(u).reshape(-1)[:n_nodes]
        t = 0.5 * (u[a] + u[b])
        e = eps_faces(t)
        G = jnp.concatenate(
            [
                radiosity_irradiation(part.view_factor, e[lo:hi], t[lo:hi])
                for part, lo, hi in zip(rad.parts, cuts[:-1], cuts[1:])
            ]
        )
        return t, e, G

    def at_gauss(t, G, test=1.0):
        """The face environment at the 2m Gauss points: coordinates there, everything else per face, twice."""
        two = lambda x: jnp.concatenate([x, x]) if hasattr(x, "shape") and jnp.ndim(x) > 0 else x  # noqa: E731
        return {"fields": {key: two(t)}, "coords": gauss, "tags": np.concatenate([tags, tags]), "G": two(G), "test": test}

    m = len(elements)

    def contributions(t, G):
        """The term's integral against each face's two hat functions, ``(ca, cb)`` of shape (m,): pointwise in the
        face (a function of that face's t and G only)."""
        c = jnp.broadcast_to(jnp.asarray(_face_eval(term, at_gauss(t, G)), dtype=jnp.float64), (2 * m,))
        w = c * 0.5 * jnp.concatenate([length, length])  # Gauss weight 1/2 per point, times the face length
        return (w * phi_a).reshape(2, m).sum(0), (w * (1.0 - phi_a)).reshape(2, m).sum(0)

    def residual(u):
        t, _, G = faces_of(u)
        ca, cb = contributions(t, G)
        return jnp.zeros(jnp.asarray(u).reshape(-1).shape[0], dtype=ca.dtype).at[a].add(ca).at[b].add(cb)

    # The exact tangent on the radiating nodes, without differentiating through the radiosity solve (k JVPs of
    # an m x m solve): with t = (u_a + u_b)/2 and G(t) = F J, M J = e sigma t^4, M = I - diag(1-e) F,
    #   M dJ = diag(h) dt,  h = d(e sigma t^4)/dt - e'(t) G   ->   dG/dt = F M^-1 diag(h),
    # and the term is pointwise in the face, so its partials in t and G are one JVP each along the ones vector.
    S = np.unique(elements)
    k = int(S.size)
    la, lb = np.searchsorted(S, elements[:, 0]), np.searchsorted(S, elements[:, 1])

    def tangent_block(u):
        t, e, G = faces_of(u)
        ones, zeros = jnp.ones_like(t), jnp.zeros_like(t)
        (ca, cb), (ca_t, cb_t) = jax.jvp(contributions, (t, G), (ones, zeros))
        _, (ca_G, cb_G) = jax.jvp(contributions, (t, G), (zeros, ones))
        de = jax.jvp(eps_faces, (t,), (ones,))[1]
        h = de * STEFAN_BOLTZMANN * t**4 + 4.0 * e * STEFAN_BOLTZMANN * t**3 - de * G
        B = jnp.zeros((k, k), dtype=ca.dtype)
        B = B.at[la, la].add(0.5 * ca_t).at[la, lb].add(0.5 * ca_t).at[lb, la].add(0.5 * cb_t).at[lb, lb].add(0.5 * cb_t)
        for part, lo, hi in zip(rad.parts, cuts[:-1], cuts[1:]):
            Fp = part.view_factor
            Minv = radiosity_resolvent(Fp, e[lo:hi])
            D = Fp @ (Minv * h[lo:hi][None, :])  # dG/dt of this medium, (m_p, m_p)
            pa, pb = la[lo:hi], lb[lo:hi]
            N = jnp.zeros((k, hi - lo), dtype=D.dtype).at[pa].add(ca_G[lo:hi, None] * D).at[pb].add(cb_G[lo:hi, None] * D)
            B = B.at[:, pa].add(0.5 * N).at[:, pb].add(0.5 * N)
        return B

    # the term must be linear in the test function: with v = 0 it has to vanish
    t0, _, G0 = faces_of(jnp.full(n_nodes, 1000.0))
    if float(jnp.max(jnp.abs(jnp.asarray(_face_eval(term, at_gauss(t0, G0, test=0.0)))))) > 0.0:
        raise ValueError(
            "d.irradiation: the radiation term must be linear in the test function (... * v * dS); "
            "with v = 0 it does not vanish."
        )

    names = "; ".join(",".join(medium) for medium in rad.media)
    c = Coupling(residual, name=f"radiation[{names}]", field_key=key)
    c._support = S
    c._tangent_block = tangent_block
    return c
