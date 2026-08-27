"""Geometry to PEEC elements — the bridge between :mod:`jno.geometry` and the integral kernel.

A partial-element method never meshes: a conductor becomes a chain of straight filaments carrying
its centreline and cross-section, and the operator is then the Neumann double integral, which is
:func:`jno.utils.solver.kernel.pair_quadratic` with ``mom = tangent × length``. So this module holds
discretisation and nothing else — no physics the kernel already has, no solver.

Everything returns arrays shaped for the kernel, so the caller writes::

    pos, mom, self_g, group = line_filaments(wire, size=0.5)
    L = float(pair_quadratic(pos, mom, lambda r: 1/r, self_g, group)) * MU0 / (4 * jnp.pi)

Filaments carry their **analytic** cross-section, which is the reason to prefer them over a meshed
conductor wherever the geometry allows. A meshed cylinder is an inscribed polygon: at seven or eight
points around a 375 µm bond wire the mesh keeps only ~88 % of the true area, which makes the wire
~14 % too resistive. A filament has no faceting to lose.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .kernel import wire_self

__all__ = ["line_filaments"]


def _leaf(shape, what):
    node = getattr(shape, "_node", None)
    prim = node[1] if (isinstance(node, tuple) and node and node[0] == "leaf") else None
    if type(prim).__name__ != what:
        got = "no plan at all" if node is None else f"a {node[0] if prim is None else type(prim).__name__!r} plan"
        raise NotImplementedError(
            f"peec: expected a single {what} primitive, got {got}. A filament discretisation needs a "
            "centreline and a cross-section; a CSG plan has neither. Build the conductor from "
            "Shape.line (a tube along a polyline), or discretise a solid as a lattice instead."
        )
    return prim


def line_filaments(shape, size: float = None, quad: int = 3):
    """Discretise a :meth:`jno.Shape.line` into filaments, with Gauss sub-points.

    Args:
        shape: a ``Shape`` whose plan is a single ``Line`` leaf.
        size: target filament length. Defaults to the shape's own ``size=``.
        quad: Gauss points per filament. One point is 7.8 % low against a closed form on the worst
            case (collinear neighbours); 2 gives 2.5 %, 3 gives 1.2 %, 8 gives 0.21 %. Three is the
            default because it is where the curve flattens against its cost.

    Returns:
        ``(pos, mom, self_g, group)`` — sub-point positions ``(N·quad, 3)``, their moments
        ``(N·quad, 3)`` (tangent × length × Gauss weight), the per-FILAMENT self term ``(N,)``, and
        the ``(N·quad,)`` element labels the kernel needs to tell "inside one element" from
        "between two".

    The polyline is subdivided so that no filament exceeds ``size``, and each original vertex stays
    a filament boundary — a bend must not fall inside a straight element.
    """
    prim = _leaf(shape, "Line")
    h = float(size if size is not None else (shape._size if shape._size is not None else 0.0))
    P = np.asarray(prim.points, dtype=float).reshape(-1, 3)
    seg = P[1:] - P[:-1]
    L = np.linalg.norm(seg, axis=1)
    if h <= 0:
        raise ValueError(
            "peec.line_filaments: no filament length. Pass size=, or give the Shape a size= when "
            "you build it — a filament count cannot be guessed from the geometry alone."
        )

    starts, tangs, lens = [], [], []
    for a, d, ln in zip(P[:-1], seg, L):
        if ln <= 0.0:
            continue
        k = max(1, int(np.ceil(ln / h)))  # vertices stay filament boundaries: subdivide within a segment
        u = d / ln
        step = ln / k
        for j in range(k):
            starts.append(a + u * (step * (j + 0.5)))
            tangs.append(u)
            lens.append(step)
    if not starts:
        raise ValueError("peec.line_filaments: the polyline has no segment longer than zero.")

    cen = np.asarray(starts)
    tan = np.asarray(tangs)
    ln = np.asarray(lens)
    n = len(ln)

    gx, gw = np.polynomial.legendre.leggauss(int(quad))
    # sub-points along each filament, and moments that sum to `tangent * length` per filament
    pos = (cen[:, None, :] + 0.5 * ln[:, None, None] * gx[None, :, None] * tan[:, None, :]).reshape(-1, 3)
    mom = (tan[:, None, :] * (ln[:, None] * gw[None, :] * 0.5)[:, :, None]).reshape(-1, 3)
    group = np.repeat(np.arange(n), int(quad))
    self_g = np.asarray(wire_self(jnp.asarray(ln), prim.r))
    return jnp.asarray(pos), jnp.asarray(mom), jnp.asarray(self_g), group
