"""Classify the boundary entities of a *built* OCC model back to auto-names + provenance.

For each boundary entity (curve in 2-D, face in 3-D) we sample a point on it and test it
against the plan's primitive predicates (:mod:`jno.geometry.primitives`). The result is a
``{entity_tag: (leaf_key, local_name)}`` map:

* ``local_name`` -- the auto-name (``left/right/top/bottom``, ``arc``, or an extrude cap
  ``front/back``) that becomes a mesh region.
* ``leaf_key`` -- which primitive the boundary came from, for ``edges_from`` provenance.

Extrude caps get reserved keys (they belong to no primitive); a lateral face inherits the
name of the base curve it was swept from by classifying its ``(x, y)`` against the 2-D base
primitives.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Reserved provenance keys for extrusion caps (belong to no primitive leaf).
_CAP_BACK = -1
_CAP_FRONT = -2


def _sample_point(bdim: int, tag: int):
    """A point in the interior of boundary entity ``(bdim, tag)`` (param-space midpoint)."""
    import gmsh
    import numpy as np

    lo, hi = gmsh.model.getParametrizationBounds(bdim, tag)
    if bdim == 1:
        t = lo[0] + 0.5 * (hi[0] - lo[0])
        return np.asarray(gmsh.model.getValue(1, tag, [t]), dtype=float)
    u = lo[0] + 0.5 * (hi[0] - lo[0])
    v = lo[1] + 0.5 * (hi[1] - lo[1])
    return np.asarray(gmsh.model.getValue(2, tag, [u, v]), dtype=float)


def _classify_point(p, leaves, dim: int, extrude_dz: Optional[float], tol: float = 1e-6):
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if dim == 3 and extrude_dz is not None:
        if abs(z) < tol:
            return (_CAP_BACK, "back")
        if abs(z - extrude_dz) < tol:
            return (_CAP_FRONT, "front")
        # Lateral face: inherit the base curve's name via the 2-D predicate at (x, y).
        for prim, _size, key in leaves:
            name = prim.classify(x, y, 0.0)
            if name is not None:
                return (key, name)
        return None
    for prim, _size, key in leaves:
        name = prim.classify(x, y, z)
        if name is not None:
            return (key, name)
    return None


def classify_boundary(dim: int, leaves, extrude_dz: Optional[float]) -> Dict[int, Tuple[int, str]]:
    """``{entity_tag: (leaf_key, local_name)}`` for every classified boundary entity.

    Requires the OCC model to be synchronized. ``dim`` is the model dimension; boundary
    entities are queried at ``dim - 1``.
    """
    import gmsh

    bdim = dim - 1
    out: Dict[int, Tuple[int, str]] = {}
    for _edim, tag in gmsh.model.getEntities(bdim):
        label = _classify_point(_sample_point(bdim, tag), leaves, dim, extrude_dz)
        if label is not None:
            out[tag] = label
    return out
