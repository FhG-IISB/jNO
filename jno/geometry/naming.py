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

import math
from typing import Dict, Tuple

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


def _rotate_point(x, y, z, axis_point, axis_dir, angle):
    """Rotate a point about the axis through ``axis_point`` with direction ``axis_dir`` by ``angle``."""
    px, py, pz = axis_point
    n = math.sqrt(axis_dir[0] ** 2 + axis_dir[1] ** 2 + axis_dir[2] ** 2) or 1.0
    ux, uy, uz = axis_dir[0] / n, axis_dir[1] / n, axis_dir[2] / n
    vx, vy, vz = x - px, y - py, z - pz
    c, s = math.cos(angle), math.sin(angle)
    dot = ux * vx + uy * vy + uz * vz
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx  # u x v
    rx = vx * c + cx * s + ux * dot * (1.0 - c)  # Rodrigues rotation
    ry = vy * c + cy * s + uy * dot * (1.0 - c)
    rz = vz * c + cz * s + uz * dot * (1.0 - c)
    return rx + px, ry + py, rz + pz


def _is_y_axis(axis_point, axis_dir):
    ax, ay, az = axis_dir
    return all(abs(c) < 1e-9 for c in axis_point) and abs(ay) > 1e-9 and abs(ax) < 1e-9 and abs(az) < 1e-9


def _revolve_profile_coords(p, axis_point, axis_dir):
    """Map a swept 3-D point to its ``(profile_x, profile_y)`` meridian coords.

    The 2-D profile lives in the z=0 plane on the positive-radius side; a revolved point keeps
    its meridian coords, so classifying against the profile primitives reuses their 2-D
    predicates. Only the x- or y-axis through the origin is supported (raises otherwise).
    """
    X, Y, Z = float(p[0]), float(p[1]), float(p[2])
    ax, ay, az = axis_dir
    at_origin = all(abs(c) < 1e-9 for c in axis_point)
    if _is_y_axis(axis_point, axis_dir):
        return math.hypot(X, Z), Y  # profile x=radius, y=height
    if at_origin and abs(ax) > 1e-9 and abs(ay) < 1e-9 and abs(az) < 1e-9:  # x-axis
        return X, math.hypot(Y, Z)  # profile x=along-axis, y=radius
    raise NotImplementedError(
        f"revolve currently supports the x- or y-axis through the origin; got axis_point={axis_point}, axis_dir={axis_dir}."
    )


def _rotate_about_axis(X, Y, Z, y_axis, s):
    c, sn = math.cos(s), math.sin(s)
    if y_axis:  # rotate about +y
        return X * c + Z * sn, Y, -X * sn + Z * c
    return X, Y * c - Z * sn, Y * sn + Z * c  # rotate about +x


def _on_profile_plane(p, axis_point, axis_dir, tol):
    """True if ``p`` lies on the original profile: the z=0 plane on the positive-radius side."""
    X, Y, Z = float(p[0]), float(p[1]), float(p[2])
    if abs(Z) >= tol:
        return False
    return (X > 0.0) if _is_y_axis(axis_point, axis_dir) else (Y > 0.0)


def _on_end_cap(p, axis_point, axis_dir, angle, tol):
    """True if ``p`` lies on the swept-end cap: rotating it back by the sweep lands on the profile.

    Wrap-free (no ``atan2``): the start cap is the profile plane; the end cap is that plane
    rotated by the sweep angle, so un-rotating by +/-angle returns it to the profile. This
    distinguishes the two caps even when they share the z=0 plane (a half-turn).
    """
    y_axis = _is_y_axis(axis_point, axis_dir)
    for s in (angle, -angle):
        if _on_profile_plane(_rotate_about_axis(p[0], p[1], p[2], y_axis, s), axis_point, axis_dir, tol):
            return True
    return False


def _classify(node, x: float, y: float, z: float, tol: float = 1e-6):
    """Classify a boundary point by walking the build-plan into each node's local frame.

    Each node transforms the point into its child's frame and recurses: undo a translate/rotate,
    project a lateral extrude point to the base plane (caps short-circuit), take meridian coords
    through a revolve (caps short-circuit), try both sides of a boolean, and finally a primitive.
    Returns ``(leaf_key, local_name)`` or ``None``. This composes for arbitrary nesting.
    """
    kind = node[0]
    if kind == "leaf":
        name = node[1].classify(x, y, z)
        return (node[2], name) if name is not None else None
    if kind in ("cut", "fuse", "inter"):
        return _classify(node[1]._node, x, y, z, tol) or _classify(node[2]._node, x, y, z, tol)
    if kind == "translate":
        dx, dy, dz = node[2]
        return _classify(node[1]._node, x - dx, y - dy, z - dz, tol)
    if kind == "rotate":
        lx, ly, lz = _rotate_point(x, y, z, node[2], node[3], -node[4])  # undo the rotation
        return _classify(node[1]._node, lx, ly, lz, tol)
    if kind == "fillet":
        # rounded blend faces match no flat predicate -> fall into `boundary`; flat faces keep names
        return _classify(node[1]._node, x, y, z, tol)
    if kind == "sweep":
        return None  # general sweep -> interior + boundary only (carve with d.tag afterwards)
    if kind == "extrude":
        dz = node[2]
        if abs(z) < tol:
            return (_CAP_BACK, "back")
        if abs(z - dz) < tol:
            return (_CAP_FRONT, "front")
        return _classify(node[1]._node, x, y, 0.0, tol)  # lateral face -> base plane
    if kind == "revolve":
        ap, ad, angle = node[2], node[3], node[4]
        if abs(angle - 2.0 * math.pi) > tol:  # partial sweep -> flat end caps
            if _on_profile_plane((x, y, z), ap, ad, tol):
                return (_CAP_BACK, "back")
            if _on_end_cap((x, y, z), ap, ad, angle, tol):
                return (_CAP_FRONT, "front")
        px, py = _revolve_profile_coords((x, y, z), ap, ad)
        return _classify(node[1]._node, px, py, 0.0, tol)  # swept face -> profile meridian
    return None


def classify_boundary(dim: int, shape) -> Dict[int, Tuple[int, str]]:
    """``{entity_tag: (leaf_key, local_name)}`` for every classified boundary entity.

    Requires the OCC model to be synchronized. Boundary entities are queried at ``dim - 1``;
    each sample point is classified by walking ``shape``'s build-plan (:func:`_classify`), so
    names survive booleans, dimension transitions, and rigid transforms at any nesting depth.
    """
    import gmsh

    bdim = dim - 1
    out: Dict[int, Tuple[int, str]] = {}
    for _edim, tag in gmsh.model.getEntities(bdim):
        p = _sample_point(bdim, tag)
        label = _classify(shape._node, float(p[0]), float(p[1]), float(p[2]))
        if label is not None:
            out[tag] = label
    return out
