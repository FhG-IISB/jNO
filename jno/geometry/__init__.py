"""Friendly gmsh-OpenCASCADE geometry authoring for jNO.

``Shape`` is an immutable build-plan (primitives + boolean operators + transforms) that
meshes on demand and (later) also serves mesh-free membership sampling for PINNs. It is
the single geometry entry that consolidates the mesh-backed and (Shapely) polygon paths.
"""

from .path import Path
from .shape import Selector, Shape

__all__ = ["Shape", "Selector", "Path"]
