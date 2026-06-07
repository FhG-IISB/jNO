from .boundary_region import BoundaryRegion
from .domain_class import domain as _domain
from .domain_data import DomainData
from .geometries import Geometries
from .mesh_utils import MeshUtils
from .meshio_mixin import MeshIOMixin
from .polygon_domain import PolygonDomain as _PolygonDomain

# Preserve historical import path for pickling/repr compatibility.
_domain.__module__ = __name__
domain = _domain

__all__ = [
    "DomainData",
    "Geometries",
    "MeshUtils",
    "BoundaryRegion",
    "MeshIOMixin",
    "domain",
    "from_array",
]

from_array = _domain.from_array

# User-facing entry point for the lazy Shapely-backed CSG domain.
# `jno.domain.csg([...])` constructs a single-polygon CSG domain;
# `jno.domain.csg.from_polygons({...})` and `.from_regions({...})` come along
# unchanged. The underlying class itself is intentionally not re-exported
# at the package top level.
_domain.csg = _PolygonDomain
