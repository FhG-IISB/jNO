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

# jno.domain(geo) now dispatches to PolygonDomain via domain.__new__.
# .csg kept as a backward-compatible alias for code that still uses it.
_domain.csg = _PolygonDomain

# Expose multi-geometry batch stacking as jno.domain.stack(n*d1, n*d2, ...)
_domain.stack = staticmethod(_PolygonDomain.stack)
