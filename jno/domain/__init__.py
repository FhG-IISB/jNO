from .boundary_region import BoundaryRegion
from .domain_class import domain as _domain
from .domain_data import DomainData
from .geometries import Geometries
from .mesh_utils import MeshUtils
from .meshio_mixin import MeshIOMixin

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
]
