from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .boundary_region import BoundaryRegion
from .domain_class import domain

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry
else:
    BaseGeometry = Any

contains_xy = None  # type: ignore[assignment]
GeometryCollection = LineString = Point = Polygon = None  # type: ignore[assignment]
triangulate = unary_union = explain_validity = None  # type: ignore[assignment]
_SHAPELY_IMPORT_ERROR: Optional[BaseException] = None


def _refresh_shapely_imports() -> None:
    global contains_xy, GeometryCollection, LineString, Point, Polygon
    global triangulate, unary_union, explain_validity, _SHAPELY_IMPORT_ERROR

    if Point is not None and GeometryCollection is not None and LineString is not None and Polygon is not None and triangulate is not None and unary_union is not None:
        return

    try:  # pragma: no cover - exercised only when the optional import is missing
        from shapely.geometry import GeometryCollection as _GeometryCollection, LineString as _LineString, Point as _Point, Polygon as _Polygon
        from shapely.ops import triangulate as _triangulate, unary_union as _unary_union
        from shapely.validation import explain_validity as _explain_validity

        try:
            from shapely import contains_xy as _contains_xy
        except Exception:
            _contains_xy = None
    except Exception as exc:  # pragma: no cover
        _SHAPELY_IMPORT_ERROR = exc
        return

    contains_xy = _contains_xy  # type: ignore[assignment]
    GeometryCollection = _GeometryCollection  # type: ignore[assignment]
    LineString = _LineString  # type: ignore[assignment]
    Point = _Point  # type: ignore[assignment]
    Polygon = _Polygon  # type: ignore[assignment]
    triangulate = _triangulate  # type: ignore[assignment]
    unary_union = _unary_union  # type: ignore[assignment]
    explain_validity = _explain_validity  # type: ignore[assignment]
    _SHAPELY_IMPORT_ERROR = None


_refresh_shapely_imports()


_GEOM_TOL = 1e-14


def _require_shapely() -> None:
    _refresh_shapely_imports()
    if _SHAPELY_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError("PolygonDomain requires shapely. Install jno with shapely support or add `shapely>=2.0`.") from _SHAPELY_IMPORT_ERROR


def _as_xy_vertices(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected ordered vertices with shape (N, 2) or (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Polygon vertices must be finite numbers")
    if arr.shape[1] == 3:
        if not np.allclose(arr[:, 2], arr[0, 2]):
            raise ValueError("PolygonDomain is 2D; (N, 3) vertices must have a constant z-coordinate")
        arr = arr[:, :2]
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if arr.shape[0] < 3:
        raise ValueError("A polygon requires at least three unique vertices")

    edges = np.roll(arr, -1, axis=0) - arr
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= _GEOM_TOL):
        raise ValueError("Polygon vertices contain a zero-length edge")
    if np.unique(np.round(arr, decimals=14), axis=0).shape[0] < 3:
        raise ValueError("A polygon requires at least three unique vertices")
    return arr


def _polygon_from_vertices(vertices: Sequence[Sequence[float]]) -> Tuple[BaseGeometry, List[BaseGeometry]]:
    _require_shapely()
    arr = _as_xy_vertices(vertices)
    poly = Polygon(arr)  # type: ignore[operator]
    if poly.is_empty or poly.area <= _GEOM_TOL:
        raise ValueError("Polygon area must be positive")
    if not poly.is_valid:
        reason = explain_validity(poly) if explain_validity is not None else "invalid geometry"
        raise ValueError(f"Invalid polygon: {reason}")
    edges = [LineString([arr[i], arr[(i + 1) % len(arr)]]) for i in range(len(arr))]  # type: ignore[operator]
    return poly, edges


def _polygon_parts(geom: BaseGeometry) -> List[BaseGeometry]:
    if geom is None or geom.is_empty:
        return []
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        return [geom] if geom.area > _GEOM_TOL else []
    if geom_type == "MultiPolygon":
        return [part for part in geom.geoms if part.area > _GEOM_TOL]
    if geom_type == "GeometryCollection":
        parts: List[BaseGeometry] = []
        for sub in geom.geoms:
            parts.extend(_polygon_parts(sub))
        return parts
    return []


def _line_parts(geom: BaseGeometry) -> List[BaseGeometry]:
    if geom is None or geom.is_empty:
        return []
    geom_type = geom.geom_type
    if geom_type in {"LineString", "LinearRing"}:
        return [geom] if geom.length > _GEOM_TOL else []
    if geom_type == "MultiLineString":
        return [part for part in geom.geoms if part.length > _GEOM_TOL]
    if geom_type == "Polygon":
        return _line_parts(geom.boundary)
    if geom_type == "MultiPolygon":
        parts: List[BaseGeometry] = []
        for sub in geom.geoms:
            parts.extend(_line_parts(sub.boundary))
        return parts
    if geom_type == "GeometryCollection":
        parts = []
        for sub in geom.geoms:
            parts.extend(_line_parts(sub))
        return parts
    return []


def _as_polygonal_geometry(geom: BaseGeometry) -> BaseGeometry:
    parts = _polygon_parts(geom)
    if not parts:
        return GeometryCollection()  # type: ignore[operator]
    return unary_union(parts)  # type: ignore[misc]


def _as_line_geometry(geom: BaseGeometry) -> BaseGeometry:
    parts = _line_parts(geom)
    if not parts:
        return GeometryCollection()  # type: ignore[operator]
    return unary_union(parts)  # type: ignore[misc]


def _segments_from_line_geometry(geom: BaseGeometry) -> np.ndarray:
    segments: List[np.ndarray] = []
    for line in _line_parts(geom):
        coords = np.asarray(line.coords, dtype=np.float64)
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            if np.linalg.norm(p1 - p0) > _GEOM_TOL:
                segments.append(np.stack([p0, p1], axis=0))
    if not segments:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.stack(segments, axis=0)


def _edge_geometries_from_region(geom: BaseGeometry) -> List[BaseGeometry]:
    return [_as_line_geometry(line) for line in _line_parts(geom.boundary)]


def _unique_segment_points(segments: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = segments.reshape(-1, 2)
    _, unique_idx = np.unique(np.round(pts, decimals=14), axis=0, return_index=True)
    return pts[np.sort(unique_idx)]


class PolygonDomain(domain):
    """Lazy Shapely-backed 2D polygon domain with true CSG operators.

    The class preserves the ``jno.domain`` variable/context contract but does
    not create a mesh. Point sets are materialized only when ``variable`` or
    ``sample`` is called with an explicit sample count.
    """

    def __init__(
        self,
        vertices: Optional[Sequence[Sequence[float]]] = None,
        *,
        name: str = "polygon",
        geometry: Optional[BaseGeometry] = None,
        regions: Optional[Mapping[str, BaseGeometry]] = None,
        source_edges: Optional[Mapping[str, Sequence[BaseGeometry]]] = None,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = False,
    ):
        _require_shapely()
        if compute_mesh_connectivity:
            compute_mesh_connectivity = False

        self._init_empty_state(
            constructor_source="PolygonDomain",
            algorithm=6,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )
        self.dimension = 2
        self.indep = ["x", "y", "t"]
        self.spatial = ["x", "y"]

        self._polygon_name = str(name)
        self._polygon_tags: Dict[str, Tuple[str, BaseGeometry]] = {}
        self._polygon_boundary_segments: Dict[str, np.ndarray] = {}
        self._polygon_boundary_normal_geometries: Dict[str, BaseGeometry] = {}
        self._area_part_cache: Dict[str, Tuple[List[BaseGeometry], np.ndarray]] = {}

        if geometry is None:
            if vertices is None:
                raise ValueError("PolygonDomain requires vertices or an internal geometry")
            polygon, edges = _polygon_from_vertices(vertices)
            regions = {self._polygon_name: polygon}
            source_edges = {self._polygon_name: edges}
            geometry = polygon
        else:
            geometry = _as_polygonal_geometry(geometry)

        self._active_geometry = geometry
        self._source_regions: Dict[str, BaseGeometry] = {str(k): _as_polygonal_geometry(v) for k, v in (regions or {}).items()}
        self._source_edges: Dict[str, List[BaseGeometry]] = {str(k): [_as_line_geometry(edge) for edge in edges] for k, edges in (source_edges or {}).items()}
        self._normal_eps = self._estimate_polygon_tol(self._active_geometry)

        if self._is_time_dependent:
            if time is None:
                raise RuntimeError("Internal error: time-dependent PolygonDomain has time=None")
            self._time_points = np.linspace(time[0], time[1], time[2])
            self._n_time = time[2]
            self.context["__time__"] = self._time_points[:, np.newaxis]
        else:
            self.context["__time__"] = np.ones((1, 1))

        self._rebuild_polygon_tags()

    @classmethod
    def from_polygons(
        cls,
        polygons: Mapping[str, Sequence[Sequence[float]]],
        *,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = False,
    ) -> "PolygonDomain":
        """Create one CSG domain from a mapping of region names to vertices."""
        _require_shapely()
        regions: Dict[str, BaseGeometry] = {}
        source_edges: Dict[str, List[BaseGeometry]] = {}
        for name, vertices in polygons.items():
            poly, edges = _polygon_from_vertices(vertices)
            key = str(name)
            if key in regions:
                raise ValueError(f"Duplicate polygon region name: {key}")
            regions[key] = poly
            source_edges[key] = edges
        active = unary_union(list(regions.values())) if regions else GeometryCollection()  # type: ignore[misc, operator]
        return cls(
            geometry=active,
            name="polygon_scene",
            regions=regions,
            source_edges=source_edges,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def from_regions(
        cls,
        regions: Mapping[str, BaseGeometry],
        *,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = False,
    ) -> "PolygonDomain":
        """Create one CSG domain from named Shapely polygonal regions."""
        _require_shapely()
        normalized_regions: Dict[str, BaseGeometry] = {}
        source_edges: Dict[str, List[BaseGeometry]] = {}
        for name, geometry in regions.items():
            key = str(name)
            if key in normalized_regions:
                raise ValueError(f"Duplicate polygon region name: {key}")
            polygonal = _as_polygonal_geometry(geometry)
            if polygonal.is_empty or polygonal.area <= _GEOM_TOL:
                continue
            normalized_regions[key] = polygonal
            source_edges[key] = _edge_geometries_from_region(polygonal)

        active = unary_union(list(normalized_regions.values())) if normalized_regions else GeometryCollection()  # type: ignore[misc, operator]
        return cls(
            geometry=active,
            name="polygon_scene",
            regions=normalized_regions,
            source_edges=source_edges,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    def _estimate_polygon_tol(self, geom: BaseGeometry) -> float:
        if geom is None or geom.is_empty:
            return 1e-10
        minx, miny, maxx, maxy = geom.bounds
        diag = float(np.hypot(maxx - minx, maxy - miny))
        return max(1e-12, 1e-8 * max(diag, 1.0))

    def _is_polygon_tag(self, tag: str) -> bool:
        return tag in self._polygon_tags

    def _rebuild_polygon_tags(self) -> None:
        self._polygon_tags = {}
        self._polygon_boundary_segments = {}
        self._polygon_boundary_normal_geometries = {}
        self._area_part_cache = {}
        self._boundary_registry = {}
        self._boundary_regions = {}
        self.avaiable_mesh_tags = []

        if not self._active_geometry.is_empty and self._active_geometry.area > _GEOM_TOL:
            self._register_interior_tag("interior", self._active_geometry)
            self._register_boundary_tag("boundary", self._active_geometry.boundary)

        active_boundary = self._active_geometry.boundary if not self._active_geometry.is_empty else GeometryCollection()  # type: ignore[operator]
        for name, region in self._source_regions.items():
            clipped_region = _as_polygonal_geometry(region.intersection(self._active_geometry))
            if not clipped_region.is_empty and clipped_region.area > _GEOM_TOL:
                self._register_interior_tag(f"interior_{name}", clipped_region)

            source_edge_geoms: List[BaseGeometry] = []
            region_is_active = not clipped_region.is_empty and clipped_region.area > _GEOM_TOL
            for edge_idx, edge in enumerate(self._source_edges.get(name, [])):
                edge = _as_line_geometry(edge)
                if edge.is_empty or edge.length <= _GEOM_TOL:
                    continue
                active_piece = _as_line_geometry(edge.intersection(active_boundary))
                normal_geometry = region if region_is_active or active_piece.is_empty else self._active_geometry
                tag = f"boundary_{name}_{edge_idx}"
                self._register_boundary_tag(tag, edge, normal_geometry=normal_geometry)
                source_edge_geoms.append(edge)
            if source_edge_geoms:
                self._register_boundary_tag(f"boundary_{name}", unary_union(source_edge_geoms), normal_geometry=region if region_is_active else self._active_geometry)  # type: ignore[misc]

    def _register_interior_tag(self, tag: str, geom: BaseGeometry) -> None:
        geom = _as_polygonal_geometry(geom)
        if geom.is_empty or geom.area <= _GEOM_TOL:
            return
        self._polygon_tags[tag] = ("interior", geom)
        self.avaiable_mesh_tags.append(tag)

    def _register_boundary_tag(self, tag: str, geom: BaseGeometry, *, normal_geometry: Optional[BaseGeometry] = None) -> None:
        geom = _as_line_geometry(geom)
        if geom.is_empty or geom.length <= _GEOM_TOL:
            return
        segments = _segments_from_line_geometry(geom)
        if len(segments) == 0:
            return
        points = _unique_segment_points(segments)
        self._polygon_tags[tag] = ("boundary", geom)
        self._polygon_boundary_segments[tag] = segments
        self._polygon_boundary_normal_geometries[tag] = normal_geometry if normal_geometry is not None else self._active_geometry
        self.avaiable_mesh_tags.append(tag)
        self._boundary_regions[tag] = BoundaryRegion(
            tag=tag,
            dim=2,
            points=points,
            edges=segments,
            tol=self._estimate_polygon_tol(geom),
        )
        self._boundary_registry[tag] = {
            "tag": tag,
            "entity_kind": "line",
            "point_indices": np.arange(len(points), dtype=int),
            "points": points,
        }

    def add_boundary_segments(self, tag: str, segments: Sequence[Sequence[Sequence[float]]], *, normal_geometry: Optional[Any] = None) -> "PolygonDomain":
        """Register an additional boundary tag from explicit line segments.

        This is intended for imported boundary-condition/radiation surfaces
        that are subsets of component boundaries rather than whole closed
        polygons.
        """
        _require_shapely()
        line_geometries = []
        for raw_segment in segments:
            arr = np.asarray(raw_segment, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] != 2 or arr.shape[1] not in (2, 3):
                raise ValueError(f"Boundary segment must have shape (2, 2) or (2, 3), got {arr.shape}")
            arr = arr[:, :2]
            if np.linalg.norm(arr[1] - arr[0]) <= _GEOM_TOL:
                continue
            line_geometries.append(LineString(arr))  # type: ignore[operator]

        if not line_geometries:
            raise ValueError(f"No positive-length segments supplied for boundary tag '{tag}'")

        if isinstance(normal_geometry, str):
            normal_geometry = self._source_regions.get(normal_geometry)
        elif isinstance(normal_geometry, Sequence) and not isinstance(normal_geometry, (bytes, bytearray)):
            normal_parts = [self._source_regions[name] for name in normal_geometry if isinstance(name, str) and name in self._source_regions]
            if normal_parts:
                normal_geometry = _as_polygonal_geometry(unary_union(normal_parts))  # type: ignore[misc]

        geom = unary_union(line_geometries)  # type: ignore[misc]
        self._register_boundary_tag(str(tag), geom, normal_geometry=normal_geometry)
        return self

    def _next_context_tag(self, tag: str) -> str:
        if tag not in self.context or tag in self._param_tags:
            return tag
        idx = 0
        while f"{tag}_{idx}" in self.context:
            idx += 1
        return f"{tag}_{idx}"

    def _area_parts_for_tag(self, tag: str, geom: BaseGeometry) -> Tuple[List[BaseGeometry], np.ndarray]:
        cached = self._area_part_cache.get(tag)
        if cached is not None:
            return cached

        parts: List[BaseGeometry] = []
        for poly in _polygon_parts(geom):
            triangles = triangulate(poly)  # type: ignore[misc]
            for tri in triangles:
                piece = _as_polygonal_geometry(tri.intersection(poly))
                parts.extend(_polygon_parts(piece))
        if not parts:
            parts = _polygon_parts(geom)
        areas = np.asarray([part.area for part in parts], dtype=np.float64)
        valid = areas > _GEOM_TOL
        parts = [part for part, keep in zip(parts, valid) if keep]
        areas = areas[valid]
        if len(parts) == 0 or float(areas.sum()) <= _GEOM_TOL:
            raise ValueError(f"Tag '{tag}' has no positive-area geometry to sample")
        probs = areas / areas.sum()
        self._area_part_cache[tag] = (parts, probs)
        return parts, probs

    def _sample_points_in_polygon(self, geom: BaseGeometry, n_samples: int) -> np.ndarray:
        minx, miny, maxx, maxy = geom.bounds
        if maxx <= minx or maxy <= miny:
            raise ValueError("Cannot sample from a degenerate polygon part")
        accepted: List[np.ndarray] = []
        remaining = n_samples
        attempts = 0
        while remaining > 0:
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError("Could not sample enough interior polygon points; check geometry validity")
            batch = max(64, remaining * 4)
            xs = np.random.uniform(minx, maxx, size=batch)
            ys = np.random.uniform(miny, maxy, size=batch)
            if contains_xy is not None:
                mask = np.asarray(contains_xy(geom, xs, ys), dtype=bool)
            else:  # pragma: no cover
                mask = np.asarray([geom.contains(Point(float(x), float(y))) for x, y in zip(xs, ys)], dtype=bool)
            pts = np.column_stack([xs[mask], ys[mask]])
            if len(pts) == 0:
                continue
            take = min(remaining, len(pts))
            accepted.append(pts[:take])
            remaining -= take
        return np.concatenate(accepted, axis=0)

    def _sample_interior(self, tag: str, geom: BaseGeometry, n_samples: int) -> np.ndarray:
        parts, probs = self._area_parts_for_tag(tag, geom)
        choices = np.random.choice(len(parts), size=n_samples, p=probs)
        result = np.zeros((n_samples, 2), dtype=np.float64)
        cursor = 0
        for part_idx in range(len(parts)):
            count = int(np.sum(choices == part_idx))
            if count == 0:
                continue
            pts = self._sample_points_in_polygon(parts[part_idx], count)
            result[cursor : cursor + count] = pts
            cursor += count
        if cursor != n_samples:
            result = result[:cursor]
        np.random.shuffle(result)
        return result

    def _sample_boundary(self, tag: str, n_samples: int, with_normals: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        segments = self._polygon_boundary_segments.get(tag)
        if segments is None or len(segments) == 0:
            raise ValueError(f"Tag '{tag}' has no boundary segments to sample")
        vectors = segments[:, 1, :] - segments[:, 0, :]
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > _GEOM_TOL
        segments = segments[valid]
        vectors = vectors[valid]
        lengths = lengths[valid]
        if len(segments) == 0:
            raise ValueError(f"Tag '{tag}' has no positive-length boundary segments to sample")
        probs = lengths / lengths.sum()
        choices = np.random.choice(len(segments), size=n_samples, p=probs)
        eps = np.finfo(np.float64).eps
        t = np.random.uniform(eps, 1.0 - eps, size=n_samples)
        p0 = segments[choices, 0, :]
        vec = vectors[choices]
        points = p0 + t[:, None] * vec
        normals = None
        if with_normals:
            normal_geometry = self._polygon_boundary_normal_geometries.get(tag, self._active_geometry)
            normals = np.stack([self._outward_normal(p0_i, p0_i + vec_i, pt_i, normal_geometry) for p0_i, vec_i, pt_i in zip(p0, vec, points)], axis=0)
        return points, normals

    def _outward_normal(self, p0: np.ndarray, p1: np.ndarray, point: np.ndarray, normal_geometry: BaseGeometry) -> np.ndarray:
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length <= _GEOM_TOL:
            return np.zeros(2, dtype=np.float64)
        tangent = tangent / length
        left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        right = -left
        eps = self._normal_eps
        left_inside = normal_geometry.contains(Point(*(point + eps * left)))
        right_inside = normal_geometry.contains(Point(*(point + eps * right)))
        if left_inside and not right_inside:
            return right
        if right_inside and not left_inside:
            return left
        if not normal_geometry.is_empty:
            rep = np.asarray(normal_geometry.representative_point().coords[0], dtype=np.float64)
            return left if np.dot(left, point - rep) >= np.dot(right, point - rep) else right
        return left

    @staticmethod
    def _segment_key(segment: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        start = tuple(np.round(segment[0], decimals=14))
        end = tuple(np.round(segment[1], decimals=14))
        return tuple(sorted((start, end)))  # type: ignore[return-value]

    def _polygon_occluder_segments(self) -> np.ndarray:
        unique_segments: Dict[Tuple[Tuple[float, float], Tuple[float, float]], np.ndarray] = {}
        for edge_list in self._source_edges.values():
            for edge in edge_list:
                for segment in _segments_from_line_geometry(edge):
                    unique_segments.setdefault(self._segment_key(segment), segment)

        if not unique_segments:
            boundary_segments = self._polygon_boundary_segments.get("boundary")
            if boundary_segments is not None:
                for segment in boundary_segments:
                    unique_segments.setdefault(self._segment_key(segment), segment)

        if not unique_segments:
            return np.zeros((0, 2, 2), dtype=np.float64)
        return np.stack(list(unique_segments.values()), axis=0)

    @staticmethod
    def _points_on_segments(points: np.ndarray, segments: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        if len(points) == 0 or len(segments) == 0:
            return np.zeros((len(points), len(segments)), dtype=bool)

        p = points[:, None, :]
        s0 = segments[None, :, 0, :]
        s1 = segments[None, :, 1, :]
        seg = s1 - s0
        rel = p - s0

        cross = seg[..., 0] * rel[..., 1] - seg[..., 1] * rel[..., 0]
        dot = rel[..., 0] * seg[..., 0] + rel[..., 1] * seg[..., 1]
        seg_len2 = seg[..., 0] * seg[..., 0] + seg[..., 1] * seg[..., 1]

        return (np.abs(cross) <= tol) & (dot >= -tol) & (dot <= seg_len2 + tol)

    def _polygon_visibility_matrix(self, points: np.ndarray) -> np.ndarray:
        segments = self._polygon_occluder_segments()
        n_points = points.shape[0]
        if n_points == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if len(segments) == 0:
            return np.ones((n_points, n_points), dtype=np.float32) - np.eye(n_points, dtype=np.float32)

        segment_start = segments[:, 0, :]
        segment_end = segments[:, 1, :]
        segment_dir = segment_end - segment_start
        point_on_segment = self._points_on_segments(points, segments)

        def cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        visibility = np.zeros((n_points, n_points), dtype=np.float32)
        for source_idx in range(n_points):
            source = points[source_idx]
            source_to_target = points - source
            source_to_target_exp = source_to_target[:, None, :]
            seg_dir_exp = segment_dir[None, :, :]
            diff_row = (segment_start - source)[None, :, :]

            denom = cross2d(source_to_target_exp, seg_dir_exp)
            parallel = np.abs(denom) < 1e-12
            denom_safe = np.where(parallel, 1.0, denom)

            t_seg = cross2d(diff_row, seg_dir_exp) / denom_safe
            t_edge = cross2d(diff_row, source_to_target_exp) / denom_safe

            eps = 1e-10
            crossings = (~parallel) & (t_seg > eps) & (t_seg < 1.0 - eps) & (t_edge > eps) & (t_edge < 1.0 - eps)
            crossings[:, point_on_segment[source_idx]] = False
            crossings &= ~point_on_segment

            visible = ~np.any(crossings, axis=1)
            visible[source_idx] = False
            visibility[source_idx] = visible.astype(np.float32)

        return visibility

    def _polygon_boundary_ds(self, tag: str, n_points: int) -> np.ndarray:
        segments = self._polygon_boundary_segments.get(tag)
        if segments is None or len(segments) == 0 or n_points <= 0:
            return np.zeros((n_points,), dtype=np.float64)
        total_length = float(np.linalg.norm(segments[:, 1, :] - segments[:, 0, :], axis=1).sum())
        if total_length <= _GEOM_TOL:
            return np.ones((n_points,), dtype=np.float64)
        return np.full((n_points,), total_length / float(n_points), dtype=np.float64)

    def _normalize_medium_tags(self, medium_tags: Optional[Sequence[str]]) -> List[str]:
        if medium_tags is None:
            medium_tags = [name for name in self._source_regions if str(name).lower() in {"gas", "air"}]
        if isinstance(medium_tags, str):
            medium_tags = [medium_tags]

        names: List[str] = []
        for raw_name in medium_tags:
            name = str(raw_name)
            for prefix in ("interior_", "boundary_"):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            if name in self._source_regions and name not in names:
                names.append(name)
        return names

    def _medium_geometry(self, medium_tags: Optional[Sequence[str]]) -> Optional[BaseGeometry]:
        names = self._normalize_medium_tags(medium_tags)
        if not names:
            return None
        medium_parts: List[BaseGeometry] = []
        infer_air = any(name.lower() == "air" for name in names)
        for name in names:
            if name.lower() != "air":
                medium_parts.append(self._source_regions[name])

        if infer_air:
            minx, miny, maxx, maxy = self._active_geometry.bounds
            scene_box = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])  # type: ignore[operator]
            occupied = [region for name, region in self._source_regions.items() if name.lower() != "air"]
            if occupied:
                inferred_air = _as_polygonal_geometry(scene_box.difference(unary_union(occupied)))  # type: ignore[misc]
            else:
                inferred_air = _as_polygonal_geometry(scene_box)
            medium_parts.append(inferred_air)

        if not medium_parts:
            return None
        medium = _as_polygonal_geometry(unary_union(medium_parts))  # type: ignore[misc]
        solid_parts = [region for name, region in self._source_regions.items() if name not in names]
        if solid_parts:
            medium = _as_polygonal_geometry(medium.difference(unary_union(solid_parts)))  # type: ignore[misc]
        return medium

    def _orient_normals_to_medium(self, points: np.ndarray, normals: np.ndarray, medium_geom: Optional[BaseGeometry]) -> np.ndarray:
        if medium_geom is None or medium_geom.is_empty:
            return normals

        oriented = np.asarray(normals, dtype=np.float64).copy()
        eps = self._normal_eps
        for idx, (point, normal) in enumerate(zip(points, oriented)):
            norm = float(np.linalg.norm(normal))
            if norm <= _GEOM_TOL:
                continue
            unit = normal / norm
            plus_inside = medium_geom.contains(Point(*(point + eps * unit)))
            minus_inside = medium_geom.contains(Point(*(point - eps * unit)))
            if minus_inside and not plus_inside:
                oriented[idx] = -normal
        return oriented

    def _filter_visibility_to_medium(self, points: np.ndarray, visibility: np.ndarray, medium_geom: Optional[BaseGeometry]) -> np.ndarray:
        """Keep only visible rays whose interiors travel through the radiation medium.

        Segment-intersection ray tracing correctly blocks rays crossing opaque
        boundaries, but intersections exactly at the source/target boundary
        points must be ignored.  That leaves a corner case where two different
        surfaces of the same solid can see each other through the solid
        interior.  Filtering a few interior points against the Gas/Air union
        removes those physically invalid paths while preserving boundary-grazing
        rays with a small buffer.
        """
        if medium_geom is None or medium_geom.is_empty or points.shape[0] == 0:
            return visibility

        filtered = np.array(visibility, copy=True)
        medium_buffer = medium_geom.buffer(max(10.0 * self._normal_eps, 1e-12))
        fractions = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)
        blocked_count = 0

        for source_idx, source in enumerate(points):
            candidate_idx = np.flatnonzero(filtered[source_idx] > 0)
            if len(candidate_idx) == 0:
                continue

            targets = points[candidate_idx]
            samples = source[None, None, :] + fractions[None, :, None] * (targets[:, None, :] - source[None, None, :])
            flat_samples = samples.reshape(-1, 2)
            if contains_xy is not None:
                inside = np.asarray(contains_xy(medium_buffer, flat_samples[:, 0], flat_samples[:, 1]), dtype=bool)
            else:  # pragma: no cover
                inside = np.asarray([medium_buffer.contains(Point(float(x), float(y))) for x, y in flat_samples], dtype=bool)
            valid = inside.reshape(len(candidate_idx), len(fractions)).all(axis=1)
            if np.any(~valid):
                blocked_count += int(np.sum(~valid))
                filtered[source_idx, candidate_idx[~valid]] = 0.0

        if blocked_count:
            self.log.info(f"Blocked {blocked_count} polygon visibility rays outside the radiation medium")
        return filtered

    def _filter_visibility_by_normals(self, points: np.ndarray, normals: np.ndarray, visibility: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Keep only rays for which both boundary normals face the segment.

        For radiative exchange, point ``j`` is only visible from point ``i`` if
        the ray direction ``i -> j`` lies in the forward hemisphere of ``i`` and
        the reverse ray ``j -> i`` lies in the forward hemisphere of ``j``.
        Perpendicular cases are excluded by requiring a strictly positive cosine
        at both ends.
        """
        if points.shape[0] == 0:
            return visibility

        vectors = points[None, :, :] - points[:, None, :]
        distances = np.linalg.norm(vectors, axis=-1)
        safe_distances = distances + np.eye(points.shape[0], dtype=np.float64)
        ray_hat = vectors / safe_distances[..., None]

        cos_i = np.sum(normals[:, None, :] * ray_hat, axis=-1)
        cos_j = -np.sum(normals[None, :, :] * ray_hat, axis=-1)
        facing = (cos_i > tol) & (cos_j > tol)

        filtered = np.array(visibility, copy=True)
        blocked = int(np.sum((filtered > 0) & (~facing)))
        filtered[~facing] = 0.0
        if blocked:
            self.log.info(f"Blocked {blocked} polygon visibility rays by boundary normal orientation")
        return filtered

    def _sampled_boundary_arrays(self, tag: str) -> Tuple[np.ndarray, np.ndarray]:
        if tag not in self.context:
            raise ValueError(f"Boundary tag '{tag}' not yet sampled. Call domain.variable('{tag}', sample=(n, None), normals=True) first.")
        if f"n_{tag}" not in self.context:
            raise ValueError(f"Boundary tag '{tag}' needs normals. Call domain.variable('{tag}', sample=(n, None), normals=True) first.")
        points = np.asarray(self.context[tag], dtype=np.float64)
        normals = np.asarray(self.context[f"n_{tag}"], dtype=np.float64)
        while points.ndim > 2:
            points = points[0]
        while normals.ndim > 2:
            normals = normals[0]
        return points, normals

    def _attach_polygon_view_factor(self, tag: str) -> None:
        kind_geom = self._polygon_tags.get(tag)
        if kind_geom is None:
            raise ValueError(f"Tag '{tag}' is not registered on PolygonDomain")
        kind, _ = kind_geom
        if kind != "boundary":
            raise ValueError(f"View factors are only available for boundary tags, got '{tag}'")
        if tag not in self.context:
            raise ValueError(f"Boundary tag '{tag}' has not been sampled yet")
        if f"n_{tag}" not in self.context:
            raise ValueError(f"Boundary tag '{tag}' requires normals before computing view factors")

        points = np.asarray(self.context[tag][0, 0], dtype=np.float64)
        # Match the mesh-domain convention: sampled boundary normals point
        # outward from the tagged region, while the view-factor kernels expect
        # normals pointing into the participating medium.
        normals = -np.asarray(self.context[f"n_{tag}"][0, 0], dtype=np.float64)
        visibility = self._polygon_visibility_matrix(points)
        visibility = self._filter_visibility_by_normals(points, normals, visibility)
        ds = self._polygon_boundary_ds(tag, points.shape[0])

        if self.dimension == 1:
            view_factor = np.asarray(self.get_view_factor_1d(points, visibility, normals, ds))
        elif self.dimension == 2:
            view_factor = np.asarray(self.get_view_factor_2d(points, visibility, normals, ds))
        else:
            view_factor = np.asarray(self.get_view_factor_3d(points, visibility, normals, ds))

        self.context[f"v_{tag}"] = visibility[None, None, ...]
        self._param_tags.add(f"v_{tag}")
        self.context[f"f_{tag}"] = view_factor[None, ...]
        self._param_tags.add(f"f_{tag}")

    def compute_enclosure_view_factor(self, tags, opaque_tags=None, medium_tags: Optional[Sequence[str]] = None):
        """Compute cross-tag polygon boundary view factors for radiative BCs.

        All *tags* must be polygon boundary tags that have already been sampled
        with normals.  The method ray-traces line-of-sight against all known
        polygon boundary segments, then stores one visibility block and one
        view-factor block for every source/target tag pair:

        ``v_<source>__<target>`` and ``f_<source>__<target>``.

        Args:
            tags: Boundary tags participating in the radiation enclosure.
            opaque_tags: Accepted for API compatibility. PolygonDomain uses all
                known polygon boundaries as opaque blockers, so this argument is
                currently informational.
            medium_tags: Region names whose union is the radiating medium.
                Normals are oriented to point into this medium before computing
                view factors. If omitted and regions named ``Gas`` or ``Air``
                exist, those are used automatically.
        """
        from ..trace import TensorTag

        if opaque_tags is None:
            opaque_tags = []
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(tag) for tag in tags]
        if not tags:
            raise ValueError("compute_enclosure_view_factor requires at least one boundary tag")

        medium_geom = self._medium_geometry(medium_tags)
        medium_names = self._normalize_medium_tags(medium_tags)
        if medium_geom is None:
            self.log.warning("No radiation medium geometry found; using sampled boundary normal orientation as-is.")

        tag_points: List[np.ndarray] = []
        tag_normals: List[np.ndarray] = []
        tag_sizes: List[int] = []
        ds_parts: List[np.ndarray] = []
        for tag in tags:
            kind_geom = self._polygon_tags.get(tag)
            if kind_geom is None:
                available = sorted(self._polygon_tags)
                raise ValueError(f"Tag '{tag}' not found on PolygonDomain. Available polygon tags: {available}")
            kind, _ = kind_geom
            if kind != "boundary":
                raise ValueError(f"Radiative view factors require boundary tags, got '{tag}'")

            points, normals = self._sampled_boundary_arrays(tag)
            normals = self._orient_normals_to_medium(points, normals, medium_geom)
            tag_points.append(points)
            tag_normals.append(normals)
            tag_sizes.append(points.shape[0])
            ds_parts.append(self._polygon_boundary_ds(tag, points.shape[0]))

        all_points = np.concatenate(tag_points, axis=0)
        all_normals = np.concatenate(tag_normals, axis=0)
        all_ds = np.concatenate(ds_parts, axis=0)
        tag_offsets: List[Tuple[int, int]] = []
        offset = 0
        for size in tag_sizes:
            tag_offsets.append((offset, offset + size))
            offset += size

        visibility = self._polygon_visibility_matrix(all_points)
        visibility = self._filter_visibility_to_medium(all_points, visibility, medium_geom)
        visibility = self._filter_visibility_by_normals(all_points, all_normals, visibility)
        for start, stop in tag_offsets:
            # Enclosure radiation blocks are used for cross-tag exchange.  Do
            # not let points from the same imported radiation surface exchange
            # with each other; use variable(tag, view_factor=True) explicitly
            # for a single-surface/self-view diagnostic.
            visibility[start:stop, start:stop] = 0.0
        if self.dimension == 1:
            view_factor = np.asarray(self.get_view_factor_1d(all_points, visibility, all_normals, all_ds))
        elif self.dimension == 2:
            view_factor = np.asarray(self.get_view_factor_2d(all_points, visibility, all_normals, all_ds))
        else:
            view_factor = np.asarray(self.get_view_factor_3d(all_points, visibility, all_normals, all_ds))

        enclosure_name = "+".join(tags)
        self.context[f"v_{enclosure_name}"] = visibility[None, None, ...]
        self._param_tags.add(f"v_{enclosure_name}")
        self.context[f"f_{enclosure_name}"] = view_factor[None, None, ...]
        self._param_tags.add(f"f_{enclosure_name}")

        result = []
        row_offset = 0
        for i, source_tag in enumerate(tags):
            row = []
            col_offset = 0
            for j, target_tag in enumerate(tags):
                row_slice = slice(row_offset, row_offset + tag_sizes[i])
                col_slice = slice(col_offset, col_offset + tag_sizes[j])

                visibility_key = f"v_{source_tag}__{target_tag}"
                factor_key = f"f_{source_tag}__{target_tag}"
                self.context[visibility_key] = visibility[row_slice, col_slice][None, None, ...]
                self.context[factor_key] = view_factor[row_slice, col_slice][None, None, ...]
                self._param_tags.add(visibility_key)
                self._param_tags.add(factor_key)
                row.append(TensorTag(tag=factor_key, domain=self))
                col_offset += tag_sizes[j]
            result.append(tuple(row))
            row_offset += tag_sizes[i]

        medium_info = f", medium=[{', '.join(medium_names)}]" if medium_names else ""
        opaque_info = f", opaque=[{', '.join(map(str, opaque_tags))}]" if opaque_tags else ""
        self.log.info(f"Computed polygon enclosure view factors for [{', '.join(tags)}]{medium_info}{opaque_info} ({all_points.shape[0]} total boundary pts)")
        return tuple(result)

    def sample(
        self,
        sample_spec: Dict[str, Tuple[int, Optional[Any]]],
        normals: bool = False,
        return_indices: bool = False,
        time_value: float | None = None,
    ):
        batch_count = self._effective_batch_count()
        last_tag = None
        last_idx = None

        for requested_tag, (n_samples, sampler) in sample_spec.items():
            if requested_tag not in self._polygon_tags:
                available = sorted(self._polygon_tags)
                raise ValueError(f"Tag '{requested_tag}' not found on PolygonDomain. Available polygon tags: {available}")
            if sampler is not None:
                raise ValueError("PolygonDomain currently uses its built-in geometric samplers; custom samplers are not supported")
            if n_samples is None:
                raise ValueError("PolygonDomain tags are lazy and require an explicit sample count, e.g. sample=(500, None)")
            if n_samples < 0:
                raise ValueError(f"n_samples must be non-negative, got {n_samples}")

            kind, geom = self._polygon_tags[requested_tag]
            if normals and kind != "boundary":
                raise ValueError(f"Normals are only available for boundary tags, got '{requested_tag}'")

            tag = self._next_context_tag(requested_tag)
            all_points = []
            all_normals = []
            for _ in range(batch_count):
                if kind == "interior":
                    pts = self._sample_interior(requested_tag, geom, n_samples)
                    nrm = None
                else:
                    pts, nrm = self._sample_boundary(requested_tag, n_samples, normals)
                all_points.append(pts)
                if normals and nrm is not None:
                    all_normals.append(nrm)

            spatial = np.stack(all_points, axis=0)
            if self._is_time_dependent:
                t_points = np.asarray(getattr(self, "_time_points", [self.time[0] if self.time else 0.0]), dtype=float)
                if time_value is not None:
                    tidx = int(np.argmin(np.abs(t_points - float(time_value))))
                    n_time = 1
                    self.context[f"__time_{tag}__"] = np.asarray([[t_points[tidx]]], dtype=np.asarray(self.context["__time__"]).dtype)
                else:
                    n_time = len(t_points)
                arr = np.broadcast_to(spatial[:, np.newaxis, :, :], (batch_count, n_time, n_samples, 2)).copy()
            else:
                arr = spatial[:, np.newaxis, :, :]
            self.context[tag] = arr

            if normals and all_normals:
                normal_arr = np.stack(all_normals, axis=0)
                n_time = arr.shape[1]
                self.context[f"n_{tag}"] = np.broadcast_to(normal_arr[:, np.newaxis, :, :], (batch_count, n_time, n_samples, 2)).copy()

            self._mesh_points[tag] = arr[0, 0].copy() if arr.size else np.zeros((0, 2), dtype=np.float64)
            last_tag = tag
            last_idx = np.arange(n_samples, dtype=int) if return_indices else None
            if self._verbose:
                if batch_count > 1:
                    self.log.info(f"Sampled {n_samples} x {batch_count} = {batch_count * n_samples} polygon points for '{tag}' with shape {arr.shape}")
                else:
                    self.log.info(f"Sampled {n_samples} polygon points for '{tag}'")

        if last_tag is None:
            raise ValueError("sample_spec must contain at least one tag")
        if return_indices:
            return self.context[last_tag], last_idx, last_tag
        return self.context[last_tag], None, last_tag

    def variable(
        self,
        tag: str,
        sample: Tuple[Optional[int], Optional[Any]] = (None, None),
        resampling_strategy=None,
        normals: bool = False,
        reverse_normals: bool = False,
        view_factor: bool = False,
        point_data: bool = False,
        split: bool = False,
        return_indices=False,
        time_value: Optional[float] = None,
    ) -> Any:
        polygon_tag = self._is_polygon_tag(tag)
        if view_factor and polygon_tag:
            kind, _ = self._polygon_tags[tag]
            if kind != "boundary":
                raise ValueError(f"View factors are only available for boundary tags, got '{tag}'")

        sampled_indices = None
        if polygon_tag:
            if isinstance(sample, tuple) and len(sample) > 0 and isinstance(sample[0], (int, type(None))):
                self.sample_dict.append([tag, sample, resampling_strategy, normals, view_factor])
                _, sampled_indices, sampled_tag = self.sample(
                    {tag: sample},
                    normals=(normals or view_factor),
                    return_indices=return_indices,
                    time_value=time_value,
                )
                tag = sampled_tag
                sample = None  # type: ignore[assignment]
            elif tag not in self.context:
                raise ValueError(f"PolygonDomain tag '{tag}' is lazy. Use variable('{tag}', sample=(n, None)) to materialize points.")

        result = super().variable(
            tag,
            sample=sample,  # type: ignore[arg-type]
            resampling_strategy=resampling_strategy,
            normals=normals,
            reverse_normals=reverse_normals,
            view_factor=False,
            point_data=point_data,
            split=split,
            return_indices=False,
            time_value=time_value,
        )

        if view_factor and polygon_tag:
            from ..trace import TensorTag

            self._attach_polygon_view_factor(tag)
            result = (*result, TensorTag(tag=f"f_{tag}", domain=self))

        if return_indices:
            return (*result, sampled_indices)
        return result

    def copy(self) -> "PolygonDomain":
        return copy.deepcopy(self)

    def __rmul__(self, n: int) -> "PolygonDomain":
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Batch count must be positive integer, got {n}")
        new = self.copy()
        current_batch = int(getattr(new, "_batch_count", getattr(new, "total_samples", 1)))
        new._batch_count = max(1, current_batch) * n
        new.total_samples = new._batch_count
        new.same_domain = False
        return new

    def __mul__(self, n: int) -> "PolygonDomain":
        return self.__rmul__(n)

    def _time_for_result(self, other: "PolygonDomain") -> Optional[Tuple[float, float, int]]:
        if self.time is None:
            return other.time
        if other.time is None:
            return self.time
        if tuple(self.time) != tuple(other.time):
            raise ValueError("Cannot combine PolygonDomain objects with different time grids")
        return self.time

    def _merged_sources(self, other: "PolygonDomain") -> Tuple[Dict[str, BaseGeometry], Dict[str, List[BaseGeometry]]]:
        regions: Dict[str, BaseGeometry] = dict(self._source_regions)
        edges: Dict[str, List[BaseGeometry]] = {name: list(vals) for name, vals in self._source_edges.items()}
        for name, region in other._source_regions.items():
            if name in regions:
                regions[name] = _as_polygonal_geometry(unary_union([regions[name], region]))  # type: ignore[misc]
            else:
                regions[name] = region
        for name, edge_list in other._source_edges.items():
            edges.setdefault(name, []).extend(edge_list)
        return regions, edges

    def _new_from_csg(self, other: "PolygonDomain", geometry: BaseGeometry, op_name: str) -> "PolygonDomain":
        regions, edges = self._merged_sources(other)
        return PolygonDomain(
            geometry=_as_polygonal_geometry(geometry),
            name=f"{self._polygon_name}_{op_name}_{other._polygon_name}",
            regions=regions,
            source_edges=edges,
            time=self._time_for_result(other),
            compute_mesh_connectivity=False,
        )

    def _coerce_other(self, other: Any) -> "PolygonDomain":
        if isinstance(other, PolygonDomain):
            return other
        return PolygonDomain(other)

    def union(self, other: Any) -> "PolygonDomain":
        other_poly = self._coerce_other(other)
        return self._new_from_csg(other_poly, self._active_geometry.union(other_poly._active_geometry), "union")

    def intersection(self, other: Any) -> "PolygonDomain":
        other_poly = self._coerce_other(other)
        return self._new_from_csg(other_poly, self._active_geometry.intersection(other_poly._active_geometry), "intersection")

    def difference(self, other: Any) -> "PolygonDomain":
        other_poly = self._coerce_other(other)
        return self._new_from_csg(other_poly, self._active_geometry.difference(other_poly._active_geometry), "difference")

    def symmetric_difference(self, other: Any) -> "PolygonDomain":
        other_poly = self._coerce_other(other)
        return self._new_from_csg(other_poly, self._active_geometry.symmetric_difference(other_poly._active_geometry), "symmetric_difference")

    def __add__(self, other: Any) -> "PolygonDomain":
        return self.union(other)

    def __or__(self, other: Any) -> "PolygonDomain":
        return self.union(other)

    def __and__(self, other: Any) -> "PolygonDomain":
        return self.intersection(other)

    def __sub__(self, other: Any) -> "PolygonDomain":
        return self.difference(other)

    def __xor__(self, other: Any) -> "PolygonDomain":
        return self.symmetric_difference(other)
