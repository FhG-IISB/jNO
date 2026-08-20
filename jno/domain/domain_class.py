from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast, overload

import cloudpickle
import jax.numpy as jnp
import meshio
import numpy as np

from ..trace import (
    RegionMask,
    TagMask,
    TensorTag,
    TestFunction,
    TrialFunction,
    Variable,
)
from ..utils.dtypes import default_np_float_dtype
from ..utils.logger import get_logger
from .boundary_region import BoundaryRegion
from .geometries import Geometries
from .mesh_utils import base_cell_type as _base_cell_type
from .meshio_mixin import MeshIOMixin
from .simplex_pool import SimplexPool

#: The space-filling cell blocks of each dimension. A mesh is expected to carry exactly one.
_VOLUME_BLOCKS_BY_DIM = {1: ("line",), 2: ("triangle", "quad"), 3: ("tetra", "hexahedron")}


def _refuse_mixed_cells(mesh, dim: int) -> None:
    """Refuse a mesh carrying more than one kind of volume cell.

    jNO assembles on a single element family: one cell array, one element table, one quadrature
    rule. A mesh mixing them is not merely unsupported -- it is *silently* unsupported, because the
    assembler takes the first block it recognises and ignores the rest. Measured on real files from
    gmsh's benchmark suite: a quarter-cylinder of 3381 tets and 1430 hexes assembles on 70 % of its
    own domain, and a plate of 61763 tets, 2744 hexes and 924 pyramids on 94 %, with no error.

    This lives in ``_apply_mesh`` rather than in the assembler's ``mesh_cell_type`` because every
    mesh that becomes a domain passes through here -- loading a file, the native constructors,
    remeshing, adaptation -- and one of ``mesh_cell_type``'s callers wraps it in a bare ``except``
    that would swallow the refusal.
    """
    cd = {}
    for name, arr in (getattr(mesh, "cells_dict", None) or {}).items():
        cd.setdefault(_base_cell_type(name), 0)
        cd[_base_cell_type(name)] += len(arr)
    present = [(n, cd[n]) for n in _VOLUME_BLOCKS_BY_DIM.get(int(dim), ()) if n in cd]
    if len(present) < 2:
        return
    total = sum(c for _n, c in present)
    kept, dropped = present[0], present[1:]
    raise NotImplementedError(
        f"this {dim}-D mesh mixes cell types: {', '.join(f'{c} {n}' for n, c in present)}. jNO "
        f"assembles on one element family, so it would use the {kept[1]} {kept[0]} cells and "
        f"silently ignore {sum(c for _n, c in dropped)} of {total} "
        f"({100 * sum(c for _n, c in dropped) / total:.0f} % of the domain). Re-mesh with a single "
        "cell type -- for gmsh, that usually means turning recombination off, or on everywhere."
    )


#: The facet block a volume cell's boundary is written into, matching what the native constructors
#: emit (``geometries.py`` / ``emit.py``): a triangle's facet is a 2-node line, a hexahedron's a
#: 4-node quad.
_FACET_BLOCK_FOR = {"line": "vertex", "triangle": "line", "quad": "line", "tetra": "triangle", "hexahedron": "quad"}


def _boundary_shells(facets: np.ndarray) -> np.ndarray:
    """Label each boundary facet by which connected shell it belongs to.

    Two facets are joined when they share an EDGE (two nodes), not merely a corner — otherwise two
    surfaces meeting at a single point would be fused. Returns ``(n_facets,)`` labels; a solid part
    gives one shell, a body with an internal void gives two (measured: a box with a spherical
    cavity gives 540 outer + 80 cavity facets), and disjoint bodies give one each.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(facets) == 0:
        return np.zeros(0, dtype=np.int64)
    nodes, inv = np.unique(facets, return_inverse=True)
    inv = inv.reshape(facets.shape)
    rows = np.repeat(np.arange(len(facets)), facets.shape[1])
    inc = coo_matrix((np.ones(rows.size), (rows, inv.ravel())), shape=(len(facets), len(nodes)))
    adj = (inc @ inc.T).tocoo()
    adj.data = (adj.data >= 2).astype(np.int8)  # >= 2 shared nodes == a shared edge
    _n, labels = connected_components(adj, directed=False)
    return np.asarray(labels, dtype=np.int64)


def _derive_region_cell_sets(mesh, dim: int):
    """Give a loaded mesh the ``interior`` / ``boundary`` tags jNO's own constructors write.

    A file built elsewhere carries gmsh *physical groups* under whatever names its author chose, or
    nothing at all — so the two tags every weak form needs (``u(*d.variable("boundary"))``) simply do
    not exist, and a perfectly good mesh loads into a domain you cannot write an equation against.

    Rather than teach every consumer about a second kind of tag, this synthesises the same
    ``cell_sets`` a native constructor would have written, before the existing tag machinery runs —
    so a derived tag is structurally identical to a native one. Names already present in the file
    are never overwritten: a mesh whose author defined ``boundary`` keeps their meaning of it.

    The boundary is topological (facets belonging to exactly one cell), which is also the only
    option for the many files that store no surface block at all. When it falls into more than one
    connected shell, each is additionally exposed as ``boundary_0``, ``boundary_1``, … so an
    internal cavity or a second body can be addressed on its own; a single-shell part gains no
    numbered tags, since they would only be noise.

    Returns ``(mesh, added)`` — the mesh (possibly with a facet block appended) and the tag names
    created, for logging.
    """
    from .mesh_utils import MeshUtils, p1_cells_dict

    # A CURVED mesh stores `tetra10` / `triangle6`, whose first `dim+1` columns are the vertices;
    # `p1_cells_dict` supplies that first-order view, so an order-2 file gets its regions too.
    cd = p1_cells_dict(mesh) if getattr(mesh, "cells_dict", None) else {}
    cell_type = next((n for n in _VOLUME_BLOCKS_BY_DIM.get(int(dim), ()) if n in cd), None)
    if cell_type is None:
        return mesh, []  # no volume cells: a surface/shell mesh, refused with its own message
    existing = dict(getattr(mesh, "cell_sets", None) or {})
    want_interior = "interior" not in existing
    want_boundary = "boundary" not in existing
    if not (want_interior or want_boundary):
        return mesh, []

    # `cells_dict` AGGREGATES same-type blocks, but `mesh.cells` keeps them split -- meshio emits one
    # block per gmsh entity, and a real CAD file has many (a sensor model here has 97). The boundary
    # is computed from the aggregate; the cell_sets must index each block separately.
    cells = np.asarray(cd[cell_type], dtype=np.int64)
    facets, _parent = MeshUtils._boundary_facets_unsorted(cells, cell_type)
    facet_type = _FACET_BLOCK_FOR[cell_type]

    blocks = list(mesh.cells)
    # The derived boundary always gets its own block rather than borrowing one the file supplies:
    # a file's surface blocks are its own named regions, generally a subset of the true boundary
    # (many files store none at all), and conflating them would silently redefine what they mean.
    facet_block = None
    if want_boundary:
        blocks.append(meshio.CellBlock(facet_type, facets))
        facet_block = len(blocks) - 1

    def _empty_per_block():
        return [np.array([], dtype=np.int64) for _ in blocks]

    vol_blocks = [i for i, b in enumerate(blocks) if _base_cell_type(b.type) == cell_type]
    sets = {k: list(v) + [np.array([], dtype=np.int64)] * (len(blocks) - len(v)) for k, v in existing.items()}
    added = []
    if want_interior:
        s = _empty_per_block()
        for i in vol_blocks:  # every block of the volume type, not just the first
            s[i] = np.arange(len(blocks[i].data), dtype=np.int64)
        sets["interior"] = s
        added.append("interior")
    if want_boundary:
        s = _empty_per_block()
        s[facet_block] = np.arange(len(blocks[facet_block].data), dtype=np.int64)
        sets["boundary"] = s
        added.append("boundary")
        labels = _boundary_shells(np.asarray(blocks[facet_block].data))
        n_shell = int(labels.max()) + 1 if labels.size else 0
        if n_shell > 1:  # only worth naming when there IS more than one
            for k in range(n_shell):
                sk = _empty_per_block()
                sk[facet_block] = np.flatnonzero(labels == k).astype(np.int64)
                sets[f"boundary_{k}"] = sk
                added.append(f"boundary_{k}")

    return meshio.Mesh(points=mesh.points, cells=blocks, cell_sets=sets, field_data=mesh.field_data), added


def _is_lazy_source(obj: Any) -> bool:
    """Whether ``obj`` is an array-like jNO can slice **without reading it whole**.

    Duck-typed on purpose -- ``.shape`` plus ``__getitem__`` is the whole contract, so h5py, zarr,
    tensorstore and ``np.memmap`` all qualify and none of them is imported here. An eager
    ``np.ndarray`` / ``jnp.ndarray`` satisfies it too, so callers must test those first: they are
    cheaper handled eagerly and some downstream paths rely on their concreteness.
    """
    if isinstance(obj, (np.ndarray, jnp.ndarray, tuple, list, str, bytes)):
        return False
    return hasattr(obj, "shape") and hasattr(type(obj), "__getitem__")


def _scalar_float(value: Any) -> float:
    """Convert a scalar-like Python/NumPy/JAX value to float for BC callbacks."""
    arr = np.asarray(value)
    if arr.shape != ():
        raise TypeError(f"Expected scalar value, got shape {arr.shape}.")
    return float(arr.item())


def _masked_sum(values, mask_cls, default, *, what: str, key: str):
    """``sum_k mask_cls(k) * values[k]`` — the one desugaring behind ``by_region`` and ``by_tag``.

    ``default`` (when nonzero) is added over the *complement* of the listed keys, so a cell/facet in no
    listed region gets it. The two callers differ only in which mask leaf they emit and in what they
    validate their keys against; the arithmetic, the empty-mapping check and the view guard are shared.
    """
    from ..trace.views import _VIEW_TYPES  # local: filled at the END of the views module

    # A typed view survives the sum -- ``Placeholder.__mul__`` yields to views, so ``RegionMask * K``
    # comes back a MatrixView and ``d.K @ grad(u)`` keeps working. That only holds while the values
    # agree on ONE view class: mixing them silently returns whichever ``_rewrap`` ran last (measured:
    # a MatrixView region plus a VectorView region yields a MatrixView, no error), and the assembler
    # then contracts the wrong rank. Physics, not a crash -- so reject it here.
    seen: Dict[Any, list] = {}
    for k, v in values.items():
        if isinstance(v, _VIEW_TYPES):
            seen.setdefault(type(v), []).append(str(k))
    if len(seen) > 1:
        kinds = ", ".join(
            f"{cls.__name__} for {sorted(ks)}" for cls, ks in sorted(seen.items(), key=lambda p: p[0].__name__)
        )
        raise ValueError(
            f"domain.{what}: the values mix view types ({kinds}). Every {key} must present the same "
            f"rank -- a coefficient cannot be a matrix on one {key} and a vector on another. Wrap the "
            "odd one out so all values agree."
        )

    expr = None
    for k, value in values.items():
        term = mask_cls(str(k)) * value
        expr = term if expr is None else expr + term
    if expr is None:
        raise ValueError(f"domain.{what}: the {{{key}: value}} mapping is empty.")
    if default is not None and default != 0:
        covered = None
        for k in values:
            m = mask_cls(str(k))
            covered = m if covered is None else covered + m
        expr = expr + default * (1.0 - covered)
    return expr


def _is_facet_predicate(where) -> bool:
    """True if ``where`` is a richer boundary-facet predicate ``f(x, n, name)`` (by param names)."""
    import inspect

    try:
        params = list(inspect.signature(where).parameters)
    except (ValueError, TypeError):
        return False
    return len(params) == 3 and params[1] in ("n", "normal", "normals") and params[2] in ("name", "names")


def _point_normals_from_facets(sub, sub_n, bpts, dim):
    """Per-point outward normal for a facet subset: average the incident facet normals at each point.

    Shared by both boundary-tag paths so they agree by construction. They did not: the **facet**
    predicate (:meth:`domain._tag_by_facet`) stored per-point normals while the ordinary **coordinate**
    predicate (:meth:`domain._register_tag_boundary_region`) did not, so a tag's normals existed only
    when the mesh happened to carry that name as a cell set. Anything reading ``normals_by_tag``
    then saw a *silently missing* entry — see :func:`jno.rcwa._z_ambient_faces`, where a zero normal
    disqualified every face and RCWA could not find its own superstrate/substrate.

    Points are matched by rounded coordinate (9 dp), the same key ``_tag_by_facet`` has always used —
    ``bpts`` comes from ``np.unique`` on those very vertices, so every point has at least one facet.
    """
    acc = {}
    for f, fn in zip(sub, sub_n):
        for v in f:
            key = tuple(np.round(v[:dim], 9))
            a = acc.get(key)
            acc[key] = fn if a is None else a + fn
    pn = np.array([acc[tuple(np.round(p[:dim], 9))] for p in bpts])
    return pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-30)


def _facet_normals(ents, dim, mesh=None):
    """Outward unit normal per boundary facet (``ents`` is ``(E, k, dim)`` facet-vertex coords).

    With ``mesh`` (a meshio mesh carrying volume cells) the orientation comes from the **topology**, via
    :func:`~jno.utils.solver.fem_facets.compute_face_normals`: a boundary facet belongs to exactly one
    cell, so "outward" is "away from that cell's opposite vertex". Exact for any shape.

    Without volume cells there is no topology to ask, and it falls back to orienting away from the
    boundary centroid. That fallback is only exact for a **star-shaped** domain: on an annulus it gives the
    inner hole a normal pointing into the material, which is the wrong sign for any law linear in ``n``.
    It was the unconditional rule here until the moving-boundary work measured it.
    """
    ents = np.asarray(ents, dtype=float)
    if dim == 2:
        t = ents[:, 1] - ents[:, 0]
        out = np.stack([t[:, 1], -t[:, 0]], axis=1)
    elif ents.shape[1] == 4:
        # A hexahedron's face is a quadrilateral and need not be planar, so no single edge pair gives
        # its normal. The cross of the two DIAGONALS does, and is the rule the facet tables already
        # use (`fem_facets.compute_face_normals`) -- keeping the two in step matters because the sign
        # below is fixed by comparing against them.
        out = np.cross(ents[:, 2] - ents[:, 0], ents[:, 3] - ents[:, 1])
    else:
        out = np.cross(ents[:, 1] - ents[:, 0], ents[:, 2] - ents[:, 0])
    out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-30)

    # The volume cell is not implied by the dimension: a 2-D mesh is triangles or quads, a 3-D one
    # tets or hexes. Ask the mesh which it holds rather than assuming the simplex.
    cells, cell_type = None, None
    for key, name in {2: (("triangle", "triangle"), ("quad", "quad"))}.get(
        int(dim), (("tetra", "tetrahedron"), ("hexahedron", "hexahedron"))
    ):
        cells = None if mesh is None else getattr(mesh, "cells_dict", {}).get(key)
        if cells is not None and len(cells):
            cell_type = name
            break
    if cells is not None and len(cells):
        from ..utils.solver.fem_facets import build_facet_connectivity, compute_face_normals

        conn = build_facet_connectivity(np.asarray(cells), cell_type)
        good = compute_face_normals(np.asarray(mesh.points), conn, np.asarray(cells), cell_type)
        # match each facet to its topological twin by centroid -- ``ents`` carries coordinates, not ids
        from scipy.spatial import cKDTree

        gmid = np.asarray(mesh.points)[np.asarray(conn.face_nodes), :dim].mean(axis=1)
        d, j = cKDTree(gmid).query(ents.mean(axis=1)[:, :dim])
        tol = 1e-8 * max(float(np.ptp(gmid)), 1.0) if gmid.size else 0.0
        if float(np.max(d, initial=0.0)) <= tol:
            return np.where((np.einsum("ij,ij->i", out, good[j]) < 0.0)[:, None], -out, out)

    ctr = ents.reshape(-1, dim).mean(axis=0)  # fallback: away from the boundary centroid
    return np.where((np.einsum("ij,ij->i", out, ents.mean(axis=1) - ctr) < 0.0)[:, None], -out, out)


def _facet_current_names(boundary_regions, mid):
    """Each facet centroid's current region name (the specific named region containing it, else
    ``"boundary"``)."""
    names = np.array(["boundary"] * len(mid), dtype=object)
    for tag, region in boundary_regions.items():
        if tag == "boundary":
            continue
        for i, p in enumerate(mid):
            if names[i] == "boundary" and region.contains(p):
                names[i] = tag
    return names


class domain(MeshIOMixin):
    """
    Mesh-based domain class for defining computational domains and sampling collocation points.

    Supports:
    - Rectangle, circle, and custom geometries via PyGmsh
    - Loading meshes from files
    - Sampling interior and boundary points
    - Time-dependent problems

    Attributes:
        _mesh_pool: Full mesh vertices per tag (private, used for sampling)
        context: Unified dict of spatial (B,N,D) and parametric (B,F) arrays for training
    """

    time: Optional[Tuple[float, float, int]]
    compute_mesh_connectivity: bool
    _mesh_pool: Dict[str, Any]
    context: Dict[str, Any]
    fem_context: Dict[str, Any]
    mesh_connectivity: Optional[Dict[str, Any]]

    @classmethod
    def _from_geometry(
        cls,
        geometry_constructor: Callable,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        return cls(
            constructor=geometry_constructor,
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def line(
        cls,
        x_range=(0, 1),
        mesh_size=0.1,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a 1D line domain."""
        return cls._from_geometry(
            Geometries.line(x_range=x_range, mesh_size=mesh_size),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def poly(
        cls,
        vertices,
        *,
        name: str = "polygon",
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = False,
    ) -> "domain":
        """Instantiate a Shapely-backed polygon CSG domain.

        ``domain.poly(...)`` returns the lazy CSG domain class (no mesh until
        sampled). For a meshed polygon, build one with ``jno.Shape.polygon(...)``
        and realize it via ``.domain()``.
        """
        from .polygon_domain import PolygonDomain

        return PolygonDomain(
            vertices,
            name=name,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def from_array(
        cls,
        tags: dict,
        compute_mesh_connectivity: bool = False,
    ) -> "domain":
        """Create a point-cloud domain from in-memory coordinate arrays.

        Avoids writing a ``.npz`` file by hand — useful for sparse sensor
        observations or any custom set of collocation points.

        Args:
            tags: Mapping of tag name to ``(N, D)`` numpy array of coordinates.
            compute_mesh_connectivity: Whether to build mesh connectivity data.

        Returns:
            Domain whose variable sets correspond to the keys of ``tags``.

        Example::

            import numpy as np
            sensors = np.random.rand(20, 2)          # 20 points in 2-D
            dom = jno.domain.from_array({"obs": sensors})
            x, y, _ = dom.variable("obs")
        """
        import os
        import tempfile

        import numpy as np

        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.close()
        np.savez(tmp.name, **tags)
        try:
            return cls(
                constructor=tmp.name,
                compute_mesh_connectivity=compute_mesh_connectivity,
            )
        finally:
            os.unlink(tmp.name)

    def __new__(cls, constructor=None, **kwargs):
        """Dispatch to PolygonDomain when constructor is a shapely geometry, vertex list, or dict."""
        if cls is domain and constructor is not None:
            if not isinstance(constructor, (str, domain)) and not callable(constructor):
                from .polygon_domain import PolygonDomain

                _POLY_KWARGS = frozenset(
                    {
                        "name",
                        "geometry",
                        "regions",
                        "source_edges",
                        "time",
                        "compute_mesh_connectivity",
                        "mesh_size",
                        "sampler",
                        "samplers",
                        "resampling_strategy",
                        "resampling_strategies",
                    }
                )
                poly_kwargs = {k: v for k, v in kwargs.items() if k in _POLY_KWARGS}
                return PolygonDomain(constructor, **poly_kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        constructor: Union[Callable, str, "domain", None] = None,
        algorithm: Optional[int] = None,
        time: Optional[Tuple[float, float, int]] = None,
        tau: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: Optional[bool] = None,
        keep_orphan_nodes: bool = False,
        **_ignored_kwargs,
    ):
        if "structured" in _ignored_kwargs:
            raise ValueError(
                "jno.domain(..., structured=True) was replaced by Shape.structured() and is no longer "
                "read. It was being swallowed by **_ignored_kwargs, so the domain came back WITHOUT a "
                "grid descriptor and the failure surfaced far away (scheme='spectral' refusing a "
                "domain the caller believed was structured). Spell it "
                "jno.Shape.rect(0, 0, 1, 1, size=h).structured().domain() instead."
            )
        """
        Initialize the domain.

        Args:
            constructor: Function accepting a pygmsh.geo.Geometry object, an existing domain,
                or a path to a meshfile
            algorithm: Gmsh meshing algorithm
            time: Tuple of (start, end, n) for physical time-dependent problems (a ``u.t`` derivative).
            tau: Tuple of (start, end, n) for a **pseudo-time load path** — identical grid machinery to
                ``time`` but flagged so ``fem.solve`` marches it as a history path (driven by ``.i(k)``)
                rather than integrating a physical time derivative. Use for quasi-static load-stepping
                (e.g. an elasto-plastic load→unload cycle). Pass ``time`` **or** ``tau``, not both.
            mesh_connectivity: Wether or not to compute the some hyperparameters about the mesh (needed for finite_difference methods)

        A geometry built with :meth:`jno.Shape.structured` is meshed as a regular lattice rather than
        by gmsh, and records its grid descriptor on :attr:`grid` / ``mesh_connectivity["grid"]``.
        """
        # `tau=` is a pseudo-time (load-path) alias of `time=`: reuse the whole grid/coordinate/tiling
        # machinery, only flag it so the solve marches it as a history path instead of integrating u.t.
        _pseudo_time = tau is not None
        if _pseudo_time:
            if time is not None:
                raise ValueError("domain: pass time= (physical time) OR tau= (pseudo-time load path), not both.")
            time = tau
        if isinstance(constructor, domain):
            existing_domain = constructor

            if algorithm is None:
                algorithm = int(getattr(existing_domain, "_algorithm", 6))

            if time is None:
                time = cast(
                    Optional[Tuple[float, float, int]],
                    getattr(existing_domain, "time", None),
                )

            if compute_mesh_connectivity is None:
                compute_mesh_connectivity = bool(getattr(existing_domain, "compute_mesh_connectivity", True))

            cloned = cloudpickle.loads(cloudpickle.dumps(existing_domain))
            self.__dict__.update(cloned.__dict__)

            self.log = get_logger()
            self._algorithm = algorithm
            self.compute_mesh_connectivity = bool(compute_mesh_connectivity)
            self._constructor_source = getattr(existing_domain, "_constructor_source", None)
            self.time = time
            self._is_time_dependent = time is not None
            self._is_pseudo_time = _pseudo_time or bool(getattr(existing_domain, "_is_pseudo_time", False))

            if getattr(existing_domain, "_is_time_dependent", False):
                base_mesh_pool: Dict[str, Any] = {}
                mesh_pool = cast(Dict[str, Any], getattr(self, "_mesh_pool", {}))

                for tag, points in mesh_pool.items():
                    if tag == "initial":
                        continue
                    if hasattr(points, "ndim") and points.ndim >= 3:
                        base_mesh_pool[tag] = np.asarray(points[0]).copy()
                    else:
                        base_mesh_pool[tag] = np.asarray(points).copy()

                self._mesh_pool = base_mesh_pool

            self.context = {k: v for k, v in self.context.items() if k != "__time__"}

            if self._is_time_dependent:
                if time is None:
                    raise RuntimeError("Internal error: time-dependent cloned domain has time=None.")
                self._add_time_dimension(time[0], time[1], time[2])
            else:
                self.context["__time__"] = np.ones((1, 1))

            spatial_dims = self.dimension
            default_spatial = ["x", "y", "z"][:spatial_dims]
            self.indep = default_spatial + ["t"]
            self.spatial = [i for i in self.indep if i != "t"]
            return

        if algorithm is None:
            algorithm = 6
        if compute_mesh_connectivity is None:
            compute_mesh_connectivity = True

        self._init_empty_state(
            constructor_source=constructor,
            algorithm=algorithm,
            time=time,
            pseudo_time=_pseudo_time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )
        # Drop mesh nodes that belong to no cell (gmsh geometry-construction points — arc/circle
        # centres, spline control points — surface as isolated vertices, giving zero rows in the
        # assembled operator). Off by default via keep_orphan_nodes=True. Applied in _apply_mesh.
        self._keep_orphan_nodes = keep_orphan_nodes

        # A `Shape.structured()` plan is meshed as a regular lattice instead of by gmsh: swap in the
        # lattice constructor and remember the grid descriptor to stamp on mesh_connectivity below.
        # `_plan` keeps the ORIGINAL shape, because the region-name/attachment collection further down
        # reads it off the constructor -- and the swapped-in closure carries neither, so a
        # `.name("steel").attach(k=2.0).structured()` plan used to lose its material without a word.
        self._structured_grid = None
        _plan = constructor if hasattr(constructor, "_node") else None
        if _plan is not None and _plan._structured is not None:
            constructor, self._structured_grid = self._structured_grid_setup(_plan)

        # Generate or load mesh / npz point cloud tags
        if isinstance(constructor, str):
            suffix = Path(constructor).suffix.lower()
            if suffix == ".npz":
                self._load_npz_tags(constructor)
                self.log.info(f"Loaded NPZ coordinate tags from {constructor}")
            else:
                self._load_mesh(constructor)
                self.log.info(f"Loaded mesh from {constructor}")  # type: ignore[attr-defined]
        elif callable(constructor):
            self._generate_mesh(constructor, algorithm)
            self.log.info(f"Loaded mesh from {constructor}")  # type: ignore[attr-defined]
            # A Shape.regions() plan carries named sub-region shapes; remember them so jno.fem
            # per-region integration can restrict a term to a region's cells (centroid membership
            # via the region shape's ``contains`` — see ``_cell_region_mask``).
            #
            # ``_region_items`` rather than the raw ``("regions", ...)`` node, because a SINGLE named
            # shape is a region too -- it is spelled as a bare leaf, and gating on the node kind meant
            # ``rect.name("a").attach(k=2.0).domain().k`` reported "no attribute 'k'": the attachment
            # was collected from nowhere and dropped without a word. ``_region_items`` already returns
            # ``((name, shape),)`` for that case, so both spellings share one path.
            #
            # Read from `_plan` (the shape as written) rather than `constructor`, which a structured
            # plan has already replaced with its lattice closure.
            _shape = _plan if _plan is not None else constructor
            _region_name = getattr(_shape, "_region_name", None)
            if getattr(_shape, "_node", (None,))[0] == "regions" or _region_name is not None:
                items = tuple(_shape._region_items())
                self._shape_regions = {name: sub for name, sub in items}
                self._collect_region_attachments(items)
            elif getattr(_shape, "_attach", None):
                # The other half of the same hole: properties attached to a shape that was never
                # named have no region to belong to, and would be silently discarded here.
                raise ValueError(
                    f"Shape.attach({', '.join(sorted(map(str, _shape._attach)))}): this shape has no "
                    "region name, so the attached propert"
                    f"{'ies have' if len(_shape._attach) > 1 else 'y has'} nothing to attach to. "
                    "Name the region first -- `shape.name('steel').attach(...)`."
                )
        else:
            raise ValueError("Must provide either geometry_func, mesh file, or NPZ tag file")

        self._apply_mesh(self.mesh)

        # Stamp the structured-grid descriptor so jno.fdm takes the direct-stencil fast path. Kept on
        # both mesh_connectivity (where the FD kernels read it) and as _grid_shape (existing convention).
        if self._structured_grid is not None and getattr(self, "mesh_connectivity", None):
            self.mesh_connectivity["grid"] = dict(self._structured_grid)
            self._grid_shape = self._structured_grid["shape"]

        # Add time dimension if needed
        if self._is_time_dependent:
            self._add_time_dimension(time[0], time[1], time[2])
        else:
            # Stationary problems: store a constant time = 1 so that
            # variable() always returns (x, y, t) consistently.
            self.context["__time__"] = np.ones((1, 1))

        # Set up independent variable names
        # dimension is now purely spatial (time is a separate axis)
        spatial_dims = self.dimension
        default_spatial = ["x", "y", "z"][:spatial_dims]
        default_indep = default_spatial + ["t"]

        self.indep = default_indep
        self.spatial = [i for i in self.indep if i != "t"]

        user_spatial_dims = len(self.spatial)
        if user_spatial_dims < spatial_dims:
            self.dimension = user_spatial_dims
            for tag, pts in self._mesh_pool.items():
                if pts.shape[-1] > self.dimension:
                    self._mesh_pool[tag] = pts[..., : self.dimension]

        # Last, so every attribute this domain will ever set is already in place to be compared against.
        self._check_attachment_clashes()

    def _init_empty_state(
        self,
        *,
        constructor_source: Any = None,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        pseudo_time: bool = False,
        compute_mesh_connectivity: bool = True,
    ) -> None:
        """Initialize common domain bookkeeping without loading a mesh.

        Subclasses that provide their own geometry/sampling backend can call
        this to get the same context, tag, batching, and logging attributes as
        regular mesh-backed domains.
        """
        super().__init__()
        self.log = get_logger()
        self._algorithm = algorithm
        self._constructor_source = constructor_source

        # Storage
        self.compute_mesh_connectivity = compute_mesh_connectivity
        self._mesh_pool = {}  # full mesh vertices per tag (M, D)
        self.context: Dict[str, Any] = {}  # unified: spatial (B,N,D) + params (B,F)
        self._param_tags: set = set()  # tags that are parametric (TensorTag)
        self.normals_by_tag: Dict[str, np.ndarray] = {}
        self._boundary_registry: Dict[str, Dict[str, Any]] = {}
        self._interface_registry: Dict[str, Dict[str, Any]] = {}
        self._tag_edges: Dict[str, np.ndarray] = {}
        self._tag_triangles: Dict[str, np.ndarray] = {}
        # A tag's facets, by node count: 2 -> _tag_edges, 3 -> _tag_triangles, 4 -> _tag_quads
        # (the boundary face of a hexahedron). Read them through `tag_facets`, which picks the
        # store the mesh actually filled instead of guessing it from the dimension.
        # 4-node cells: a quadrilateral. In 2-D that is a VOLUME cell (the mesh itself);
        # in 3-D it is a hexahedron's boundary face.
        self._tag_quads: Dict[str, np.ndarray] = {}
        # Node-count -> store, in the order a facet lookup should try them.
        self._tag_facet_stores = ("_tag_edges", "_tag_triangles", "_tag_quads")
        self._boundary_regions: Dict[str, BoundaryRegion] = {}
        # User-defined predicate regions from domain.tag(name, where): name -> spatial predicate.
        # Consulted by _make_tag_location_fn (FEM boundary location-fn, auto-restricted to the
        # boundary) and by tag() to build the sampling pool.
        self._tag_predicates: Dict[str, Any] = {}
        # Precomputed simplex pools (segments / triangles + optional normals)
        # for in-JIT collocation sampling — populated by ``_build_simplex_pools``
        # after each ``_apply_mesh``.
        self._simplex_pools: Dict[str, SimplexPool] = {}
        # self._boundary_predicates: Dict[str, Callable] = {}

        # Neural operator storage
        self.parameters: Dict[str, Any] = {}
        self.arrays: Dict[str, np.ndarray] = {}
        self.tag_indices: Dict[str, np.ndarray] = {}
        self.avaiable_mesh_tags: List[str] = []  # names of the tags from the mesh generator
        self._boundary_loop_tags: set = set()  # tags extracted from line cells (boundary loops)
        self.mesh_connectivity = None  # precomputed mesh connectivity data
        # Resampling support
        self._mesh_points: Dict[str, np.ndarray] = {}  # Full mesh points for resampling
        self._mesh_pool_groups: Dict[str, List[Tuple[int, Any]]] = {}  # Per-tag sampling groups as (batch_count, points)
        self._normal_pool_groups: Dict[
            str, List[Tuple[int, np.ndarray]]
        ] = {}  # Per-tag normal groups as (batch_count, normals)
        self._resampling_strategies: Dict[str, Any] = {}  # Tag -> ResamplingStrategy
        self._sub_domains: List[Dict[str, Any]] = []  # metadata from merged sub-domains
        self._batch_domain_map: Optional[np.ndarray] = None  # maps batch index → sub-domain index

        # Configuration
        self.dimension: int = 2
        self.total_samples: int = 1
        self.time = time
        self._is_time_dependent = time is not None
        self._is_pseudo_time = pseudo_time
        self._verbose = True
        self.same_domain = False

        # Tracking
        self.index_tags: List[str] = []
        self.normal_tags: List[str] = []
        self.reference_solutions: List[Callable] = []
        self.sample_dict: List = []

        # Meshio mesh
        self.mesh = None

    def _load_npz_tags(self, npz_file: str):
        """Load per-tag coordinate arrays from a .npz file.

        Expected NPZ structure:
        - Each key is a tag name (e.g. "Air", "WallS", ...)
        - Each value is a numeric array of shape (N, D), with D >= 1
        """
        npz_path = Path(npz_file)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")

        loaded = np.load(npz_path, allow_pickle=False)
        tags = list(loaded.files)
        if not tags:
            raise ValueError(f"No arrays found in NPZ file: {npz_file}")

        self.mesh = None
        self._mesh_pool = {}
        self.avaiable_mesh_tags = []

        inferred_dim: Optional[int] = None
        for tag in tags:
            arr = np.asarray(loaded[tag], dtype=np.float64)

            if arr.ndim != 2:
                raise ValueError(f"NPZ tag '{tag}' must have shape (N, D), got {arr.shape}")
            if arr.shape[0] == 0:
                # Keep empty tags to preserve user intent.
                if inferred_dim is None:
                    inferred_dim = 2
                arr = np.zeros((0, inferred_dim), dtype=np.float64)
            elif arr.shape[1] < 1:
                raise ValueError(f"NPZ tag '{tag}' must have at least one coordinate column, got {arr.shape}")

            if inferred_dim is None and arr.shape[0] > 0:
                inferred_dim = int(arr.shape[1])

            self._mesh_pool[tag] = arr
            self.context[tag] = arr[None, None, ...]
            self.avaiable_mesh_tags.append(tag)

        if inferred_dim is None:
            inferred_dim = 2
        self.dimension = inferred_dim

    def summary(self) -> "domain":
        """Log a human-readable summary of the domain configuration.

        Returns:
            Self for method chaining.
        """
        lines = ["─── Domain Summary ───"]
        lines.append(f"  Spatial dimension : {self.dimension}D  ({', '.join(self.spatial)})")
        lines.append(f"  Time-dependent    : {self._is_time_dependent}")
        if self._is_time_dependent and self.time is not None:
            t0, t1, nt = self.time
            lines.append(f"  Time range        : [{t0}, {t1}]  ({nt} steps)")
        lines.append(f"  Batch / samples   : {self.total_samples}")

        if self._mesh_pool:
            lines.append(f"  Mesh tags ({len(self._mesh_pool)}):")
            for tag, pts in self._mesh_pool.items():
                lines.append(f"    • {tag:20s}  shape {pts.shape}")

        if self._boundary_registry:
            lines.append(f"  Boundary tags ({len(self._boundary_registry)}):")
            coord_labels = self.spatial if self.spatial else [f"x{i}" for i in range(self.dimension)]
            for tag in sorted(self._boundary_registry):
                pts = self._boundary_registry[tag].get("points")
                parts: list = [f"    • {tag}"]
                if pts is not None and len(pts) > 0:
                    pts_xy = np.asarray(pts)[:, : self.dimension]
                    extents = []
                    for axis, label in enumerate(coord_labels):
                        lo, hi = float(pts_xy[:, axis].min()), float(pts_xy[:, axis].max())
                        if hi - lo < 1e-10:
                            extents.append(f"{label}={lo:.3g}")
                        else:
                            extents.append(f"{label}=[{lo:.3g},{hi:.3g}]")
                    parts.append("  " + "  ".join(extents))
                normals = self.normals_by_tag.get(tag)
                if normals is None:
                    ctx_n = self.context.get(f"n_{tag}")
                    if ctx_n is not None:
                        normals = np.asarray(ctx_n).reshape(-1, self.dimension)
                if normals is not None and len(normals) > 0:
                    mean_n = np.asarray(normals).mean(axis=0)[: self.dimension]
                    n_str = "(" + ", ".join(f"{v:+.2f}" for v in mean_n) + ")"
                    parts.append(f"  n={n_str}")
                lines.append("".join(parts))

        if self._param_tags:
            lines.append(f"  Tensor tags ({len(self._param_tags)}):")
            for tag in sorted(self._param_tags):
                arr = self.context.get(tag)
                shape_str = str(arr.shape) if arr is not None else "(not set)"
                lines.append(f"    • {tag:20s}  shape {shape_str}")

        if self.parameters:
            lines.append(f"  Scalar parameters ({len(self.parameters)}):")
            for k, v in self.parameters.items():
                lines.append(f"    • {k} = {v}")

        lines.append("──────────────────────")
        msg = "\n".join(lines)
        self.log.info(msg)
        return self

    def __lt__(self, other: Tuple[str, Any]) -> "domain":
        """Attach parameters or arrays using < operator."""
        if not isinstance(other, tuple) or len(other) != 2:
            raise ValueError("Use: domain < (name, value)")

        name, value = other

        if not isinstance(name, str):
            raise ValueError("Name must be a string")

        if isinstance(value, (int, float)):
            self.parameters[name] = float(value)
            self.log.info(f"Attached parameter '{name}' = {value}")
        elif isinstance(value, np.ndarray):
            self.arrays[name] = value.astype(default_np_float_dtype())
            self.log.info(f"Attached array '{name}' with shape {value.shape}")
        elif isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=default_np_float_dtype())
            self.arrays[name] = arr
            self.log.info(f"Attached array '{name}' with shape {arr.shape}")
        else:
            raise ValueError(f"Value must be scalar or array, got {type(value)}")

        return self

    def __rmul__(self, n: int) -> "domain":
        """Batch the domain n times: 2 * domain samples 2x independently.

        When sample() is called, it will sample n times and concatenate results.

        Example:
            domain = 10 * domain.from_mesh(domain.rect, {...}, 0.05)
            domain.sample({"interior": (100, None)})  # Results in 1000 interior points
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Batch count must be positive integer, got {n}")
        current_batch = int(getattr(self, "_batch_count", getattr(self, "total_samples", 1)))
        if self._mesh_pool_groups:
            for tag, groups in self._mesh_pool_groups.items():
                self._mesh_pool_groups[tag] = [(count * n, points) for count, points in groups]
            for tag, groups in self._normal_pool_groups.items():
                self._normal_pool_groups[tag] = [(count * n, normals) for count, normals in groups]
        self._batch_count = max(1, current_batch) * n
        self.total_samples = self._batch_count
        self.same_domain = not any(len(groups) > 1 for groups in self._mesh_pool_groups.values())
        return self

    def __mul__(self, n: int) -> "domain":
        """Batch the domain n times: domain * 2 samples 2x independently."""
        return self.__rmul__(n)

    def _check_lazy_tensor_layout(self, tag, handle):
        """The lazy counterpart of :meth:`_normalize_tensor_time_axis` — *validate*, never rewrite.

        The eager path inserts the missing ``(B, T, ...)`` time axis for you. A lazy handle cannot be
        rewritten without reading it, which is the one thing it exists to avoid, so the same layout
        rule is enforced as an error instead: store the array with its time axis, or pass it eagerly.
        Refusing here is what keeps the silent-data-loss guarantee the eager path gives — the
        alternative is that ``H`` is read as the timestep count and most of the field is dropped.
        """
        shape = tuple(int(s) for s in handle.shape)
        b = self._effective_batch_count()
        if len(shape) < 4 or shape[0] not in (b, 1):
            return  # not routed as a spatial tensor; nothing to check (see _normalize_tensor_time_axis)

        n_t = 1
        if getattr(self, "_is_time_dependent", False):
            t_ctx = self.context.get("__time__")
            n_t = int(t_ctx.shape[0]) if t_ctx is not None and hasattr(t_ctx, "shape") else 1
        if shape[1] in (n_t, 1):
            return

        raise ValueError(
            f"domain.variable({tag!r}, <lazy {type(handle).__name__}>): the source has shape {shape}, whose "
            f"axis 1 is {shape[1]}, but context tensors are (B, T, ...) and this domain has {n_t} timestep(s), "
            f"so axis 1 must be {n_t} or 1. An eager array is reshaped for you; a lazy source is not, because "
            f"that would read it. Store it with the time axis (shape {(shape[0], n_t) + shape[1:]}), or pass "
            f"an eager array."
        )

    def _normalize_tensor_time_axis(self, tag, tensor):
        """Give a grid-valued tensor tag the time axis the layout requires, or refuse.

        Context tensors are ``(B, T, ...)``. The compiler peels ``B`` with a vmap and then
        ``scan_over_time`` infers the time extent as ``max(v.shape[0])`` over every remaining value
        with ``ndim >= 3`` -- so a tensor attached as ``(B, H, W, C)``, the shape a user actually
        has, gets ``H`` read as the number of timesteps. One "timestep" then reaches the expression
        and the rest of the field is **silently discarded**: measured, a ``(4, 8, 5, 1)`` attach
        arrives at the evaluator as ``(5, 1)``.

        Nothing about that is discoverable -- the docstring above documents the *leading* dimension
        conventions and says nothing about axis 1 -- so it is normalized here, where the batch
        count, the time extent and the array are all still in hand.

        Only tensors the compiler routes as **spatial** (``shape[0] in (B, 1)``) are touched; a
        "shared" tag is never vmapped and so never reaches the time inference. Tensors of rank < 4
        are left alone for the same reason: after the batch axis is peeled they fall below the
        ``ndim >= 3`` test, so a parameter like ``(B, 1, 1)`` is untouched.
        """
        b = self._effective_batch_count()
        if tensor.ndim < 4 or tensor.shape[0] not in (b, 1):
            return tensor

        n_t = 1
        if getattr(self, "_is_time_dependent", False):
            t_ctx = self.context.get("__time__")
            n_t = int(t_ctx.shape[0]) if t_ctx is not None and hasattr(t_ctx, "shape") else 1

        if tensor.shape[1] in (n_t, 1):
            return tensor  # already carries a time axis (or a broadcast one)

        if getattr(self, "_is_time_dependent", False):
            raise ValueError(
                f"domain.variable({tag!r}, ...): the array has shape {tuple(tensor.shape)}, whose axis 1 is "
                f"{tensor.shape[1]}, but this domain has {n_t} timesteps. Context tensors are (B, T, ...), so "
                f"axis 1 must be {n_t} (one entry per step) or 1 (shared across steps). Insert the time axis "
                f"explicitly -- arr[:, None, ...] to share one field across all steps."
            )

        # Steady domain: T is 1 by definition, so there is exactly one thing axis 1 can be.
        return tensor[:, None, ...]

    def _effective_batch_count(self) -> int:
        """Infer the current batch size from metadata and existing batched context."""
        declared = int(getattr(self, "_batch_count", getattr(self, "total_samples", 1)))
        inferred = 1

        # Infer from already-batched context entries (skip shared time axis).
        for tag, data in self.context.items():
            if tag == "__time__" or not hasattr(data, "shape"):
                continue
            shape = data.shape
            if len(shape) >= 1:
                inferred = max(inferred, int(shape[0]))

        return max(1, declared, inferred)

    def _sampling_groups_for_tag(self, tag: str) -> List[Tuple[int, Any, Optional[np.ndarray]]]:
        """Return per-batch sampling sources for a tag as ``(count, points, normals)``."""
        point_groups = self._mesh_pool_groups.get(tag)
        normal_groups = self._normal_pool_groups.get(tag)

        if point_groups:
            groups: List[Tuple[int, Any, Optional[np.ndarray]]] = []
            for idx, (count, points) in enumerate(point_groups):
                normals = None
                if normal_groups and idx < len(normal_groups):
                    normals = normal_groups[idx][1]
                groups.append((count, points, normals))
            return groups

        points = self._mesh_pool[tag]
        normals = self.normals_by_tag.get(tag)
        count = int(getattr(self, "_batch_count", getattr(self, "total_samples", 1))) if self.same_domain else 1
        return [(max(1, count), points, normals)]

    def draw_candidates(self, tag: str):
        """Return (points, normals_or_None) candidate pool for resampling.

        Collects all sampling groups so merged domains expose the full union
        of their node sets.  Time-dependent pools (T, N, D) are reduced to
        their spatial slice (N, D) since spatial coordinates are shared across
        timesteps.
        """
        import numpy as _np

        if tag not in self._mesh_pool:
            return None, None
        groups = self._sampling_groups_for_tag(tag)
        all_pts, all_nrm = [], []
        has_normals = False
        for _, grp_pts, grp_nrm in groups:
            p = _np.asarray(grp_pts)
            if p.ndim == 3:  # (T, N, D) — time-dep: extract spatial slice
                p = p[0]
            all_pts.append(p)
            if grp_nrm is not None:
                n = _np.asarray(grp_nrm)
                if n.ndim == 3:
                    n = n[0]
                all_nrm.append(n)
                has_normals = True
            else:
                all_nrm.append(None)
        pts = _np.concatenate(all_pts, axis=0) if len(all_pts) > 1 else all_pts[0].copy()
        if has_normals and all(n is not None for n in all_nrm):
            nrm = _np.concatenate(all_nrm, axis=0)
        else:
            nrm = None
        return pts, nrm

    def __add__(self, other: "domain") -> "domain":
        """Merge another domain into this one (stacks along batch dimension).

        For time-dependent problems, ``_mesh_pool`` entries have shape
        ``(T, N, D_spatial)`` and do **not** get stacked (they represent the
        same spatial mesh).  ``context`` entries are concatenated along the
        batch axis (axis 0).  ``"__time__"`` is shared and not stacked.
        """
        self_groups = {
            tag: [(count, points) for count, points, _ in self._sampling_groups_for_tag(tag)]
            for tag in self._mesh_pool.keys()
        }
        other_groups = {
            tag: [(count, points) for count, points, _ in other._sampling_groups_for_tag(tag)]
            for tag in other._mesh_pool.keys()
        }
        self_normal_groups = {
            tag: [(count, normals) for count, _, normals in self._sampling_groups_for_tag(tag) if normals is not None]
            for tag in self._mesh_pool.keys()
        }
        other_normal_groups = {
            tag: [(count, normals) for count, _, normals in other._sampling_groups_for_tag(tag) if normals is not None]
            for tag in other._mesh_pool.keys()
        }

        for tag, points in other._mesh_pool.items():
            if tag not in self._mesh_pool:
                self._mesh_pool[tag] = points
            # else: keep self's mesh pool (same mesh)

        for tag in set(self_groups) | set(other_groups):
            merged_groups = list(self_groups.get(tag, [])) + list(other_groups.get(tag, []))
            if merged_groups:
                self._mesh_pool_groups[tag] = merged_groups

        for tag in set(self_normal_groups) | set(other_normal_groups):
            merged_groups = list(self_normal_groups.get(tag, [])) + list(other_normal_groups.get(tag, []))
            if merged_groups:
                self._normal_pool_groups[tag] = merged_groups

        for tag, data in other.context.items():
            if tag == "__time__":
                # Time array is shared, not batched
                self.context[tag] = data
                continue
            if tag in self.context:
                self.context[tag] = np.concatenate([self.context[tag], data], axis=0)
            else:
                self.context[tag] = data

        if not hasattr(self, "_parameter_list"):
            self._parameter_list = {k: [v] for k, v in self.parameters.items()}
        for name, value in other.parameters.items():
            if name in self._parameter_list:
                self._parameter_list[name].append(value)
            else:
                self._parameter_list[name] = [value]

        for name, values in self._parameter_list.items():
            self.parameters[name] = np.array(values, dtype=default_np_float_dtype())

        # Keep batch metadata consistent after domain stacking.
        self_batch = int(getattr(self, "_batch_count", getattr(self, "total_samples", 1)))
        other_batch = int(getattr(other, "_batch_count", getattr(other, "total_samples", 1)))
        self._batch_count = max(1, self_batch) + max(1, other_batch)
        self.total_samples = self._batch_count
        self.same_domain = False

        # Track sub-domain metadata for FD / mesh-connectivity routing.
        self._sub_domains.append(
            {
                "mesh_connectivity": other.mesh_connectivity,
                "batch_count": max(1, other_batch),
            }
        )

        # Build batch → domain index map: 0 = primary, 1 = first sub, …
        primary_count = max(1, self_batch)
        parts = [np.full(primary_count, 0, dtype=int)]
        for i, sd in enumerate(self._sub_domains):
            parts.append(np.full(sd["batch_count"], i + 1, dtype=int))
        self._batch_domain_map = np.concatenate(parts)

        return self

    # FEM/ variational interface

    @property
    def built_mesh(self) -> "meshio.Mesh":
        """The constructed mesh, guaranteed non-``None``.

        Use this instead of ``.mesh`` when reading mesh data (``.points``,
        ``.cells_dict``): ``.mesh`` is ``meshio.Mesh | None`` until a mesh is
        built/loaded, so a type checker flags member access on it. This accessor
        asserts the mesh exists and returns it, clearing that noise. Raises
        ``RuntimeError`` if called before ``build_mesh()`` (or a mesh load).
        """
        if self.mesh is None:
            raise RuntimeError("Mesh has not been built — call build_mesh() (or load a mesh) first.")
        return self.mesh

    @property
    def _domain_mesh_connectivities(self) -> List:
        """Return mesh connectivities for the primary and all sub-domains."""
        mcs = [self.mesh_connectivity]
        for sd in self._sub_domains:
            mcs.append(sd["mesh_connectivity"])
        return mcs

    @property
    def grid(self):
        """The regular-lattice descriptor of a :meth:`jno.Shape.structured` domain, else ``None``.

        ``{"shape": (Nx, Ny[, Nz]), "spacing": (hx, hy[, hz]), "origin": (x0, y0[, z0])}`` with
        ``shape`` in **nodes** (one more per axis than the cell count passed to ``structured(n=...)``).
        Nodes are stored in C order — ``idx = ((i·Ny + j)·Nz + k)`` — so a nodal field reshapes
        straight to ``grid["shape"]``::

            d = jno.Shape.rect(0, 0, 1, 1).structured(n=63).domain()
            u.reshape(d.grid["shape"])          # (64, 64)

        This is also the key ``jno.fdm`` reads to take its assembly-free stencil path.
        """
        return dict(self._structured_grid) if getattr(self, "_structured_grid", None) else None

    def _estimate_boundary_tol(self, pts: np.ndarray) -> float:
        pts = np.asarray(pts)
        if pts.size == 0:
            return 1e-8
        bbox_min = np.min(pts, axis=0)
        bbox_max = np.max(pts, axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        return max(1e-8, 1e-10 * max(diag, 1.0))

    def boundary_tags(self):
        """
        Return the available boundary tag names on the mesh.

        Returns
        -------
        list[str]
            Registered boundary tags that can be used for Dirichlet or
            Neumann conditions.
        """
        return sorted(self._boundary_registry.keys())

    def interface_tags(self, *regions: str):
        """Internal material-interface tags from a :meth:`Shape.regions` domain.

        With no arguments, every interface tag (``"a|b"``) — facet regions between two materials, on
        which you can impose a coupling or flux condition. They are deliberately *not* part of
        :meth:`boundary_tags`, which is the outer boundary only.

        Given **two region names**, the two *sides* of that interface, in the order asked::

            lo, hi = d.interface_tags("substrate", "coating")   # ("...substrate", "...coating")
            a, b = d.variable(lo, split=True), d.variable(hi, split=True)
            fem = jno.fem([..., T(a[0], a[1]) - T(b[0], b[1])])   # glue the two bodies

        A ``conforming=False`` domain meshes each region separately, so the shared surface exists
        **twice** — once per body, spatially coincident, with different node layouts. No ``domain.tag``
        predicate can separate them (they occupy the same points), which is why the emitter names them
        ``"a|b.a"`` / ``"a|b.b"`` and why they are reached through their region names rather than
        geometrically. Returning them in the order asked means the caller chooses which side a tie
        eliminates, rather than inferring it from alphabetical order.

        **Order matters, and there is a right answer.** In ``u(A) - u(B)`` the first region is the
        **secondary**: its interface DOFs are eliminated in favour of an interpolation from the main. So
        the secondary must be the more finely meshed side, or the fine mesh's resolution at the interface
        is discarded. Measured on a coating/substrate tie with 81 nodes against 10::

            secondary = the 81-node side  ->  interface value exact       (0.00% error)
            secondary = the 10-node side  ->  interface value off by 10.62%

        Nothing detects this today, and the wrong choice produces a plausible number rather than an
        error -- so pass the finer region first.
        """
        registry = getattr(self, "_interface_registry", {}) or {}
        if not regions:
            return sorted(registry)
        if len(regions) != 2:
            raise TypeError(f"domain.interface_tags: pass no regions, or exactly two; got {len(regions)}.")

        pair = "|".join(sorted(str(r) for r in regions))
        sides = {t.rpartition(".")[2]: t for t in registry if t.rpartition(".")[0] == pair}
        if not sides:
            if pair in registry:
                raise ValueError(
                    f"domain.interface_tags({regions[0]!r}, {regions[1]!r}): this is a CONFORMING "
                    f"interface, so the two regions share one surface ({pair!r}) and there are no "
                    "separate sides to tie. Build the domain with Shape.regions(..., conforming=False) "
                    "to mesh each body independently."
                )
            known = sorted({t.rpartition(".")[0] or t for t in registry})
            raise ValueError(
                f"domain.interface_tags: no interface between {regions[0]!r} and {regions[1]!r}. "
                f"Known interfaces: {known}. Two regions only share one when they touch."
            )
        missing = [r for r in regions if str(r) not in sides]
        if missing:
            raise ValueError(
                f"domain.interface_tags: {pair!r} has sides {sorted(sides)}, which does not include "
                f"{missing}. Pass the two region names that form the interface."
            )
        return tuple(sides[str(r)] for r in regions)

    def dirichlet(self, tags, values=None):
        """
        Create a symbolic Dirichlet boundary-condition descriptor.

        Parameters
        ----------
        tags : str | list[str]
            Boundary tag or tags where the condition is applied.
        values : callable | list[callable] | dict[int, callable] | None, optional
            Prescribed boundary values. ``None`` gives a homogeneous condition.

        Returns
        -------
        object
            Boundary-condition descriptor for use with ``init_fem(..., bcs=...)``.
        """
        try:
            from ..utils.solver.fem_route import dirichlet as _dirichlet_bc
        except ImportError as e:
            raise ImportError(
                "FEM support is not available. Install the FEM/dev extras to use domain.dirichlet(...) and init_fem(...)."
            ) from e
        return _dirichlet_bc(tags, values)

    def neumann(self, tags):
        """
        Create a symbolic Neumann boundary-condition descriptor.

        Parameters
        ----------
        tags : str | list[str]
            Boundary tag or tags where the condition is active.

        Returns
        -------
        object
            Boundary-condition descriptor for use with ``init_fem(..., bcs=...)``.
        """
        try:
            from ..utils.solver.fem_route import neumann as _neumann_bc
        except ImportError as e:
            raise ImportError(
                "FEM support is not available. Install the FEM/dev extras to use domain.neumann(...) and init_fem(...)."
            ) from e
        return _neumann_bc(tags)

    def periodic(self, *pairs):
        """
        Create a periodic boundary-condition descriptor.

        Example
        -------
            domain.init_fem(
                bcs=[domain.periodic(("left", "right"), ("bottom", "top"))],
            )
        """
        try:
            from ..utils.solver.fem_route import periodic as _periodic_bc
        except ImportError as e:
            raise ImportError(
                "FEM support is not available. Install the FEM/dev extras to use domain.periodic(...) and init_fem(...)."
            ) from e
        return _periodic_bc(*pairs)

    def _build_dirichlet_bc_info(self, dirichlet_tags, dirichlet_value_fns=None, vec: int = 1):
        """
        Build JAX-FEM Dirichlet boundary data from tagged user input.

        Parameters
        ----------
        dirichlet_tags : list[str]
            Boundary tags with Dirichlet constraints.
        dirichlet_value_fns : dict | None, optional
            Mapping from tag to value function definition.
        vec : int, default=1
            Number of field components.

        Returns
        -------
        list
            JAX-FEM ``dirichlet_bc_info`` in the form
            ``[location_fns, vec_ids, value_fns]``.
        """
        if dirichlet_value_fns is None:
            dirichlet_value_fns = {}

        loc_fns = []
        vec_ids = []
        val_fns = []

        def zero_fn(p):
            return 0.0

        for tag in dirichlet_tags:
            loc_fn = self._make_tag_location_fn(tag)
            if loc_fn is None:
                self.log.warning(f"Dirichlet tag '{tag}' not found in mesh tags. Skipping.")
                continue

            spec = dirichlet_value_fns.get(tag, None)

            # Case 1: no user-specified BC -> zero on all components
            if spec is None:
                for c in range(vec):
                    loc_fns.append(loc_fn)
                    vec_ids.append(c)
                    val_fns.append(zero_fn)
                continue

            # Case 2: scalar callable -> apply only to component 0
            # (keeps old scalar behaviour unchanged)
            if callable(spec):
                loc_fns.append(loc_fn)
                vec_ids.append(0)
                val_fns.append(spec)
                continue

            # Case 3: list/tuple of callables, one per component
            if isinstance(spec, (list, tuple)):
                if len(spec) != vec:
                    raise ValueError(f"Dirichlet BC for tag '{tag}' has {len(spec)} component functions, but vec={vec}.")
                for c, fn in enumerate(spec):
                    if not callable(fn):
                        raise TypeError(f"Dirichlet BC entry for tag '{tag}', component {c} is not callable.")
                    loc_fns.append(loc_fn)
                    vec_ids.append(c)
                    val_fns.append(fn)
                continue

            # Case 4: dict {component_id: callable}
            if isinstance(spec, dict):
                for c in sorted(spec.keys()):
                    fn = spec[c]
                    if not callable(fn):
                        raise TypeError(f"Dirichlet BC entry for tag '{tag}', component {c} is not callable.")
                    loc_fns.append(loc_fn)
                    vec_ids.append(int(c))
                    val_fns.append(fn)
                continue

            raise TypeError(f"Unsupported Dirichlet BC specification for tag '{tag}': {type(spec).__name__}")

        if len(loc_fns) == 0:
            return [[lambda p: False], [0], [lambda p: 0.0]]

        return [loc_fns, vec_ids, val_fns]

    def variational_symbols(self, value_shape=(), names=("u", "phi"), order=1, complex=False, space="Lagrange"):
        """
        Return generic variational symbols.

        ``order`` is the element polynomial degree for this field (1 = P1, 2 = P2).
        It is per-field (mixed methods like Taylor-Hood use different orders for
        different fields); the domain mesh stays linear and the FEM assembly mesh is
        promoted as needed.

        Parameters
        ----------
        value_shape : tuple, default=()
            Shape of the field value at one spatial point:
            ()    -> scalar
            (2,)  -> 2D vector
            (3,)  -> 3D vector
        names : tuple[str, str]
            Names of the trial and test symbols.

        Returns
        -------
        (trial, test)
            Symbolic variational placeholders carrying shape metadata.

        Examples
        --------
        Scalar Poisson:
            u, phi = domain.fem_symbols()

        2D vector elasticity:
            u, v = domain.fem_symbols(value_shape=(2,))

        3D vector elasticity:
            u, v = domain.fem_symbols(value_shape=(3,))
        """
        trial_name, test_name = names
        if complex:
            # A complex field is carried as TWO real fields (re, im) — the FEM-friendly
            # representation. The user writes the weak form with ordinary complex algebra
            # (`*` is the complex product, `1j`, `.conj`, `.real`/`.imag`); `jno.fem`
            # lowers `weak.real` onto the coupled (multifield) real system it already
            # assembles. Re-trial pairs with re-test, im-trial with im-test.
            from ..trace.views import ComplexPair

            re_tr = TrialFunction(name=f"{trial_name}_re", value_shape=value_shape, order=order, space=space)
            im_tr = TrialFunction(name=f"{trial_name}_im", value_shape=value_shape, order=order, space=space)
            re_te = TestFunction(name=f"{test_name}_re", value_shape=value_shape, order=order, space=space)
            im_te = TestFunction(name=f"{test_name}_im", value_shape=value_shape, order=order, space=space)
            re_te.field_key = re_tr.field_key
            im_te.field_key = im_tr.field_key
            for _s in (re_tr, im_tr, re_te, im_te):
                _s._domain = self
                # Mark these as members of a complex (re, im) pair. The real-equivalent weak form
                # couples re/im test functions within a single additive term, which the native
                # assembler's one-test-field-per-term classifier rejects; jno.fem routes any form
                # touching a complex field to the real-equivalent block path.
                _s._complex_field_member = True
            return (ComplexPair(re_tr, im_tr), ComplexPair(re_te, im_te))
        trial = TrialFunction(name=trial_name, value_shape=value_shape, order=order, space=space)
        test = TestFunction(name=test_name, value_shape=value_shape, order=order, space=space)
        test.field_key = trial.field_key  # one field per fem_symbols() call (pairs u<->phi)
        # Carry the owning domain so a consumer can recover the mesh / FE space from a
        # symbol alone -- e.g. jno.np.parameter(phi) sizing a field parameter to the
        # space (mirrors how Variable carries its _domain).
        trial._domain = self
        test._domain = self
        return (trial, test)

    def fem_symbols(self, value_shape=(), names=("u", "phi"), order=1, complex=False, space="Lagrange"):
        """
        Backward-compatible alias for variational_symbols().

        ``space`` selects the element family: ``"Lagrange"`` (default, nodal) or a non-nodal
        family on triangles — ``"RT"`` (H(div) Raviart-Thomas), ``"N1curl"`` (H(curl) Nedelec),
        ``"Hermite"`` (C0 value+gradient), ``"Argyris"`` (C1 conforming biharmonic) or ``"Morley"``
        (non-conforming biharmonic, cheap: 6 DOF). Non-nodal families assemble through the native
        push-forward engine.

        Examples
        --------
        Scalar:
            u, phi = domain.fem_symbols()

        Vector:
            u, v = domain.fem_symbols(value_shape=(2,))

        Mixed order (Taylor-Hood: P2 velocity, P1 pressure):
            u, v = domain.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
            p, q = domain.fem_symbols(names=("p", "q"))  # order=1

        Complex field (e.g. time-harmonic Maxwell / Helmholtz) — one symbol that is a
        genuine complex trial/test (carried as two coupled real fields under the hood):
            E, v = domain.fem_symbols(value_shape=(2,), complex=True)
            weak = curl(E) * curl(v) - k2 * E.dot(v) - J.dot(v)   # `*` = complex product
            fem  = jno.fem([weak.real, *bcs])                     # lowers to the real coupled solve
        """
        return self.variational_symbols(value_shape=value_shape, names=names, order=order, complex=complex, space=space)

    def test_function(self, value_shape=(), name="phi", order=1):
        """Return only the weak-form test function.

        Intended for NN-first weak VPINN authoring:
            phi = domain.test_function()
            u   = net(...)
            weak = ...
        """
        return TestFunction(name=name, value_shape=value_shape, order=order)

    def trial_function(self, value_shape=(), name="u", order=1):
        """Advanced helper for explicit FEM-only authoring."""
        return TrialFunction(name=name, value_shape=value_shape, order=order)

    def unknown(self, value_shape=(), name="u"):
        """The discrete **unknown solution field** on this domain's mesh — a *valued* P1 nodal field
        for strong-form / collocation methods (``jno.fdm``, …), the counterpart to the *symbolic*
        trial from :meth:`fem_symbols`.

        Where ``fem_symbols()`` gives an abstract weak-form ``TrialFunction`` (valued only during FE
        assembly), ``unknown()`` gives a field whose DOFs *are* the unknown, so it supports strong-form
        derivatives (``u.d2(x, scheme=...)``) and is the object a strong-form solver solves for::

            u = domain.unknown()
            jno.fdm([-u.d2(x) - u.d2(y) - f, u(xb, yb) - g]).solve()
        """
        from ..architectures.models import parameter

        sym = TrialFunction(name=name, value_shape=value_shape, order=1)
        sym._domain = self  # so parameter() sizes a P1 nodal field to this mesh's DOFs
        return parameter(sym)

    def _register_variational_sample(
        self,
        sample_tag: str,
        support: str,
        region_id: str,
        context_tag: str | None = None,
    ):
        """
        Register one sampled quadrature/surface tag as a variational region.

        Parameters
        ----------
        sample_tag : str
            User-facing / variable-facing tag used in domain.variable(...)
            e.g. "fem_gauss", "gauss_right", "gauss_wall_3"
        support : str
            "volume" or "boundary"
        region_id : str
            Geometry-level region id, e.g. "volume", "right", "wall_3", ...
        context_tag : str | None
            Internal context key if different from sample_tag.
        """
        if not hasattr(self, "_variational_sampling_registry"):
            self._variational_sampling_registry = {}

        self._variational_sampling_registry[sample_tag] = {
            "support": support,
            "region_id": region_id,
            "context_tag": context_tag if context_tag is not None else sample_tag,
        }

    def point_region(self, name: str, xy) -> "domain":
        """Register a single-node boundary region at the mesh vertex nearest ``xy``.

        Unlike :meth:`region` (which selects whole boundary *segments*), this pins
        one mesh vertex so a Dirichlet term such as ``p(domain.variable(name)) - 0``
        constrains exactly that node. The canonical use is fixing the pressure
        null space of a pure-Dirichlet Stokes problem so the saddle system is
        non-singular and solvable directly (no ``lstsq``/zero-mean workaround).

        Parameters
        ----------
        name : str
            Tag for the pinned node; usable in :meth:`variable` and as a Dirichlet
            region in :func:`jno.fem`.
        xy : array-like
            Target coordinates; the nearest mesh vertex is pinned (vertices are
            shared by P1 and P2 fields, so the pin lands on a node of either).
        """
        from .boundary_region import BoundaryRegion

        if self.mesh is None:
            raise ValueError("Mesh must be loaded before registering a point region.")
        pts = np.asarray(self.mesh.points)[:, : self.dimension]
        target = np.asarray(xy, dtype=float).reshape(-1)[: self.dimension]
        nid = int(((pts - target) ** 2).sum(axis=1).argmin())
        coord = pts[nid : nid + 1].copy()  # (1, D)
        bbox = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        tol = 1e-6 * bbox if bbox > 0 else 1e-7
        # Time-dependent domains store pools as (n_time, n_pts, D); broadcast the pin node
        # across the time slices so domain.variable(name) samples it like any other region.
        interior_pool = self._mesh_pool.get("interior")
        if interior_pool is not None and np.asarray(interior_pool).ndim == 3:
            n_time = int(np.asarray(interior_pool).shape[0])
            self._mesh_pool[name] = np.broadcast_to(coord, (n_time,) + coord.shape).copy()  # (n_time, 1, D)
        else:
            self._mesh_pool[name] = coord
        self._boundary_regions[name] = BoundaryRegion(
            tag=name, dim=self.dimension, points=coord, edges=None, triangles=None, tol=tol
        )
        self._register_variational_sample(sample_tag=name, support="boundary", region_id=name, context_tag=name)
        return self

    def init_fem_native(self, *, element_type: str = "TRI3", quad_degree: int = 2, bcs=None, vec: int = 1) -> "domain":
        """FEM-context init for the VPINN / grouped-weak-form path.

        Populates ``self.fem_context`` (the quadrature/shape-function/boundary tensors the
        grouped-weak-form evaluator reads), seeds the volume + per-Neumann-tag quadrature pools, and
        registers the variational samples, building the tensors from the native Lagrange element +
        facet machinery (P1/P2 nodal bases, affine geometry).
        """
        import numpy as onp

        from ..utils.solver.fem_native import build_native_fem_context
        from ..utils.solver.fem_route import expand_bcs

        if self.mesh is None:
            raise ValueError("Mesh must be loaded before initializing FEM context.")
        self._variational_initialized = True
        self._variational_sampling_registry = {}

        dirichlet_tags, dirichlet_value_fns, neumann_tags, _periodic = expand_bcs(bcs or [], vec=vec)

        # Dirichlet node ids: P1 geometry nodes matching each Dirichlet tag's location predicate.
        pts = onp.asarray(self.mesh.points)[:, : self.dimension]
        dirichlet_node_ids: List[int] = []
        for tag in dirichlet_tags:
            loc_fn = self._make_tag_location_fn(tag)
            if loc_fn is None:
                continue
            for i, p in enumerate(pts):
                try:
                    inside = bool(loc_fn(p))
                except Exception:
                    inside = False
                if inside:
                    dirichlet_node_ids.append(i)

        fem_context, vol_quad, surf_quad_by_tag, surf_normals_by_tag = build_native_fem_context(
            self,
            element_type=element_type,
            quad_degree=quad_degree,
            vec=vec,
            neumann_tags=list(neumann_tags),
            dirichlet_node_ids=dirichlet_node_ids,
        )

        self._fem_backend = "native"
        self._fem_element_type = element_type
        self._fem_quad_degree = quad_degree
        self._fem_default_vec = vec
        self._fem_solver_enabled = True
        self.fem_context = fem_context

        # Volume quadrature sampling (mirrors init_fem; time-broadcast when the domain carries a window).
        if getattr(self, "_is_time_dependent", False):
            n_time = int(getattr(self, "_n_time", len(getattr(self, "_time_points", [0.0]))))
            self._mesh_pool["fem_gauss"] = onp.broadcast_to(vol_quad[None, :, :], (n_time, *vol_quad.shape)).copy()
        else:
            self._mesh_pool["fem_gauss"] = vol_quad
        self._register_variational_sample(
            sample_tag="fem_gauss", support="volume", region_id="volume", context_tag="fem_gauss"
        )

        # Per-Neumann-tag boundary quadrature sampling + outward normals.
        for tag, qpts in surf_quad_by_tag.items():
            self._mesh_pool[f"gauss_{tag}"] = jnp.asarray(qpts)
            nrm = surf_normals_by_tag.get(tag)
            if nrm is not None and hasattr(self, "normals_by_tag"):
                self.normals_by_tag[f"gauss_{tag}"] = onp.asarray(nrm)
            self._register_variational_sample(
                sample_tag=f"gauss_{tag}", support="boundary", region_id=tag, context_tag=f"gauss_{tag}"
            )

        self._fem_dirichlet_tags = list(dirichlet_tags)
        self._fem_neumann_tags = list(neumann_tags)
        self._fem_dirichlet_value_fns = dirichlet_value_fns if dirichlet_value_fns is not None else {}

        self.context.update(
            {
                k: v
                for k, v in self.fem_context.items()
                if not (getattr(v, "ndim", 0) >= 1 and getattr(v, "shape", (1,))[0] == 0)
            }
        )
        return self

    def _make_tag_location_fn(self, tag):
        """
        Build a point-membership function for a boundary tag.

        Parameters
        ----------
        tag : str
            Boundary tag name.

        Returns
        -------
        callable | None
            Function returning whether a point belongs to the tagged region,
            or ``None`` if the tag is unknown.
        """
        # A domain.tag(name, where) region resolves to: the predicate AND on the domain boundary.
        # the assembler applies a location-fn to EVERY node, so a bare predicate selecting a thick region
        # would also pin interior dofs (a thick boundary predicate pinned the interior velocity and
        # silently zeroed the interior pressure rows). Intersecting with the full-boundary region
        # keeps Dirichlet boundary-restricted, while the exact predicate misses no boundary node
        # (per-facet proximity alone can miss nodes on a curved boundary). NB: such a predicate is
        # evaluated under JAX here, so it must be jax-traceable (jno.np / arithmetic, not bare numpy).
        where = getattr(self, "_tag_predicates", {}).get(tag, None)
        if where is not None:
            dim = self.dimension
            full = self._boundary_regions.get("boundary", None)

            def _loc(p):  # location functions take a single point argument
                import jax.numpy as jnp

                p = jnp.asarray(p)
                pred = where(*(p[..., i] for i in range(dim)))
                return pred if full is None else (pred & full.contains(p))

            return _loc

        region = self._boundary_regions.get(tag, None)
        if region is None:
            return None
        return lambda p: region.contains(p)

    def tag_node_mask(self, tag, points):
        """Boolean mask of which ``points`` belong to ``tag`` -- the float64-safe resolution.

        The companion to :meth:`_make_tag_location_fn`, and the one an assembler should use when it
        already holds concrete coordinates. A ``domain.tag(name, where)`` predicate is evaluated **in
        numpy float64** here rather than under JAX, because ``jnp.asarray`` truncates coordinates to
        float32 unless x64 is enabled -- and a tag tolerance finer than float32 eps then matches no
        point at all. ``d.tag("right", lambda x: x > 1 - 1e-9)`` on a domain reaching x = 1 is the
        canonical case: in float32 ``1 - 1e-9`` rounds to exactly 1.0, the strict ``>`` is false for
        every node, and an essential condition bound to that tag is dropped in silence.

        Only the *user* predicate moves off the JAX path. The boundary restriction it is intersected
        with stays where it was -- a tolerance-based proximity test against the region's own points,
        which float32 cannot break.

        Returns ``None`` when the tag is unknown, matching ``_make_tag_location_fn``.
        """
        import jax
        import jax.numpy as jnp

        loc = self._make_tag_location_fn(tag)
        if loc is None:
            return None
        pts = np.asarray(points)
        n = int(pts.shape[0])

        def _vmapped(fn, num_args=1):
            # Chunked for the same reason the callers chunked: a geometric region predicate tests one
            # point against every boundary facet, so an unchunked vmap materialises (n x n_facets) --
            # 804 MB on a realistic 3-D mesh, which exhausts the device before assembly starts.
            pj, chunk, out = jnp.asarray(pts), 512, []
            for s in range(0, n, chunk):
                blk = pj[s : min(s + chunk, n)]
                hit = jax.vmap(fn)(blk) if num_args == 1 else jax.vmap(fn)(blk, jnp.arange(s, s + blk.shape[0]))
                out.append(np.asarray(hit).reshape(-1))
            return np.concatenate(out) if out else np.zeros(0, dtype=bool)

        where = (getattr(self, "_tag_predicates", {}) or {}).get(tag)
        if where is not None:
            pts64 = np.asarray(points, dtype=np.float64)
            dim = int(self.dimension)
            mask = np.asarray(where(*(pts64[:, i] for i in range(dim)))).reshape(-1).astype(bool)
            full = self._boundary_regions.get("boundary", None)
            if full is not None:
                mask &= _vmapped(full.contains).astype(bool)
            return mask

        num_args = loc.__code__.co_argcount if hasattr(loc, "__code__") else 1
        return _vmapped(loc, num_args).astype(bool)

    def tag(self, name, where, region=None):
        """Define a named region from a **spatial** predicate ``where(x, y[, z]) -> bool``.

        ``region=`` restricts the predicate to **one body's** nodes. It exists because a spatial
        predicate cannot always separate what you mean: a ``Shape.regions(..., conforming=False)``
        domain meshes each body independently, so the shared surface exists *twice* -- two coincident
        sets of nodes at identical coordinates. No function of ``(x, y, z)`` can tell them apart.
        Naming which body owns the facet supplies the missing discriminator::

            d.tag("film_face", lambda x, y: abs(y - 1.0) < 1e-9, region="coating")
            d.tag("base_face", lambda x, y: abs(y - 1.0) < 1e-9, region="substrate")
            fem = jno.fem([..., T(a[0], a[1]) - T(b[0], b[1])])   # glue the two bodies

        It is equally the answer to "the part of the outer boundary belonging to *this* body", which a
        coordinate predicate also cannot express once two bodies touch.

        One general method for naming any subset of the domain -- interior *or* boundary. The
        region is abstract (predicate-based), so it is carried even without a mesh: the PINN
        sampler draws the points satisfying ``where`` each step. After ``build_mesh`` it also maps
        to mesh nodes/facets, and a ``jno.fem`` boundary condition bound to ``name`` is applied on
        exactly the boundary facets where ``where`` holds (FEM location-functions are evaluated only
        against the boundary, so a bare predicate selects the right subset -- there is no
        interior/boundary flag). Untagged boundary stays natural (do-nothing), which is how you get a
        natural outflow on a complex geometry.

        Only the **spatial** coordinates are passed to ``where`` (never time): a region is the same
        at every time level of a time-dependent domain. A shapely geometry may be passed instead of
        a callable. Returns ``self`` (chainable). To name a region *and* grab its coordinates in one
        line, pass the predicate straight to :meth:`variable` instead::

            xl, yl, zl, _ = dom.variable("left", where=lambda x, y, z: x < 1e-6)

        Example::

            dom.tag("inlet",    lambda x, y: x < 1e-6)            # boundary subset -> Dirichlet there
            dom.tag("cylinder", lambda x, y: (x-0.8)**2 + (y-0.5)**2 < 0.21**2)
            # an untagged outlet edge is left as a natural (do-nothing) outflow
        """
        self._tag_predicates = getattr(self, "_tag_predicates", {})
        if callable(where) and _is_facet_predicate(where):
            # Richer boundary-facet predicate f(x, n, name): coords + outward normal + current name.
            return self._tag_by_facet(name, where)
        if not callable(where):  # accept a shapely geometry
            geom = where

            def where(x, y, _g=geom):  # noqa: ANN001
                from shapely import contains_xy as _cxy

                return np.asarray(_cxy(_g, np.asarray(x), np.asarray(y)))

        self._tag_predicates[name] = where
        if region is not None:
            # Stored, not applied here: `_register_tag_boundary_region` works on facet COORDINATES and
            # dedups them, so the two coincident sides of a non-conforming interface are already
            # indistinguishable by the time it runs. The restriction is resolved where node IDs still
            # exist -- see `fem_native._boundary_node_ids`.
            known = set(getattr(self, "tag_indices", {}) or {})
            if str(region) not in known:
                raise ValueError(
                    f"tag({name!r}, region={region!r}): unknown region. Known: {sorted(known)}. "
                    "`region=` names the BODY that owns the facets (a Shape.regions name), which is "
                    "what tells two coincident interface surfaces apart."
                )
            self._tag_regions = getattr(self, "_tag_regions", {})
            self._tag_regions[name] = str(region)
        self._materialize_tag_pool(name, where)
        self._register_tag_boundary_region(name, where, region)
        # Lazy mesh-free sampling: register the parent geometry so PolygonDomain.sample can draw the
        # region with sample=(n, None); _sample_interior filters by the predicate (resampled each step).
        poly_tags = getattr(self, "_polygon_tags", None)
        geom = getattr(self, "_active_geometry", None)
        if poly_tags is not None and geom is not None and name not in poly_tags:
            poly_tags[name] = ("interior", geom)
        if name not in self.avaiable_mesh_tags:
            self.avaiable_mesh_tags.append(name)
        return self

    def _tag_by_facet(self, name, where):
        """Name a boundary subset by a richer predicate ``f(x, n, name)``: facet centroids,
        outward facet normals, and each facet's current region name (in/exclude in one predicate).

        Additive path (only for facet predicates): selects boundary facets and registers a
        ``BoundaryRegion`` + sampling pool + per-tag normals, so ``variable(name, normals=True)``
        and a ``jno.fem`` BC bound to ``name`` work. Needs a meshed boundary.
        """
        full = self._boundary_regions.get("boundary")
        dim = self.dimension
        ents = None if full is None else full.facets
        if ents is None or len(ents) == 0:
            raise ValueError(f"tag({name!r}): a facet predicate f(x, n, name) needs a meshed boundary.")
        ents = np.asarray(ents)  # (E, k, dim) facet-vertex coordinates
        mid = ents.mean(axis=1)  # (E, dim) facet centroids
        nrm = _facet_normals(ents, dim, getattr(self, "mesh", None))  # (E, dim) outward facet normals
        names = _facet_current_names(self._boundary_regions, mid)  # (E,) each facet's current name
        keep = np.asarray(where(mid, nrm, names)).reshape(-1).astype(bool)
        if not keep.any():
            raise ValueError(f"tag({name!r}): the facet predicate f(x, n, name) selected no boundary facets.")
        sub, sub_n = ents[keep], nrm[keep]
        bpts = np.unique(sub.reshape(-1, sub.shape[-1])[:, :dim], axis=0)
        self._boundary_regions[name] = BoundaryRegion.from_facets(name, dim, bpts, sub, tol=full.tol)
        # Time-dependent domains store pools as (n_time, n_pts, D) and `sample` indexes the SPATIAL axis
        # as `group_points[:, idx, :]`. Storing the bare (n_pts, D) here made `variable(name)` raise
        # "too many indices" on any time-dependent domain, so a facet predicate could name a boundary but
        # never read its coordinates or normals. Same broadcast the pin-node path above uses.
        _interior_pool = self._mesh_pool.get("interior")
        if _interior_pool is not None and np.asarray(_interior_pool).ndim == 3:
            _n_time = int(np.asarray(_interior_pool).shape[0])
            self._mesh_pool[name] = np.broadcast_to(bpts, (_n_time,) + bpts.shape).copy()
        else:
            self._mesh_pool[name] = bpts
        # per-point outward normals for variable(name, normals=True): average the facet normals at each point
        self.normals_by_tag[name] = _point_normals_from_facets(sub, sub_n, bpts, dim)
        self.tag_indices[name] = np.arange(len(bpts))
        if name not in self.avaiable_mesh_tags:
            self.avaiable_mesh_tags.append(name)
        return self

    def by_region(self, values, *, default=None):
        """A coefficient whose value depends on which region a mesh cell is in.

        ``values`` is a ``{region: value}`` mapping; the returned coefficient evaluates, on each cell,
        to the value of the region that cell's **centroid** lies in. Write a multi-region weak form as a
        **single** equation over the whole ``interior`` instead of one term per region::

            k = d.by_region({"steel": 16.0, "air": 0.026})     # per-region conductivity
            Q = d.by_region(heat_source, default=0.0)          # 0 in any unlisted region
            heat = k * (T.x*s.x + T.y*s.y) - Q * s             # one equation, all regions

        It is *general* -- a value can be any coefficient (a python scalar, a ``jno.fn`` field, or a
        trainable ``jno.np.parameter``), so the same primitive expresses conductivity, a source, a
        density, a reaction rate, an elastic modulus, ... and trainable per-region values compose for
        free (``d.by_region({**k, "air": nu*0.026})``).

        Each region must be a geometry part (``from_regions``) or a ``domain.tag`` predicate. ``default``
        is the value for cells in no listed region; ``default=None`` (the strict default) requires the
        regions to cover every value of interest -- a cell in an unlisted region simply contributes ``0``,
        and an unknown region name raises. Desugars to ``sum_r RegionMask(r) * values[r]`` -- the proven
        per-region integration path (``tests/test_fem_per_region.py``); ``jno.fem`` logs the expansion.
        """
        # ``_shape_regions`` are the named sub-regions of a ``Shape.regions`` / ``a.name(..) + b.name(..)``
        # plan. They belong here for the same reason geometry parts do -- ``RegionMask`` resolves them
        # through the same centroid-membership path (``_cell_region_mask``). Without them a
        # Shape-built multi-material domain could not use ``by_region`` at all.
        valid = (
            set(getattr(self, "_source_regions", {}) or {})
            | set(getattr(self, "_tag_predicates", {}) or {})
            | set(getattr(self, "_shape_regions", {}) or {})
        )
        unknown = [r for r in values if r not in valid]
        if unknown:
            raise ValueError(
                f"domain.by_region: unknown region(s) {sorted(unknown)}; each key must be a geometry part, "
                f"a Shape.regions sub-region, or a domain.tag predicate. Known regions: {sorted(valid)}."
            )
        expr = _masked_sum(values, RegionMask, default, what="by_region", key="region")
        # NB: get_logger's first positional is a log *directory* -- get_logger(__name__) literally
        # creates a folder named `jno.domain.domain_class/` in the caller's cwd.
        get_logger().info(f"by_region: per-region coefficient over {len(values)} region(s): {sorted(map(str, values))}")
        return expr

    def by_tag(self, values, *, default=None):
        """A **surface** coefficient whose value depends on which boundary tag a facet carries — the
        mirror of :meth:`by_region`, for the boundary rather than the volume.

        ``values`` is a ``{tag: value}`` mapping. The returned coefficient evaluates, on each boundary
        facet, to the value of the tag owning it, so a mixed-boundary condition is **one** term over the
        whole boundary instead of one term per tag::

            d.tag("wall", lambda x, y: x < 1e-9)
            d.tag("lid",  lambda x, y: y > 1 - 1e-9)
            h = d.by_tag({"wall": 25.0, "lid": 5.0})       # per-tag film coefficient
            xb, yb, _ = d.variable("boundary", split=True)
            ub, vb = u.bind(x=xb, y=yb), v.bind(x=xb, y=yb)
            robin = h * (ub - T_inf) * vb                   # one equation, both tags

        A facet belongs to a tag by the assembler's own facet selection — the same rule that decides
        which facets a Dirichlet condition on that tag pins — so the two can never disagree. As in
        ``by_region``, a value can be any coefficient (scalar, symbolic expression, trainable
        ``jno.np.parameter``, typed view), and ``default`` fills the facets in no listed tag.

        Desugars to ``sum_t TagMask(t) * values[t]``. **Surface terms only**: used in a volume term, on
        a non-nodal space, or in 1-D it raises rather than integrating over the wrong thing. A tag that
        owns no boundary facet on this mesh raises at assembly rather than contributing silent zero.
        """
        valid = set(getattr(self, "_tag_predicates", {}) or {}) | set(getattr(self, "_boundary_regions", {}) or {})
        unknown = [t for t in values if t not in valid]
        if unknown:
            raise ValueError(
                f"domain.by_tag: unknown tag(s) {sorted(unknown)}; each key must be a boundary tag "
                f"(``domain.tag(name, where)``). Known tags: {sorted(valid)}."
            )
        expr = _masked_sum(values, TagMask, default, what="by_tag", key="tag")
        get_logger().info(f"by_tag: per-tag surface coefficient over {len(values)} tag(s): {sorted(map(str, values))}")
        return expr

    def _collect_region_attachments(self, items):
        """Index ``Shape.attach(...)`` properties as ``{property: {region: value}}`` for ``__getattr__``."""
        attached: dict = {}
        kinds: dict = {}
        for name, sub in items:
            for prop, value in (getattr(sub, "_attach", None) or {}).items():
                attached.setdefault(str(prop), {})[str(name)] = value
                kinds[str(name)] = "volume"  # a Shape region is a body: always per-cell
        self._region_attachments = attached
        self._attachment_kind = kinds

    def attach(self, target: str, **props):
        """Declare material properties on an existing **region or boundary tag**, after the domain is built.

        The runtime counterpart of :meth:`Shape.attach`, and the only way to attach to a
        ``domain.tag`` — or to a mesh-file domain, which has no ``Shape`` to declare them on::

            d.tag("wall", lambda x, y: x < 1e-9)
            d.tag("lid",  lambda x, y: y > 1 - 1e-9)
            d.attach("wall", h=25.0).attach("lid", h=5.0)
            robin = d.h * (ub - T_inf) * vb          # ONE term over the whole boundary

        Whether a target is a **volume** or a **surface** quantity is decided here, once, from what the
        target actually owns on this mesh -- a tag owning boundary facets is a surface, a tag owning
        only cells is a volume, and a tag owning both is ambiguous and raises rather than guessing.
        ``d.<prop>`` then emits the matching coefficient (:meth:`by_region` or :meth:`by_tag`).

        Returns ``self`` so declarations chain. Repeated calls merge (last wins).
        """
        target = str(target)
        kind = self._attachment_target_kind(target)
        attached = self.__dict__.setdefault("_region_attachments", {})
        kinds = self.__dict__.setdefault("_attachment_kind", {})
        for prop, value in props.items():
            attached.setdefault(str(prop), {})[target] = value
        kinds[target] = kind
        self._check_attachment_clashes()
        return self

    def _attachment_target_kind(self, target: str) -> str:
        """``"volume"`` or ``"surface"`` for an attach target — resolved from what it owns on this mesh.

        A ``domain.tag`` is deliberately general (the docstring of :meth:`tag` says so): it names any
        subset, interior *or* boundary. So the kind cannot be read off the name, and guessing it wrong
        picks the wrong mask -- a surface coefficient integrated over cells, or the reverse, which is
        wrong physics and not an error anywhere. Decide it once, here, and refuse the ambiguous case.
        """
        if target in (getattr(self, "_shape_regions", {}) or {}) or target in (getattr(self, "_source_regions", {}) or {}):
            return "volume"
        if target not in (getattr(self, "_tag_predicates", {}) or {}) and target not in (
            getattr(self, "_boundary_regions", {}) or {}
        ):
            known_r = sorted(
                set(getattr(self, "_shape_regions", {}) or {}) | set(getattr(self, "_source_regions", {}) or {})
            )
            known_t = sorted(getattr(self, "_tag_predicates", {}) or {})
            raise ValueError(f"domain.attach: unknown target {target!r}. Known regions: {known_r}; known tags: {known_t}.")
        has_facets = target in (getattr(self, "_boundary_regions", {}) or {})
        has_cells = False
        try:
            from ..utils.solver.fem_utils import _cell_region_mask

            has_cells = bool(np.asarray(_cell_region_mask(self, target)).any())
        except Exception:  # noqa: BLE001 - no mesh yet, or a predicate this path cannot evaluate
            has_cells = False
        if has_facets and has_cells:
            raise ValueError(
                f"domain.attach: tag {target!r} owns both interior cells and boundary facets, so whether "
                f"its properties are volume or surface quantities is ambiguous. Split it into two tags, "
                f"or build the coefficient explicitly with d.by_region({{...}}) / d.by_tag({{...}})."
            )
        if not has_facets and not has_cells:
            raise ValueError(
                f"domain.attach: tag {target!r} owns neither a cell nor a boundary facet on this mesh, so "
                f"nothing attached to it could ever be integrated. Check the tag's predicate."
            )
        return "surface" if has_facets else "volume"

    def _check_attachment_clashes(self):
        """Reject an attached name that collides with a real ``domain`` attribute, at build time.

        It has to be build time, and it has to be the *end* of ``__init__``: ``__getattr__`` only runs
        when normal lookup *fails*, so an attached ``mesh`` or ``tag`` would be silently shadowed by
        the attribute of the same name -- the user would get a bound method where they asked for a
        coefficient, with nothing raised anywhere. Checking both the class (methods, properties) and
        the instance dict is what makes it catch ``mesh``, which is only ever set on the instance.
        """
        attached = self.__dict__.get("_region_attachments") or {}
        clashes = sorted(p for p in attached if hasattr(type(self), p) or p in self.__dict__)
        if clashes:
            raise ValueError(
                f"attach: {clashes} collide with existing jno.domain attributes and would be "
                f"unreachable as `d.{clashes[0]}`. Rename the attached propert"
                f"{'ies' if len(clashes) > 1 else 'y'}."
            )

    def __getattr__(self, name):
        """Resolve ``d.<prop>`` for a property attached via :meth:`Shape.attach`.

        Only reached when normal attribute lookup fails, so it can never shadow a real method; the
        build-time check in ``_check_attachment_clashes`` covers the reverse direction.
        """
        # `self.__dict__` directly, never `getattr`/`self._x` -- during __init__ (and under copy /
        # pickle, which build an instance without running it) any attribute miss lands here, and
        # touching another missing attribute would recurse forever.
        attached = self.__dict__.get("_region_attachments") or {}
        if name in attached:
            values = attached[name]
            kinds = self.__dict__.get("_attachment_kind") or {}
            declared = {kinds.get(t, "volume") for t in values}
            if len(declared) > 1:
                raise AttributeError(
                    f"domain.{name}: declared on both a volume region and a boundary tag "
                    f"({sorted(values)}), so it has no single meaning -- a coefficient is integrated "
                    f"over cells or over facets, not both. Use d.by_region({{...}}) and "
                    f"d.by_tag({{...}}) explicitly for the two halves."
                )
            resolved = {t: self._resolve_attached(v) for t, v in values.items()}
            if declared == {"surface"}:
                # No completeness rule on the boundary: tags are not a partition of it, and untagged
                # boundary is deliberately natural (do-nothing) in jNO. Facets no tag claims simply
                # contribute nothing to this coefficient -- documented, not silently patched.
                return self.by_tag(resolved)
            known = set(self.__dict__.get("_shape_regions") or {})
            missing = sorted(known - set(values))
            if missing:
                raise AttributeError(
                    f"domain.{name}: region(s) {missing} never attached a '{name}'. Every region must "
                    f"declare it -- add .attach({name}=...) to each, or use "
                    f"d.by_region({{...}}, default=...) explicitly if some regions genuinely have none."
                )
            return self.by_region(resolved)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def attached(self, name: str) -> dict:
        """The raw ``{region: value}`` mapping for a property declared with :meth:`Shape.attach`.

        ``d.<name>`` gives the per-region *coefficient*, ready for a weak form. This gives the values
        themselves, for the consumers that need a mapping rather than an expression -- most notably
        ``enclosure.emissivity({tag: eps})``, since enclosure element tags are region names::

            eps = gap.emissivity(d.attached("eps"))

        Unlike ``d.<name>`` this does not require every region to have declared the property: an
        enclosure only spans some of them, and a mapping is allowed to be partial.
        """
        attached = self.__dict__.get("_region_attachments") or {}
        if name not in attached:
            raise KeyError(f"domain.attached: no region declared {name!r}. Declared properties: {sorted(attached)}.")
        return dict(attached[name])

    def _resolve_attached(self, value):
        """A plain function attached as a property value is called here with this domain's spatial
        variables, so ``.attach(k=lambda r, z: 2.0 + 0.5*z)`` becomes a symbolic coefficient.

        It cannot be resolved at ``Shape.attach`` time: a spatially varying value has to be built from
        ``d.variable(...)``, and the domain does not exist while the geometry plan is being written.
        ``isroutine`` rather than ``callable`` on purpose -- symbolic expressions define ``__call__``
        (that is how ``u(x, y)`` binds), so ``callable`` would try to invoke them as size functions.
        """
        if not inspect.isroutine(value):
            return value
        coords = self.variable("interior", split=True)
        return value(*coords[: self.dimension])

    def _register_tag_boundary_region(self, name, where, region=None):
        """If any **boundary** facets satisfy ``where``, register a ``BoundaryRegion`` for ``name``
        so a ``jno.fem`` term bound to it classifies as a boundary (Dirichlet / Neumann) condition
        and ``variable(name, normals=True)`` works. Interior-only tags add nothing here (they stay
        interior sampling regions). **1-D vertices**; 2-D edges; 3-D triangles.

        With ``region=`` the search also covers **interface** facets. It has to: a non-conforming
        interface is deliberately kept out of the ``"boundary"`` region -- otherwise a plain
        ``u(boundary) - g`` would pin it and silently solve two disconnected bodies -- so a predicate
        over the interface plane would otherwise match nothing at all and the tag would never register.
        The facets of the two sides are coincident, so this cannot separate them (it dedups by
        coordinate); that is resolved later against node IDs in ``fem_native._boundary_node_ids``.
        """
        full = self._boundary_regions.get("boundary")
        if full is None:
            return
        dim = self.dimension
        if dim == 1:
            # A 1-D boundary facet IS a vertex, so there are no edges/triangles to search and the
            # endpoints themselves are the entities. Without this branch the `triangles` lookup below
            # found nothing and returned, leaving `d.tag` with no boundary region at all -- silently:
            # the tag still sampled, `d.variable(name)` still worked, and nothing complained until
            # `jno.fem` rejected `u(tag) - g` as a whole-domain residual and blamed the residual.
            _bp = np.asarray(full.points, dtype=float).reshape(len(np.asarray(full.points)), -1)
            _keep = np.asarray(where(_bp[:, 0])).reshape(-1).astype(bool)
            if not _keep.any():
                return
            self._boundary_regions[name] = BoundaryRegion(tag=name, dim=1, points=_bp[_keep][:, :1], tol=full.tol)
            # The outward sign (-1 at the left end, +1 at the right) is taken from the "boundary"
            # region's OWN per-point normals rather than recomputed, so a tagged endpoint can never
            # disagree with the built-in tag naming the same point.
            _bn = self.normals_by_tag.get("boundary")
            if _bn is not None and len(np.asarray(_bn)) == len(_bp):
                self.normals_by_tag[name] = np.asarray(_bn).reshape(len(_bp), -1)[_keep]
            return
        blocks = [full.facets]
        if region is not None:
            blocks += [r.facets for t, r in self._boundary_regions.items() if "|" in t and r.facets is not None]
        blocks = [np.asarray(b) for b in blocks if b is not None and len(b)]
        # Every block must be the same kind of facet to stack: they are, on the single-cell-type mesh
        # jNO assembles on, but drop any that is not rather than raising out of numpy.
        blocks = [b for b in blocks if b.shape[1:] == blocks[0].shape[1:]] if blocks else blocks
        ents = np.concatenate(blocks, axis=0) if blocks else None
        if ents is None or len(ents) == 0:
            return
        ents = np.asarray(ents)  # (E, k, dim)
        mid = ents.mean(axis=1)  # facet centroids (E, dim)
        keep = np.asarray(where(*[mid[:, i] for i in range(dim)])).reshape(-1).astype(bool)
        if not keep.any():
            return
        sub = ents[keep]
        bpts = np.unique(sub.reshape(-1, sub.shape[-1])[:, :dim], axis=0)
        self._boundary_regions[name] = BoundaryRegion.from_facets(name, dim, bpts, sub, tol=full.tol)
        # Per-point outward normals, exactly as the facet-predicate path stores them. Without this a
        # coordinate-tagged boundary had a region but NO entry in ``normals_by_tag``, so every reader
        # of that dict silently saw nothing for the tag — the mesh cell-set path was the only source,
        # which is why the miss only surfaced once a tag was re-derived from a predicate instead.
        self.normals_by_tag[name] = _point_normals_from_facets(
            sub, _facet_normals(sub, dim, getattr(self, "mesh", None)), bpts, dim
        )

    def _materialize_tag_pool(self, name, where):
        """Populate ``_mesh_pool[name]`` with the spatial points (interior + boundary) satisfying
        ``where`` -- so ``variable(name)`` can sample it. Spatial coords only; on a time-dependent
        domain the same spatial selection is carried at every time level."""
        dim = self.dimension
        pools = {k: np.asarray(self._mesh_pool[k]) for k in ("interior", "boundary") if k in self._mesh_pool}
        time_dep = bool(self._is_time_dependent and self.time is not None)
        n_time = int(self.time[2]) if time_dep else 0

        if pools:
            # Meshed: keep the existing mesh nodes (interior + boundary) satisfying the predicate.
            ref = pools.get("interior", next(iter(pools.values())))
            time_dep = ref.ndim >= 3 or time_dep
            n_time = ref.shape[0] if ref.ndim >= 3 else n_time
            spatial = lambda a: (a[0] if a.ndim >= 3 else a)[:, :dim]  # noqa: E731  time-invariant coords
            cand = np.unique(np.concatenate([spatial(a) for a in pools.values()], axis=0), axis=0)
        else:
            # Mesh-free: rejection-sample the domain geometry, then filter by the predicate, so the
            # PINN sampler can draw from the region each step (subsampling this candidate pool).
            sampler = getattr(self, "_sample_points_in_polygon", None)
            geom = getattr(self, "_active_geometry", None)
            if sampler is None or geom is None:
                return  # cannot materialise yet; nothing to sample from
            cand = np.asarray(sampler(geom, 8000))[:, :dim]

        mask = np.asarray(where(*[cand[:, i] for i in range(dim)])).reshape(-1).astype(bool)
        sel = cand[mask]
        if time_dep and n_time > 0:
            self._mesh_pool[name] = np.broadcast_to(sel[None], (n_time, sel.shape[0], dim)).copy()
        else:
            self._mesh_pool[name] = sel

    def assemble_weak_form(self, expr, **kwargs):
        """Assemble a symbolic weak-form expression into a VPINN GroupedAssembly."""
        from ..utils.solver.weak_form import assemble_weak_form

        return assemble_weak_form(self, expr, **kwargs)

    # Generators
    def _structured_grid_setup(self, shape):
        """Resolve a :meth:`jno.Shape.structured` plan into a lattice constructor.

        Returns ``(constructor, grid_meta)`` where ``constructor(geo) -> (meshio.Mesh, dim, ds)`` builds
        the regular grid over the rectangle (:meth:`Geometries.equi_distant_rect`) or box
        (:meth:`Geometries.equi_distant_box`) — the ``left/right/bottom/top[/front/back]`` +
        ``boundary`` + ``interior`` cell sets come with it — and ``grid_meta`` is the
        ``{"shape": (Nx, Ny[, Nz]), "spacing": (hx, hy[, hz]), "origin": (...)}`` descriptor stamped
        onto ``mesh_connectivity["grid"]`` and exposed as :attr:`domain.grid`. That is the key
        ``jno.fdm``'s kernels read to take the assembly-free 5-/7-point-stencil path (node order
        ``idx = ((i·Ny + j)·Nz + k)``).

        The cell comes from the plan's own :meth:`jno.Shape.quad` choice: a lattice is the one 3-D plan
        that CAN be hex-meshed, so ``.structured().quad()`` is how a hexahedral mesh is spelled.

        Refuses by name rather than falling back to gmsh — a caller who then reads ``domain.grid`` or
        expects hexes would otherwise fail somewhere else, having silently solved on another mesh.
        """
        from ..geometry.primitives import Box, Rect

        node = getattr(shape, "_node", None)
        prim = node[1] if (isinstance(node, tuple) and node and node[0] == "leaf") else None
        if not isinstance(prim, (Rect, Box)):
            _what = (
                f"a {type(prim).__name__.lower()}"
                if prim is not None
                else ("no plan at all" if node is None else f"a {node[0]!r} plan")
            )
            raise NotImplementedError(
                f"Shape.structured() needs a single axis-aligned rect/box; this is {_what}. Mesh it "
                "unstructured (drop .structured()), or decompose it into mapped blocks. Cut-cell and "
                "transfinite/swept structured meshing are planned."
            )
        if int(getattr(shape, "_mesh_order", 1)) > 1:
            raise NotImplementedError(
                "Shape.structured().curved() is not supported: a curved lattice cell is a 9-/27-node "
                "block the lattice builder does not emit. Use .curved() on an unstructured plan, or "
                "raise the BASIS order instead (jno.fem(..., order=2) on a straight mesh)."
            )
        cells = shape.cell_choices()
        if len(cells) > 1:
            raise NotImplementedError(
                f"Shape.structured(): this plan asks for more than one cell type ({', '.join(sorted(cells))})."
            )
        tensor = "quad" in cells

        size = getattr(shape, "_size", None)
        counts = tuple(getattr(shape, "_structured", ()) or ())
        if not counts:
            if callable(size):
                raise NotImplementedError(
                    "Shape.structured() derives its cell counts from the shape's size=, and a spatially "
                    "varying size=<callable> (a graded mesh) has no single count. Pass explicit counts "
                    "-- .structured(n=32) or .structured(n=(32, 16)) -- or drop .structured()."
                )
            h = float(size) if isinstance(size, (int, float)) and size > 0 else 0.1
            if not (isinstance(size, (int, float)) and size > 0):
                self.log.info(f"Shape.structured(): no scalar size= on the shape; defaulting to spacing h={h}.")
        else:
            h = None

        def _axis(lo, hi, i):
            """``(lo, hi, n_cells)``; >= 2 cells so the 3-point edge stencil is defined."""
            a, b = sorted((float(lo), float(hi)))
            n = counts[i] if counts else int(round((b - a) / h))
            return a, b, max(2, int(n))

        if isinstance(prim, Rect):
            x_lo, x_hi, nx = _axis(prim.x0, prim.x1, 0)
            y_lo, y_hi, ny = _axis(prim.y0, prim.y1, 1)
            constructor = Geometries.equi_distant_rect(
                x_range=(x_lo, x_hi), y_range=(y_lo, y_hi), nx=nx, ny=ny, cell="quad" if tensor else "triangle"
            )
            grid_meta = {
                "shape": (nx + 1, ny + 1),
                "spacing": ((x_hi - x_lo) / nx, (y_hi - y_lo) / ny),
                "origin": (x_lo, y_lo),
            }
        else:  # Box (3-D)
            x_lo, x_hi, nx = _axis(prim.x0, prim.x1, 0)
            y_lo, y_hi, ny = _axis(prim.y0, prim.y1, 1)
            z_lo, z_hi, nz = _axis(prim.z0, prim.z1, 2)
            constructor = Geometries.equi_distant_box(
                x_range=(x_lo, x_hi),
                y_range=(y_lo, y_hi),
                z_range=(z_lo, z_hi),
                nx=nx,
                ny=ny,
                nz=nz,
                cell="hex" if tensor else "tetra",
            )
            grid_meta = {
                "shape": (nx + 1, ny + 1, nz + 1),
                "spacing": ((x_hi - x_lo) / nx, (y_hi - y_lo) / ny, (z_hi - z_lo) / nz),
                "origin": (x_lo, y_lo, z_lo),
            }

        # A named single-region plan (`rect.name("steel").attach(k=2.0)`) gets its region as a mesh tag
        # on the gmsh path, and the lattice builders know nothing about region names -- so without this
        # `d.k` resolved but `d.variable("steel")` did not. The plan is one primitive, so the region IS
        # the whole interior.
        region = getattr(shape, "_region_name", None)
        if region is not None:
            _inner = constructor

            def constructor(geo, _inner=_inner, _name=str(region)):  # noqa: F811
                mesh, dim, ds = _inner(geo)
                mesh.cell_sets[_name] = [np.asarray(b, dtype=np.int64) for b in mesh.cell_sets["interior"]]
                return mesh, dim, ds

        return constructor, grid_meta

    def _generate_mesh(self, geometry_func: Callable, algorithm: int):
        """Generate mesh using PyGmsh."""
        import pygmsh

        explicit_dim = None

        with pygmsh.geo.Geometry() as geo:
            mesh, explicit_dim, ds = geometry_func(geo)

            if not isinstance(mesh, meshio.Mesh):
                mesh = geo.generate_mesh(dim=explicit_dim, algorithm=algorithm, verbose=False)

        self.mesh = mesh
        self.dimension = explicit_dim
        self.ds = ds

    def _drop_orphan_nodes(self, mesh):
        """Compact out mesh points that support no finite element (mesh hygiene).

        gmsh routinely emits geometry-construction points -- circle/arc centres, spline control
        points -- as isolated 0-D ``vertex`` cells. Such a node supports no basis function: it sits
        in the DOF vector as a zero row/column, making the assembled operator singular. The correct
        fix is to remove it (not to pin it). So: keep only the >=1-D cell blocks (line/triangle/tetra
        -- the ones FEM assembles over), drop the 0-D ``vertex``/``point`` blocks, and compact any
        node then left unreferenced. Renumbering is transparent to boundary regions -- physical
        groups / ``cell_sets`` index *cells*, and a dropped node was in none of the kept ones.
        Returns the mesh unchanged when there is nothing to remove.
        """
        import meshio

        pts = np.asarray(getattr(mesh, "points", None))
        if pts is None or pts.size == 0 or not mesh.cells:
            return mesh
        # In a 1-D domain the boundary *is* its endpoint vertices, so the 0-D block is a real
        # boundary region (named left/right), not a construction-point orphan -- keep it. Only in
        # >=2-D are stray vertex/point cells the singular geometry-construction points to drop.
        drop_types = () if getattr(self, "dimension", None) == 1 else ("vertex", "point")
        keep_blocks = [i for i, cb in enumerate(mesh.cells) if cb.type not in drop_types]
        fem_cells = [mesh.cells[i] for i in keep_blocks]
        if not fem_cells:
            return mesh
        # "which nodes are referenced" is a membership question, so it is a scatter into a flag
        # array (O(n)), not a sort (O(n log n)): np.unique over 3.1M connectivity entries cost
        # 0.29 s of a 26.7 s build to answer it. flatnonzero already returns them in order.
        seen = np.zeros(len(pts), dtype=bool)
        for cb in fem_cells:
            seen[np.asarray(cb.data).reshape(-1)] = True
        keep = np.flatnonzero(seen).astype(np.int64)
        if keep.size == len(pts) and len(keep_blocks) == len(mesh.cells):
            return mesh  # every node supports an element and there are no 0-D blocks -> nothing to do
        remap = np.full(len(pts), -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep), dtype=np.int64)
        cells = [meshio.CellBlock(cb.type, remap[np.asarray(cb.data)]) for cb in fem_cells]
        # cell_sets / cell_data are lists parallel to mesh.cells -> keep only the surviving blocks
        sub = lambda d: {k: [v[i] for i in keep_blocks] for k, v in (d or {}).items()}
        point_data = {k: np.asarray(v)[keep] for k, v in (mesh.point_data or {}).items()}
        n_dropped = len(pts) - len(keep)
        if n_dropped:
            self.log.info(f"Dropped {n_dropped} orphan mesh node(s) (geometry-construction points, no element)")
        return meshio.Mesh(
            pts[keep],
            cells,
            cell_sets=sub(mesh.cell_sets),
            cell_data=sub(mesh.cell_data),
            field_data=mesh.field_data,
            point_data=point_data,
        )

    def _apply_mesh(self, mesh) -> None:
        """Run the post-mesh pipeline on a freshly attached mesh.

        Extracts boundary indices into ``_boundary_registry`` and (if
        ``compute_mesh_connectivity`` is set) builds ``mesh_connectivity``.
        Shared by the mesh-backed ``domain.__init__`` path and by
        ``PolygonDomain.build_mesh``.
        """
        if mesh is None:
            boundary_indices = np.asarray([], dtype=np.int64)
        else:
            if not getattr(self, "_keep_orphan_nodes", False):
                mesh = self._drop_orphan_nodes(mesh)
            _refuse_mixed_cells(mesh, int(self.dimension))
            # AFTER `_drop_orphan_nodes`, which renumbers nodes, and BEFORE the tag machinery, so a
            # derived tag goes through exactly the same path as one the file defined.
            mesh, _added = _derive_region_cell_sets(mesh, int(self.dimension))
            if _added:
                self.log.info(f"Derived mesh regions {_added} (not defined by the file)")
            self.mesh = mesh
            boundary_indices = self._extract_points_from_mesh(mesh)
            self._build_simplex_pools()

        if mesh is not None and self.compute_mesh_connectivity:
            self.mesh_connectivity, msg = self._preprocess_mesh_connectivity(mesh, self.dimension, boundary_indices)
            self.log.info(msg)

    def _reset_custom_tag_state(self) -> None:
        """Drop the **mesh-dependent** state of predicate-tagged regions so a re-tag on a freshly
        attached mesh re-derives them cleanly instead of layering onto stale entries from the old
        mesh. Called on remesh (see ``_domain_from_arrays``): the spatial predicates
        (``_tag_predicates``) are KEPT so callers re-materialize via ``tag(name, pred)``.

        Without this, re-tagging a surface region (Neumann / Robin / absorbing) on top of its stale
        boundary-region entry corrupts the assembled flux term -- e.g. an absorbing box's source
        collapses after the first remesh -- which silently breaks any field-parameter adaptive
        inverse-design loop.
        """
        names = set(getattr(self, "_tag_predicates", {}))
        if not names:
            return
        for attr in ("_boundary_regions", "tag_indices", "normals_by_tag", "_mesh_pool"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                for n in names:
                    store.pop(n, None)
        ctx = getattr(self, "context", None)
        if isinstance(ctx, dict):
            for n in names:
                ctx.pop(n, None)
                ctx.pop(f"n_{n}", None)
        amt = getattr(self, "avaiable_mesh_tags", None)
        if isinstance(amt, list):
            self.avaiable_mesh_tags = [t for t in amt if t not in names]

    def refine(self, vertex_size, **mmg_options):
        """Locally remesh **in place** to a per-vertex target edge size (metric-based).

        ``vertex_size`` is a ``(n_vertices,)`` array giving the desired edge length at
        each current mesh vertex, or an ``(n_vertices, 3)`` **anisotropic** metric tensor
        (e.g. from ``hessian_metric``) that also orients the refinement; the mesh is adapted
        (via Mmg) so it is equidistributed, refining where it shrinks.  Geometric corners
        are preserved.
        Spatial tag predicates (``domain.tag(...)``) survive, so ``jno.fem`` conditions
        bound to a tag resolve geometrically on the new nodes.  Returns ``self``.

        This is the mesh-modification primitive behind ``FEM.solve(adapt=...)``; call it
        directly to apply a hand-built size field.  Requires the optional ``mmgpy``
        dependency (``pip install mmgpy``).
        """
        from ..utils.solver.fem_adapt import remesh_with_mmg

        remesh_with_mmg(self, vertex_size, copy=False, **mmg_options)
        return self

    def _remesh_periodic(self, pairs) -> bool:
        """Re-mesh IN PLACE from the stored ``jno.Shape`` geometry, making the named opposite-face ``pairs``
        (``[(main, secondary), ...]``) conforming via gmsh ``setPeriodic`` — so a Nédélec (edge) field's
        per-edge DOFs line up one-to-one across the periodic faces. Inferred from the constraint list (the
        periodic ties), never requested explicitly. Idempotent: returns ``False`` (a no-op) when the domain
        is not ``Shape``-backed (a user-supplied conforming mesh is used as-is) or the pairs are already
        applied; ``True`` after a re-mesh."""
        from ..geometry.shape import Shape

        shape = getattr(self, "_constructor_source", None)
        if not isinstance(shape, Shape):
            return False  # no geometry to re-mesh; rely on the mesh as given (fails loudly if non-conforming)
        want = frozenset(frozenset(p) for p in pairs)
        if want <= getattr(self, "_periodic_meshed", frozenset()):
            return False  # already conforming for these face pairs
        from ..geometry.emit import build as _emit_build

        mesh, _dim, ds = _emit_build(shape, periodic=list(pairs))
        self.mesh, self.ds = mesh, ds
        if hasattr(self, "_reset_custom_tag_state"):
            self._reset_custom_tag_state()
        self._apply_mesh(mesh)
        for name, pred in list(getattr(self, "_tag_predicates", {}).items()):  # re-materialize predicate tags
            self.tag(name, pred)
        if getattr(self, "_is_time_dependent", False) and self.time is not None:
            self._add_time_dimension(*self.time)  # re-broadcast the fresh (N, D) tags across time (idempotent)
        self._periodic_meshed = getattr(self, "_periodic_meshed", frozenset()) | want
        self.log.info(f"Re-meshed periodic (conforming) for face pairs {sorted(tuple(sorted(p)) for p in pairs)}")
        return True

    def tag_facets(self, name: str):
        """A tag's boundary facets as ``(n_facets, n_nodes_per_facet)`` node ids, or ``None``.

        The facets of a tag live in one of three stores according to how many nodes they have:
        2 (an edge), 3 (a triangular face), or 4 (a hexahedron's quadrilateral face). Which one
        depends on the *cell* the mesh is built from, not on the dimension — a 3-D mesh has
        triangular faces if it is tetrahedral and quadrilateral ones if it is hexahedral — so
        callers ask here instead of picking a store by dimension, which is what silently returned
        nothing for a hexahedral mesh.
        """
        for store in getattr(self, "_tag_facet_stores", ("_tag_edges", "_tag_triangles", "_tag_quads")):
            facets = (getattr(self, store, {}) or {}).get(name)
            if facets is not None and len(facets):
                return np.asarray(facets)
        return None

    def _build_simplex_pools(self) -> None:
        """Populate ``self._simplex_pools`` from ``_tag_edges`` / ``_tag_triangles``.

        (Tags whose facets are quadrilaterals — a hexahedral mesh's boundary — get no pool: the
        pools are barycentric, and sampling a quad needs a bilinear map. They are refused where
        they are used rather than silently sampled as if they were triangles.)

        See also :meth:`tag_facets`, which returns a tag's facets whatever their node count.

        Runs after ``_extract_points_from_mesh`` so the per-tag cell-membership
        tables are filled in.  For each mesh tag we materialise a
        ``SimplexPool``:

        * ``_tag_triangles[tag]`` (dim-2 interior) → V=3 barycentric pool.
        * ``_tag_edges[tag]`` (1-D interior or 2-D boundary) → V=2 lerp pool;
          if the tag has a per-point normal in ``normals_by_tag`` we average
          the two endpoint normals to get a per-segment normal.  Falls back to
          a geometric normal (rotate the segment vector 90° and pick the
          half-space pointing away from the mesh centroid) when no per-point
          normals are available.

        Existing pools for tags not touched by the current mesh (e.g. the
        Shapely-side pools that ``PolygonDomain._register_*_tag`` populated
        before a later ``build_mesh()``) are left intact — only tags present
        in ``_tag_triangles`` / ``_tag_edges`` are overwritten.
        """
        points = getattr(self, "points", None)
        if points is None or len(points) == 0:
            return
        points_d = points[:, : self.dimension]
        mesh_centroid = points_d.mean(axis=0) if len(points_d) else None

        # Track which tags get a triangle pool in this build so the segment
        # loop knows to skip them (a tag with both triangles + boundary edges
        # is sampled as an interior — boundary edges are auxiliary).
        triangle_tags_this_build: set = set()

        # 2-D interior tags (triangle pool, no normals).
        for tag, tri_indices in self._tag_triangles.items():
            tri_coords = points_d[tri_indices]
            if tri_coords.ndim == 3 and tri_coords.shape[1] == 3 and tri_coords.shape[2] == 2:
                self._simplex_pools[tag] = SimplexPool.from_triangles(tri_coords)
                triangle_tags_this_build.add(tag)

        # 1-D interior + 2-D boundary tags (segment pool, with normals on dim=2).
        for tag, edge_indices in self._tag_edges.items():
            if tag in triangle_tags_this_build:
                continue  # this tag is being sampled as an interior
            seg_coords = points_d[edge_indices]
            if seg_coords.ndim != 3 or seg_coords.shape[1] != 2:
                continue

            normals = None
            if self.dimension == 2 and mesh_centroid is not None:
                vectors = seg_coords[:, 1, :] - seg_coords[:, 0, :]
                lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
                lengths = np.where(lengths > 0, lengths, 1.0)
                tangents = vectors / lengths
                cand = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
                midpoints = 0.5 * (seg_coords[:, 0, :] + seg_coords[:, 1, :])
                outward = midpoints - mesh_centroid
                flip = (cand * outward).sum(axis=-1) < 0.0
                cand[flip] *= -1.0
                normals = cand.astype(np.float32)

            self._simplex_pools[tag] = SimplexPool.from_segments(seg_coords, normals=normals)

    @staticmethod
    def _cells_of(block_data, indices, offset):
        """Rows of ``block_data`` picked out by global cell ``indices`` less ``offset``, in order.

        The Python loop this replaces ran once per CELL, and the interior tag names every cell in the
        mesh: a 1.03M-triangle 2-D mesh paid 1.03M ``set.update`` calls and built 1.03M tuples only to
        hand them straight back to ``np.array`` -- 1.05 s of a 26.7 s build. Out-of-range indices are
        dropped rather than raising, exactly as the loop's bounds test did.
        """
        data = np.asarray(block_data)
        local = np.asarray(indices, dtype=np.int64).reshape(-1) - int(offset)
        return data[local[(local >= 0) & (local < len(data))]]

    def _extract_points_from_mesh(self, mesh):
        """Extract points and normals from mesh and organize by tag."""
        points = mesh.points[:, : self.dimension]
        self.points = points
        self._mesh_pool = {}
        self._boundary_registry = {}
        self._interface_registry: Dict[str, Dict[str, Any]] = {}
        self.tag_indices = {}
        self._tag_edges = {}
        self._tag_triangles = {}
        self._tag_quads = {}
        self._boundary_regions = {}

        if self.dimension > 1:
            boundary_normals, boundary_indices = self.get_boundary_normals(mesh)
            boundary_normals = boundary_normals[:, : self.dimension]
        else:
            left_boundary = np.where(points[:, 0] == np.min(points[:, 0]))[0]
            right_boundary = np.where(points[:, 0] == np.max(points[:, 0]))[0]

            boundary_indices = np.stack([left_boundary, right_boundary]).flatten()

            boundary_normals = np.array([[-1], [1]])

        # node id -> row in ``boundary_normals``, as a lookup ARRAY rather than a dict: it is read
        # once per point of every tag, and the interior tag holds every node in the mesh. A later
        # duplicate wins, exactly as it did when this was a dict comprehension.
        normal_pos_of = np.full(len(points), -1, dtype=np.int64)
        if len(boundary_indices):
            normal_pos_of[np.asarray(boundary_indices, dtype=np.int64)] = np.arange(len(boundary_indices))

        if hasattr(mesh, "cell_sets") and mesh.cell_sets:
            # Compute cumulative offsets: cell_sets may use global cell indices
            block_offsets = {}
            cumulative = 0
            for b_idx, cell_block in enumerate(mesh.cells):
                block_offsets[(b_idx, cell_block.type)] = cumulative
                cumulative += len(cell_block.data)

            # Also build a per-type offset map for easier lookup
            type_to_blocks = {}
            for b_idx, cell_block in enumerate(mesh.cells):
                if cell_block.type not in type_to_blocks:
                    type_to_blocks[cell_block.type] = []
                type_to_blocks[cell_block.type].append((b_idx, cell_block))

            for name, cell_data in mesh.cell_sets.items():
                if name.startswith("gmsh:"):
                    continue

                self.avaiable_mesh_tags.append(name)

                # lists of ARRAYS (one per contributing block), concatenated below
                tag_point_blocks: List[np.ndarray] = []
                tag_edge_blocks: List[np.ndarray] = []
                tag_tri_blocks: List[np.ndarray] = []
                tag_quad_blocks: List[np.ndarray] = []  # a hexahedron's boundary face
                # A tag backed by volume cells (block 0 = the spatial-fill element for this
                # dimension) is a region/interior, never a boundary — so it must not pick up PCA
                # "boundary" normals (that would mislabel an interior sub-region as a boundary tag).
                # Boundary tags live in the facet block (block 1); point clouds (block 0 = vertex)
                # are not volume-filling, so they still get PCA normals.
                # A dimension has more than one space-filling cell: 2-D is a triangle OR a quad,
                # 3-D a tet OR a hexahedron. Testing against a single simplex name made every cell
                # of a tensor-product mesh look like a facet, so its volume region was never
                # recognised as one.
                _vol_elems = {1: ("line",), 2: ("triangle", "quad"), 3: ("tetra", "hexahedron")}.get(self.dimension, ())
                block0_is_volume = bool(mesh.cells) and _base_cell_type(mesh.cells[0].type) in _vol_elems
                has_volume_cells = False

                if isinstance(cell_data, dict):
                    for cell_type, indices in cell_data.items():
                        if len(indices) == 0:
                            continue

                        # Handle vertex (point) cells specially
                        if cell_type == "vertex":
                            for b_idx, cell_block in enumerate(mesh.cells):
                                if cell_block.type == "vertex":
                                    sel = self._cells_of(cell_block.data, indices, block_offsets.get((b_idx, "vertex"), 0))
                                    if len(sel):  # vertex data contains the point index
                                        tag_point_blocks.append(sel.reshape(len(sel), -1)[:, 0])
                        else:
                            if block0_is_volume and _base_cell_type(cell_type) in _vol_elems and len(indices) > 0:
                                has_volume_cells = True
                            for b_idx, cell_block in enumerate(mesh.cells):
                                if cell_block.type == cell_type:
                                    sel = self._cells_of(cell_block.data, indices, block_offsets.get((b_idx, cell_type), 0))
                                    if len(sel):
                                        tag_point_blocks.append(sel.ravel())
                                        # A curved facet ("line3"/"triangle6") is still an edge/face;
                                        # downstream these arrays are P1 (2- and 3-node), so truncate
                                        # to the vertex columns rather than change every consumer.
                                        _bt = _base_cell_type(cell_block.type)
                                        if _bt == "line":
                                            tag_edge_blocks.append(sel[:, :2])
                                        elif _bt == "triangle":
                                            tag_tri_blocks.append(sel[:, :3])
                                        elif _bt == "quad":
                                            tag_quad_blocks.append(sel[:, :4])
                else:
                    # Handle list-style cell_data. meshio cell-set indices
                    # can be either block-local (manually written `.inp`
                    # files) or global gmsh cell IDs. Per-cell-set rule: if
                    # ``max(indices) >= block_len`` the indices must be
                    # global (a local index can't exceed block_len-1), so
                    # subtract ``min(indices)`` to convert to local;
                    # otherwise treat as local. This is robust to gmsh's
                    # internal numbering not aligning with meshio's
                    # cumulative block ordering.
                    for block_idx, indices in enumerate(cell_data):
                        if indices is None or len(indices) == 0:
                            continue
                        if block_idx == 0 and block0_is_volume:
                            has_volume_cells = True
                        if block_idx < len(mesh.cells):
                            cell_block = mesh.cells[block_idx]
                            block_len = len(cell_block.data)
                            idx_array = np.asarray(indices)
                            if idx_array.max() >= block_len:
                                sub = int(idx_array.min())
                            else:
                                sub = 0

                            sel = self._cells_of(cell_block.data, idx_array, sub)
                            if len(sel):
                                if cell_block.type == "vertex":
                                    tag_point_blocks.append(sel.reshape(len(sel), -1)[:, 0])
                                else:
                                    tag_point_blocks.append(sel.ravel())
                                    _bt = _base_cell_type(cell_block.type)
                                    if _bt == "line":
                                        tag_edge_blocks.append(sel[:, :2])
                                    elif _bt == "triangle":
                                        tag_tri_blocks.append(sel[:, :3])
                                    elif _bt == "quad":
                                        tag_quad_blocks.append(sel[:, :4])

                tag_edges = np.concatenate(tag_edge_blocks, axis=0) if tag_edge_blocks else np.zeros((0, 2), int)
                tag_tris = np.concatenate(tag_tri_blocks, axis=0) if tag_tri_blocks else np.zeros((0, 3), int)
                tag_quads = np.concatenate(tag_quad_blocks, axis=0) if tag_quad_blocks else np.zeros((0, 4), int)
                # np.unique both de-duplicates and sorts, which is what `sorted(set(...))` did
                tag_points = np.unique(np.concatenate(tag_point_blocks)) if tag_point_blocks else np.zeros(0, int)

                if len(tag_quads):
                    self._tag_quads[name] = np.asarray(tag_quads, dtype=int)
                if len(tag_tris):
                    self._tag_triangles[name] = np.asarray(tag_tris, dtype=int)
                if len(tag_edges):
                    self._tag_edges[name] = np.asarray(tag_edges, dtype=int)

                if len(tag_points):
                    if len(tag_edges):
                        self._boundary_loop_tags.add(name)
                        # the chain walk is a Python graph traversal: hand it plain ints, as before
                        indices_list = self._chain_edges_to_loop(tag_edges.tolist())
                        # The walk runs over P1 edges, so it returns only their vertices. A CURVED
                        # facet also carries midside nodes, which are genuine boundary DOFs -- without
                        # them a Dirichlet condition would pin the corners of each facet and leave its
                        # interior free. Append rather than merge, so the ordered loop stays a prefix
                        # for the consumers that rely on it. Empty for a straight mesh, where the chain
                        # already covers every node, so this path is unchanged there.
                        _extra = np.setdiff1d(tag_points, np.asarray(indices_list, dtype=int))
                        if _extra.size:
                            indices_list = np.concatenate([np.asarray(indices_list, dtype=int), _extra])
                    else:
                        indices_list = np.asarray(tag_points, dtype=int)

                    indices_list = np.asarray(indices_list, dtype=int)
                    self.tag_indices[name] = indices_list
                    self._mesh_pool[name] = points[indices_list]

                    # Only attach per-point normals when every point in this tag
                    # has one — otherwise the tag mixes boundary and interior
                    # points (e.g. the gmsh "interior" surface tag includes the
                    # boundary nodes too), and storing a partial normal array
                    # creates a shape mismatch against _mesh_pool[name].
                    found = normal_pos_of[indices_list]
                    normal_positions = found[found >= 0]
                    if len(normal_positions) == len(indices_list) and len(indices_list) > 0:
                        self.normals_by_tag[name] = boundary_normals[normal_positions]
                    elif len(normal_positions) == 0 and not has_volume_cells:
                        tag_pt_coords = points[indices_list, : self.dimension]
                        if len(tag_pt_coords) > 1:
                            tag_normals, _ = self._compute_normals_pca(
                                points,
                                indices_list,
                                self.dimension,
                                k=min(8, len(tag_pt_coords)),
                                mesh=mesh,
                            )
                            self.normals_by_tag[name] = tag_normals[:, : self.dimension]

                    # --- Generic boundary registry entry ---
                    is_boundary_tag = False
                    entity_kind = None

                    if self.dimension == 2 and len(tag_edges) > 0:
                        is_boundary_tag = True
                        entity_kind = "line"
                    elif self.dimension == 3 and len(tag_tris) > 0:
                        is_boundary_tag = True
                        entity_kind = "triangle"
                    elif self.dimension == 3 and len(tag_quads) > 0:
                        # A hexahedral mesh's boundary facet. Without this the tag fell through to the
                        # normals fallback below and registered as `boundary_points`, so its region
                        # carried no entities and `contains` degraded to a point-distance test.
                        is_boundary_tag = True
                        entity_kind = "quad"
                    elif name in self.normals_by_tag:
                        # fallback: still treat as a boundary-like tag if normals exist
                        is_boundary_tag = True
                        entity_kind = "boundary_points"

                    if is_boundary_tag:

                        def _coords(facets):
                            return points[np.asarray(facets, dtype=int)][:, :, : self.dimension] if len(facets) else None

                        # pred = self._boundary_predicates.get(name, None)
                        tol = self._estimate_boundary_tol(points[indices_list][:, : self.dimension])

                        self._boundary_regions[name] = BoundaryRegion(
                            tag=name,
                            dim=self.dimension,
                            points=points[indices_list][:, : self.dimension],
                            edges=_coords(tag_edges),
                            triangles=_coords(tag_tris),
                            quads=_coords(tag_quads),
                            tol=tol,
                        )

                        # An internal material interface (auto-named "a|b" by Shape.regions) is a facet
                        # region you can impose a coupling/flux condition on, but it is NOT the outer
                        # boundary -- keep it out of boundary_tags() so `dirichlet(boundary_tags())`
                        # never pins it. It stays queryable via d.variable("a|b") / interface_tags().
                        registry = self._interface_registry if "|" in name else self._boundary_registry
                        registry[name] = {
                            "tag": name,
                            "entity_kind": entity_kind,
                            "point_indices": indices_list,
                            "points": points[indices_list],
                        }
        # self._register_default_box_boundary_predicates()
        return boundary_indices

    def _add_time_dimension(self, t_start: float, t_end: float, n_time: int = 100):
        """Add time dimension to all point sets.

        After this call the mesh pool stores arrays with shape
        ``(T, N, D_spatial)`` — spatial coordinates tiled across T time
        steps.  A separate ``_time_points`` array of shape ``(T,)`` holds
        the time values.  The ``"initial"`` tag is a special case with
        ``T=1`` at ``t=t_start``.

        ``self.dimension`` is **not** incremented because time is handled
        as a separate axis, not as an extra spatial column.
        """
        self._time_points = np.linspace(t_start, t_end, n_time)  # (T,)
        self._n_time = n_time
        new_mesh_pool = {}
        for tag, points in self._mesh_pool.items():
            # The "initial" tag is always (re)derived from "interior" below, so
            # skip any pre-existing one (idempotent across re-meshing).
            if tag == "initial":
                continue

            pts = np.asarray(points)
            # Already time-broadcast (e.g. a second build_mesh on the same
            # time-dependent domain) — keep the (T, N, D) pool as is.
            if pts.ndim >= 3:
                new_mesh_pool[tag] = pts
                continue

            # points has shape (N, D_spatial)
            if tag == "interior":
                # Initial tag: spatial points at t=0 → (1, N, D_spatial)
                new_mesh_pool["initial"] = pts[np.newaxis, :, :]  # T=1

            # Tile spatial points across T time steps → (T, N, D_spatial)
            new_mesh_pool[tag] = np.broadcast_to(
                pts[np.newaxis, :, :],  # (1, N, D_spatial)
                (n_time, *pts.shape),  # (T, N, D_spatial)
            ).copy()  # copy so it's contiguous

        self._mesh_pool = new_mesh_pool
        # NOTE: self.dimension stays as D_spatial — time is a separate axis

        # Store time array in context so Variable("__time__") can be created.
        # Shape: (T, 1) — will be broadcast to (B, T, 1) during sample().
        self.context["__time__"] = self._time_points[:, np.newaxis]  # (T, 1)

        return None

    # The dominant call ``x, y, _ = dom.variable("interior")`` returns a tuple of
    # coordinate ``Variable``s; typing it (rather than ``Any``) is what makes the
    # whole traced-DSL chain — ``x.d(x)``, ``u.scalar``, … — discoverable in an
    # IDE. The tensor-tag / flagged forms stay ``Any`` (unchanged permissiveness).
    @overload
    def variable(
        self,
        tag: str,
        sample: Tuple[Optional[int], Optional[Callable]] = (None, None),
        resampling_strategy=None,
        normals: bool = False,
        reverse_normals: bool = False,
        view_factor: bool = False,
        point_data: bool = False,
        split: bool = False,
        return_indices: Literal[False] = False,
        time_value: Optional[float] = None,
        where=None,
        region=None,
    ) -> "tuple[Variable, ...]": ...

    @overload
    def variable(
        self,
        tag: str,
        sample: Any = (None, None),
        resampling_strategy=None,
        normals: bool = False,
        reverse_normals: bool = False,
        view_factor: bool = False,
        point_data: bool = False,
        split: bool = False,
        return_indices: bool = False,
        time_value: Optional[float] = None,
        where=None,
        region=None,
    ) -> Any: ...
    @property
    def cell_size(self):
        """Element size ``h`` as a symbol usable directly in a weak form.

        Isotropic per-cell size ``|det J|^(1/dim)`` (an edge-length scale), resolved at each
        quadrature point during FEM assembly. This is the handle for mesh-dependent stabilization
        — e.g. SUPG/GLS for advection-dominated transport::

            h    = dom.cell_size
            tau  = h / (2 * beta.norm())
            supg = tau * (beta[0]*ui.x + beta[1]*ui.y) * (beta[0]*vi.x + beta[1]*vi.y)

        Resolved on the native 2D/3D assembler's volume terms. A ``cell_size`` coefficient adds no
        trial/test gradient, so a stabilized term still classifies (``term_kind``) by its u/v
        gradient structure, and ``h`` is geometry (constant w.r.t. the unknown) so differentiable
        assembly is unaffected. (Not meaningful for PINN / boundary-facet terms.)
        """
        if "cell_size" not in self.context:
            # Placeholder so the Variable constructs; the real per-cell h is packed at assembly time
            # (jno/utils/solver/fem_native.py) and overrides this everywhere it is actually used.
            self.context["cell_size"] = np.ones((1, 1), dtype=default_np_float_dtype())
        return Variable(tag="cell_size", dim=[0, 1], domain=self, axis="spatial")

    # ------------------------------------------------------------------
    # Per-cell mesh geometry as trace nodes (differentiable in the mesh)
    # ------------------------------------------------------------------
    #: Linear volume-cell blocks per dimension, simplex first. The order is the lookup order, so a
    #: mesh carrying both (which :meth:`Shape.cell_choices` refuses to build) resolves to the simplex.
    _TOPO_BLOCKS = {1: ("line",), 2: ("triangle", "quad"), 3: ("tetra", "hexahedron")}

    def _cells_topo(self):
        """``(cells, kind)`` — the mesh's linear volume cells, simplex **or** tensor-product.

        ``kind`` is the meshio block name, so a caller can branch on cell shape rather than infer it
        from the column count. Sibling of :meth:`_cells_p1` rather than a replacement: that one
        promises a simplex and a great deal of code reads it that way, while everything *geometric*
        -- volume, angles, aspect, facets, patches -- has a tensor-product counterpart and should
        take this instead.
        """
        mesh = self.built_mesh
        cd = getattr(mesh, "cells_dict", {})
        dim = int(self.dimension)
        for key in self._TOPO_BLOCKS.get(dim, ()):
            cells = cd.get(key)
            if cells is not None:
                return np.asarray(cells, dtype=np.int64), key
        want = " or ".join(repr(k) for k in self._TOPO_BLOCKS.get(dim, ()))
        raise ValueError(f"domain: no {want} cells on this {dim}-D mesh — cell geometry is unavailable.")

    def _cells_p1(self):
        """``(n_cells, dim+1)`` vertex ids of the P1 simplices."""
        mesh = self.built_mesh
        key = "triangle" if self.dimension == 2 else ("tetra" if self.dimension == 3 else "line")
        cells = getattr(mesh, "cells_dict", {}).get(key)
        if cells is None:
            raise ValueError(f"domain: no {key!r} cells on this mesh — cell geometry is unavailable.")
        return np.asarray(cells, dtype=np.int64)

    def _moving_points(self):
        """``(arg_exprs, rebuild)`` for the mesh coordinates *as the optimiser currently has them*.

        ``rebuild(*vals)`` returns the ``(n_points, dim)`` array with every ``.trainable()``
        coordinate scattered in. Passing ``arg_exprs`` as the traced arguments of a
        :func:`jno.fn` node is what makes a geometric quantity differentiable w.r.t. the mesh:
        the node depends on the coordinate parameters, so ``∂g/∂X`` flows without anyone writing
        a shape derivative. With no trainable coordinates the mesh is fixed and ``arg_exprs`` is
        empty, which is still valid — the quantity is then simply a constant.
        """
        dim = int(self.dimension)
        pts0 = jnp.asarray(np.asarray(self.mesh.points)[:, :dim])
        specs = list(getattr(self, "_trainable_coords", None) or [])
        meta = [(jnp.asarray(s["ids"], dtype=jnp.int32), int(s["axis"])) for s in specs]

        def rebuild(*vals):
            pts = pts0
            for (ids, axis), v in zip(meta, vals):
                pts = pts.at[ids, axis].set(jnp.asarray(v).reshape(-1))
            return pts

        return [s["expr"] for s in specs], rebuild

    def cell_volume(self):
        """Per-cell volume (area in 2-D) as a ``(n_cells,)`` node, differentiable in the mesh.

        ``|det J| / d!`` from the cell's own vertices, so it tracks ``.trainable()`` coordinates.
        The handle for an element-size constraint in shape or topology optimisation::

            g = (d.cell_volume() / ((2 - rho) * v_max)).pnorm(50)
            jno.core([compliance, jno.le(g, 1.0)])
        """
        import jno as _jno

        dim = int(self.dimension)
        cells_np, kind = self._cells_topo()
        cells = jnp.asarray(cells_np, dtype=jnp.int32)
        args, rebuild = self._moving_points()
        fact = {1: 1.0, 2: 2.0, 3: 6.0}[dim]  # d! — the simplex volume factor

        def _vol(*vals):
            v = rebuild(*vals)[cells]  # (n_cells, dim+1, dim)
            jac = jnp.stack([v[:, i + 1] - v[:, 0] for i in range(dim)], axis=-1)
            return jnp.abs(jnp.linalg.det(jac)) / fact

        def _vol_quad(*vals):
            """The shoelace area, which is exact for a quadrilateral of ANY shape.

            ``|det J| / 2`` is the simplex formula and would read a quadrilateral as the parallelogram
            spanned by two of its edges -- right only when it happens to be one. The shoelace sum over
            the four corners in mesh order is exact for any simple polygon, convex or not, and is
            linear in each coordinate, so it differentiates cleanly w.r.t. the moving nodes.
            """
            v = rebuild(*vals)[cells]  # (n_cells, 4, 2)
            x, y = v[..., 0], v[..., 1]
            xn, yn = jnp.roll(x, -1, axis=-1), jnp.roll(y, -1, axis=-1)
            return jnp.abs(jnp.sum(x * yn - xn * y, axis=-1)) / 2.0

        if kind == "hexahedron":
            raise NotImplementedError(
                "domain.cell_volume(): hexahedral cells are not supported yet — the quadrilateral "
                "(2-D) and simplex paths are. A hexahedron's volume is not |det J| / 6 and its faces "
                "need not be planar, so it needs its own formula rather than the simplex one."
            )
        return _jno.fn(_vol_quad if kind == "quad" else _vol, args, name="cell_volume")

    def cell_angles(self, eps: float = 1e-12):
        """Per-cell angles in radians, differentiable in the mesh. **2-D and 3-D.**

        The dihedral angle is the tetrahedron's counterpart of a triangle's interior angle, so one
        name covers both and a mesh-quality constraint is the *same expression* in either dimension:

        * **triangles** → ``(n_cells, 3)``, the interior angle at each vertex, from the inverse
          cosine of its two incident edge vectors;
        * **quadrilaterals** → ``(n_cells, 4)``, the interior angle at each corner, in ``(0, 2π)``
          so that a **reflex** corner reads above ``π`` rather than being folded back below it;
        * **tetrahedra** → ``(n_cells, 6)``, the dihedral angle across each edge, from the inward
          normals of the two faces meeting there.

        Both are bounded below by the same form, since a dihedral is also under ``π``::

            g = ((2 * jno.np.pi - d.cell_angles()) / (2 * jno.np.pi - theta_min)).pnorm(50)
            jno.core([compliance, jno.le(g, 1.0)])

        **The one thing that does differ, and it is physics rather than API.** On a **triangle** a
        lower bound induces an upper bound, because three angles summing to ``π`` leave no room for a
        large one once the small ones are held up. That argument is special to the triangle:

        * a **tet's** six dihedrals sum to a *variable* quantity strictly inside ``(2π, 3π)`` —
          measured 2.207π to 2.467π on a unit box, against 2.35π for the regular tet — so a **cap**,
          two faces folded flat together, satisfies a minimum-angle bound while being degenerate;
        * a **quadrilateral's** four angles sum to exactly ``2π``, which is enough room for three
          corners at 45° and a fourth at 225°. That cell is non-convex and still has positive area,
          so nothing else in the constraint set objects to it either.

        In both cases bound the maximum explicitly as well::

            jno.le((d.cell_angles() / theta_max).pnorm(50, normalize=True), 1.0)   # tets and quads

        **What this does not catch.** A *needle* — a small base with a distant apex — keeps every
        dihedral moderate while being badly conditioned; :meth:`cell_aspect` is what sees it. The two
        are complements: a *sliver* (four near-coplanar vertices) holds its face angles above 20°
        while its dihedrals collapse to 0.03° and 179.9°, and a needle does the converse. This
        matters because of Babuška & Aziz (*SIAM J. Numer. Anal.* **13** (1976) 214–226): a needle
        still satisfies the maximum-angle condition, so it costs conditioning, while a sliver and a
        cap violate it and destroy interpolation accuracy outright.

        ``eps`` guards the denominator; it does not otherwise shift the angles. Reference: Jung, Yun
        & Kim, *Computers & Structures* **331** (2026) 108403, Sec. 2.3.3 — eq. (21)/(24) is the
        triangle case, and the tetrahedral one is the extension their Sec. 2.3.2 leaves open.
        """
        import itertools

        import jno as _jno

        dim = int(self.dimension)
        if dim not in (2, 3):
            raise NotImplementedError(f"domain.cell_angles(): 2-D or 3-D; this domain is {dim}-D.")
        cells_np, kind = self._cells_topo()
        cells = jnp.asarray(cells_np, dtype=jnp.int32)
        args, rebuild = self._moving_points()

        def _ang_quad(*vals):
            """The four interior angles, in ``(0, 2π)`` -- a **reflex** corner reads above ``π``.

            The ``arccos`` of two incident edges cannot express a reflex angle: its range is
            ``[0, π]``, so a non-convex ("dart") quadrilateral reports its 225° corner as 135° and a
            minimum-angle bound sees nothing wrong. A quadrilateral CAN go non-convex while keeping a
            positive area, which a triangle cannot, so this is a real failure mode of a deformable
            quad mesh rather than a nicety. ``atan2`` against the signed corner cross-product gives
            the true interior angle, oriented by the cell's own signed area so mesh winding does not
            matter. Checked by the four angles summing to ``2π``, reflex corners included.
            """
            v = rebuild(*vals)[cells]  # (n_cells, 4, 2)
            x, y = v[..., 0], v[..., 1]
            sgn = jnp.sign(jnp.sum(x * jnp.roll(y, -1, axis=-1) - jnp.roll(x, -1, axis=-1) * y, axis=-1))
            sgn = jnp.where(sgn == 0.0, 1.0, sgn)[:, None]
            a = jnp.roll(v, 1, axis=1) - v  # towards the previous corner
            b = jnp.roll(v, -1, axis=1) - v  # towards the next corner
            cross = b[..., 0] * a[..., 1] - b[..., 1] * a[..., 0]
            th = jnp.arctan2(sgn * cross, jnp.sum(a * b, axis=-1))
            return jnp.where(th < 0.0, th + 2.0 * np.pi, th)  # (n_cells, 4)

        def _ang_2d(*vals):
            v = rebuild(*vals)[cells]  # (n_cells, 3, 2)
            out = []
            for i in range(3):
                a = v[:, (i + 1) % 3] - v[:, i]
                b = v[:, (i + 2) % 3] - v[:, i]
                cos = jnp.sum(a * b, axis=-1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + eps)
                out.append(jnp.arccos(jnp.clip(cos, -1.0, 1.0)))
            return jnp.stack(out, axis=-1)  # (n_cells, 3)

        # Edge (i, j) is shared by the two faces that omit the other two vertices, which is what
        # pairs an edge with the normals whose angle is its dihedral.
        edges = list(itertools.combinations(range(4), 2))

        def _inward_normal(v, k):
            """Unit normal of the face opposite vertex ``k``, oriented towards the cell interior."""
            keep = [i for i in range(4) if i != k]
            n = jnp.cross(v[:, keep[1]] - v[:, keep[0]], v[:, keep[2]] - v[:, keep[0]])
            n = n / (jnp.linalg.norm(n, axis=-1, keepdims=True) + eps)
            s = jnp.sign(jnp.sum(n * (v[:, k] - v[:, keep[0]]), axis=-1, keepdims=True))
            return n * jnp.where(s == 0.0, 1.0, s)

        def _ang_3d(*vals):
            v = rebuild(*vals)[cells]  # (n_cells, 4, 3)
            nrm = [_inward_normal(v, k) for k in range(4)]
            out = []
            for i, j in edges:
                a, b = [k for k in range(4) if k not in (i, j)]
                cos = jnp.sum(nrm[a] * nrm[b], axis=-1)
                # Both normals point inward, so the dihedral is π minus the angle between them.
                out.append(np.pi - jnp.arccos(jnp.clip(cos, -1.0, 1.0)))
            return jnp.stack(out, axis=-1)  # (n_cells, 6)

        if kind == "hexahedron":
            raise NotImplementedError(
                "domain.cell_angles(): hexahedral cells are not supported yet — quadrilaterals (2-D) "
                "and simplices are. A hexahedron's faces need not be planar, so 'the' dihedral across "
                "an edge is not well defined without first deciding how to split them."
            )
        return _jno.fn(_ang_quad if kind == "quad" else (_ang_2d if dim == 2 else _ang_3d), args, name="cell_angles")

    def cell_aspect(self, eps: float = 1e-30):
        """Per-cell **aspect ratio** as a ``(n_cells,)`` node, differentiable in the mesh. 2-D and 3-D.

        The longest edge over the inradius, scaled so a **regular** simplex is exactly ``1.0`` and a
        stretched one is larger (a sliver diverges). This is the quantity a mesh-quality condition is
        written on, and unlike :meth:`cell_angles` it is dimension-generic, so the same expression
        works on triangles and tetrahedra::

            fem.solve(adapt=jno.solve.relocate(objective=...)
                                     .remesh(criterion=jno.le(d.cell_aspect(), 6.0)))

        Differentiable in the vertex positions like its neighbours here (checked against central
        differences at 2.7e-10), which is what a constrained optimiser needs. It is **not** usable as
        ``relocate(objective=...)`` yet: that path assembles a weak term, and a per-cell node is not
        one -- it fails on the runtime coordinate parameters rather than being refused, so the shape
        of the fix is a per-cell objective path, not a message.

        Contrast :meth:`cell_size`, which is ``|det J|^(1/dim)`` -- an isotropic SIZE. It cannot see
        stretch at all: a sliver and a regular element of the same area share it.

        ``eps`` guards the inradius denominator on a collapsed element; it does not otherwise shift
        the ratio. Reference: Shewchuk, *What Is a Good Linear Finite Element? Interpolation,
        Conditioning, Anisotropy, and Quality Measures* (2002), §2 -- the length/inradius family.
        """
        import itertools

        import jno as _jno

        dim = int(self.dimension)
        if dim not in (2, 3):
            raise NotImplementedError(f"domain.cell_aspect(): 2-D or 3-D; this domain is {dim}-D.")
        cells_np, kind = self._cells_topo()
        if kind == "hexahedron":
            raise NotImplementedError(
                "domain.cell_aspect(): hexahedral cells are not supported yet — quadrilaterals (2-D) and simplices are."
            )
        if kind == "quad":
            # The SAME measure, longest edge over the inradius: r = 2A/P is the inradius of any
            # tangential polygon and reads as an inscribed-circle scale for the rest, exactly as
            # `dim * vol / surf` already does for a simplex. Only the normalising constant differs,
            # because the reference cell does: a unit square has A = 1, P = 4, so r = 1/2 and the
            # longest edge is 1, giving 2.0 where an equilateral triangle gives 2*sqrt(3). The
            # measure is 1.0 on ANY square, grows with elongation (1.5 on a 1x2 rectangle) and with
            # skew, and diverges as the cell collapses.
            cells_q = jnp.asarray(cells_np, dtype=jnp.int32)
            args_q, rebuild_q = self._moving_points()

            def _asp_quad(*vals):
                v = rebuild_q(*vals)[cells_q]  # (n_cells, 4, 2)
                e = jnp.roll(v, -1, axis=1) - v  # the four edges, in mesh order
                lengths = jnp.linalg.norm(e, axis=-1)
                x, y = v[..., 0], v[..., 1]
                area = jnp.abs(jnp.sum(x * jnp.roll(y, -1, axis=-1) - jnp.roll(x, -1, axis=-1) * y, axis=-1)) / 2.0
                inradius = 2.0 * area / (jnp.sum(lengths, axis=-1) + eps)
                return jnp.max(lengths, axis=-1) / (2.0 * inradius + eps)

            return _jno.fn(_asp_quad, args_q, name="cell_aspect")
        cells = jnp.asarray(cells_np, dtype=jnp.int32)
        args, rebuild = self._moving_points()
        fact = {2: 2.0, 3: 6.0}[dim]  # d!
        fac_fact = {2: 1.0, 3: 2.0}[dim]  # (d-1)!
        # longest-edge / inradius for a REGULAR simplex, so the measure reads 1.0 there: 2*sqrt(3) on
        # a triangle, 2*sqrt(6) on a tetrahedron.
        norm = {2: 2.0 * np.sqrt(3.0), 3: 2.0 * np.sqrt(6.0)}[dim]
        pairs = list(itertools.combinations(range(dim + 1), 2))

        def _asp(*vals):
            v = rebuild(*vals)[cells]  # (n_cells, dim+1, dim)
            longest = jnp.max(jnp.stack([jnp.linalg.norm(v[:, j] - v[:, i], axis=-1) for i, j in pairs], axis=-1), axis=-1)
            jac = jnp.stack([v[:, i + 1] - v[:, 0] for i in range(dim)], axis=-1)
            vol = jnp.abs(jnp.linalg.det(jac)) / fact
            # Facet measures by the Gram determinant, which is the one formula that covers both
            # dimensions: a triangle's facet is an edge (length) and a tet's is a triangle (area).
            surf = 0.0
            for k in range(dim + 1):
                keep = [i for i in range(dim + 1) if i != k]
                e = jnp.stack([v[:, keep[i + 1]] - v[:, keep[0]] for i in range(dim - 1)], axis=-1)
                gram = jnp.einsum("cij,cik->cjk", e, e)
                surf = surf + jnp.sqrt(jnp.clip(jnp.linalg.det(gram), 0.0, None)) / fac_fact
            inradius = dim * vol / (surf + eps)  # r = d V / S for a d-simplex
            return longest / (norm * inradius + eps)

        return _jno.fn(_asp, args, name="cell_aspect")

    def measure(self):
        """Total volume (area in 2-D) of the domain, as a node differentiable in the mesh.

        The denominator of a volume fraction. A node rather than a cached float because the mesh
        deforms: anything summed over element measures has to track the moving geometry.
        """
        return self.cell_volume().sum

    def transfer_cell_field(self, values, target, *, outside=0.0, points=None):
        """Carry a per-element field onto another mesh, by centroid containment.

        The transfer a **reanalysis** needs: an optimisation whose mesh coordinates are design
        variables can lower its objective either by improving the structure or by distorting
        elements until they under-integrate strain energy, and it cannot tell those apart. The
        only way to find out is to put the converged density on a fresh, undistorted mesh and
        re-solve. Measured once on a design reporting ``C = 77.97``: the same field on a clean
        mesh gave ``205.24``, a 163 % over-report, from a picture that looked entirely correct.

        Each target element takes the value of whichever source element contains its centroid,
        which is exact for a piecewise-constant field up to the target's own resolution -- so
        refine the target well past the source.

        Args:
            values: ``(n_cells,)`` field on THIS domain's elements.
            target: The domain to transfer onto.
            outside: Value for target centroids that fall outside this mesh (disjoint domains).
            points: Optional ``(n_points, dim)`` node positions to use for THIS mesh instead of
                its own — pass the *deformed* coordinates when the source mesh has moved under
                ``.trainable()``, since the domain still holds the positions it was built with.

        Returns:
            ``(target_n_cells,)`` numpy array.

        Triangles in 2-D and tetrahedra in 3-D. The point location underneath
        (:func:`fem_adapt._locate_in_cells`) is dimension-generic on a simplex -- ``dim + 1``
        vertices and one ``dim x dim`` solve for the barycentric coordinates -- so the tetrahedral
        case is the same code, not a second implementation.

        References:
            Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, Fig. 7c-d, where the
            gap is +17.6 % for conventional elements and -0.5 % with the enriched formulation.
        """
        dim = int(self.dimension)
        if dim not in (2, 3):
            raise NotImplementedError(f"domain.transfer_cell_field(): simplices in 2-D or 3-D; this domain is {dim}-D.")
        if int(target.dimension) != dim:
            raise ValueError(
                f"domain.transfer_cell_field(): the target is {int(target.dimension)}-D but this domain is "
                f"{dim}-D; a per-element field cannot cross dimensions."
            )
        src_cells, _ = self._cells_topo()
        src_pts = np.asarray(self.mesh.points)[:, :dim] if points is None else np.asarray(points)[:, :dim]
        vals = np.asarray(values).reshape(-1)
        if vals.size != src_cells.shape[0]:
            raise ValueError(
                f"domain.transfer_cell_field(): values has {vals.size} entries but this mesh has "
                f"{src_cells.shape[0]} cells."
            )
        tgt_cells, _ = target._cells_topo()
        tgt_centroids = np.asarray(target.mesh.points)[:, :dim][tgt_cells].mean(axis=1)

        from jno.utils.solver.fem_adapt import _locate_in_cells

        # 4-tuple: `_locate_in_cells` grew a reference-coordinate return with the quad/hex elements
        # (271363d3), after this call site was written against the 3-tuple.
        #
        # `k` is how many nearest CENTROIDS are tested for containment, and 32 is calibrated on
        # triangles. It is not enough on tetrahedra: a tet is pointier than a triangle, so its
        # centroid sits further from parts of the cell, and the containing tet can rank well down
        # the centroid ordering. Measured on a 138-tet box against its own refinement, one target
        # centroid of 767 -- at (1.545, 0.996, 1.015), the middle of the mesh, not a boundary case
        # -- had its true tet at rank exactly 32, so a k=32 search missed it by one and the point
        # was silently reported outside the mesh and given `outside`.
        #
        # So escalate, and only on the points that missed: the common path stays at k=32, and a
        # handful of stragglers get a wider search rather than every query paying for one (the
        # candidate solve is (Q, k, dim, dim), so a blanket k=128 would quadruple its memory).
        # Genuinely exterior points -- disjoint domains, which `outside=` exists for -- never
        # resolve, hence the cap: past it they are accepted as outside, which is the right answer
        # for them and a bounded cost for everyone.
        k, k_max = 32, min(512, int(src_cells.shape[0]))
        owner, _w, _ref, inside = _locate_in_cells(src_pts, src_cells, tgt_centroids, tol=1e-9, k=k)
        owner, inside = np.asarray(owner).copy(), np.asarray(inside).copy()
        while not inside.all() and k < k_max:
            k = min(k * 4, k_max)
            miss = ~inside
            o_m, _w_m, _r_m, in_m = _locate_in_cells(src_pts, src_cells, tgt_centroids[miss], tol=1e-9, k=k)
            owner[miss], inside[miss] = o_m, in_m
        return np.where(inside, vals[owner], float(outside))

    def _interior_facets(self):
        """Interior facets — those shared by exactly two cells. Edges in 2-D, triangles in 3-D.

        Returns a dict of host-side numpy arrays:
            ``cells`` ``(n_facets, 2)`` int — the two cells meeting at each facet.
            ``nodes`` ``(n_facets, n_face_nodes)`` int — the facet's corner nodes, for its measure
            (2 in 2-D, giving a length; 3 in 3-D, giving a triangle area).

        Boundary facets are excluded: they carry no density jump, since there is no element on the
        far side. This is the traversal a perimeter functional needs (Haber, Jog & Bendsoe,
        *Struct. Optim.* **11**, 1996, 1-12) — in 3-D the same functional measures the material
        boundary's AREA rather than a length, which is what its target has to be given in.

        Named for the facet rather than the edge because that is the dimension-generic word, and the
        one :mod:`jno.utils.solver.fem_facets` already uses; the local-face tables come from there
        (:func:`_face_table`) rather than being written out again per cell type.
        """
        from jno.utils.solver.fem_facets import _face_table

        cells, kind = self._cells_topo()
        dim = int(self.dimension)
        # `_face_table` already carries the tensor-product tables, so a quadrilateral's four edges
        # come from the same place a triangle's three do; only the block name has to be translated.
        cell_key = {"triangle": "triangle", "quad": "quad", "tetra": "tetrahedron"}.get(kind)
        if cell_key is None or dim not in (2, 3):
            raise NotImplementedError(
                f"domain._interior_facets(): triangles, quadrilaterals or tetrahedra; got {kind!r} in {dim}-D."
            )
        local_faces, n_face_nodes = _face_table(cell_key)
        lf = np.asarray([f[:n_face_nodes] for f in local_faces], dtype=np.int64)  # (n_local, n_fn)
        n_cells, n_local = cells.shape[0], lf.shape[0]

        # Canonical key per (cell, local facet): the sorted corner ids, so the two cells sharing a
        # facet produce the same row whatever order they list it in.
        keys = np.sort(cells[:, lf], axis=-1).reshape(n_cells * n_local, n_face_nodes)
        uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inverse = np.asarray(inverse).ravel()

        # The two owners of each shared facet: order the slots by facet id, then a facet used twice
        # occupies two adjacent positions. A stable sort keeps that pair in cell order.
        slot_cell = np.repeat(np.arange(n_cells, dtype=np.int64), n_local)
        order = np.argsort(inverse, kind="stable")
        cell_sorted = slot_cell[order]
        starts = np.searchsorted(inverse[order], np.arange(len(uniq), dtype=np.int64))
        shared = np.where(counts == 2)[0]
        return {
            "cells": np.stack([cell_sorted[starts[shared]], cell_sorted[starts[shared] + 1]], axis=1),
            "nodes": uniq[shared],
        }

    def _facet_ridges(self):
        """:meth:`_interior_facets`, grouped by the **ridge** each facet borders.

        A ridge is the facet's own facet — a *point* in 2-D, an *edge* in 3-D — so two interior
        facets sharing a ridge are neighbours on the material boundary, and the angle between them
        is that boundary's discrete curvature. This is the traversal a bending functional needs,
        the way :meth:`_interior_facets` alone is the traversal a perimeter needs.

        Returns :meth:`_interior_facets`' ``cells`` and ``nodes``, plus:
            ``facet_ridge`` ``(n_facets, n_local_ridges)`` int — which ridge each of the facet's
                sides belongs to. 2 per facet in 2-D (its endpoints), 3 in 3-D (its edges).
            ``ridge_nodes`` ``(n_ridges, n_ridge_nodes)`` int — the ridge's own corners: 1 in 2-D,
                2 in 3-D. Its measure follows — a point counts 1, an edge contributes its length.

        **A ridge collects every incident interior facet, not two.** In 3-D a mesh edge is shared
        by a whole fan of tetrahedra, hence by a fan of faces — 6 on average, not 2 — so there is
        no such thing as "the" neighbour across it. Which of them the material boundary actually
        runs through is a property of the *density*, not of the mesh, and so cannot be decided
        here; a functional over this table has to weight the pairs by their density jump and let
        the ones that carry no boundary contribute nothing. Grouping rather than pairing is what
        leaves that open.
        """
        facets = self._interior_facets()
        nodes = facets["nodes"]  # (F, n_face_nodes)
        # The facet's own facets. An edge's ridges are its two endpoints; a triangle's are its
        # three edges. Both are "drop one corner", which is why the tables are one line each.
        local = np.array([[0], [1]] if nodes.shape[1] == 2 else [[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        keys = np.sort(nodes[:, local], axis=-1).reshape(-1, local.shape[1])
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        return {
            **facets,
            "facet_ridge": np.asarray(inverse).reshape(nodes.shape[0], local.shape[0]),
            "ridge_nodes": uniq,
        }

    def patch_filter(self):
        """The patch filter of eq. (17)-(19) as a pure ``(n_cells,) -> (n_cells,)`` callable.

        Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, Sec. 2.3.2. It maps the
        design density to the **physical** density, and it is a manufacturability operator rather
        than a smoothing one: it drives to zero exactly those elements whose surroundings make the
        layout unbuildable, and leaves every other element untouched.

        Around each vertex of element ``k`` the elements form a patch, walked counterclockwise
        (:meth:`_patch_topology`). For that patch,

            f_k = [ prod_{i=2}^{N-2} { 1 - r_i (1 - mean(r_1..r_{i-1})) (1 - mean(r_{i+1}..r_{N-1})) }
                    * { 1 - r_k (1 - (r_1 + r_{N-1}) / 2) } ] ^ (1 / (N - 2))

        with ``r_1 ... r_{N-1}`` the other elements in order. The product detects a **one-node
        connection** -- a dense element joined to the rest of the structure through this vertex
        alone -- and the final factor a **single dense element** in an otherwise void patch, which
        puts a sharp corner on the boundary. Both drive ``f_k`` to zero; a sound patch leaves it at
        one. The physical density is ``rho_bar_k = rho_k (f^I + f^J + f^K) / P_k`` over the
        element's three vertices, ``P_k`` counting only the patches with at least three elements
        (eq. 19). With SIMP downstream a suppressed element's stiffness vanishes, so the
        configuration removes itself.

        **Boundary patches drop the final factor** (Fig. 2d): one-node connections are still
        suppressed there, but a single dense element on the domain boundary is *preserved*, since
        it costs little smoothness and helps keep a natural boundary profile. The paper states this
        rule in prose without an equation, so the placement is our reading; the value stays 1 when
        nothing is detected either way, so both branches remain on one scale.

        Two ways to use it, and a design needs both::

            rho.constrain(d.patch_filter())   # the PHYSICS sees rho_bar (paramax reparameterises
                                              # before every forward pass; MMA still steps in rho)
            rho_bar = rho.patch()             # the same map as a trace node, for constraints,
                                              # reporting and `crux.eval`

        The filter is non-local -- an element's physical density depends on its whole
        neighbourhood -- so it cannot be written inside the weak form, where the kernel sees one
        element at a time. That is why the physics route is a reparameterisation.
        """
        topo = self._patch_topology()
        others = jnp.asarray(topo["others"], dtype=jnp.int32)  # (K, 3, M) with -1 padding
        n_int = jnp.asarray(topo["size"], dtype=jnp.int32)  # (K, 3)
        interior = jnp.asarray(~topo["boundary"])  # (K, 3)
        m = others.shape[-1]
        pos = jnp.arange(m, dtype=jnp.int32)[None, None, :]  # array index p; r_{p+1} of the paper

        def _patch(rv):
            r = jnp.asarray(rv).reshape(-1)
            n = n_int  # (K, 3) patch size N
            vals = jnp.where(others >= 0, r[jnp.maximum(others, 0)], 0.0)  # (K, 3, M)
            live = (pos < (n - 1)[..., None]).astype(vals.dtype)  # entries 1..N-1 are real
            vals = vals * live
            csum = jnp.cumsum(vals, axis=-1)  # csum[p] = sum_{j<=p+1} r_j
            total = jnp.sum(vals, axis=-1, keepdims=True)

            # Factor i of the product, at array index p = i - 1, live for 1 <= p <= N-3.
            denom_pre = jnp.maximum(pos, 1).astype(vals.dtype)  # i - 1
            pre = jnp.concatenate([jnp.zeros_like(csum[..., :1]), csum[..., :-1]], -1) / denom_pre
            denom_suf = jnp.maximum((n - 2)[..., None] - pos, 1).astype(vals.dtype)  # N - i - 1
            suf = (total - csum) / denom_suf
            term = 1.0 - vals * (1.0 - pre) * (1.0 - suf)
            active = ((pos >= 1) & (pos <= (n - 3)[..., None])).astype(vals.dtype)
            prod = jnp.prod(jnp.where(active > 0, term, 1.0), axis=-1)  # (K, 3)

            # The final factor: k itself dense between two void neighbours in the patch.
            r_first = vals[..., 0]
            r_last = jnp.take_along_axis(vals, jnp.clip(n - 2, 0, m - 1)[..., None], axis=-1)[..., 0]
            rk = r[:, None]
            last = 1.0 - rk * (1.0 - 0.5 * (r_first + r_last))
            last = jnp.where(interior, last, 1.0)  # preserved on the boundary (Fig. 2d)

            valid = n >= 3
            expo = 1.0 / jnp.maximum(n.astype(vals.dtype) - 2.0, 1.0)
            f = jnp.where(valid, jnp.clip(prod * last, 0.0, None) ** expo, 0.0)
            p_k = jnp.sum(valid.astype(f.dtype), axis=-1)  # (K,)
            # P_k = 0 means no patch around this element reaches three elements; there is nothing
            # to judge it by, so it passes through unchanged rather than becoming 0/0.
            return r * jnp.where(p_k > 0, jnp.sum(f, axis=-1) / jnp.maximum(p_k, 1.0), 1.0)

        return _patch

    # Local edges of a tetrahedron as vertex-index pairs. Only the six pairs matter here (a fan is
    # keyed by the edge's two GLOBAL vertices, not by a local slot), so this is written out rather
    # than taken from BASIX_TET_EDGES -- nothing indexes these against an element table.
    _TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    def _patch_topology(self):
        """The element patches eq. (17)-(19) walks — vertex fans in 2-D, edge fans in 3-D.

        Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403. The formula's indices
        ``rho_{k,1} ... rho_{k,N-1}`` are a walk around a patch, with ``rho_{k,1}`` and
        ``rho_{k,N-1}`` the two elements sharing a facet with ``k``, so the patch has to come back
        as an ORDER, not a set.

        Returns a dict of numpy arrays, all host-side and computed once:
            ``others``   ``(n_cells, P, Nmax-1)`` int — the ordered other-element ids, ``-1`` padded.
            ``size``     ``(n_cells, P)`` int — ``N``, the patch's element count, including ``k``.
            ``boundary`` ``(n_cells, P)`` bool — whether that patch is open (touches the boundary).

        ``P`` is the number of patches an element belongs to: **3 vertices** of a triangle, **4** of
        a quadrilateral, or the **6 edges** of a tetrahedron.

        **Quadrilaterals use vertex patches, like triangles, and are the sharpest case measured.**
        Nothing in the walk is simplex-specific — the fan is ordered by the angle of each incident
        cell's centroid about the shared vertex, which reads a quad exactly as it reads a triangle.
        What changes is the patch SIZE, and eq. (18) is a geometric mean over ``N - 2`` factors, so
        size is what governs its contrast: a structured quad mesh has valence 4 against a
        triangulation's ~6, and ``tests/test_patch_filter_scaling.py`` measures a hinge scoring
        **0.001** of solid at ``N = 4`` against 0.04 at ``N = 6``. A deforming structured grid also
        holds that valence fixed wherever its nodes go, so unlike an unstructured mesh the criterion
        behaves identically at every vertex.

        **Why edges and not vertices in 3-D.** Around an interior edge, a tet's four faces contain
        both endpoints in exactly two cases, so the fan's dual graph is 2-regular — a cycle — and
        eq. (18) transfers verbatim with ``N^I -> N^edge``. Around an interior *vertex* the dual is
        3-regular and there is no total order to walk at all, and the patch is far larger:
        ``4T/V ~ 27`` against ``6T/E ~ 5.2`` for the edge fan on a Delaunay tetrahedral mesh. That
        size is the real obstacle rather than the ordering — eq. (18) is a geometric mean, so its
        contrast collapses as the patch grows, measured in ``tests/test_patch_filter_scaling.py``
        as a hinge scoring 0.04 of solid at ``N=6`` but 0.85 at ``N=27``. The edge fan lands at
        ``N ~ 5``, where the criterion is at its sharpest.

        The **ordering is taken once, from the current mesh**. Connectivity is fixed under
        ``.trainable()`` coordinates (relocation, not remeshing), so the patches themselves never
        change; only a distortion large enough to reorder the fan would invalidate the walk, which
        the geometric constraints of eq. (24)-(28) exist to prevent.
        """
        cells, kind = self._cells_topo()
        dim = int(self.dimension)
        if kind == "tetra":
            return self._edge_fan_topology(cells)
        if dim != 2 or kind not in ("triangle", "quad"):
            raise NotImplementedError(
                f"domain._patch_topology(): triangles or quadrilaterals in 2-D, tetrahedra in 3-D; got {kind!r} in {dim}-D."
            )
        pts = np.asarray(self.mesh.points)[:, :dim]
        n_cells, n_vert = cells.shape
        centroids = pts[cells].mean(axis=1)

        incident: List[List[int]] = [[] for _ in range(pts.shape[0])]
        for k, cell in enumerate(cells):
            for n in cell:
                incident[int(n)].append(k)

        # A vertex is on the domain boundary iff one of its incident edges is used by one cell only.
        # The cell's edges are its consecutive corner pairs, which is the same statement for a
        # triangle and a quadrilateral -- both list their corners in cyclic order.
        local_edges = tuple((i, (i + 1) % n_vert) for i in range(n_vert))
        edge_count: dict = {}
        for cell in cells:
            for a, b in local_edges:
                e = (int(min(cell[a], cell[b])), int(max(cell[a], cell[b])))
                edge_count[e] = edge_count.get(e, 0) + 1
        on_boundary = np.zeros(pts.shape[0], dtype=bool)
        for (a, b), c in edge_count.items():
            if c == 1:
                on_boundary[a] = on_boundary[b] = True

        ccw = {}
        for n, ks in enumerate(incident):
            if not ks:
                continue
            rel = centroids[ks] - pts[n]
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            ccw[n] = [ks[i] for i in np.argsort(ang)]

        n_max = max((len(v) for v in ccw.values()), default=1)
        others = np.full((n_cells, n_vert, max(n_max - 1, 1)), -1, dtype=np.int64)
        size = np.zeros((n_cells, n_vert), dtype=np.int64)
        boundary = np.zeros((n_cells, n_vert), dtype=bool)
        for k, tri in enumerate(cells):
            for v, n in enumerate(tri):
                ring = ccw[int(n)]
                q = ring.index(k)
                rest = ring[q + 1 :] + ring[:q]  # counterclockwise, starting adjacent to k
                others[k, v, : len(rest)] = rest
                size[k, v] = len(ring)
                boundary[k, v] = bool(on_boundary[int(n)])
        return {"others": others, "size": size, "boundary": boundary}

    def _edge_fan_topology(self, cells):
        """The 3-D half of :meth:`_patch_topology`: the tetrahedra around each edge, in fan order.

        For a tet ``(a,b,c,d)`` containing edge ``(a,b)``, only faces ``(a,b,c)`` and ``(a,b,d)``
        contain both endpoints, so each tet in the fan has exactly two neighbours and the fan is a
        **cycle** — the property that makes eq. (18)'s walk well defined without any angular sort.
        Two tets are adjacent in it when they share a face through the edge, which is exactly when
        their *far* vertices (the two that are not ``a`` or ``b``) overlap. That is the whole
        traversal: no geometry is read, only connectivity, which is why the fan is valid for any
        node positions the optimiser reaches.

        A **boundary** edge is an open chain rather than a cycle — its end tets have one neighbour —
        and is flagged so :meth:`patch_filter` applies the Fig. 2c-d boundary rule.
        """
        n_cells = cells.shape[0]
        edges: dict = {}
        for k in range(n_cells):
            t = cells[k]
            for slot, (i, j) in enumerate(self._TET_EDGES):
                a, b = int(t[i]), int(t[j])
                edges.setdefault((min(a, b), max(a, b)), []).append((k, slot))

        rings: dict = {}
        n_max = 1
        for (a, b), members in edges.items():
            ks = [k for k, _ in members]
            # The two vertices of each tet that are NOT on the edge; sharing one means sharing the
            # face through the edge, i.e. being neighbours in the fan.
            far = [tuple(int(v) for v in cells[k] if int(v) not in (a, b)) for k in ks]
            by_vertex: dict = {}
            for idx, pair in enumerate(far):
                for v in pair:
                    by_vertex.setdefault(v, []).append(idx)
            adj: List[List[int]] = [[] for _ in ks]
            for lst in by_vertex.values():
                if len(lst) == 2:  # an interior face: exactly two tets meet across it
                    adj[lst[0]].append(lst[1])
                    adj[lst[1]].append(lst[0])
            closed = all(len(nb) == 2 for nb in adj)
            if closed and len(ks) > 2:
                # 2-regularity is the structural claim this rests on; a violation means a
                # non-manifold mesh or a mis-extracted patch, and would silently give eq. (18) a
                # walk that revisits an element.
                assert sum(len(nb) for nb in adj) == 2 * len(ks), f"edge ({a},{b}) fan is not a cycle"
            start = 0 if closed else next(i for i in range(len(ks)) if len(adj[i]) < 2)
            order, seen, cur = [start], {start}, start
            while len(order) < len(ks):
                nxt = next((j for j in adj[cur] if j not in seen), None)
                if nxt is None:  # a fan split by a non-manifold edge; keep what was reachable
                    break
                order.append(nxt)
                seen.add(nxt)
                cur = nxt
            rings[(a, b)] = ([ks[i] for i in order], closed)
            n_max = max(n_max, len(order))

        n_slots = len(self._TET_EDGES)
        others = np.full((n_cells, n_slots, max(n_max - 1, 1)), -1, dtype=np.int64)
        size = np.zeros((n_cells, n_slots), dtype=np.int64)
        boundary = np.zeros((n_cells, n_slots), dtype=bool)
        for (a, b), members in edges.items():
            ring, closed = rings[(a, b)]
            for k, slot in members:
                if k not in ring:  # dropped by a broken traversal above
                    continue
                q = ring.index(k)
                rest = ring[q + 1 :] + ring[:q]  # fan order, starting adjacent to k
                others[k, slot, : len(rest)] = rest
                size[k, slot] = len(ring)
                boundary[k, slot] = not closed
        return {"others": others, "size": size, "boundary": boundary}

    def variable(
        self,
        tag: str,
        sample: Union[Tuple[Optional[int], Optional[Callable]], np.ndarray, jnp.ndarray] = (None, None),
        resampling_strategy=None,
        normals: bool = False,
        reverse_normals: bool = False,
        view_factor: bool = False,
        point_data: bool = False,
        split: bool = False,
        return_indices=False,
        time_value: Optional[float] = None,
        where=None,
        region=None,
    ) -> Any:
        """Create Variable placeholders for a tagged point set or tensor.

        Args:
            tag: Name of the point set (e.g., 'interior', 'boundary')
                 or tensor tag (e.g., 'diffusivity')
            where: Optional spatial predicate ``f(x, y[, z]) -> bool`` (or a shapely geometry).
                 When given, the region ``tag`` is defined on the fly -- exactly as
                 :meth:`tag` would -- and then its coordinates are returned, so naming a
                 region and grabbing its coords is a single call::

                     xl, yl, zl, _ = dom.variable("left", where=lambda x, y, z: x < 1e-6)
            sample: Optional sampling specification for this tag:
                    - (n_samples, sampler) tuple to trigger sampling
                    - np.ndarray / jnp.ndarray to register a tensor tag

                When ``sample`` is an array, the leading dimension determines
                how the tensor is routed by the compiler:

                  * ``shape[0] == B`` (the domain's effective batch count) —
                    vmapped over the batch axis, one row per sample.
                  * ``shape[0] == 1`` — broadcast across the batch.
                  * ``shape[0]`` anything else — *shared*: the full array is
                    exposed at every step (use this for labeled supervised
                    data, lookup tables, or gather indices that are not
                    aligned with the physics batch).

            resampling_strategy: Optional ResamplingStrategy for adaptive point selection
            normals: If True, also compute and return normal vectors for this tag
            reverse_normals: If True, flip the sign of the normal vectors
            return_indices: Wether or not to return the indices of the sampled points

        Returns:
            For point sets: Tuple of Variable placeholders, one per dimension.
            For point sets with normals=True: with ``split=True``, the flat tuple of coordinate + normal
                *component* variables ``(x, y, [z], t, nx, ny, [nz])``; with ``split=False`` (the default),
                the boundary normal as a **single vector** Variable — so it drops straight into vector ops,
                e.g. ``u.vector.cross(domain.variable(region, normals=True))``. (Coordinates in that case come
                from a separate ``domain.variable(region)`` / ``split=True`` call.)
            For tensor tags: Single TensorTag placeholder.
        """

        # Define-and-fetch: a predicate names the region here, then we return its coordinates.
        # `region=` restricts it to one body -- the only way to pick a single side of a
        # non-conforming interface, whose two surfaces share coordinates exactly. See `tag`.
        if where is not None:
            self.tag(tag, where, region=region)
        elif region is not None:
            raise TypeError("domain.variable: `region=` applies only with `where=`, which is what it restricts.")

        # Optional sampling / tensor-tag attachment
        if sample is not None:
            if _is_lazy_source(sample):
                # A LAZY source: anything array-like that jNO can slice without reading -- an
                # h5py.Dataset, a zarr.Array, an np.memmap. Stored as the handle itself, never
                # materialized here, so the dataset may exceed memory. It is read one batch at a
                # time by `core.solve(offload_data=True)`, which is the only path that can stream
                # it; the on-device path has to hold the whole array and refuses (see core.solve).
                if point_data:
                    self.context[tag] = sample
                else:
                    self._check_lazy_tensor_layout(tag, sample)
                    self.context[tag] = sample
                    self._param_tags.add(tag)
            elif isinstance(sample, jnp.ndarray) or isinstance(sample, np.ndarray):
                # Attach as tensor tag (parameter field) or point data.
                # Three shape conventions for tensor tags are documented above
                # and routed in jno/trace_compiler.py at attach time; we do
                # not validate the leading dim here.
                if point_data:
                    self.context[tag] = sample
                else:
                    tensor = jnp.asarray(sample)
                    if tensor.ndim < 1:
                        tensor = tensor.reshape(1, 1)
                    self.context[tag] = self._normalize_tensor_time_axis(tag, tensor)
                    self._param_tags.add(tag)
            elif not isinstance(sample, tuple):
                raise TypeError(
                    f"domain.variable({tag!r}, sample=...): expected a sampling spec `(n, sampler)`, an array, "
                    f"or a lazy array-like exposing BOTH `.shape` and `__getitem__` (h5py.Dataset, zarr.Array, "
                    f"np.memmap). Got {type(sample).__name__}"
                    + (" -- it has `.shape` but is not indexable." if hasattr(sample, "shape") else ".")
                )

        # ------------------------------------------------------------------
        # Clean API for initial condition:
        #   x0, y0, t0 = domain.variable("initial", split=True)
        #
        # requested_tag = public tag returned to the user.
        # source_tag    = internal mesh/context tag used for sampling.
        #
        # For time-dependent domains, "initial" means t = self.time[0].
        # If an explicit "initial" mesh pool exists, sample from it.
        # Otherwise fall back to "interior" but materialize the result
        # under the public tag "initial".
        # ------------------------------------------------------------------
        requested_tag = tag
        source_tag = requested_tag

        if requested_tag == "initial" and self._is_time_dependent and self.time is not None:
            if time_value is None:
                time_value = float(self.time[0])

            if requested_tag not in self._mesh_pool and "interior" in self._mesh_pool:
                source_tag = "interior"

        sample_tag = source_tag

        if (
            sample_tag in self._mesh_pool.keys()
            and isinstance(sample, tuple)
            and len(sample) > 0
            and isinstance(sample[0], (int, type(None)))
        ):
            # Sample points for this tag on demand.
            self.sample_dict.append([sample_tag, (None, None), resampling_strategy, normals, view_factor])

            # Pass time_value through so "initial" / fixed-time slices can be materialized.
            points, idx, sampled_tag = self.sample(
                {sample_tag: sample},
                normals,
                return_indices,
                time_value=time_value,
            )

            # If user asked for "initial" but we sampled from "interior",
            # expose the sampled/sliced data under "initial".
            if requested_tag != sampled_tag:
                if sampled_tag in self.context:
                    self.context[requested_tag] = self.context[sampled_tag]

                sampled_time_tag = f"__time_{sampled_tag}__"
                requested_time_tag = f"__time_{requested_tag}__"

                if sampled_time_tag in self.context:
                    self.context[requested_time_tag] = self.context[sampled_time_tag]
                elif time_value is not None:
                    base_time = self.context.get("__time__", np.asarray([[time_value]], dtype=default_np_float_dtype()))
                    dtype = np.asarray(base_time).dtype
                    self.context[requested_time_tag] = np.asarray([[time_value]], dtype=dtype)

                if normals and f"n_{sampled_tag}" in self.context:
                    self.context[f"n_{requested_tag}"] = self.context[f"n_{sampled_tag}"]

                tag = requested_tag
            else:
                tag = sampled_tag

            # Safety: if a fixed time was requested, ensure the tag-specific
            # temporal context exists even when sample(...) did not create it.
            if time_value is not None:
                requested_time_tag = f"__time_{tag}__"
                if requested_time_tag not in self.context:
                    base_time = self.context.get("__time__", np.asarray([[time_value]], dtype=default_np_float_dtype()))
                    dtype = np.asarray(base_time).dtype
                    self.context[requested_time_tag] = np.asarray([[time_value]], dtype=dtype)

        # Store resampling strategy if provided
        if resampling_strategy is not None:
            self._resampling_strategies[tag] = resampling_strategy

        # Check if it's a parametric (TensorTag) entry
        if tag in self._param_tags:
            if split:
                return tuple(TensorTag(tag=tag, dim_index=i, domain=self) for i in range(sample.shape[-1]))  # type: ignore[attr-defined,union-attr]
            else:
                return TensorTag(tag=tag, domain=self)

        if point_data:
            if split:
                return tuple(Variable(tag=tag, dim=[i, i + 1], domain=self) for i in range(sample.shape[-1]))  # type: ignore[attr-defined,union-attr]
            else:
                return Variable(tag=tag, dim=[0, None], domain=self)

        if tag not in self.context:
            available = sorted(k for k in self.context.keys() if not k.startswith("__"))
            mesh_keys = sorted(getattr(self, "_mesh_pool", {}).keys())
            if tag in mesh_keys:
                hint = (
                    f"Tag '{tag}' exists in the mesh pool but has not been sampled yet. "
                    f"Call domain.variable('{tag}', sample=(n, sampler)) to materialize it."
                )
            else:
                hint = (
                    f"Tag '{tag}' is not in the mesh pool or context. "
                    f"Available context tags: {available}. "
                    f"Available mesh-pool tags: {mesh_keys}."
                )
            raise ValueError(hint)

        fem_meta = None
        if getattr(self, "_variational_initialized", False):
            fem_meta = getattr(self, "_variational_sampling_registry", {}).get(tag, None)

        # Create Variable placeholder for each spatial dimension
        coord_vars: List[Any] = [
            Variable(tag=tag, dim=[i, i + 1], domain=self, axis="spatial", fem_meta=fem_meta) for i in range(self.dimension)
        ]

        # Always add temporal variable.
        # If a time-specialized tag exists for this sampled point set
        # (e.g. "__time_interior_0__"), use it; otherwise fall back to "__time__".
        time_tag = f"__time_{tag}__" if f"__time_{tag}__" in self.context else "__time__"

        coord_vars.append(
            Variable(
                tag=time_tag,
                dim=[0, 1],
                domain=self,
                axis="temporal",
                fem_meta=None,
            )
        )

        if normals:
            if f"n_{tag}" not in self.context:
                raise ValueError(
                    f"domain.variable('{tag}', normals=True): no outward normals found for "
                    f"tag '{tag}'. Normals are only available for boundary tags on mesh-based "
                    "domains. Check that the tag refers to a boundary region."
                )
            if reverse_normals:
                self.context[f"n_{tag}"] = -self.context[f"n_{tag}"]
            # ``split`` controls the *shape* of the normal (mirroring the tensor-tag path): ``split=True``
            # appends the normal's scalar components to the flat ``(x, y, [z], t, nx, ny, [nz])`` tuple;
            # ``split=False`` (the default) returns the boundary normal as a **single vector** so it drops
            # straight into vector ops, e.g. ``u.vector.cross(d.variable(region, normals=True))`` — no need
            # to slice the components out and rebuild them. (Coordinates are still available via
            # ``d.variable(region, ..., split=True)`` or a plain ``d.variable(region)`` call.)
            if not split and not view_factor and not return_indices:
                return Variable(tag=f"n_{tag}", dim=[0, len(self.spatial)], domain=self)
            coord_vars += [Variable(tag=f"n_{tag}", dim=[i, i + 1], domain=self) for i in range(len(self.spatial))]

        if view_factor and hasattr(self, "mesh_connectivity"):
            # Only take the first batch index
            Nrm = -self.context[f"n_{tag}"][0, ...]  # Reverse the normals
            P = self.context[tag][0, ...]

            ds = self.mesh_connectivity["nodal_ds"][self.mesh_connectivity["boundary_indices"]]

            if ds.shape[0] != P.shape[0]:
                ds = self.ds * np.ones(P.shape[0])
                self.log.warning("Size of elements is constant due to mismatch in boundary array.")

            all_bp = self.mesh_connectivity["boundary_points"]
            all_VM = self.mesh_connectivity["VM"]
            subset_bp = P[0]

            # Check if all points are in the global boundary points (outer boundary)
            # If not, compute a local visibility matrix for internal boundaries
            point_to_idx = {tuple(pt): i for i, pt in enumerate(all_bp)}
            points_in_boundary = [tuple(pt) in point_to_idx for pt in subset_bp]

            if all(points_in_boundary):
                # All points are on outer boundary, use global visibility matrix
                subset_indices = np.array([point_to_idx[tuple(pt)] for pt in subset_bp])
                subset_VM = all_VM[np.ix_(subset_indices, subset_indices)]
            else:
                # Internal boundary - compute local visibility matrix
                # Order points into a proper closed polygon first
                order = self._order_boundary_loop(subset_bp)
                ordered_bp = subset_bp[order]
                edges = np.array(
                    [[i, (i + 1) % len(ordered_bp)] for i in range(len(ordered_bp))],
                    dtype=np.int32,
                )
                ordered_VM = self.get_visibility_matrix_raytrace(ordered_bp, edges, n_ray_samples=3)
                # Map VM back to original point order
                inv_order = np.argsort(order)
                subset_VM = np.asarray(ordered_VM)[np.ix_(inv_order, inv_order)]

            if self.dimension == 1:
                VF = self.get_view_factor_1d(P[0], subset_VM, Nrm[0], ds)
            elif self.dimension == 2:
                VF = self.get_view_factor_2d(P[0], subset_VM, Nrm[0], ds)
            elif self.dimension == 3:
                VF = self.get_view_factor_3d(P[0], subset_VM, Nrm[0], ds)
            else:
                raise ValueError(
                    f"view_factor=True is only supported for spatial dimension 1, 2, or 3 (got dimension={self.dimension})."
                )

            # view_factor stores tensors with a hard-coded batch size of 1.
            # Reject the batched-domain case explicitly instead of silently
            # producing wrong results when the domain has been merged via `+`.
            batch_count = getattr(self, "_batch_count", 1) or 1
            if batch_count > 1:
                raise NotImplementedError(
                    f"view_factor=True is not supported on batched domains "
                    f"(_batch_count={batch_count}). Compute view factors on a "
                    "single-batch domain or open an issue if you need this."
                )
            self.context[f"v_{tag}"] = subset_VM[None, None, ...]
            self._param_tags.add(f"v_{tag}")
            self.context[f"f_{tag}"] = VF[None, ...]
            self._param_tags.add(f"f_{tag}")
            coord_vars += [TensorTag(tag=f"f_{tag}", domain=self)]

        # if view_factor and not hasattr(self, "mesh_connectivity"):
        #    self.log.error("In order to calcuate the view factor please set compute_mesh_connectivity in the domain initialization to true.")

        if return_indices:
            coord_vars += [idx]

        return tuple(coord_vars)

    def distance_function(
        self,
        tag: str = "interior",
        boundary_tags=None,
        name=None,
    ):
        """Compute minimum distance from boundary to each sampled point at *tag*.

        Returns a Variable placeholder of shape (N, 1) per batch — suitable
        for hard Dirichlet BC enforcement::

            d = dom.distance_function("interior")
            u_phys = net(jno.np.concat([x, y], axis=-1)) * d  # u=0 on ∂Ω

        Args:
            tag:           Source tag whose points get distances computed.
            boundary_tags: Tags used as boundary reference. Defaults to
                           ["boundary"] if present, else all non-interior tags.
            name:          Context key for the result. Auto-generated if None.
        """
        import numpy as _np

        # --- resolve interior points ---
        if tag in self._mesh_pool:
            pts = _np.asarray(self._mesh_pool[tag])
        elif tag in self.context:
            raw = self.context[tag]
            # context shape is (B, 1, N, D) or (1, 1, N, D); use first batch
            pts = _np.asarray(raw[0, 0])
        else:
            raise ValueError(f"distance_function: tag '{tag}' not found in mesh_pool or context.")

        pts_spatial = pts[:, : self.dimension]

        # --- resolve boundary points ---
        if boundary_tags is None:
            if "boundary" in self._mesh_pool:
                boundary_tags = ["boundary"]
            else:
                boundary_tags = [t for t in self._mesh_pool if t not in (tag, "initial")]

        bdry_parts = [_np.asarray(self._mesh_pool[t])[:, : self.dimension] for t in boundary_tags if t in self._mesh_pool]
        if not bdry_parts:
            raise ValueError(f"distance_function: no boundary tags found (tried {boundary_tags}).")
        bdry_pts = _np.vstack(bdry_parts)

        # --- compute distances ---
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(bdry_pts)
            dists, _ = tree.query(pts_spatial)
        except ImportError:
            diffs = pts_spatial[:, None, :] - bdry_pts[None, :, :]
            dists = _np.sqrt((diffs**2).sum(-1)).min(-1)

        # Store as (1, 1, N, 1) — same layout as other context entries
        dist_arr = dists.astype(_np.float32)[_np.newaxis, _np.newaxis, :, _np.newaxis]
        dist_tag = name or f"__dist_{tag}__"
        self.context[dist_tag] = dist_arr
        self._param_tags.add(dist_tag)

        return Variable(tag=dist_tag, dim=[0, 1], domain=self, axis="spatial")

    def __getitem__(self, tag: str) -> Tuple[Variable, ...]:
        """Shorthand for domain.variable(tag).

        Example:
            x, y = domain['interior']
        """
        return self.variable(tag)

    @staticmethod
    def _chain_edges_to_loop(edges):
        """Chain a set of ``(a, b)`` edge pairs into one ordered path of point indices.

        Handles both a **closed loop** (every node degree 2 -> N edges give N nodes) and an **open
        chain** (a box face: two endpoints of degree 1 -> N edges give N+1 nodes). The open case must
        keep *both* endpoints: a shared corner is an endpoint of a face's chain, and dropping it (the
        old closed-loop-only walk did) removes the corner from that face's tag -- which silently breaks
        multidirectional periodicity, where a corner must belong to every face it lies on.

        Args:
            edges: List of 2-tuples (global point indices) forming a single connected path/loop.

        Returns:
            np.ndarray of global point indices in traversal order.
        """
        from collections import defaultdict

        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        # Start at an endpoint (degree 1) for an open chain; anywhere for a closed loop.
        endpoints = [n for n, nbrs in adj.items() if len(nbrs) == 1]
        start = endpoints[0] if endpoints else edges[0][0]
        visited = {start}
        order = [start]
        current = start
        while True:
            nxt = next((nb for nb in adj[current] if nb not in visited), None)
            if nxt is None:
                break
            visited.add(nxt)
            order.append(nxt)
            current = nxt
        return np.array(order, dtype=int)

    @staticmethod
    def _extract_volume_boundary(triangles):
        """Extract boundary edges from a set of triangles.

        Boundary edges appear in exactly one triangle (interior edges appear
        in two).

        Args:
            triangles: (N, 3) array of triangle vertex indices.

        Returns:
            List of (a, b) edge tuples forming the boundary.
        """
        from jno.domain.mesh_utils import MeshUtils

        edges = MeshUtils._get_boundary_elements(np.asarray(triangles), "triangle")
        return [tuple(e) for e in edges.tolist()]

    @staticmethod
    def _chain_edges_to_loops(edges):
        """Chain a set of (a, b) edge pairs into one or more ordered loops.

        Unlike ``_chain_edges_to_loop`` which assumes a single loop, this
        handles disconnected boundaries (e.g. an annular region has two
        boundary loops).

        Args:
            edges: List of 2-tuples (global point indices).

        Returns:
            List of np.ndarray, each an ordered loop of global point indices.
        """
        from collections import defaultdict

        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)

        visited_global = set()
        loops = []
        for start in adj:
            if start in visited_global:
                continue
            loop = [start]
            visited_global.add(start)
            current = start
            while True:
                nxt = None
                for nb in adj[current]:
                    if nb not in visited_global:
                        nxt = nb
                        break
                if nxt is None:
                    break
                visited_global.add(nxt)
                loop.append(nxt)
                current = nxt
            loops.append(np.array(loop, dtype=int))
        return loops

    @staticmethod
    def _order_boundary_loop(pts):
        """Order 2D boundary points into a proper closed polygon.

        Uses angular sorting from the centroid.  This is exact for convex
        loops (rectangles, circles, etc.) and a good heuristic for mildly
        non-convex ones.

        Args:
            pts: (N, 2) array of boundary points.

        Returns:
            order: (N,) index array such that ``pts[order]`` forms a
                   proper polygon.
        """
        n = len(pts)
        if n <= 2:
            return np.arange(n)

        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        return np.argsort(angles)

    def enclosure(
        self,
        tags,
        *,
        axisymmetric=False,
        n_quad=6,  # see build_enclosure: the closed-form azimuth has no near-field refinement
        n_phi=16,
        opaque_tags=None,
        medium_tags=None,
        enforce_closure=False,
        closure_iters=200,
        occlude=True,
        inward=False,
        r_min=None,
        near_field=True,
    ):
        """Build an :class:`~jno.domain.enclosure.Enclosure` from radiating boundary ``tags``.

        Discretises the listed boundary surfaces into mesh-edge **elements** (aligned to FEM nodes) and
        returns a handle exposing the element-to-element view factor ``enclosure.view_factor`` (fully
        geometry-determined — self-view included), per-element ``areas``/``normals``, the global node
        ``elements``, and an F-quality gate (``.check()`` / ``.quality()``). Write the grey-body
        radiosity in ``jno.np`` on top of ``enclosure.view_factor``; see ``docs/fem.md``.

        Args:
            tags: Boundary tags forming the enclosure (each a radiating surface; tags only group
                elements for per-surface emissivity — they never block exchange).
            axisymmetric: Treat the 2D mesh as a meridional ``(r, z)`` half-plane (bodies of revolution).
            n_quad: Gauss points per element for the double-area view-factor quadrature.
            opaque_tags: Optional boundary tags that block rays (occluders) without radiating.
            medium_tags: Optional transparent (meshed) medium regions, by geometry-part name. When given,
                ``tags`` are solid geometry-part names and the radiating elements are the internal
                solid|medium **interface** edges (the common furnace case where the gas/air gap is meshed),
                with normals pointing out of the solid into the medium. When ``None`` (default), the
                radiating elements are domain **boundary** edges on ``tags`` (an un-meshed/vacuum gap).
            inward: Boundary mode only. When ``True``, element normals point *into* the meshed domain —
                use when the radiating ``tags`` are the outer walls of a **meshed cavity** (an oven /
                furnace filled with a transparent fluid) and radiation crosses the meshed interior, so
                the facing walls see one another. Default ``False`` (normals out of the mesh, vacuum gap).
            r_min: Axisymmetric only. Near-field FLOOR for the ring kernel (``R^2 -> R^2 + r_min^2``) — a
                fudge that caps the kernel for near-coincident rings rather than integrating them properly.
                Defaults to half the median element length. Note that this default is itself a ~12%
                systematic error on the analytic concentric-cylinder view factors; pass a value to override.
        """
        from .enclosure import build_enclosure

        return build_enclosure(
            self,
            tags,
            axisymmetric=axisymmetric,
            n_quad=n_quad,
            n_phi=n_phi,
            opaque_tags=opaque_tags,
            medium_tags=medium_tags,
            enforce_closure=enforce_closure,
            closure_iters=closure_iters,
            occlude=occlude,
            inward=inward,
            r_min=r_min,
            near_field=near_field,
        )

    def compute_enclosure_view_factor(self, tags, opaque_tags=None):
        """Compute view factors over a combined radiation enclosure.

        All listed tags must have been sampled (with ``normals=True``) before
        calling this method.  The method combines the points from every tag,
        computes normals directly from the loop geometry (more reliable than
        PCA for internal boundaries), auto-orients them to point into the gas
        region, and then builds the visibility and view-factor matrices.

        Args:
            tags: List of tag names that form the enclosure, e.g.
                  ``["interior_boundary", "interior_boundary_outer"]``.
            opaque_tags: Optional list of tag names whose boundary edges block
                  rays but whose points do **not** participate in the view-factor
                  matrix.  Use this for solid obstacles inside the enclosure,
                  e.g. ``opaque_tags=["solid0", "solid1"]``.

        Returns:
            Nested list of ``TensorTag`` view-factor matrices.  For *n* tags
            the result is an *n x n* list-of-lists::

                [[F_AA, F_AB],
                 [F_BA, F_BB]]

            where ``F_AB`` has shape ``(N_A, N_B)`` and gives the view factor
            from each point on tag A to points on tag B.

        Example::

            (F00, F01), (F10, F11) = domain.compute_enclosure_view_factor(
                ["interior_boundary", "interior_boundary_outer"],
                opaque_tags=["solid0", "solid1"],
            )
            bc_rad0 = ... - eps * sigma * (u_b0**4 - jnn.sum(F00 * u_b0**4) - jnn.sum(F01 * u_b1**4))
        """
        if opaque_tags is None:
            opaque_tags = []

        # ----- 1. Gather ordered points per tag -------------------------------
        tag_pts = []  # list of (N_i, D)
        tag_sizes = []
        for tag in tags:
            if tag not in self.context:
                raise ValueError(f"Tag '{tag}' not yet sampled.  Call domain.variable('{tag}') first.")

            # Use _mesh_pool directly if available (already in loop order)
            if tag in self._mesh_pool:
                pts = np.array(self._mesh_pool[tag], dtype=np.float64)
                if pts.ndim > 2:
                    pts = pts[0]  # time-dep: (T, N, D) -> (N, D)
            else:
                pts = np.asarray(self.context[tag], dtype=np.float64)
                while pts.ndim > 2:
                    pts = pts[0]
                # Fallback: angular sort
                order = self._order_boundary_loop(pts)
                pts = pts[order]

            tag_pts.append(pts)
            tag_sizes.append(pts.shape[0])

        all_pts = np.concatenate(tag_pts, axis=0)  # (N_total, D)
        N_total = all_pts.shape[0]

        # ----- 2. Build edge list for ray-tracing (per-tag closed loops) ------
        edges = []
        offset = 0
        for n in tag_sizes:
            for i in range(n):
                edges.append([offset + i, offset + (i + 1) % n])
            offset += n

        # ----- 2b. Gather opaque obstacle edges (block rays, no VF) -----------
        # Opaque tags add blocking edges.  Two kinds are supported:
        #   - Boundary-loop tags (line cells): use directly as a closed loop.
        #   - Volume tags (triangle cells): extract boundary edges of the
        #     triangulated region automatically.
        opaque_loop_pts = []
        for otag in opaque_tags:
            if otag in self._boundary_loop_tags:
                # --- Boundary loop tag: already ordered ---
                if otag in self._mesh_pool:
                    opts = np.array(self._mesh_pool[otag], dtype=np.float64)
                    if opts.ndim > 2:
                        opts = opts[0]
                else:
                    self.log.warning(f"Opaque tag '{otag}' not in mesh pool, skipping.")
                    continue
                opaque_loop_pts.append(opts)
            elif otag in self._tag_triangles:
                # --- Volume tag: extract boundary edges from triangles ---
                tris = self._tag_triangles[otag]
                bnd_edges = self._extract_volume_boundary(tris)
                if len(bnd_edges) == 0:
                    self.log.warning(f"Opaque tag '{otag}': no boundary edges found. Skipping.")
                    continue
                # Chain edges into one or more loops
                loops = self._chain_edges_to_loops(bnd_edges)
                pts = np.asarray(self.points, dtype=np.float64)
                for loop_indices in loops:
                    opaque_loop_pts.append(pts[loop_indices])
            else:
                self.log.warning(
                    f"Opaque tag '{otag}' has no line or triangle cells. "
                    f"Available boundary loops: {sorted(self._boundary_loop_tags)}, "
                    f"volume tags: {sorted(self._tag_triangles.keys())}. Skipping."
                )
                continue

        # Append opaque points and their closed-loop edges.
        #
        # Subtlety: when a volume tag (e.g. solid0) is opaque, its mesh
        # boundary edges lie on the *same geometric curve* as one of the
        # participating tag loops, but they use different point positions
        # (original mesh vertices vs. resampled boundary points).  If we
        # add them blindly, the duplicate overlapping edges block all
        # rays.  Fix: detect and skip any opaque loop whose geometry
        # coincides with a participating loop (Hausdorff distance < tol).
        if opaque_loop_pts:
            from scipy.spatial import cKDTree

            # Build per-tag kd-trees for coincidence testing
            tag_trees = []
            for pts in tag_pts:
                tag_trees.append(cKDTree(pts))

            # Tolerance: use mesh spacing as proxy (average edge length
            # of the first participating loop)
            ref = tag_pts[0]
            edge_lens = np.linalg.norm(np.diff(ref, axis=0, append=ref[:1]), axis=1)
            tol = edge_lens.mean() * 0.5

            extra_pts = []
            next_idx = N_total
            kept_loops = 0
            skipped_loops = 0

            for loop_coords in opaque_loop_pts:
                # Check if this loop coincides with ANY participating loop
                coincides = False
                for tree_i in tag_trees:
                    dists, _ = tree_i.query(loop_coords)
                    if dists.max() < tol:
                        coincides = True
                        break

                if coincides:
                    skipped_loops += 1
                    continue  # skip – same boundary, different discretisation

                kept_loops += 1
                n_op = len(loop_coords)
                # All points are truly new (not on any participating loop)
                start_idx = next_idx
                for k in range(n_op):
                    extra_pts.append(loop_coords[k])
                next_idx += n_op
                for k in range(n_op):
                    edges.append([start_idx + k, start_idx + (k + 1) % n_op])

            if extra_pts:
                raytrace_pts = np.concatenate([all_pts, np.array(extra_pts, dtype=np.float64)], axis=0)
            else:
                raytrace_pts = all_pts

            if skipped_loops:
                self.log.info(f"Opaque: kept {kept_loops} loop(s), skipped {skipped_loops} coincident loop(s)")
        else:
            raytrace_pts = all_pts

        edges = np.array(edges, dtype=np.int32)

        # ----- 3. Compute normals from loop geometry --------------------------
        # Much more reliable than PCA for internal boundaries.
        # For each point, average the normals of its two adjacent edges.
        tag_nrm = []
        for loop_pts in tag_pts:
            n = len(loop_pts)
            normals = np.zeros_like(loop_pts)
            for i in range(n):
                # Forward and backward edge tangents
                t_fwd = loop_pts[(i + 1) % n] - loop_pts[i]
                t_bwd = loop_pts[i] - loop_pts[(i - 1) % n]
                # 2D: outward normal for CCW polygon = rotate tangent 90° CW
                n_fwd = np.array([t_fwd[1], -t_fwd[0]])
                n_bwd = np.array([t_bwd[1], -t_bwd[0]])
                avg = n_fwd + n_bwd
                norm = np.linalg.norm(avg)
                if norm > 1e-12:
                    normals[i] = avg / norm
                else:
                    normals[i] = n_fwd / (np.linalg.norm(n_fwd) + 1e-30)
            tag_nrm.append(normals)

        # ----- 4. Orient normals to point INTO the gas region -----------------
        # Use only the PARTICIPATING edges for the ray-cast test.
        # Opaque edges must NOT be included here -- they would change the
        # parity and flip normals for boundaries whose gas side is between
        # the participating loop and the opaque loop.
        participating_edges = []
        offset = 0
        for n in tag_sizes:
            for i in range(n):
                participating_edges.append([offset + i, offset + (i + 1) % n])
            offset += n
        participating_edges = np.array(participating_edges, dtype=np.int32)

        E0_p = all_pts[participating_edges[:, 0]]
        E1_p = all_pts[participating_edges[:, 1]]

        def _ray_cast_inside(test_pts):
            """Even-odd ray cast over PARTICIPATING enclosure edges only."""
            x = test_pts[:, 0:1]
            y = test_pts[:, 1:2]
            y0 = E0_p[np.newaxis, :, 1]
            y1 = E1_p[np.newaxis, :, 1]
            x0 = E0_p[np.newaxis, :, 0]
            x1 = E1_p[np.newaxis, :, 0]
            straddles = (y0 > y) != (y1 > y)
            dy = y1 - y0
            dy_safe = np.where(np.abs(dy) < 1e-14, 1.0, dy)
            x_int = x0 + (x1 - x0) * (y - y0) / dy_safe
            crossings = straddles & (x < x_int)
            n_cross = np.sum(crossings.astype(np.int32), axis=1)
            return (n_cross % 2) == 1

        for loop_idx, (pts, nrm) in enumerate(zip(tag_pts, tag_nrm)):
            # Sample several non-corner points and vote
            n = len(pts)
            n_test = min(8, n)
            test_indices = np.linspace(0, n - 1, n_test + 2, dtype=int)[1:-1]
            gas_votes = 0
            for ti in test_indices:
                test_pt = pts[ti] + 1e-4 * nrm[ti]
                inside = _ray_cast_inside(test_pt[None, :])
                gas_votes += 1 if inside[0] else -1
            if gas_votes < 0:
                tag_nrm[loop_idx] = -nrm

        all_nrm = np.concatenate(tag_nrm, axis=0)

        # ----- 5. Visibility matrix over combined set -------------------------
        # Run ray-tracing over ALL points (participating + opaque) so opaque
        # edges block rays, then slice out only the participating rows/cols.
        VM_full = np.asarray(self.get_visibility_matrix_raytrace(raytrace_pts, edges, n_ray_samples=3))
        VM = np.array(VM_full[:N_total, :N_total], copy=True)  # writable copy

        # ----- 5b. Block self-visibility through enclosed solid ---------------
        # For a convex boundary loop enclosing a solid region, rays between
        # two points on the same loop pass through the solid interior without
        # crossing any boundary edge (the adjacency mask skips edges touching
        # source/target).  Fix: for each boundary loop whose interior is solid
        # (normals point OUTWARD, away from centroid), test every same-loop
        # visible pair's midpoint; if it falls inside the polygon, block it.
        offset = 0
        for loop_idx, (lpts, nrm, n) in enumerate(zip(tag_pts, tag_nrm, tag_sizes)):
            centroid = lpts.mean(axis=0)
            # Use a mid-side sample point (not a corner) to test normal dir
            sample_idx = n // 4
            to_centroid = centroid - lpts[sample_idx]
            dot_val = np.dot(to_centroid, nrm[sample_idx])
            if dot_val >= 0:
                # Normals point toward centroid → interior is gas, not solid
                offset += n
                continue

            # Interior is solid — block same-loop pairs whose midpoint is inside
            # Compute all pairwise midpoints (n, n, 2)
            mid_x = (lpts[:, 0:1] + lpts[:, 0:1].T) / 2  # (n, n)
            mid_y = (lpts[:, 1:2] + lpts[:, 1:2].T) / 2

            # Even-odd ray-cast point-in-polygon for all midpoints at once
            loop_e0 = lpts  # (n, 2)
            loop_e1 = np.roll(lpts, -1, axis=0)  # (n, 2)
            inside = np.zeros((n, n), dtype=bool)
            for e in range(n):
                ey0, ey1 = loop_e0[e, 1], loop_e1[e, 1]
                ex0, ex1 = loop_e0[e, 0], loop_e1[e, 0]
                straddle = (ey0 > mid_y) != (ey1 > mid_y)
                dy = ey1 - ey0
                if abs(dy) < 1e-14:
                    continue
                x_int = ex0 + (ex1 - ex0) * (mid_y - ey0) / dy
                inside ^= straddle & (mid_x < x_int)

            # Zero out VM entries for through-solid pairs
            blk = VM[offset : offset + n, offset : offset + n]
            n_blocked = int((inside & (blk > 0)).sum())
            blk[inside] = 0
            if n_blocked:
                self.log.info(
                    f"Blocked {n_blocked} self-visible pairs through solid interior for loop {loop_idx} ({tags[loop_idx]})"
                )
            offset += n

        # ----- 6. Element sizes (constant ds assumed for internal boundaries) -
        ds = self.ds * np.ones(N_total)

        # ----- 7. View-factor matrix ------------------------------------------
        # Normals now point INTO the gas (the participating medium).  The VF
        # formula expects exactly this convention:
        #   cos_i = dot(Nrm_i, r_hat_{i->j})  >0 when j is in the outward
        #           hemisphere of surface i
        #   cos_j = -dot(Nrm_j, r_hat_{i->j}) >0 when i is in the outward
        #           hemisphere of surface j
        # No negation is needed because the normals already point correctly.
        if self.dimension == 1:
            VF = np.asarray(self.get_view_factor_1d(all_pts, VM, all_nrm, ds))
        elif self.dimension == 2:
            VF = np.asarray(self.get_view_factor_2d(all_pts, VM, all_nrm, ds))
        else:
            VF = np.asarray(self.get_view_factor_3d(all_pts, VM, all_nrm, ds))

        # ----- 8. Store combined VM for plotting ------------------------------
        enclosure_name = "+".join(tags)
        self.context[f"v_{enclosure_name}"] = VM[None, None, ...]
        self._param_tags.add(f"v_{enclosure_name}")

        # ----- 9. Extract per-tag cross-blocks --------------------------------
        result = []
        row_offset = 0
        for i, tag_i in enumerate(tags):
            row = []
            col_offset = 0
            for j, tag_j in enumerate(tags):
                block = VF[
                    row_offset : row_offset + tag_sizes[i],
                    col_offset : col_offset + tag_sizes[j],
                ]
                key = f"f_{tag_i}__{tag_j}"
                self.context[key] = block[None, None, ...]
                self._param_tags.add(key)
                row.append(TensorTag(tag=key, domain=self))
                col_offset += tag_sizes[j]
            result.append(tuple(row))
            row_offset += tag_sizes[i]

        opaque_info = f", opaque=[{', '.join(opaque_tags)}]" if opaque_tags else ""
        self.log.info(f"Computed enclosure view factor for [{', '.join(tags)}]{opaque_info} ({N_total} total boundary pts)")

        return tuple(result)

    def _sample_one_source(
        self,
        tag: str,
        n_samples: int,
        sampler,
        batch_count: int,
        same_domain: bool,
        mesh_pool: Dict[str, Any],
        normals_by_tag: Dict[str, np.ndarray],
        normals: bool,
    ):
        """Sample *n_samples* points from a single mesh source.

        Returns ``(samples, normals_or_None)`` where shapes are
        ``(batch_count, 1, n_samples, D)`` for steady-state.
        """
        is_time_dep = self._is_time_dependent

        if tag not in mesh_pool:
            return None, None

        available_points = mesh_pool[tag]
        normals_available = tag in normals_by_tag
        if normals_available and normals:
            available_normals = normals_by_tag[tag]

        n_available = available_points.shape[1] if is_time_dep else available_points.shape[0]

        effective_n = n_samples
        if effective_n is None:
            effective_n = n_available
        if effective_n > n_available:
            self.log.warning(
                f"Requested {effective_n} samples for '{tag}' but only "
                f"{n_available} available in sub-domain. Using all points."
            )
            effective_n = n_available

        all_samples = []
        all_normals = []

        if not same_domain:
            for _ in range(batch_count):
                if sampler is not None:
                    if callable(sampler):
                        idx = sampler(available_points, effective_n)
                    elif isinstance(sampler, np.ndarray):
                        idx = sampler
                else:
                    if n_available != effective_n:
                        idx = np.random.choice(n_available, size=effective_n, replace=False)
                    else:
                        idx = np.arange(n_available)

                is_time_expanded = getattr(available_points, "ndim", 0) == 3

                if is_time_expanded:
                    all_samples.append(available_points[:, idx, :])
                else:
                    all_samples.append(available_points[idx])

                if normals_available and normals:
                    all_normals.append(available_normals[idx])

            stacked = np.stack(all_samples, axis=0)
            if not is_time_expanded:
                stacked = stacked[:, np.newaxis, :, :]
            nrm_stacked = None
            if normals_available and normals:
                nrm_stacked = np.stack(all_normals, axis=0)
                if not is_time_expanded:
                    nrm_stacked = nrm_stacked[:, np.newaxis, :, :]
            return stacked, nrm_stacked
        else:
            if sampler is not None:
                idx = sampler(available_points, effective_n)
            else:
                if n_available != effective_n:
                    idx = np.random.choice(n_available, size=effective_n, replace=False)
                else:
                    idx = np.arange(n_available)

            is_time_expanded = getattr(available_points, "ndim", 0) == 3

            if is_time_expanded:
                sampled_pts = available_points[:, idx, :]
            else:
                sampled_pts = available_points[idx][np.newaxis, :, :]

            result = np.broadcast_to(
                sampled_pts[np.newaxis, ...],
                (batch_count, *sampled_pts.shape),
            ).copy()

            nrm_result = None
            if normals_available and normals:
                sampled_nrm = available_normals[idx]
                if not is_time_expanded:
                    sampled_nrm = sampled_nrm[np.newaxis, :, :]
                nrm_result = np.broadcast_to(
                    sampled_nrm[np.newaxis, ...],
                    (batch_count, *sampled_nrm.shape),
                ).copy()

            return result, nrm_result

    def sample(
        self,
        sample_spec: Dict[str, Tuple[int, Optional[Callable]]],
        normals: bool = False,
        return_indices: bool = False,
        time_value: float | None = None,
    ):
        """
        Sample points from the domain.

        Args:
            sample_spec: Dictionary mapping tag names to (n_samples, optional_sampler)
                        Example: {"interior": (2000, None), "boundary": (500, None)}

                        For time-dependent problems, use "initial" to sample points at t=0:
                        Example: {"interior": (2000, None), "initial": (500, None)}

        If domain was batched (e.g., 10 * domain), samples n_samples for each
        batch independently and concatenates results.

        Shapes stored in ``self.context``:

        * **Always**: ``(B, T, N, D_spatial)`` for spatial tags.
          For steady-state problems T=1.
        * **Time-dependent only**: ``(T, 1)`` for ``"__time__"``
          (shared across batches).
        """

        batch_count = self._effective_batch_count()
        is_time_dep = self._is_time_dependent

        def _apply_time_value_to_sampled(tag_out, arr):
            """If time_value is given, reduce (B,T,N,D) to (B,1,N,D)."""
            if time_value is None or not is_time_dep:
                return arr

            t_points = np.asarray(getattr(self, "_time_points", [self.time[0]]), dtype=float)
            tidx = int(np.argmin(np.abs(t_points - float(time_value))))

            arr = np.asarray(arr)
            arr = arr[:, tidx : tidx + 1, :, :]

            # Create a tag-specific time context, e.g. "__time_initial__".
            self.context[f"__time_{tag_out}__"] = np.asarray(
                [[t_points[tidx]]],
                dtype=np.asarray(self.context.get("__time__", np.asarray([[time_value]]))).dtype,
            )

            return arr

        for tag, (n_samples, sampler) in sample_spec.items():
            # Handle special "initial" tag for time-dependent problems
            source_tag = tag
            if tag == "initial" and self._is_time_dependent and self.time is not None:
                if "initial" not in self._mesh_pool and "interior" in self._mesh_pool:
                    source_tag = "interior"

            if source_tag not in self._mesh_pool:
                available = list(self._mesh_pool.keys())
                self.log.error(f"Tag '{tag}' not found. Available: {available}")

            sampling_groups = self._sampling_groups_for_tag(source_tag)
            normals_available = normals and any(group_normals is not None for _, _, group_normals in sampling_groups)

            available_points = sampling_groups[0][1]
            n_available_by_group = []
            for _, group_points, _ in sampling_groups:
                if is_time_dep:
                    n_available_by_group.append(group_points.shape[1])
                else:
                    n_available_by_group.append(group_points.shape[0])
            n_available = min(n_available_by_group)

            ii = 0
            og_tag = tag
            while tag in self.context and tag not in self._param_tags:
                tag = og_tag + f"_{ii}"
                ii += 1

            if n_samples is None:
                n_samples = n_available

            if n_samples > n_available:
                self.log.warning(
                    f"Requested {n_samples} samples for '{tag}' but only {n_available} available across all batches. Using all shared points."
                )
                n_samples = n_available

            all_samples = []
            all_normals = []

            if not self.same_domain:
                for group_count, group_points, group_normals in sampling_groups:
                    group_n_available = group_points.shape[1] if is_time_dep else group_points.shape[0]
                    group_n_samples = min(n_samples, group_n_available)
                    for _ in range(group_count):
                        if sampler is not None:
                            if callable(sampler):
                                idx = sampler(group_points, group_n_samples)
                            elif isinstance(sampler, np.ndarray):
                                idx = sampler
                        else:
                            if group_n_available != group_n_samples:
                                idx = np.random.choice(
                                    group_n_available,
                                    size=group_n_samples,
                                    replace=False,
                                )
                            else:
                                idx = np.arange(group_n_available)

                        if is_time_dep:
                            # Index spatial axis: (T, N, D) → (T, n_samples, D)
                            all_samples.append(group_points[:, idx, :])
                        else:
                            # (N, D) → (n_samples, D)
                            all_samples.append(group_points[idx])

                        if normals_available and group_normals is not None:
                            all_normals.append(group_normals[idx])

                # Stack → (B, T, N, D) for time-dep, (B, N, D) for steady
                stacked = np.stack(all_samples, axis=0)
                if not is_time_dep:
                    # (B, N, D) → (B, 1, N, D)  — T=1 for steady-state
                    stacked = stacked[:, np.newaxis, :, :]
                self.context[tag] = _apply_time_value_to_sampled(tag, stacked)

                if normals_available and all_normals:
                    nrm_stacked = np.stack(all_normals, axis=0)  # (B, N, D)
                    if is_time_dep:
                        # Normals are geometric (time-independent), but the coords are tiled to (B, T, N, D);
                        # tile the normals across T too so the eval's time-scan slices them per step. Without
                        # this they stay (B, N, D) and the scan collapses them to a single point (a silently
                        # wrong, constant boundary flux).
                        t_steps = int(np.asarray(stacked).shape[1])
                        nrm_stacked = np.broadcast_to(
                            nrm_stacked[:, np.newaxis], (nrm_stacked.shape[0], t_steps, *nrm_stacked.shape[1:])
                        ).copy()
                    else:
                        nrm_stacked = nrm_stacked[:, np.newaxis, :, :]  # (B, 1, N, D) — T=1 for steady
                    self.context[f"n_{tag}"] = nrm_stacked

            else:
                # Sample once -> broadcast to all batches
                _, available_points, available_normals = sampling_groups[0]
                if sampler is not None:
                    idx = sampler(available_points, n_samples)
                else:
                    if n_available != n_samples:
                        idx = np.random.choice(n_available, size=n_samples, replace=False)
                    else:
                        idx = np.arange(n_available)

                if is_time_dep:
                    # (T, N, D) → (T, n_samples, D) → broadcast to (B, T, n_samples, D)
                    sampled_pts = available_points[:, idx, :]
                else:
                    # (N, D) → (1, n_samples, D)  — T=1 for steady-state
                    sampled_pts = available_points[idx][np.newaxis, :, :]

                self.context[tag] = np.broadcast_to(
                    sampled_pts[np.newaxis, ...],
                    (batch_count, *sampled_pts.shape),
                )
                self.context[tag] = _apply_time_value_to_sampled(tag, self.context[tag])

                if normals_available and available_normals is not None:
                    sampled_nrm = available_normals[idx]  # (n_samples, D)
                    if is_time_dep:
                        # tile the (time-independent) normals across T so the eval's time-scan slices them
                        # per step (matching the coords); otherwise the scan collapses them to one point.
                        t_steps = int(sampled_pts.shape[0])  # sampled_pts is (T, n_samples, D) here
                        sampled_nrm = np.broadcast_to(sampled_nrm[np.newaxis], (t_steps, *sampled_nrm.shape))
                    else:
                        sampled_nrm = sampled_nrm[np.newaxis, :, :]  # (1, n_samples, D) — T=1 for steady
                    self.context[f"n_{tag}"] = np.broadcast_to(
                        sampled_nrm[np.newaxis, ...],
                        (batch_count, *sampled_nrm.shape),
                    )

            if self._verbose:
                if is_time_dep:
                    tag_arr = np.asarray(self.context[tag])
                    # Time-dependent tag arrays are typically (B, T_tag, N, D).
                    # Use tag-specific T so regions like 'initial' (T=1) log correctly.
                    n_time = int(tag_arr.shape[1]) if tag_arr.ndim >= 3 else 1
                    per_batch_total = n_samples * n_time
                    if batch_count > 1:
                        grand_total = per_batch_total * batch_count
                        self.log.info(
                            f"Sampled {n_samples} spatial points x {n_time} timesteps x {batch_count} batches "
                            f"= {grand_total} spatiotemporal points for '{tag}' with shape {self.context[tag].shape}"
                        )
                    else:
                        self.log.info(
                            f"Sampled {n_samples} spatial points x {n_time} timesteps "
                            f"= {per_batch_total} spatiotemporal points for '{tag}'"
                        )
                    continue
                if batch_count > 1:
                    self.log.info(
                        f"Sampled {n_samples} x {batch_count} = {batch_count * n_samples} points for '{tag}' with shape {self.context[tag].shape}"
                    )
                else:
                    self.log.info(f"Sampled {n_samples} points for '{tag}'")

        if return_indices:
            return self.context[tag], idx, tag
        else:
            return self.context[tag], None, tag

    def plot(
        self,
        save_path: str = "./runs/domain.png",
        figsize: Tuple[int, int] = (10, 8),
        show_normals: bool = True,
        arrow_scale: float = 0.05,
        interactive: bool = False,
    ):
        """Plot the sampled points and normals.

        Args:
            save_path: Output path. For 3D, ``.html`` enables interactive view.
            figsize: Figure size (width, height)
            show_normals: Whether to display normal vectors as arrows
            arrow_scale: Scale factor for normal vector arrows. In 3D this is
                interpreted relative to the tag bounding-box diagonal.
            interactive: If True and in 3D, export interactive Plotly HTML
                (zoom/rotate/pan) with sampled points and normal vectors.
        """
        import os

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if not self.context:
            self.log.warning("No sampled points to plot")
            return

        # Get spatial dimension (exclude time)
        spatial_dim = len(self.spatial)

        def _extract_points(arr):
            """Normalize sampled point arrays to shape (N, D)."""
            a = np.asarray(arr)
            # Typical shapes: (B,T,N,D), (B,N,D), (N,D)
            if a.ndim == 4:
                return a[0, 0]
            if a.ndim == 3:
                return a[0]
            if a.ndim == 2:
                return a
            return None

        def _extract_normals(arr):
            """Normalize sampled normal arrays to shape (N, D)."""
            a = np.asarray(arr)
            # Typical shapes: (B,T,N,D), (B,N,D), (N,D)
            if a.ndim == 4:
                return a[0, 0]
            if a.ndim == 3:
                return a[0]
            if a.ndim == 2:
                return a
            return None

        # Interactive 3D path for easier exploration (zoom/rotate/pan).
        if spatial_dim == 3 and (interactive or save_path.lower().endswith(".html")):
            try:
                import plotly.graph_objects as go
            except Exception as e:
                raise ImportError("plotly is required for interactive 3D plotting") from e

            traces = []
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]

            for i, (tag, points) in enumerate(self.context.items()):
                if tag.startswith("n_") or tag in self._param_tags or tag == "__time__":
                    continue

                pts = _extract_points(points)
                if pts is None or pts.ndim != 2 or pts.shape[-1] < 3:
                    continue

                color = colors[i % len(colors)]
                n_points = pts.shape[0]

                traces.append(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="markers",
                        marker=dict(size=2.5, color=color, opacity=0.75),
                        name=f"{tag} ({n_points})",
                    )
                )

                if show_normals and f"n_{tag}" in self.context:
                    normals = _extract_normals(self.context[f"n_{tag}"])
                    if normals is None or normals.ndim != 2 or normals.shape[-1] < 3:
                        continue

                    m = min(len(pts), len(normals))
                    if m == 0:
                        continue

                    pts_m = pts[:m]
                    nrm_m = normals[:m]

                    # Downsample only for very dense clouds to keep HTML responsive.
                    max_arrows = 8000
                    step = max(1, m // max_arrows)
                    pts_s = pts_m[::step]
                    nrm_s = nrm_m[::step]

                    diag = float(np.linalg.norm(np.ptp(pts_m, axis=0)))
                    scale3d = (arrow_scale * diag) if diag > 0 else arrow_scale

                    x0, y0, z0 = pts_s[:, 0], pts_s[:, 1], pts_s[:, 2]
                    x1 = x0 + scale3d * nrm_s[:, 0]
                    y1 = y0 + scale3d * nrm_s[:, 1]
                    z1 = z0 + scale3d * nrm_s[:, 2]

                    # Draw normals as lightweight line segments.
                    xs = np.empty(3 * len(x0))
                    ys = np.empty(3 * len(y0))
                    zs = np.empty(3 * len(z0))
                    xs[0::3], xs[1::3], xs[2::3] = x0, x1, np.nan
                    ys[0::3], ys[1::3], ys[2::3] = y0, y1, np.nan
                    zs[0::3], zs[1::3], zs[2::3] = z0, z1, np.nan

                    traces.append(
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=0.7,
                            name=f"{tag} normals",
                        )
                    )

            fig = go.Figure(data=traces)
            fig.update_layout(
                title="Sampled Points (Interactive 3D)",
                template="plotly_white",
                scene=dict(
                    xaxis_title=self.spatial[0],
                    yaxis_title=self.spatial[1],
                    zaxis_title=self.spatial[2],
                    aspectmode="data",
                ),
                legend=dict(itemsizing="constant"),
            )

            if not save_path.lower().endswith(".html"):
                root, _ = os.path.splitext(save_path)
                save_path = f"{root}.html"
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.write_html(save_path, include_plotlyjs="cdn")
            self.log.info(f"Saved interactive domain plot to {save_path}")
            return

        # Create figure
        if spatial_dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        # Plot points by tag
        for i, (tag, points) in enumerate(self.context.items()):
            # Skip normal tags, parameter tags, and time tags
            if tag.startswith("n_") or tag in self._param_tags or tag == "__time__":
                continue

            color = colors[i % len(colors)]

            pts = _extract_points(points)
            if pts is None or pts.ndim != 2:
                continue
            n_points = pts.shape[0]

            if spatial_dim == 1:
                # 1D: plot as points on a line
                ax.scatter(
                    pts[:, 0],
                    np.zeros(n_points),
                    c=[color],
                    s=10,
                    alpha=0.7,
                    label=f"{tag} ({n_points})",
                )

                # Plot normals if available
                if show_normals and f"n_{tag}" in self.context:
                    normals = _extract_normals(self.context[f"n_{tag}"])
                    if normals is None or normals.ndim != 2 or normals.shape[-1] < 1:
                        continue
                    for j in range(n_points):
                        ax.arrow(
                            pts[j, 0],
                            0,
                            normals[j, 0] * arrow_scale,
                            0,
                            head_width=0.02,
                            head_length=0.01,
                            fc=color,
                            ec=color,
                            alpha=0.8,
                            linewidth=1.5,
                        )

            elif spatial_dim == 2:
                # 2D: scatter plot
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=[color],
                    s=10,
                    alpha=0.7,
                    label=f"{tag} ({n_points})",
                )

                # Plot normals if available
                if show_normals and f"n_{tag}" in self.context:
                    normals = _extract_normals(self.context[f"n_{tag}"])
                    if normals is None or normals.ndim != 2 or normals.shape[-1] < 2:
                        continue
                    ax.quiver(
                        pts[:, 0],
                        pts[:, 1],
                        normals[:, 0],
                        normals[:, 1],
                        color=color,
                        alpha=0.6,
                        scale=1 / arrow_scale,
                        width=0.003,
                        label=f"{tag} normals",
                    )

            elif spatial_dim == 3:
                # 3D: scatter plot
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    c=[color],
                    s=10,
                    alpha=0.7,
                    label=f"{tag} ({n_points})",
                )

                # Plot normals if available
                if show_normals and f"n_{tag}" in self.context:
                    normals = _extract_normals(self.context[f"n_{tag}"])
                    if normals is None or normals.ndim != 2 or normals.shape[-1] < 3:
                        continue
                    m = min(len(pts), len(normals))
                    diag = float(np.linalg.norm(np.ptp(pts[:m], axis=0)))
                    scale3d = (arrow_scale * diag) if diag > 0 else arrow_scale
                    ax.quiver(
                        pts[:m, 0],
                        pts[:m, 1],
                        pts[:m, 2],
                        normals[:m, 0],
                        normals[:m, 1],
                        normals[:m, 2],
                        color=color,
                        alpha=0.6,
                        length=scale3d,
                        normalize=True,
                        label=f"{tag} normals",
                    )

        # Set labels
        if spatial_dim == 1:
            ax.set_xlabel(self.spatial[0])
            if self._is_time_dependent:
                ax.set_ylabel("time")
            else:
                ax.set_ylabel("(placeholder)")
                ax.set_ylim(-0.1, 0.1)
        elif spatial_dim == 2:
            ax.set_xlabel(self.spatial[0])
            ax.set_ylabel(self.spatial[1])
            ax.set_aspect("equal")
        elif spatial_dim == 3:
            ax.set_xlabel(self.spatial[0])
            ax.set_ylabel(self.spatial[1])
            ax.set_zlabel(self.spatial[2])

        # Time info
        time_info = ""
        if self._is_time_dependent and spatial_dim > 1:
            if "t" in self.context:
                t_vals = self.context["t"]
                if t_vals.ndim == 3:
                    t_vals = t_vals[0, :, 0]
                elif t_vals.ndim == 2:
                    t_vals = t_vals[0, :]
                time_info = f" (t ∈ [{t_vals.min():.3f}, {t_vals.max():.3f}])"

        ax.set_title(f"Sampled Points: {time_info}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        import matplotlib.pyplot as plt  # already imported above, but safe

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.log.info(f"Saved domain plot to {save_path}")

        # --- Visibility fan plots for tags with view factors ---
        vf_tags = [k[2:] for k in self.context if k.startswith("v_")]
        if vf_tags and spatial_dim == 2:
            self._plot_visibility_fans(save_path, vf_tags, figsize=figsize)

    def _plot_visibility_fans(self, base_save_path: str, tags, figsize=(10, 8), n_show: int = 25):
        """Plot visibility fans for boundary tags that have view factors.

        Combines all boundary tags into a single 5x5 grid. For each source
        point, draws lines to every visible boundary point across all tags.

        Args:
            base_save_path: Base path for saving (``_visibility`` is appended)
            tags: List of tag names that have visibility matrices
            figsize: Ignored (fixed 5x5 layout)
            n_show: Total number of source points across all tags (default 25)
        """
        import os

        import matplotlib.pyplot as plt

        # Collect all boundary data across tags
        tag_data = []  # list of (tag_label, pts, VM)
        for tag in tags:
            vm_key = f"v_{tag}"
            if vm_key not in self.context:
                continue

            VM = np.asarray(self.context[vm_key])
            while VM.ndim > 2:
                VM = VM[0]

            # Combined enclosure tag (e.g. "interior_boundary+interior_boundary_outer")
            if "+" in tag:
                sub_tags = tag.split("+")
                sub_pts = []
                for st in sub_tags:
                    if st not in self.context:
                        continue
                    p = np.asarray(self.context[st])
                    if p.ndim == 4:
                        p = p[0, 0]
                    elif p.ndim == 3:
                        p = p[0]
                    sub_pts.append(p)
                if not sub_pts:
                    continue
                pts = np.concatenate(sub_pts, axis=0)
            else:
                if tag not in self.context:
                    continue
                pts = np.asarray(self.context[tag])
                if pts.ndim == 4:
                    pts = pts[0, 0]
                elif pts.ndim == 3:
                    pts = pts[0]

            if pts.shape[0] > 0:
                tag_data.append((tag, pts, VM))

        if not tag_data:
            return

        # Collect all boundary points for background rendering
        all_pts = np.concatenate([pts for _, pts, _ in tag_data], axis=0)

        # Distribute n_show slots across tags proportionally to point count
        total_bnd = sum(pts.shape[0] for _, pts, _ in tag_data)
        source_specs = []  # list of (tag, pts, VM, local_idx)
        remaining = n_show
        for i, (tag, pts, VM) in enumerate(tag_data):
            if i == len(tag_data) - 1:
                n_tag = remaining
            else:
                n_tag = max(1, int(round(n_show * pts.shape[0] / total_bnd)))
                remaining -= n_tag
            n_tag = min(n_tag, pts.shape[0])
            indices = np.linspace(0, pts.shape[0] - 1, n_tag, dtype=int)
            for idx in indices:
                source_specs.append((tag, pts, VM, idx))

        n_total = len(source_specs)
        ncols, nrows = 5, 5
        n_total = min(n_total, nrows * ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

        tag_names = ", ".join(t for t, _, _ in tag_data)
        fig.suptitle(
            f"Visibility Fans — {tag_names}  ({total_bnd} boundary pts)",
            fontsize=14,
            y=1.01,
        )

        for i, ax in enumerate(axes.flat):
            if i >= n_total:
                ax.set_visible(False)
                continue

            tag, pts, VM, idx = source_specs[i]
            n_bnd = pts.shape[0]

            visible = np.where(VM[idx] == 1)[0]
            visible = visible[visible != idx]

            # Draw all boundary points from all tags as light background
            ax.scatter(
                all_pts[:, 0],
                all_pts[:, 1],
                c="lightgrey",
                s=6,
                zorder=1,
                edgecolors="none",
            )

            # Lines to visible points
            for j in visible:
                ax.plot(
                    [pts[idx, 0], pts[j, 0]],
                    [pts[idx, 1], pts[j, 1]],
                    color="lime",
                    alpha=0.15,
                    lw=0.5,
                    zorder=2,
                )

            # Visible points
            ax.scatter(
                pts[visible, 0],
                pts[visible, 1],
                c="green",
                s=12,
                zorder=3,
                edgecolors="none",
            )
            # Source point
            ax.scatter(
                pts[idx, 0],
                pts[idx, 1],
                c="red",
                marker="*",
                s=150,
                zorder=5,
                edgecolors="k",
                linewidths=0.5,
            )

            n_vis = len(visible)
            ax.set_title(f"{tag} i={idx}, sees {n_vis}/{n_bnd - 1}", fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)

        fig.tight_layout()

        base, ext = os.path.splitext(base_save_path)
        fan_path = f"{base}_visibility{ext}"
        fig.savefig(fan_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log.info(f"Saved visibility fan plot to {fan_path}")
