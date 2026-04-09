from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable, Any, Union

import cloudpickle
import jax
import jax.numpy as jnp
import meshio
import numpy as np

from ..trace import Variable, TensorTag, Literal, BinaryOp, FunctionCall, Jacobian, TestFunction, TrialFunction, Constant, FemLinearSystem, Assembly
from ..utils.logger import get_logger
from .boundary_region import BoundaryRegion
from .domain_data import DomainData
from .geometries import Geometries
from .meshio_mixin import MeshIOMixin


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
    def rect(
        cls,
        x_range=(0, 1),
        y_range=(0, 1),
        mesh_size=0.1,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a rectangular domain.

        Args:
            x_range: Inclusive spatial extent along x.
            y_range: Inclusive spatial extent along y.
            mesh_size: Target mesh size passed to pygmsh.
            algorithm: Gmsh meshing algorithm.
            time: Optional tuple of ``(start, end, n_steps)`` for time-dependent problems.
            compute_mesh_connectivity: Whether to precompute mesh connectivity metadata.
        """
        return cls._from_geometry(
            Geometries.rect(x_range=x_range, y_range=y_range, mesh_size=mesh_size),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def polygon(
        cls,
        vertices,
        mesh_size=0.1,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a 2D polygon domain from an arbitrary list of vertices."""
        return cls._from_geometry(
            Geometries.polygon(vertices=vertices, mesh_size=mesh_size),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def equi_distant_rect(
        cls,
        x_range=(0, 1),
        y_range=(0, 1),
        nx=10,
        ny=10,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a structured rectangular triangulation."""
        return cls._from_geometry(
            Geometries.equi_distant_rect(x_range=x_range, y_range=y_range, nx=nx, ny=ny),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def poseidon(
        cls,
        nx: int = 128,
        ny: int = 128,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate the structured Poseidon-style 2D grid."""
        return cls._from_geometry(
            Geometries.poseidon(nx=nx, ny=ny),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def cube(
        cls,
        x_range=(0, 1),
        y_range=(0, 1),
        z_range=(0, 1),
        mesh_size=0.1,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a cubic 3D domain."""
        return cls._from_geometry(
            Geometries.cube(x_range=x_range, y_range=y_range, z_range=z_range, mesh_size=mesh_size),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def disk(
        cls,
        center=(0, 0),
        radius=1.0,
        mesh_size=0.1,
        num_points=32,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a polygonal disk domain."""
        return cls._from_geometry(
            Geometries.disk(center=center, radius=radius, mesh_size=mesh_size, num_points=num_points),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def l_shape(
        cls,
        size=1.0,
        mesh_size=0.1,
        separate_boundary=False,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate an L-shaped domain."""
        return cls._from_geometry(
            Geometries.l_shape(size=size, mesh_size=mesh_size, separate_boundary=separate_boundary),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def rectangle_with_hole(
        cls,
        outer_size=1.0,
        hole_size=0.4,
        mesh_size=0.1,
        separate_boundary=False,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a rectangle with a single rectangular hole."""
        return cls._from_geometry(
            Geometries.rectangle_with_hole(
                outer_size=outer_size,
                hole_size=hole_size,
                mesh_size=mesh_size,
                separate_boundary=separate_boundary,
            ),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def rect_pml(
        cls,
        x_range=(0, 1),
        y_range=(0, 1),
        mesh_size=0.1,
        pml_thickness_top=0.2,
        pml_thickness_bottom=0.2,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a rectangle with top and bottom PML regions."""
        return cls._from_geometry(
            Geometries.rect_pml(
                x_range=x_range,
                y_range=y_range,
                mesh_size=mesh_size,
                pml_thickness_top=pml_thickness_top,
                pml_thickness_bottom=pml_thickness_bottom,
            ),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    @classmethod
    def rectangle_with_holes(
        cls,
        outer_size=(2.0, 1.0),
        holes=None,
        mesh_size=0.1,
        separate_boundary=True,
        *,
        algorithm: int = 6,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: bool = True,
    ) -> "domain":
        """Instantiate a rectangle with multiple rectangular holes."""
        return cls._from_geometry(
            Geometries.rectangle_with_holes(
                outer_size=outer_size,
                holes=holes,
                mesh_size=mesh_size,
                separate_boundary=separate_boundary,
            ),
            algorithm=algorithm,
            time=time,
            compute_mesh_connectivity=compute_mesh_connectivity,
        )

    def __init__(
        self,
        constructor: Union[Callable, str, "domain", None] = None,
        algorithm: Optional[int] = None,
        time: Optional[Tuple[float, float, int]] = None,
        compute_mesh_connectivity: Optional[bool] = None,
    ):
        """
        Initialize the domain.

        Args:
            constructor: Function accepting a pygmsh.geo.Geometry object, an existing domain,
                or a path to a meshfile
            algorithm: Gmsh meshing algorithm
            time: Tuple of (start, end) for time-dependent problems
            mesh_connectivity: Wether or not to compute the some hyperparameters about the mesh (needed for finite_difference methods)
        """
        if isinstance(constructor, domain):
            existing_domain = constructor
            if algorithm is None:
                algorithm = getattr(existing_domain, "_algorithm", 6)
            if time is None:
                time = existing_domain.time
            if compute_mesh_connectivity is None:
                compute_mesh_connectivity = existing_domain.compute_mesh_connectivity
            constructor = getattr(existing_domain, "_constructor_source", None)

        if algorithm is None:
            algorithm = 6
        if compute_mesh_connectivity is None:
            compute_mesh_connectivity = True

        super().__init__()
        self.log = get_logger()
        self._algorithm = algorithm
        self._constructor_source = constructor

        # Storage
        self.compute_mesh_connectivity = compute_mesh_connectivity
        self._mesh_pool: Dict[str, Any] = {}  # full mesh vertices per tag (M, D)
        self.context: Dict[str, Any] = {}  # unified: spatial (B,N,D) + params (B,F)
        self._param_tags: set = set()  # tags that are parametric (TensorTag)
        self.normals_by_tag: Dict[str, np.ndarray] = {}
        self._boundary_registry: Dict[str, Dict[str, Any]] = {}
        self._tag_edges: Dict[str, np.ndarray] = {}
        self._tag_triangles: Dict[str, np.ndarray] = {}
        self._boundary_regions: Dict[str, BoundaryRegion] = {}
        # self._boundary_predicates: Dict[str, Callable] = {}

        # Neural operator storage
        self.parameters: Dict[str, Any] = {}
        self.arrays: Dict[str, np.ndarray] = {}
        self.tag_indices: Dict[str, np.ndarray] = {}
        self.avaiable_mesh_tags: List[str] = []  # names of the tags from the mesh generator
        self._boundary_loop_tags: set = set()  # tags extracted from line cells (boundary loops)
        self.mesh_connectivity: Optional[Dict[str, Any]] = None  # precomputed mesh connectivity data
        # Resampling support
        self._mesh_points: Dict[str, np.ndarray] = {}  # Full mesh points for resampling
        self._resampling_strategies: Dict[str, Any] = {}  # Tag -> ResamplingStrategy

        # Configuration
        self.dimension: int = 2
        self.total_samples: int = 1
        self.time = time
        self._is_time_dependent = time is not None
        self._verbose = True
        self.same_domain = False

        # Tracking
        self.index_tags: List[str] = []
        self.normal_tags: List[str] = []
        self.reference_solutions: List[Callable] = []
        self.sample_dict: List = []

        # Meshio mesh
        self.mesh = None

        # Generate or load mesh
        if isinstance(constructor, str):
            self._load_mesh(constructor)
            self.log.info(f"Loaded mesh from the constructor function: {len(self.mesh.points)} points")  # type: ignore[attr-defined]
        elif callable(constructor):
            self._generate_mesh(constructor, algorithm)
            self.log.info(f"Loaded mesh from {constructor}: {len(self.mesh.points)} points")  # type: ignore[attr-defined]
        else:
            raise ValueError("Must provide either geometry_func or mesh_file")

        boundary_indices = self._extract_points_from_mesh(self.mesh)

        # Preprocess mesh for finite differences
        if self.mesh is not None and self.compute_mesh_connectivity:
            self.mesh_connectivity, msg = self._preprocess_mesh_connectivity(self.mesh, self.dimension, boundary_indices)
            self.log.info(msg)

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
            self.arrays[name] = value.astype(np.float32)
            self.log.info(f"Attached array '{name}' with shape {value.shape}")
        elif isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=np.float32)
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
        self._batch_count = n
        self.total_samples = n
        self.same_domain = True
        return self

    def __mul__(self, n: int) -> "domain":
        """Batch the domain n times: domain * 2 samples 2x independently."""
        return self.__rmul__(n)

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

    def __add__(self, other: "domain") -> "domain":
        """Merge another domain into this one (stacks along batch dimension).

        For time-dependent problems, ``_mesh_pool`` entries have shape
        ``(T, N, D_spatial)`` and do **not** get stacked (they represent the
        same spatial mesh).  ``context`` entries are concatenated along the
        batch axis (axis 0).  ``"__time__"`` is shared and not stacked.
        """
        for tag, points in other._mesh_pool.items():
            if tag not in self._mesh_pool:
                self._mesh_pool[tag] = points
            # else: keep self's mesh pool (same mesh)

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
            self.parameters[name] = np.array(values, dtype=np.float32)

        # Keep batch metadata consistent after domain stacking.
        self_batch = int(getattr(self, "_batch_count", getattr(self, "total_samples", 1)))
        other_batch = int(getattr(other, "_batch_count", getattr(other, "total_samples", 1)))
        self._batch_count = max(1, self_batch) + max(1, other_batch)
        self.total_samples = self._batch_count
        self.same_domain = False

        return self

    # FEM/ variational interface

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
            from ..utils.fem_route import dirichlet as _dirichlet_bc
        except ImportError as e:
            raise ImportError("FEM support is not available. Install the FEM/dev extras to use " "domain.dirichlet(...) and init_fem(...).") from e
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
            from ..utils.fem_route import neumann as _neumann_bc
        except ImportError as e:
            raise ImportError("FEM support is not available. Install the FEM/dev extras to use " "domain.neumann(...) and init_fem(...).") from e
        return _neumann_bc(tags)

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
                    raise ValueError(f"Dirichlet BC for tag '{tag}' has {len(spec)} component functions, " f"but vec={vec}.")
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

    def variational_symbols(self, value_shape=(), names=("u", "phi")):
        """
        Return generic variational symbols.

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
        return (
            TrialFunction(name=trial_name, value_shape=value_shape),
            TestFunction(name=test_name, value_shape=value_shape),
        )

    def fem_symbols(self, value_shape=(), names=("u", "phi")):
        """
        Backward-compatible alias for variational_symbols().

        Examples
        --------
        Scalar:
            u, phi = domain.fem_symbols()

        Vector:
            u, v = domain.fem_symbols(value_shape=(2,))
        """
        return self.variational_symbols(value_shape=value_shape, names=names)

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

    def init_fem(
        self,
        element_type: str = "TRI3",
        quad_degree: int = 2,
        neumann_tags: List[str] = [],
        dirichlet_tags: List[str] = [],
        dirichlet_value_fns: dict | None = None,
        fem_solver: bool = False,
        vec: int = 1,
        bcs=None,
    ) -> "domain":
        """
        Initialize the JAX-FEM data associated with this domain.

        This sets up the FEM mesh, boundary-condition data, quadrature data,
        and cached tensors needed for weak-form assembly and FEM solves.

        Parameters
        ----------
        element_type : str, default="TRI3"
            Finite-element type used by JAX-FEM.
        quad_degree : int, default=2
            Quadrature degree for volume and boundary integration.
        neumann_tags : list[str], optional
            Boundary tags used for Neumann or surface terms.
        dirichlet_tags : list[str], optional
            Boundary tags with Dirichlet constraints.
        dirichlet_value_fns : dict | None, optional
            Prescribed Dirichlet value functions by boundary tag.
        fem_solver : bool, default=False
            Whether to enable FEM-solver mode.
        vec : int, default=1
            Number of field components.
        bcs : list | None, optional
            New-style boundary-condition descriptors.

        Returns
        -------
        domain
            The current domain with FEM context initialized.
        """
        if self.mesh is None:
            raise ValueError("Mesh must be loaded before initializing FEM context.")
        self._variational_initialized = True
        self._variational_sampling_registry = {}
        import jax.numpy as jnp
        import numpy as onp
        from jax_fem.problem import Problem
        from jax_fem.generate_mesh import Mesh
        from scipy.spatial import KDTree  # Ensuring this is available locally
        from ..utils.fem_route import expand_bcs

        if bcs is not None:
            try:
                from ..utils.fem_route import expand_bcs
            except ImportError as e:
                raise ImportError("FEM support is not available. Install the FEM/dev extras to use init_fem(...).") from e

            if dirichlet_tags or neumann_tags or dirichlet_value_fns is not None:
                raise ValueError("Use either 'bcs=[...]' or the legacy " "'dirichlet_tags/neumann_tags/dirichlet_value_fns' arguments, not both.")

            dirichlet_tags, dirichlet_value_fns, neumann_tags = expand_bcs(bcs, vec=vec)
        meshio_type_map = {
            "TRI3": "triangle",
            "QUAD4": "quad",
            "TET4": "tetra",
        }
        meshio_type = meshio_type_map.get(element_type)
        jax_mesh = Mesh(self.mesh.points[:, : self.dimension], self.mesh.cells_dict[meshio_type])

        # --- Location functions for Neumann ---
        location_fns = []
        valid_tags = []

        for tag in neumann_tags:
            loc_fn = self._make_tag_location_fn(tag)
            if loc_fn is None:
                self.log.warning(f"Neumann tag '{tag}' not found in mesh tags. Skipping.")
                continue

            valid_tags.append(tag)
            location_fns.append(loc_fn)
        dirichlet_bc_info = self._build_dirichlet_bc_info(
            dirichlet_tags,
            dirichlet_value_fns,
            vec=vec,
        )

        class DummyProblem(Problem):
            def get_tensor_map(self):
                return lambda x: x

            def get_mass_map(self):
                return lambda x: x

            def get_surface_maps(self):
                return [lambda u, x: jnp.zeros((1,))] * len(location_fns)

        prob = DummyProblem(
            jax_mesh,
            vec=vec,
            dim=self.dimension,
            ele_type=element_type,
            gauss_order=quad_degree,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns,
        )
        self._fem_solver_enabled = bool(fem_solver)
        self._jaxfem_solver_context = {
            "mesh": jax_mesh,
            "element_type": element_type,
            "quad_degree": quad_degree,
            "location_fns": location_fns,
            "valid_neumann_tags": list(valid_tags),
            "dirichlet_tags": list(dirichlet_tags),
            "dirichlet_bc_info": dirichlet_bc_info,
            "dummy_problem": prob,
            "dim": self.dimension,
            "default_vec": vec,
        }

        fe = prob.fes[0]

        # --- Identify Dirichlet node indices ---
        dirichlet_node_ids: List[int] = []
        for node_inds in getattr(fe, "node_inds_list", []):
            dirichlet_node_ids.extend(np.asarray(node_inds).reshape(-1).tolist())

        dirichlet_nodes = jnp.array(sorted(set(dirichlet_node_ids)), dtype=jnp.int32) if dirichlet_node_ids else jnp.array([], dtype=jnp.int32)
        # --- Precompute Constants for Fast Assembly ---
        shape_vals_jax = jnp.asarray(fe.shape_vals)
        shape_grads_jax = jnp.asarray(fe.shape_grads)
        JxW_jax = jnp.asarray(fe.JxW)
        cells_jax = jnp.asarray(fe.cells, dtype=jnp.int32)
        flat_cells = cells_jax.flatten()

        # Precompute Volume Normalization Areas (runs ONLY ONCE!)
        local_areas = jnp.einsum("cq,qa->ca", JxW_jax, shape_vals_jax)
        global_areas = jax.ops.segment_sum(local_areas.flatten(), flat_cells, num_segments=fe.num_total_nodes)
        num_cells, num_quads, num_local_nodes, dim = shape_grads_jax.shape
        N_flat = jnp.tile(shape_vals_jax[None, ...], (num_cells, 1, 1)).reshape(-1, num_local_nodes)
        dN_dx_flat = shape_grads_jax.reshape(-1, num_local_nodes, dim)

        # Extract standard Volume Context
        self.fem_context = {
            "cells": cells_jax,
            "flat_cells": flat_cells,
            "global_areas": global_areas,
            "N_flat": N_flat,
            "dN_dx_flat": dN_dx_flat,
            "JxW": JxW_jax,
            "num_total_nodes": fe.num_total_nodes,
            "boundary_nodes": jnp.asarray(self._extract_points_from_mesh(self.mesh), dtype=jnp.int32),
            "dirichlet_nodes": dirichlet_nodes,
            "surface_data": {},
        }
        self._mesh_pool["fem_gauss"] = jnp.asarray(fe.get_physical_quad_points()).reshape(-1, self.dimension)
        self._register_variational_sample(
            sample_tag="fem_gauss",
            support="volume",
            region_id="volume",
            context_tag="fem_gauss",
        )
        self._fem_dirichlet_tags = list(dirichlet_tags)
        self._fem_neumann_tags = list(neumann_tags)
        self._fem_dirichlet_value_fns = dirichlet_value_fns if dirichlet_value_fns is not None else {}
        # --- Extract Native jax-fem Nanson Scales for Neumann ---
        for i, tag in enumerate(valid_tags):
            inds = prob.boundary_inds_list[i]
            if len(inds) > 0:
                _, nanson_scale = fe.get_face_shape_grads(inds)

                parent_cells = fe.cells[inds[:, 0]]
                face_ids = inds[:, 1]
                face_shape_vals = fe.face_shape_vals[face_ids]

                physical_coos = onp.take(fe.points, fe.cells, axis=0)
                selected_coos = physical_coos[inds[:, 0]]
                physical_face_quads = onp.einsum("fqn,fnd->fqd", face_shape_vals, selected_coos)

                # Precompute boundary normalization areas for this Neumann tag
                local_boundary_areas = jnp.einsum("fq,fqn->fn", jnp.asarray(nanson_scale), jnp.asarray(face_shape_vals))
                global_boundary_areas = jax.ops.segment_sum(local_boundary_areas.flatten(), jnp.asarray(parent_cells, dtype=jnp.int32).flatten(), num_segments=fe.num_total_nodes)

                quad_pts_flat_np = onp.asarray(physical_face_quads).reshape(-1, self.dimension)
                quad_pts_flat = jnp.asarray(quad_pts_flat_np)

                # Map quadrature points to boundary normals using nearest tagged boundary point.
                quad_normals = None
                if tag in self.normals_by_tag and tag in self._mesh_pool:
                    tag_pts_np = onp.asarray(self._mesh_pool[tag])[:, : self.dimension]
                    tag_nrm_np = onp.asarray(self.normals_by_tag[tag])[:, : self.dimension]

                    if len(tag_pts_np) == len(tag_nrm_np) and len(tag_pts_np) > 0:
                        tree = KDTree(tag_pts_np)
                        _, nn_idx = tree.query(quad_pts_flat_np)
                        quad_normals_np = tag_nrm_np[onp.asarray(nn_idx, dtype=int)]
                        quad_normals = jnp.asarray(quad_normals_np)

                        # expose normals on the quadrature tag so domain.variable("gauss_tag", normals=True) works
                        self.normals_by_tag[f"gauss_{tag}"] = quad_normals_np

                self.fem_context["surface_data"][tag] = {
                    "flat_parent_nodes": jnp.asarray(parent_cells, dtype=jnp.int32).flatten(),
                    "face_shape_vals": jnp.asarray(face_shape_vals),
                    "nanson_scale": jnp.asarray(nanson_scale),
                    "global_boundary_areas": global_boundary_areas,
                    "quad_points": quad_pts_flat,
                    "quad_normals": quad_normals,
                }

                self._mesh_pool[f"gauss_{tag}"] = quad_pts_flat
                self._register_variational_sample(
                    sample_tag=f"gauss_{tag}",
                    support="boundary",
                    region_id=tag,
                    context_tag=f"gauss_{tag}",
                )
                self.log.info(f"jax-fem Nanson extraction: Matched {len(inds)} faces for '{tag}'")
        # ====================================================================
        # THE FIX: Pad FEM arrays with Batch (B=1) and Time (T=1) dimensions
        # This prevents trace_compiler from mistaking the spatial/node
        # dimensions for the Batch dimension during vmap.
        # ====================================================================

        keys_to_pad = ["cells", "flat_cells", "global_areas", "N_flat", "dN_dx_flat", "JxW", "boundary_nodes", "dirichlet_nodes"]
        for key in keys_to_pad:
            if key in self.fem_context and hasattr(self.fem_context[key], "ndim"):
                self.fem_context[key] = jnp.expand_dims(self.fem_context[key], axis=(0, 1))

        # Also pad the nested jax-fem surface/neumann arrays if they exist
        for tag_name, s_data in self.fem_context.get("surface_data", {}).items():
            for skey, s_arr in s_data.items():
                if hasattr(s_arr, "ndim"):
                    s_data[skey] = jnp.expand_dims(s_arr, axis=(0, 1))
                # Expose the integration arrays to jNO's trace evaluator
        self.context.update(self.fem_context)

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
        region = self._boundary_regions.get(tag, None)
        if region is None:
            return None
        return lambda p: region.contains(p)

    def assemble_weak_form(self, expr, target="vpinn", **kwargs):
        """
        Assemble a symbolic weak form for the requested backend.

        Parameters
        ----------
        expr : object
            Symbolic weak-form expression.
        target : str, default="vpinn"
            Assembly backend or lowering target.
        **kwargs
            Additional backend-specific options.

        Returns
        -------
        object
            Assembled backend-specific representation of the weak form.
        """
        from ..utils.weak_form import assemble_weak_form

        return assemble_weak_form(self, expr, target=target, **kwargs)

    # Generators
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

    def _extract_points_from_mesh(self, mesh):
        """Extract points and normals from mesh and organize by tag."""
        index_to_normal_pos = {}
        points = mesh.points[:, : self.dimension]
        self.points = points
        self._mesh_pool = {}
        self._boundary_registry = {}
        self.tag_indices = {}
        self._tag_edges = {}
        self._tag_triangles = {}
        self._boundary_regions = {}

        if self.dimension > 1:
            boundary_normals, boundary_indices = self.get_boundary_normals(mesh)
            boundary_normals = boundary_normals[:, : self.dimension]
            index_to_normal_pos = {int(idx): int(pos) for pos, idx in enumerate(boundary_indices)}
        else:
            left_boundary = np.where(points[:, 0] == np.min(points[:, 0]))[0]
            right_boundary = np.where(points[:, 0] == np.max(points[:, 0]))[0]

            boundary_indices = np.stack([left_boundary, right_boundary]).flatten()
            index_to_normal_pos = {int(idx): int(pos) for pos, idx in enumerate(boundary_indices)}

            boundary_normals = np.array([[-1], [1]])

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

                tag_points = set()
                tag_edges = []
                tag_tris = []

                if isinstance(cell_data, dict):
                    for cell_type, indices in cell_data.items():
                        if len(indices) == 0:
                            continue

                        # Handle vertex (point) cells specially
                        if cell_type == "vertex":
                            for b_idx, cell_block in enumerate(mesh.cells):
                                if cell_block.type == "vertex":
                                    for idx in indices:
                                        local_idx = int(idx) - block_offsets.get((b_idx, "vertex"), 0)
                                        if 0 <= local_idx < len(cell_block.data):
                                            # vertex data contains the point index
                                            point_idx = int(cell_block.data[local_idx].flatten()[0])
                                            tag_points.add(point_idx)
                        else:
                            for b_idx, cell_block in enumerate(mesh.cells):
                                if cell_block.type == cell_type:
                                    offset = block_offsets.get((b_idx, cell_type), 0)
                                    for idx in indices:
                                        local_idx = int(idx) - offset
                                        if 0 <= local_idx < len(cell_block.data):
                                            cell = cell_block.data[local_idx]
                                            tag_points.update(cell.flatten())
                                            if cell_block.type == "line":
                                                tag_edges.append(tuple(cell))
                                            elif cell_block.type == "triangle":
                                                tag_tris.append(tuple(cell))
                else:
                    # Handle list-style cell_data
                    for block_idx, indices in enumerate(cell_data):
                        if indices is None or len(indices) == 0:
                            continue
                        if block_idx < len(mesh.cells):
                            cell_block = mesh.cells[block_idx]

                            if cell_block.type == "vertex":
                                for idx in indices:
                                    if 0 <= idx < len(cell_block.data):
                                        point_idx = int(cell_block.data[idx].flatten()[0])
                                        tag_points.add(point_idx)
                            else:
                                for idx in indices:
                                    if 0 <= idx < len(cell_block.data):
                                        cell = cell_block.data[idx]
                                        tag_points.update(cell.flatten())
                                        if cell_block.type == "line":
                                            tag_edges.append(tuple(cell))
                                        elif cell_block.type == "triangle":
                                            tag_tris.append(tuple(cell))

                if tag_tris:
                    self._tag_triangles[name] = np.array(tag_tris, dtype=int)
                if tag_edges:
                    self._tag_edges[name] = np.array(tag_edges, dtype=int)

                if tag_points:
                    if tag_edges:
                        self._boundary_loop_tags.add(name)
                        indices_list = self._chain_edges_to_loop(tag_edges)
                    else:
                        indices_list = np.array(sorted(tag_points), dtype=int)

                    indices_list = np.asarray(indices_list, dtype=int)
                    self.tag_indices[name] = indices_list
                    self._mesh_pool[name] = points[indices_list]

                    normal_positions = np.array([index_to_normal_pos[i] for i in indices_list if i in index_to_normal_pos])
                    if len(normal_positions) > 0:
                        self.normals_by_tag[name] = boundary_normals[normal_positions]
                    else:
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
                    elif name in self.normals_by_tag:
                        # fallback: still treat as a boundary-like tag if normals exist
                        is_boundary_tag = True
                        entity_kind = "boundary_points"

                    if is_boundary_tag:
                        edge_coords = None
                        tri_coords = None

                        if len(tag_edges) > 0:
                            edge_arr = np.asarray(tag_edges, dtype=int)
                            edge_coords = points[edge_arr][:, :, : self.dimension]

                        if len(tag_tris) > 0:
                            tri_arr = np.asarray(tag_tris, dtype=int)
                            tri_coords = points[tri_arr][:, :, : self.dimension]

                        # pred = self._boundary_predicates.get(name, None)
                        tol = self._estimate_boundary_tol(points[indices_list][:, : self.dimension])

                        self._boundary_regions[name] = BoundaryRegion(
                            tag=name,
                            dim=self.dimension,
                            points=points[indices_list][:, : self.dimension],
                            edges=edge_coords,
                            triangles=tri_coords,
                            tol=tol,
                        )

                        self._boundary_registry[name] = {
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
            # points has shape (N, D_spatial)

            if tag == "interior":
                # Initial tag: spatial points at t=0 → (1, N, D_spatial)
                new_mesh_pool["initial"] = points[np.newaxis, :, :]  # T=1

            # Tile spatial points across T time steps → (T, N, D_spatial)
            new_mesh_pool[tag] = np.broadcast_to(
                points[np.newaxis, :, :],  # (1, N, D_spatial)
                (n_time, *points.shape),  # (T, N, D_spatial)
            ).copy()  # copy so it's contiguous

        self._mesh_pool = new_mesh_pool
        # NOTE: self.dimension stays as D_spatial — time is a separate axis

        # Store time array in context so Variable("__time__") can be created.
        # Shape: (T, 1) — will be broadcast to (B, T, 1) during sample().
        self.context["__time__"] = self._time_points[:, np.newaxis]  # (T, 1)

        return None

    def add_tensor_tag(self, name: str, tensor: Union[np.ndarray, jnp.ndarray]) -> "domain":
        """Attach a tensor to this domain for parametric PDEs.

        Tensor tags allow parameters to vary across batched domains.
        The first dimension is the batch dimension and must match the
        domain's batch count exactly.

        Args:
            name: Name for this tensor (used in vars(name))
            tensor: Array with shape (B, ...) where B equals the domain's batch count.

        Returns:
            Self for method chaining.

        Example:
            domain = 2 * domain.from_mesh(...)
            domain.add_tensor_tag('diffusivity', jnp.array([[1.0], [2.0]]))  # shape (2, 1)
            a = domain.variable('diffusivity')  # Returns TensorTag
        """
        tensor = jnp.asarray(tensor)
        if tensor.ndim < 1:
            tensor = tensor.reshape(1, 1)

        # Validate batch dimension - must match exactly
        batch_count = self._effective_batch_count()
        tensor_batch = tensor.shape[0]
        if tensor_batch != batch_count:
            self.log.warning(f"Tensor '{name}' has batch dimension {tensor_batch}, but domain has " f"effective batch count {batch_count}. Was this intended?")

        self.context[name] = tensor
        self._param_tags.add(name)
        return self

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
        return_indices=False,
    ) -> Any:
        """Create Variable placeholders for a tagged point set or tensor.

        Args:
            tag: Name of the point set (e.g., 'interior', 'boundary')
                 or tensor tag (e.g., 'diffusivity')
            sample: Optional sampling specification for this tag:
                    - (n_samples, sampler) tuple to trigger sampling
                    - jax.numpy array to register a tensor tag

            resampling_strategy: Optional ResamplingStrategy for adaptive point selection
            normals: If True, also compute and return normal vectors for this tag
            reverse_normals: If True, flip the sign of the normal vectors
            return_indices: Wether or not to return the indices of the sampled points

        Returns:
            For point sets: Tuple of Variable placeholders, one per dimension.
            For point sets with normals=True: Tuple of variables + normal variables.
            For tensor tags: Single TensorTag placeholder.
        """

        # Optional sampling / tensor-tag attachment
        if sample is not None:
            if isinstance(sample, jnp.ndarray) or isinstance(sample, np.ndarray):
                # Attach as tensor tag (parameter field) or point data
                if point_data:
                    self.context[tag] = sample
                else:
                    self.add_tensor_tag(tag, sample)

        if tag in self._mesh_pool.keys() and isinstance(sample, tuple) and len(sample) > 0 and isinstance(sample[0], (int, type(None))):
            # Sample points for this tag on demand
            # Save sample dict for inference
            self.sample_dict.append([tag, (None, None), resampling_strategy, normals, view_factor])
            points, idx, tag = self.sample({tag: sample}, normals, return_indices)

        # Store resampling strategy if provided
        if resampling_strategy is not None:
            self._resampling_strategies[tag] = resampling_strategy

        # Check if it's a parametric (TensorTag) entry
        if tag in self._param_tags:
            if split:
                return tuple(TensorTag(tag=tag, dim_index=i, domain=self) for i in range(sample.shape[-1]))  # type: ignore[attr-defined]
            else:
                return TensorTag(tag=tag, domain=self)

        if point_data:
            if split:
                return tuple(Variable(tag=tag, dim=[i, i + 1], domain=self) for i in range(sample.shape[-1]))  # type: ignore[attr-defined]
            else:
                return Variable(tag=tag, dim=[0, None], domain=self)

        if tag not in self.context:
            available = list(self.context.keys())
            raise ValueError(f"Tag '{tag}' not found. Did you call sample() first? Available: {available}")

        fem_meta = None
        if getattr(self, "_variational_initialized", False):
            fem_meta = getattr(self, "_variational_sampling_registry", {}).get(tag, None)

        # Create Variable placeholder for each spatial dimension
        coord_vars: List[Any] = [Variable(tag=tag, dim=[i, i + 1], domain=self, axis="spatial", fem_meta=fem_meta) for i in range(self.dimension)]

        # Always add temporal variable (constant 1 for stationary problems)
        coord_vars.append(
            Variable(
                tag="__time__",
                dim=[0, 1],
                domain=self,
                axis="temporal",
                fem_meta=None,
            )
        )

        if normals:
            if reverse_normals:
                self.context[f"n_{tag}"] = -self.context[f"n_{tag}"]
            coord_vars += [Variable(tag=f"n_{tag}", dim=[i, i + 1], domain=self) for i in range(len(self.spatial))]

        if view_factor and hasattr(self, "mesh_connectivity"):

            # Only take the first batch index
            Nrm = -self.context[f"n_{tag}"][0, ...]  # Reverse the normals
            P = points[0, ...]

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
                edges = np.array([[i, (i + 1) % len(ordered_bp)] for i in range(len(ordered_bp))], dtype=np.int32)
                ordered_VM = self.get_visibility_matrix_raytrace(ordered_bp, edges, n_ray_samples=3)
                # Map VM back to original point order
                inv_order = np.argsort(order)
                subset_VM = np.asarray(ordered_VM)[np.ix_(inv_order, inv_order)]

            if self.dimension == 1:
                VF = self.get_view_factor_1d(P[0], subset_VM, Nrm[0], ds)
            if self.dimension == 2:
                VF = self.get_view_factor_2d(P[0], subset_VM, Nrm[0], ds)
            elif self.dimension == 3:
                VF = self.get_view_factor_3d(P[0], subset_VM, Nrm[0], ds)

            # TODO: Fix if used for multiple domains !
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

    def __getitem__(self, tag: str) -> Tuple[Variable, ...]:
        """Shorthand for domain.variable(tag).

        Example:
            x, y = domain['interior']
        """
        return self.variable(tag)

    @staticmethod
    def _chain_edges_to_loop(edges):
        """Chain a set of (a, b) edge pairs into an ordered loop.

        Args:
            edges: List of 2-tuples (global point indices) forming a closed loop.

        Returns:
            np.ndarray of global point indices in loop order.
        """
        from collections import defaultdict

        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        # Walk the graph
        start = edges[0][0]
        visited = {start}
        order = [start]
        current = start
        for _ in range(len(edges) - 1):
            for nb in adj[current]:
                if nb not in visited:
                    visited.add(nb)
                    order.append(nb)
                    current = nb
                    break
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
        from collections import Counter

        edge_count = Counter()
        for tri in triangles:
            for i in range(3):
                e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
                edge_count[e] += 1
        return [e for e, c in edge_count.items() if c == 1]

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
                self.log.warning(f"Opaque tag '{otag}' has no line or triangle cells. " f"Available boundary loops: {sorted(self._boundary_loop_tags)}, " f"volume tags: {sorted(self._tag_triangles.keys())}. Skipping.")
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
                self.log.info(f"Opaque: kept {kept_loops} loop(s), " f"skipped {skipped_loops} coincident loop(s)")
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
                self.log.info(f"Blocked {n_blocked} self-visible pairs through solid " f"interior for loop {loop_idx} ({tags[loop_idx]})")
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
                block = VF[row_offset : row_offset + tag_sizes[i], col_offset : col_offset + tag_sizes[j]]
                key = f"f_{tag_i}__{tag_j}"
                self.context[key] = block[None, None, ...]
                self._param_tags.add(key)
                row.append(TensorTag(tag=key, domain=self))
                col_offset += tag_sizes[j]
            result.append(tuple(row))
            row_offset += tag_sizes[i]

        opaque_info = f", opaque=[{', '.join(opaque_tags)}]" if opaque_tags else ""
        self.log.info(f"Computed enclosure view factor for [{', '.join(tags)}]{opaque_info} " f"({N_total} total boundary pts)")

        return tuple(result)

    def sample(self, sample_spec: Dict[str, Tuple[int, Optional[Callable]]], normals: bool = False, return_indices: bool = False):
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

        for tag, (n_samples, sampler) in sample_spec.items():
            # Handle special "initial" tag for time-dependent problems
            if tag not in self._mesh_pool:
                available = list(self._mesh_pool.keys())
                self.log.error(f"Tag '{tag}' not found. Available: {available}")

            normals_avaiable = tag in self.normals_by_tag

            available_points = self._mesh_pool[tag]
            if normals_avaiable and normals:
                available_normals = self.normals_by_tag[tag]  # Pull the normals for this tag

            # n_available is the number of *spatial* points
            if is_time_dep:
                # _mesh_pool[tag] has shape (T, N, D_spatial)
                n_available = available_points.shape[1]
            else:
                # _mesh_pool[tag] has shape (N, D_spatial)
                n_available = available_points.shape[0]

            ii = 0
            og_tag = tag
            while tag in self.context and tag not in self._param_tags:
                tag = og_tag + f"_{ii}"
                ii += 1

            # Store full mesh points for resampling
            if tag not in self._mesh_points:
                self._mesh_points[tag] = available_points.copy()

            if n_samples is None:
                n_samples = n_available

            if n_samples > n_available:
                self.log.warning(f"Requested {n_samples} samples for '{tag}' but only {n_available} available. Using all points.")
                n_samples = n_available

            all_samples = []
            all_normals = []

            if not self.same_domain:
                for _ in range(batch_count):
                    if sampler is not None:
                        if callable(sampler):
                            idx = sampler(available_points, n_samples)
                        elif isinstance(sampler, np.ndarray):
                            idx = sampler
                    else:
                        if n_available != n_samples:
                            idx = np.random.choice(n_available, size=n_samples, replace=False)
                        else:
                            idx = np.arange(n_available)

                    if is_time_dep:
                        # Index spatial axis: (T, N, D) → (T, n_samples, D)
                        all_samples.append(available_points[:, idx, :])
                    else:
                        # (N, D) → (n_samples, D)
                        all_samples.append(available_points[idx])

                    if normals_avaiable and normals:
                        all_normals.append(available_normals[idx])

                # Stack → (B, T, N, D) for time-dep, (B, N, D) for steady
                stacked = np.stack(all_samples, axis=0)
                if not is_time_dep:
                    # (B, N, D) → (B, 1, N, D)  — T=1 for steady-state
                    stacked = stacked[:, np.newaxis, :, :]
                self.context[tag] = stacked

                if normals_avaiable and normals:
                    nrm_stacked = np.stack(all_normals, axis=0)
                    if not is_time_dep:
                        nrm_stacked = nrm_stacked[:, np.newaxis, :, :]
                    self.context[f"n_{tag}"] = nrm_stacked

            else:
                # Sample once -> broadcast to all batches
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

                if normals_avaiable and normals:
                    sampled_nrm = available_normals[idx]
                    if not is_time_dep:
                        sampled_nrm = sampled_nrm[np.newaxis, :, :]
                    self.context[f"n_{tag}"] = np.broadcast_to(
                        sampled_nrm[np.newaxis, ...],
                        (batch_count, *sampled_nrm.shape),
                    )

            if self._verbose:
                if batch_count > 1:
                    self.log.info(f"Sampled {n_samples} x {batch_count} = {batch_count * n_samples} points for '{tag}' with shape {self.context[tag].shape}")
                else:
                    self.log.info(f"Sampled {n_samples} points for '{tag}'")

        if return_indices:
            return self.context[tag], idx, tag
        else:
            return self.context[tag], None, tag

    def plot(self, save_path: str = "./runs/domain.png", figsize: Tuple[int, int] = (10, 8), show_normals: bool = True, arrow_scale: float = 0.05, interactive: bool = False):
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
                ax.scatter(pts[:, 0], np.zeros(n_points), c=[color], s=10, alpha=0.7, label=f"{tag} ({n_points})")

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
                ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=10, alpha=0.7, label=f"{tag} ({n_points})")

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
        import matplotlib.pyplot as plt
        import os

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
        fig.suptitle(f"Visibility Fans — {tag_names}  ({total_bnd} boundary pts)", fontsize=14, y=1.01)

        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        for i, ax in enumerate(axes.flat):
            if i >= n_total:
                ax.set_visible(False)
                continue

            tag, pts, VM, idx = source_specs[i]
            n_bnd = pts.shape[0]

            visible = np.where(VM[idx] == 1)[0]
            not_visible = np.where(VM[idx] == 0)[0]
            visible = visible[visible != idx]

            # Draw all boundary points from all tags as light background
            ax.scatter(all_pts[:, 0], all_pts[:, 1], c="lightgrey", s=6, zorder=1, edgecolors="none")

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
            ax.scatter(pts[visible, 0], pts[visible, 1], c="green", s=12, zorder=3, edgecolors="none")
            # Source point
            ax.scatter(pts[idx, 0], pts[idx, 1], c="red", marker="*", s=150, zorder=5, edgecolors="k", linewidths=0.5)

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
