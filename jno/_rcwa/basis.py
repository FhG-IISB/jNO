"""Owned lattice/expansion utilities for the RCWA backend.

This is jNO's own implementation of the fmmax basis-convention subset that
``jno.rcwa`` consumes: lattice vectors, expansion generation and transverse
wavevectors. The math and conventions -- expansion sort order (stable |G|
sort, zeroth order first), the reciprocal-lattice convention and the index
semantics of `transverse_wavevectors` -- match fmmax 1.7.1 exactly, so the
seam swap keeps parity; numerical parity is pinned by
``tests/test_rcwa_backend_basis.py`` with fmmax as a test-only oracle.

Unlike fmmax, which runs part of the expansion generation through jnp,
`generate_expansion` is entirely eager host-side numpy: the lattice vectors
are concrete at engine build and the resulting coefficient grid is cached, so
the traced solve path never touches numpy control flow.
"""

import dataclasses
import enum
import operator
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as onp


@dataclasses.dataclass
class LatticeVectors:
    """A pair of 2-vectors, for the real-space or the reciprocal-space lattice.

    Attributes:
        u: The first primitive lattice vector, shape ``(..., 2)``.
        v: The second primitive lattice vector, same shape as ``u``.
    """

    u: jnp.ndarray
    v: jnp.ndarray

    def __post_init__(self) -> None:
        _validate_lattice_vector(self.u, "u")
        _validate_lattice_vector(self.v, "v")

    @property
    def reciprocal(self) -> "LatticeVectors":
        """The reciprocal lattice (jnp, differentiable in the lattice)."""
        return _reciprocal(self)


@enum.unique
class Truncation(enum.Enum):
    """How Fourier orders are truncated."""

    #: Orders within a circle whose area matches the requested term count.
    CIRCULAR = "circular"

    #: Orders within a parallelogram matched to the reciprocal spacing.
    PARALLELOGRAMIC = "parallelogramic"


@dataclasses.dataclass
class Expansion:
    """The integer reciprocal-lattice coefficients of the Fourier expansion.

    `basis_coefficients` is stored as a plain numpy int array so that it can
    serve as a static pytree aux node (hashable/equatable) -- the expansion is
    a property of the truncation geometry, fixed at engine build.

    Attributes:
        basis_coefficients: Integer coefficients ``(num_terms, 2)`` of the
            primitive reciprocal-lattice vectors, sorted by magnitude with the
            zeroth order first.
        num_terms: The number of terms in the expansion.
    """

    basis_coefficients: onp.ndarray

    def __post_init__(self) -> None:
        coeffs = self.basis_coefficients
        if onp.asarray(coeffs).ndim != 2 or onp.asarray(coeffs).shape[-1] != 2:
            raise ValueError(f"`basis_coefficients` must have shape (num_terms, 2) but got {onp.shape(coeffs)}.")
        if not onp.issubdtype(onp.asarray(coeffs).dtype, onp.integer):
            raise ValueError(f"`basis_coefficients` must be an integer array but got dtype {onp.asarray(coeffs).dtype}.")

    def __hash__(self) -> int:
        return hash(self.basis_coefficients.tobytes())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Expansion):
            return False
        if self.basis_coefficients.shape != other.basis_coefficients.shape:
            return False
        return bool(onp.all(self.basis_coefficients == other.basis_coefficients))

    @property
    def num_terms(self) -> int:
        """The number of terms in the expansion."""
        return self.basis_coefficients.shape[-2]


def min_array_shape_for_expansion(expansion: Expansion) -> Tuple[int, int]:
    """The minimum grid shape ``(nx, ny)`` whose FFT resolves every term.

    A grid with fewer points than ``2 * max|coefficient| + 1`` per axis cannot
    represent the highest requested order without aliasing.
    """
    coeffs = expansion.basis_coefficients
    return (
        int(2 * onp.amax(onp.abs(coeffs[:, 0])) + 1),
        int(2 * onp.amax(onp.abs(coeffs[:, 1])) + 1),
    )


def validate_shape_for_expansion(shape: Tuple[int, ...], expansion: Expansion) -> None:
    """Raises if `shape`'s final two axes are insufficient for `expansion`."""
    min_shape = min_array_shape_for_expansion(expansion)
    if len(shape) < 2 or any(d < dmin for d, dmin in zip(shape[-2:], min_shape)):
        raise ValueError(
            f"`shape` with final two axes {shape[-2:]} is insufficient for "
            f"`expansion`; the minimum final-two shape is {min_shape}."
        )


def generate_expansion(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
    truncation: Truncation = Truncation.CIRCULAR,
) -> Expansion:
    """Generates the expansion for the specified real-space lattice.

    Runs eagerly on the host with pure numpy; `approximate_num_terms` must be
    an integer and the lattice vectors must be concrete. The returned
    coefficients are sorted by magnitude (stable sort), with the zeroth order
    first. To keep the expansion symmetric the term count may differ from
    `approximate_num_terms`.

    Args:
        primitive_lattice_vectors: The primitive vectors of the real-space
            lattice, each of shape ``(2,)``.
        approximate_num_terms: The approximate number of terms.
        truncation: The truncation scheme, CIRCULAR by default.

    Returns:
        The ``Expansion``.

    Note:
        Bit-for-bit ordering parity with fmmax is guaranteed except for a
        single order sitting exactly on the truncation circle of a skewed,
        non-axis-aligned lattice: there numpy's interop with jnp arrays can
        round the boundary magnitude differently within fmmax itself. The
        engine's axis-aligned lattices are unaffected, and the term set is
        always identical.
    """
    try:
        approximate_num_terms = operator.index(approximate_num_terms)
    except TypeError as e:
        raise ValueError(f"`approximate_num_terms` must be an integer, got {approximate_num_terms!r}.") from e
    if approximate_num_terms <= 0:
        raise ValueError(f"`approximate_num_terms` must be positive, got {approximate_num_terms}.")

    # The coefficient generators work with the reciprocal lattice, as in fmmax.
    # Convert to host numpy: `generate_expansion` is an eager build-time step.
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    u = onp.asarray(reciprocal_vectors.u)
    v = onp.asarray(reciprocal_vectors.v)
    if u.ndim != 1 or u.shape != (2,) or v.ndim != 1 or v.shape != (2,):
        raise ValueError(
            f"generate_expansion requires lattice vectors of shape (2,), but got "
            f"{u.shape} and {v.shape}. An expansion is generated once per concrete "
            f"lattice at engine build."
        )

    if truncation == Truncation.CIRCULAR:
        basis_coefficients = _basis_coefficients_circular(u, v, approximate_num_terms)
    elif truncation == Truncation.PARALLELOGRAMIC:
        basis_coefficients = _basis_coefficients_parallelogramic(u, v, approximate_num_terms)
    else:
        raise ValueError(f"Unknown `truncation`, got {truncation}.")
    return Expansion(basis_coefficients)


def _grid_float_dtype(lattice_dtype: onp.dtype) -> onp.dtype:
    """The dtype fmmax's grid arithmetic ends up in.

    fmmax multiplies its integer coefficient grid by the (float) reciprocal
    lattice through JAX promotion: with x64 disabled an int64 grid is
    canonicalized to int32 first (so a ``float32`` lattice gives float32
    products), while with x64 enabled int64 promotes with any float to
    float64. The magnitude ordering depends on the last ulp of these rounded
    values (exactly-tied lattice vectors differ only by rounding), so the
    ordered coefficient grid is identical only when this dtype is reproduced.

    NumPy's own ``promote_types`` cannot be used here: NEP 50 raises
    ``int32 + float32`` to float64, whereas JAX keeps float32, and matching
    that last-ulp behavior is what guarantees ordering parity.
    """
    return onp.dtype(jnp.promote_types(onp.int64, onp.dtype(lattice_dtype)))


def transverse_wavevectors(
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
) -> jnp.ndarray:
    """The transverse wavevectors for all expansion terms.

    The zeroth-order term sits at `in_plane_wavevector`; each other term is
    shifted by the corresponding reciprocal-lattice vector.

    Args:
        in_plane_wavevector: The zeroth-order transverse wavevector, shape
            ``(..., 2)``.
        primitive_lattice_vectors: The real-space lattice vectors.
        expansion: The expansion defining the term ordering.

    Returns:
        The transverse wavevectors, shape ``batch + (num_terms, 2)`` where
        ``batch`` is the broadcast of the leading dims of `in_plane_wavevector`
        and the lattice vectors.
    """
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    kx = in_plane_wavevector[..., 0, jnp.newaxis] + 2 * jnp.pi * (
        expansion.basis_coefficients[:, 0] * reciprocal_vectors.u[..., 0, jnp.newaxis]
        + expansion.basis_coefficients[:, 1] * reciprocal_vectors.v[..., 0, jnp.newaxis]
    )
    ky = in_plane_wavevector[..., 1, jnp.newaxis] + 2 * jnp.pi * (
        expansion.basis_coefficients[:, 0] * reciprocal_vectors.u[..., 1, jnp.newaxis]
        + expansion.basis_coefficients[:, 1] * reciprocal_vectors.v[..., 1, jnp.newaxis]
    )
    batch_shape = jnp.broadcast_shapes(
        in_plane_wavevector.shape[:-1],
        primitive_lattice_vectors.u.shape[:-1],
    )
    assert kx.shape == batch_shape + (expansion.num_terms,)
    return jnp.stack([kx, ky], axis=-1)


# ------------------------------------------------------------------------------
# Basis-coefficient generation (eager host-side numpy).
# ------------------------------------------------------------------------------


def _basis_coefficients_circular(u: onp.ndarray, v: onp.ndarray, approximate_num_terms: int) -> onp.ndarray:
    """Coefficients of reciprocal vectors inside the area-matched circle.

    All vectors with magnitude below ``sqrt(num_terms * |u x v| / pi)`` are
    kept, so the set has approximately `approximate_num_terms` members. The
    grid arithmetic runs in fmmax's promoted dtype (see `_grid_float_dtype`),
    and the magnitudes use plain numpy norms as fmmax's ``np.linalg.norm`` on
    the jnp grid does (it does not dispatch into jnp). Rounding-sensitive ties
    in the sort therefore match fmmax bit for bit.
    """
    dtype = _grid_float_dtype(u.dtype)
    g = onp.arange(-(approximate_num_terms // 2), approximate_num_terms // 2 + 1)
    G1, G2 = onp.meshgrid(g, g, indexing="ij")
    G1 = G1.flatten()
    G2 = G2.flatten()
    # Magnitudes run in fmmax's promoted dtype; the coefficient grid stays integer.
    G1f = G1.astype(dtype)
    G2f = G2.astype(dtype)
    u = u.astype(dtype)
    v = v.astype(dtype)

    vectors = G1f[:, onp.newaxis] * u[onp.newaxis, :] + G2f[:, onp.newaxis] * v[onp.newaxis, :]
    magnitude = onp.linalg.norm(vectors, axis=-1)

    # fmmax divides by ``np.pi`` inside jnp, which treats the numpy scalar as a
    # weak type: with x64 disabled the whole ``max_magnitude`` chain stays
    # float32 (numpy would instead promote to float64). Matching that dtype is
    # what decides a coincidentally-boundary order, so use a dtype-matched pi.
    pi = onp.float32(onp.pi) if dtype == onp.float32 else onp.float64(onp.pi)
    max_magnitude = onp.sqrt(approximate_num_terms * onp.abs(_cross(u, v)) / pi)
    mask = magnitude < max_magnitude

    G1 = G1[mask]
    G2 = G2[mask]
    magnitude = magnitude[mask]
    G = onp.stack([G1, G2], axis=-1)

    order = onp.argsort(magnitude, kind="stable")
    return G[order, :]


def _basis_coefficients_parallelogramic(u: onp.ndarray, v: onp.ndarray, approximate_num_terms: int) -> onp.ndarray:
    """Coefficients of reciprocal vectors inside the matched parallelogram.

    Chooses ``(nu, nv)`` with ``(2*nu + 1) * (2*nv + 1) ~= approximate_num_terms``
    and ``nu / nv = |v| / |u|`` (in the reciprocal basis), so that the sides of
    the parallelogram of lattice points have equal length in k-space. Grid
    arithmetic runs in fmmax's promoted dtype as in `_basis_coefficients_circular`.
    """
    dtype = _grid_float_dtype(u.dtype)
    u = u.astype(dtype)
    v = v.astype(dtype)

    ku_spacing = onp.sqrt(onp.sum(onp.abs(u) ** 2))
    kv_spacing = onp.sqrt(onp.sum(onp.abs(v) ** 2))

    def _solve_quadratic(ratio):
        a = 4 * ratio
        b = 2 * (ratio + 1)
        c = 1 - approximate_num_terms
        nu = (-b + onp.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return int(onp.around(nu))

    nu = _solve_quadratic(ku_spacing / kv_spacing)
    nv = _solve_quadratic(kv_spacing / ku_spacing)

    G1, G2 = onp.meshgrid(
        onp.arange(-nu, nu + 1),
        onp.arange(-nv, nv + 1),
        indexing="ij",
    )
    G1 = G1.flatten()
    G2 = G2.flatten()
    G = onp.stack([G1, G2], axis=-1)

    G1f = G1.astype(dtype)
    G2f = G2.astype(dtype)
    vectors = G1f[:, onp.newaxis] * u[onp.newaxis, :] + G2f[:, onp.newaxis] * v[onp.newaxis, :]
    magnitude = onp.linalg.norm(vectors, axis=-1)

    order = onp.argsort(magnitude, kind="stable")
    return G[order, :]


def _cross(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """The signed area spanned by the 2-vectors `x` and `y` (``x[0]y[1]-x[1]y[0]``)."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _reciprocal(lattice_vectors: LatticeVectors) -> LatticeVectors:
    """Computes the reciprocal lattice pair (jnp, differentiable in the lattice).

    Follows the fmmax convention: ``u' = (v_y, -v_x) / (u x v)`` and
    ``v' = (-u_y, u_x) / (u x v)`` with the 2-D cross product, giving the two
    vectors that generate the reciprocal lattice.
    """
    cross_product = _cross(lattice_vectors.u, lattice_vectors.v)
    uprime = jnp.stack([lattice_vectors.v[..., 1], -lattice_vectors.v[..., 0]], axis=-1) / cross_product[..., jnp.newaxis]
    vprime = jnp.stack([-lattice_vectors.u[..., 1], lattice_vectors.u[..., 0]], axis=-1) / cross_product[..., jnp.newaxis]
    return LatticeVectors(u=uprime, v=vprime)


def _validate_lattice_vector(x, name: str) -> None:
    """Checks that `x` has a trailing dimension of 2 (allowing leading batch axes)."""
    shape = jnp.shape(x)
    if len(shape) == 0 or shape[-1] != 2:
        raise ValueError(f"`{name}` must have a trailing dimension of 2, but got shape {shape}.")


# ------------------------------------------------------------------------------
# Pytree registration: LatticeVectors is a (differentiable) leaf container,
# Truncation and Expansion are static aux nodes.
# ------------------------------------------------------------------------------


class _HashableArray:
    """Hashable wrapper for a numpy array, used as a static pytree aux node."""

    def __init__(self, value: onp.ndarray):
        self.value = value

    def __hash__(self) -> int:
        return hash((self.value.shape, self.value.dtype, self.value.tobytes()))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _HashableArray):
            return False
        return (
            self.value.shape == other.value.shape
            and self.value.dtype == other.value.dtype
            and bool(onp.all(self.value == other.value))
        )


def _unflatten_expansion(aux_tuple: Tuple[_HashableArray], leaves: Tuple) -> Expansion:
    """Reconstructs an `Expansion` from its static aux node."""
    del leaves
    return Expansion(basis_coefficients=aux_tuple[0].value)


jax.tree_util.register_pytree_node(
    LatticeVectors,
    lambda lv: ((lv.u, lv.v), None),
    lambda _, uv: LatticeVectors(*uv),
)

jax.tree_util.register_pytree_node(
    Truncation,
    lambda x: ((), x.value),
    lambda value, _: Truncation(value),
)

jax.tree_util.register_pytree_node(
    Expansion,
    lambda e: ((), (_HashableArray(e.basis_coefficients),)),
    unflatten_func=_unflatten_expansion,
)
