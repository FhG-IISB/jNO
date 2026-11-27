"""The Fourier basis of a doubly periodic unit cell.

A cell with primitive lattice vectors ``a1``, ``a2`` has the reciprocal basis ``b1``, ``b2`` with
``a_i . b_j = delta_ij``, so every Floquet-Bloch harmonic has the transverse wavevector

    k_mn = k_in + 2 pi (m b1 + n b2)

for integers ``(m, n)``. An :class:`Expansion` is the truncated, ordered set of those integer pairs: sorted by
``|m b1 + n b2|`` with the zeroth order first, so that index 0 is always the specular order. Two truncations are
offered:

* circular (the default): all orders inside the k-space circle whose area holds the requested number of terms
  (each order occupies one reciprocal cell of area ``|b1 x b2|``); this keeps the resolution isotropic;
* parallelogramic: ``|m| <= M``, ``|n| <= N`` with ``(2M + 1)(2N + 1)`` close to the requested count and
  ``M |b1| ~ N |b2|``, the set most other RCWA codes use (needed to compare with them).

The expansion is built eagerly on the host from the concrete lattice and is a static pytree node, so the traced
solve never sees it as data. The lattice vectors themselves are an ordinary (differentiable) pytree.
"""

import dataclasses
import enum
import operator
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as onp

#: Tie tolerance on |G|, relative to the larger reciprocal vector. Orders whose |G| differ by less are treated as
#: tied: they keep the enumeration order (m major, then n, both ascending). Symmetry-equivalent orders then tie
#: exactly also when the lattice was given in float32 (0.7 is 0.69999999 there, which moves |G| by about 1e-7), so
#: the order is the same in c64 and c128; orders that close are interchangeable for the truncation anyway.
_TIE = 1e-6


@dataclasses.dataclass
class LatticeVectors:
    """The two primitive vectors ``u``, ``v`` of a 2-D lattice (each of shape ``(..., 2)``)."""

    u: jnp.ndarray
    v: jnp.ndarray

    def __post_init__(self) -> None:
        for name, x in (("u", self.u), ("v", self.v)):
            shape = jnp.shape(x)
            if len(shape) == 0 or shape[-1] != 2:
                raise ValueError(f"`{name}` must have a trailing dimension of 2, but got shape {shape}.")

    @property
    def reciprocal(self) -> "LatticeVectors":
        """The reciprocal basis ``(b1, b2)`` with ``a_i . b_j = delta_ij`` (without the factor ``2 pi``)."""
        area = self.u[..., 0] * self.v[..., 1] - self.u[..., 1] * self.v[..., 0]
        b1 = jnp.stack([self.v[..., 1], -self.v[..., 0]], axis=-1) / area[..., None]
        b2 = jnp.stack([-self.u[..., 1], self.u[..., 0]], axis=-1) / area[..., None]
        return LatticeVectors(u=b1, v=b2)


@enum.unique
class Truncation(enum.Enum):
    """How the set of Fourier orders is truncated."""

    CIRCULAR = "circular"
    PARALLELOGRAMIC = "parallelogramic"


@dataclasses.dataclass(eq=False)
class Expansion:
    """The ordered integer orders ``(m, n)`` of a truncated Fourier expansion.

    Attributes:
        basis_coefficients: ``(num_terms, 2)`` integers, sorted by ``|m b1 + n b2|``, the zeroth order first.
    """

    basis_coefficients: onp.ndarray

    def __post_init__(self) -> None:
        c = onp.asarray(self.basis_coefficients)
        if c.ndim != 2 or c.shape[-1] != 2:
            raise ValueError(f"`basis_coefficients` must have shape (num_terms, 2) but got {c.shape}.")
        if not onp.issubdtype(c.dtype, onp.integer):
            raise ValueError(f"`basis_coefficients` must be an integer array but got dtype {c.dtype}.")
        self.basis_coefficients = c

    def __hash__(self) -> int:
        return hash((self.basis_coefficients.shape, self.basis_coefficients.tobytes()))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Expansion)
            and self.basis_coefficients.shape == other.basis_coefficients.shape
            and bool(onp.all(self.basis_coefficients == other.basis_coefficients))
        )

    @property
    def num_terms(self) -> int:
        """The number of orders."""
        return self.basis_coefficients.shape[0]


def min_array_shape_for_expansion(expansion: Expansion) -> Tuple[int, int]:
    """The smallest real-space grid ``(nx, ny)`` that represents every order without aliasing: an FFT of ``N``
    points holds the harmonics ``-floor(N/2) .. floor((N-1)/2)``, so ``N >= 2 max|m| + 1``."""
    c = expansion.basis_coefficients
    return int(2 * onp.max(onp.abs(c[:, 0])) + 1), int(2 * onp.max(onp.abs(c[:, 1])) + 1)


def validate_shape_for_expansion(shape: Tuple[int, ...], expansion: Expansion) -> None:
    """Raises if the last two axes of ``shape`` are too small for ``expansion``."""
    need = min_array_shape_for_expansion(expansion)
    if len(shape) < 2 or any(d < dmin for d, dmin in zip(shape[-2:], need)):
        raise ValueError(
            f"`shape` with final two axes {shape[-2:]} is insufficient for `expansion`; the minimum final-two "
            f"shape is {need}."
        )


def generate_expansion(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
    truncation: Truncation = Truncation.CIRCULAR,
) -> Expansion:
    """The truncated order set for a concrete real-space lattice (host-side numpy, at build).

    Both truncations are symmetric under ``(m, n) -> (-m, -n)``, so the term count may differ slightly from
    ``approximate_num_terms``.
    """
    try:
        count = operator.index(approximate_num_terms)
    except TypeError as e:
        raise ValueError(f"`approximate_num_terms` must be an integer, got {approximate_num_terms!r}.") from e
    if count <= 0:
        raise ValueError(f"`approximate_num_terms` must be positive, got {count}.")
    rec = primitive_lattice_vectors.reciprocal
    b1 = onp.asarray(rec.u, dtype=onp.float64)
    b2 = onp.asarray(rec.v, dtype=onp.float64)
    if b1.shape != (2,) or b2.shape != (2,):
        raise ValueError(
            f"generate_expansion requires lattice vectors of shape (2,), but got {b1.shape} and {b2.shape}. An "
            "expansion is generated once per concrete lattice at engine build."
        )
    if truncation == Truncation.CIRCULAR:
        # count reciprocal cells of area |b1 x b2| fill the circle of radius sqrt(count |b1 x b2| / pi); an order
        # within the tie tolerance of the circle counts as on it and is left out, whatever the precision
        half = count // 2
        m, n = _grid(half, half)
        radius = onp.sqrt(count * abs(b1[0] * b2[1] - b1[1] * b2[0]) / onp.pi)
        keep = _magnitude(m, n, b1, b2) < radius - _TIE * max(onp.linalg.norm(b1), onp.linalg.norm(b2))
        m, n = m[keep], n[keep]
    elif truncation == Truncation.PARALLELOGRAMIC:
        # (2M+1)(2N+1) ~ count with M |b1| = N |b2|: M = r N, r = |b2|/|b1|
        ratio = onp.linalg.norm(b2) / onp.linalg.norm(b1)
        m, n = _grid(_parallelogram_extent(count, ratio), _parallelogram_extent(count, 1.0 / ratio))
    else:
        raise ValueError(f"Unknown `truncation`, got {truncation}.")
    return Expansion(onp.stack([m, n], axis=-1)[_sorted_with_ties(_magnitude(m, n, b1, b2), b1, b2)])


def transverse_wavevectors(
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
) -> jnp.ndarray:
    """``k_in + 2 pi (m b1 + n b2)`` for every order, shape ``batch + (num_terms, 2)``.

    ``batch`` broadcasts the leading axes of ``in_plane_wavevector`` (shape ``(..., 2)``) and the lattice vectors.
    """
    rec = primitive_lattice_vectors.reciprocal
    c = jnp.asarray(expansion.basis_coefficients)
    g = c[:, 0, None] * rec.u[..., None, :] + c[:, 1, None] * rec.v[..., None, :]  # (..., num_terms, 2)
    return in_plane_wavevector[..., None, :] + 2 * jnp.pi * g


def _sorted_with_ties(mag: onp.ndarray, b1, b2) -> onp.ndarray:
    """Indices sorting ``mag`` ascending, where runs of values within the tie tolerance of their neighbour form one
    tied group that keeps the input (enumeration) order. Grouping by consecutive gaps, not by rounding, so that no
    pair of nearly equal values is split by a rounding boundary."""
    idx = onp.argsort(mag, kind="stable")
    gaps = onp.diff(mag[idx]) > _TIE * max(onp.linalg.norm(b1), onp.linalg.norm(b2))
    group = onp.concatenate([[0], onp.cumsum(gaps)])
    return idx[onp.lexsort((idx, group))]


def _grid(m_max: int, n_max: int) -> Tuple[onp.ndarray, onp.ndarray]:
    m, n = onp.meshgrid(onp.arange(-m_max, m_max + 1), onp.arange(-n_max, n_max + 1), indexing="ij")
    return m.ravel(), n.ravel()


def _magnitude(m, n, b1, b2) -> onp.ndarray:
    return onp.linalg.norm(m[:, None] * b1[None, :] + n[:, None] * b2[None, :], axis=-1)


def _parallelogram_extent(count: int, ratio: float) -> int:
    """The half-width ``M`` (rounded) solving ``(2M + 1)(2M / ratio + 1) = count``, i.e.
    ``4 M^2 + 2 (ratio + 1) M + ratio (1 - count) = 0`` after multiplying by ``ratio``."""
    a, b, c = 4.0, 2.0 * (ratio + 1.0), ratio * (1.0 - count)
    return int(onp.around((-b + onp.sqrt(b * b - 4 * a * c)) / (2 * a)))


# -- pytree registration: the lattice is a differentiable leaf pair, the truncation and expansion are static. --


class _Static:
    """A static pytree aux holder comparing and hashing like the expansion it carries."""

    def __init__(self, expansion: Expansion):
        self.expansion = expansion

    def __hash__(self) -> int:
        return hash(self.expansion)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Static) and self.expansion == other.expansion


jax.tree_util.register_pytree_node(LatticeVectors, lambda lv: ((lv.u, lv.v), None), lambda _, uv: LatticeVectors(*uv))
jax.tree_util.register_pytree_node(Truncation, lambda t: ((), t.value), lambda value, _: Truncation(value))
jax.tree_util.register_pytree_node(Expansion, lambda e: ((), _Static(e)), lambda aux, _: aux.expansion)
