from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .fem_utils import (
    _normalize_dirichlet_value,
)

# --------------------------------
# FEM boundary-condition helpers
# --------------------------------


@dataclass(frozen=True)
class DirichletBC:
    """
    Essential boundary-condition descriptor for FEM assembly.

    Instances are created through `dirichlet(...)` and later normalized by
    `expand_bcs(...)` during `domain.init_fem(...)`.

    Parameters
    ----------
    tags:
        Boundary tag names on which the Dirichlet condition is applied.
    values:
        Boundary value specification. Supported forms are handled by
        `_normalize_dirichlet_value(...)` and include `None`, scalars,
        callables, component lists/tuples, and component dictionaries.
    """

    tags: tuple[str, ...]
    values: object = None


@dataclass(frozen=True)
class NeumannBC:
    """
    Natural boundary-condition descriptor for FEM assembly.

    Instances are created through `neumann(...)`. The tags mark boundary
    regions whose weak-form boundary terms should be included in surface
    assembly.

    Parameters
    ----------
    tags:
        Boundary tag names treated as natural/surface regions.
    """

    tags: tuple[str, ...]


@dataclass(frozen=True)
class PeriodicBC:
    """
    Periodic boundary-condition descriptor.

    Each pair identifies a main and secondary boundary whose degrees of freedom
    must be identified through a prolongation matrix.
    """

    pairs: tuple[tuple[str, str], ...]


def periodic(*pairs):
    """
    Create periodic boundary-condition pairings.

    Example
    -------
    periodic(
        ("left", "right"),
        ("bottom", "top"),
    )
    """
    normalized = []

    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("Each periodic pair must be `(main_tag, secondary_tag)`.")

        normalized.append(
            (
                str(pair[0]),
                str(pair[1]),
            )
        )

    if len(normalized) == 0:
        raise ValueError("At least one periodic boundary pair is required.")

    return PeriodicBC(
        pairs=tuple(normalized),
    )


def _as_tags(tags) -> tuple[str, ...]:
    """
    Normalize one boundary tag or a sequence of tags into a non-empty tuple.

    Parameters
    ----------
    tags:
        Either a string tag or a sequence of tag-like objects.

    Returns
    -------
    tuple[str, ...]
        Normalized tuple of boundary tag strings.
    """
    if isinstance(tags, str):
        return (tags,)
    if isinstance(tags, Sequence):
        out = tuple(str(t) for t in tags)
        if len(out) == 0:
            raise ValueError("Boundary tag list cannot be empty.")
        return out
    raise TypeError(f"Boundary tags must be a string or a sequence of strings, got {type(tags).__name__}.")


def dirichlet(tags, values=None):
    """
    Create a Dirichlet boundary-condition descriptor.

    This is the public helper used in FEM setup, for example:

        domain.init_fem(
            bcs=[
                domain.dirichlet("left", 0.0),
                domain.dirichlet(["bottom", "top"], {"x": 0.0, "y": 1.0}),
            ]
        )

    Parameters
    ----------
    tags:
        Boundary tag or list of boundary tags.
    values:
        Boundary value specification. For scalar unknowns, this can be a scalar
        or callable. For vector unknowns, this can also be a component list,
        tuple, or dictionary.

    Returns
    -------
    DirichletBC
        Boundary-condition descriptor consumed by `expand_bcs(...)`.
    """
    return DirichletBC(tags=_as_tags(tags), values=values)


def neumann(tags):
    """
    Create a Neumann/natural boundary-condition descriptor.

    This marks boundary regions whose weak-form boundary terms should be
    assembled through surface kernels.

    Parameters
    ----------
    tags:
        Boundary tag or list of boundary tags.

    Returns
    -------
    NeumannBC
        Boundary-condition descriptor consumed by `expand_bcs(...)`.
    """
    return NeumannBC(tags=_as_tags(tags))


def expand_bcs(bcs, vec: int):
    """
    Normalize user boundary-condition descriptors for FEM initialization.

    Returns
    -------
    tuple
        ``(dirichlet_tags, dirichlet_value_fns, neumann_tags, periodic_pairs)``.
        ``periodic_pairs`` is a list of ``(main_tag, secondary_tag)`` tuples.
    """
    dirichlet_tags = []
    dirichlet_value_fns = {}
    neumann_tags = []
    periodic_pairs = []

    for bc in bcs:
        if isinstance(bc, DirichletBC):
            for tag in bc.tags:
                if tag not in dirichlet_tags:
                    dirichlet_tags.append(tag)
                dirichlet_value_fns[tag] = _normalize_dirichlet_value(bc.values, vec)
        elif isinstance(bc, NeumannBC):
            for tag in bc.tags:
                if tag not in neumann_tags:
                    neumann_tags.append(tag)
        elif isinstance(bc, PeriodicBC):
            periodic_pairs.extend(bc.pairs)
        else:
            raise TypeError(
                f"Unsupported BC entry '{type(bc).__name__}'. Use dirichlet(...), neumann(...) or periodic(...)."
            )

    return dirichlet_tags, dirichlet_value_fns, neumann_tags, periodic_pairs
