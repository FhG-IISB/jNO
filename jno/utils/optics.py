"""Beer-Lambert attenuation along a beam through a meshed body -- the optical path, as a linear map.

A weakly absorbed beam (1064 nm in water, say, where the absorption length is centimetres) deposits heat
through the VOLUME, not on the surface, and how much reaches a point depends on what the beam passed
through to get there:

    I(x) = I0 exp(-tau(x)),    tau(x) = \\int_0^{s(x)} alpha(x - s e) ds,    Q(x) = alpha(x) I(x)

That is non-local: it cannot be written as a formula in the term list, because it depends on the whole
chord behind each point. But it is LINEAR in the absorption field, so it is a matrix, and the matrix
depends only on the mesh and the beam direction. Build it once per mesh on the host (point location),
apply it in JAX -- so ``tau`` stays differentiable in the field that produced it.

The chord is found by SAMPLING rather than by intersecting rays with the boundary: samples that land
outside the mesh contribute nothing, so a non-convex body, or two droplets with a gap between them, need
no special handling and shadowing comes out for free. The cost is that the quadrature sees the body's
edge as a step, which is first-order accurate in the sample spacing.

These helpers stay private -- they are the beam's GEOMETRY, and jNO does not write your physics for you.
The public spelling that puts them in a weak form is ``jno.derived``, which turns any pure-JAX rule on the
state into a nodal field usable anywhere a field is::

    nodes, w = beam_paths(pts, cells, direction, pts)              # host geometry, built once
    tau = jno.derived(lambda T: optical_depth(alpha(T), nodes, w), inputs=[u], on=u)
    Q   = alpha_of(u) * I0 * jno.np.exp(-tau)                      # tau reads as an ordinary field

which is what makes the two-way coupling -- a hotter body absorbs more, so the deposited power depends on
the field it is producing -- an ordinary term rather than a special case. The tables above are fixed for
the life of the ``jno.fem``, so on a MOVING mesh they go stale: rebuild them and rebuild the problem.
"""

from __future__ import annotations

import numpy as np


def beam_paths(
    points: np.ndarray,
    cells: np.ndarray,
    direction,
    targets: np.ndarray,
    *,
    n_samples: int = 128,
    span: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(nodes (T, N, n_local), weights (T, N, n_local))`` for the optical depth at each target.

    The beam travels along ``direction``; the path to a target runs BACKWARDS from it, so the samples
    sit at ``x_t - s e`` for ``s`` in ``(0, span)``, midpoint rule. ``span`` defaults to the mesh's own
    extent along the beam, which is the longest chord any target can have.

    Apply with :func:`optical_depth`. Host only, and pure geometry: no field values enter here.
    """
    from .solver.fem_adapt import _locate_in_cells

    X = np.asarray(points, dtype=float)
    dim = X.shape[1]
    e = np.asarray(direction, dtype=float).reshape(-1)
    if e.shape[0] != dim:
        raise ValueError(f"beam direction has {e.shape[0]} components for a {dim}-D mesh.")
    norm = float(np.linalg.norm(e))
    if not norm > 0.0:
        raise ValueError("beam direction must be a non-zero vector.")
    e = e / norm
    T = np.asarray(targets, dtype=float).reshape(-1, dim)
    if n_samples < 1:
        raise ValueError(f"beam_paths needs at least one sample along the ray; got {n_samples}.")

    proj = X @ e
    L = float(span) if span is not None else float(proj.max() - proj.min())
    if not L > 0.0:
        raise ValueError("the mesh has no extent along the beam direction; pass span= explicitly.")
    ds = L / n_samples
    s = (np.arange(n_samples, dtype=float) + 0.5) * ds  # midpoint rule
    q = T[:, None, :] - s[None, :, None] * e[None, None, :]  # (T, N, dim)

    cell_idx, w, _ref, inside = _locate_in_cells(X, np.asarray(cells), q.reshape(-1, dim), tol=1e-9, k=32)
    w = np.where(np.asarray(inside)[:, None], np.asarray(w), 0.0)  # off the body: no material, no absorption
    nodes = np.asarray(cells)[cell_idx]  # (T*N, n_local)
    n_local = nodes.shape[1]
    return nodes.reshape(len(T), n_samples, n_local), (w * ds).reshape(len(T), n_samples, n_local)


def optical_depth(alpha_nodes, nodes: np.ndarray, weights: np.ndarray):
    """``tau`` at each target: the path integral of a nodal absorption field. Differentiable in it."""
    import jax.numpy as jnp

    a = jnp.asarray(alpha_nodes).reshape(-1)
    return jnp.sum(jnp.asarray(weights) * a[jnp.asarray(nodes)], axis=(1, 2))
