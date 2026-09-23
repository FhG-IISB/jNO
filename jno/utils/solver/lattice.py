"""Read the per-node stencil of a **matrix-free** operator on a structured grid, by colouring.

A finite-difference operator on a lattice couples each node to a bounded window of neighbours. Probe it
with one vector per colour of that window and the whole operator falls out -- every coefficient of every
row, interior and boundary alike, without assembling anything or knowing which PDE produced it:

    y_c = A e_c ,   e_c[j] = 1 where j ≡ c (mod P) ,   P = the window's width per axis

Within one window there is **exactly one** offset ``o`` with ``i + o ≡ c (mod P)``, so ``y_c[i]`` *is* the
coefficient of that offset in row ``i``. ``prod(P)·nf`` matvecs give the complete operator (9 for the
2-D five-point stencil with one-sided boundary rows, 16 for its 3-D counterpart, times the field count).

That is what an operator-dependent multigrid needs: the true diagonal (or point block) for its smoother,
and coarse operators formed from the operator itself rather than from a rediscretised model problem. It
also assembles a sparse matrix on a grid in ``prod(P)·nf`` matvecs, where the mesh path pays a Python
graph colouring plus a search over stencil radii.

The window is *measured*, not assumed: candidates grow until the reconstruction reproduces the operator's
action on a random vector, so a fourth-order stencil, an upwind bias or a wide boundary closure is found
rather than silently truncated.

References: the colouring argument is Curtis, Powell & Reid, *J. Inst. Math. Appl.* 13 (1974) 117 (sparse
Jacobians by differencing), specialised to the regular structure of a lattice, where the distance-2
colouring is periodic and known in closed form. Using the probed stencils to build coarse operators is
black-box multigrid: J. E. Dendy, *J. Comput. Phys.* 48 (1982) 366.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np

#: Windows tried, in order of size: the compact stencil, then one-sided room for a boundary closure, then
#: wider ones. A window ``(lo, hi)`` means offsets ``lo <= o <= hi`` on every axis.
WINDOWS = ((-1, 1), (-1, 2), (-2, 2), (-2, 3), (-3, 3), (-3, 4), (-4, 4))


def offsets(lo: int, hi: int, dim: int):
    """Every offset vector of the window ``[lo, hi]^dim``, in C order."""
    return tuple(itertools.product(range(lo, hi + 1), repeat=dim))


def shift(x: jnp.ndarray, offset, periodic=()) -> jnp.ndarray:
    """``x[i + offset]`` on the grid ``x`` (one array per field: the leading axis is the field), reading
    zero outside it -- or wrapping on the axes flagged ``periodic``. Rolls and masks, so XLA partitions it
    as a halo exchange when the grid is sharded."""
    out = x
    for axis, o in enumerate(offset):
        if o == 0:
            continue
        ax = axis + 1  # axis 0 is the field
        out = jnp.roll(out, -o, axis=ax)
        if axis >= len(periodic) or not periodic[axis]:
            n = out.shape[ax]
            idx = jnp.arange(n)
            live = (idx < n - o) if o > 0 else (idx >= -o)  # the entries that did not wrap
            out = out * live.reshape((1,) * ax + (n,) + (1,) * (out.ndim - ax - 1))
    return out


def apply_stencil(S: jnp.ndarray, window, x: jnp.ndarray, periodic=()) -> jnp.ndarray:
    """Apply a probed stencil: ``y[f, i] = Σ_g Σ_o S[f, g, i, o] · x[g, i + o]``.

    ``S`` is ``(nf, nf, *shape, n_offsets)`` and ``x`` is ``(nf, *shape)``; the offsets are
    :func:`offsets` of ``window`` in the same order.
    """
    lo, hi = window
    dim = x.ndim - 1
    y = jnp.zeros_like(x)
    for w, o in enumerate(offsets(lo, hi, dim)):
        xs = shift(x, o, periodic)  # (nf, *shape)
        y = y + jnp.einsum("fg...,g...->f...", S[..., w], xs)
    return y


def _colour_masks(shape, period, dtype):
    """One ``(*shape,)`` indicator per colour: 1 at the nodes whose lattice index is ``≡ c (mod period)``."""
    grids = np.indices(shape)
    masks = []
    for c in itertools.product(*(range(p) for p in period)):
        m = np.ones(shape, dtype=bool)
        for axis, ca in enumerate(c):
            m &= grids[axis] % period[axis] == ca
        masks.append(jnp.asarray(m, dtype))
    return masks


def probe(matvec, shape, nf=1, *, window=None, periodic=(), dtype=None, verify=True, seed=0):
    """The per-node stencil of ``matvec`` on the lattice ``shape``.

    ``matvec`` takes and returns a flat vector of ``nf * prod(shape)`` entries, blocked by field and in C
    order within a field -- the layout ``jno.fdm`` solves in. Returns ``(window, S)`` with ``S`` of shape
    ``(nf, nf, *shape, n_offsets)``.

    With ``window=None`` the candidates of :data:`WINDOWS` are tried in order and the first whose
    reconstruction reproduces ``matvec`` on a random vector (to ``√ε`` relative) is returned; passing a
    window skips the search. ``verify=False`` skips the check when the window is given.

    Cost: ``prod(window width)·nf`` matvecs per candidate, plus one for the check. A periodic axis needs
    its length to be a multiple of the window's width, so that a node's window holds one offset per colour;
    it raises otherwise rather than folding two coefficients into one.
    """
    shape = tuple(int(s) for s in shape)
    dim = len(shape)
    dtype = jnp.result_type(float) if dtype is None else dtype
    n = int(np.prod(shape))
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(nf * n), dtype=dtype)
    ref = matvec(v) if verify else None

    for cand in WINDOWS if window is None else (window,):
        lo, hi = cand
        period = tuple(hi - lo + 1 for _ in range(dim))
        for axis, p in enumerate(period):
            if axis < len(periodic) and periodic[axis] and shape[axis] % p:
                raise ValueError(
                    f"jno lattice probe: periodic axis {axis} has {shape[axis]} nodes, which is not a multiple "
                    f"of the stencil window's width {p}, so one colour would carry two of a row's coefficients. "
                    "Use a grid whose periodic axes are a multiple of the width, or pass an explicit window."
                )
        masks = _colour_masks(shape, period, dtype)
        S = _probe_window(matvec, shape, nf, lo, hi, masks, dtype)
        if not verify:
            return cand, S
        got = apply_stencil(S, cand, v.reshape(nf, *shape), periodic).reshape(-1)
        scale = float(jnp.linalg.norm(ref)) or 1.0
        if float(jnp.linalg.norm(got - ref)) <= float(np.sqrt(np.finfo(dtype).eps)) * scale:
            return cand, S
        if window is not None:
            raise ValueError(
                f"jno lattice probe: the window {cand} does not reproduce this operator (relative error "
                f"{float(jnp.linalg.norm(got - ref)) / scale:.2e}). Its rows reach further than the window."
            )
    raise ValueError(
        f"jno lattice probe: no window up to {WINDOWS[-1]} reproduces this operator's action -- its rows are "
        "wider than a finite-difference stencil (a spectral scheme couples every node on an axis), so it has "
        "no per-node stencil to read."
    )


def _probe_window(matvec, shape, nf, lo, hi, masks, dtype):
    """One pass of colours: ``S[f, g, i, w]`` from ``matvec`` applied to each colour of each field.

    For colour ``c`` the seed is 1 exactly at the nodes ``≡ c (mod period)``, so row ``i``'s output is the
    coefficient of the single window offset ``o ≡ c − i (mod period)``. That offset's index in the window is
    the same for every field pair, so one scatter per output field places the whole colour's result.
    """
    dim = len(shape)
    offs = offsets(lo, hi, dim)
    period = hi - lo + 1
    idx = tuple(jnp.asarray(a) for a in np.indices(shape))  # lattice index per node, per axis
    blocks = [[jnp.zeros(shape + (len(offs),), dtype) for _ in range(nf)] for _ in range(nf)]
    for g in range(nf):  # the field the seed vector carries
        for c, mask in enumerate(masks):
            colour = np.unravel_index(c, (period,) * dim)
            out = matvec(jnp.zeros((nf,) + shape, dtype).at[g].set(mask).reshape(-1)).reshape((nf,) + shape)
            w = jnp.zeros(shape, dtype=int)  # this colour's offset index, per node
            for a in range(dim):
                w = w * period + (colour[a] - idx[a] - lo) % period
            for f in range(nf):
                blocks[f][g] = blocks[f][g].at[idx + (w,)].set(out[f])
    return jnp.stack([jnp.stack(row) for row in blocks])
