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

import jax
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


def _axis_colours(n: int, lo: int, hi: int, periodic: bool):
    """A colouring of one axis in which the ``hi − lo + 1`` nodes of any row's window all differ.

    ``i mod P`` does that for an open axis, and for a periodic one whose length is a multiple of the width.
    Where it is not -- 32 unique nodes with a three-wide stencil -- the last ``n mod P`` nodes would see two
    window neighbours of the same colour across the seam, so they get colours of their own.

    Returns ``(colour_of_node, n_colours, offset_of[colour, node])``; the offset table holds the one window
    offset a node reads for a colour, or ``NONE`` where that colour touches none of its window.
    """
    period = hi - lo + 1
    colour = np.arange(n) % period
    n_colours = period
    tail = n % period if periodic else 0
    if tail:
        colour[n - tail :] = period + np.arange(tail)
        n_colours = period + tail
    off = np.full((n_colours, n), _NONE, dtype=int)
    for o in range(lo, hi + 1):
        j = np.arange(n) + o
        inside = (j >= 0) & (j < n) if not periodic else np.ones(n, bool)
        j = j % n if periodic else np.clip(j, 0, n - 1)
        rows = np.nonzero(inside)[0]
        if np.any(off[colour[j[rows]], rows] != _NONE):
            raise ValueError(
                f"jno lattice probe: the colouring of an axis of {n} nodes with window [{lo}, {hi}] is not "
                "proper -- two of a row's neighbours share a colour. Pass an explicit window."
            )
        off[colour[j[rows]], rows] = o
    return colour, n_colours, off


#: Marks "this colour touches none of the node's window" in an offset table.
_NONE = -10_000


def _accumulate_colours(matvec, shape, nf, window, periodic, dtype, init, update):
    """Fold ``update`` over every colour of every seed field, inside ONE compiled loop per seed field.

    ``update(carry, g, out, w, live) -> carry`` sees the operator's output for that colour, the window
    offset each node reads (a flat index into :func:`offsets`), and where that offset exists. Every consumer
    -- the full stencil, and the reductions a multigrid level needs -- is a different fold over this.

    The colours run inside ``lax.fori_loop`` rather than as a Python loop: a hierarchy probes a dozen levels,
    and dispatching (and compiling) a program per colour per level cost 12 s of setup at 1M nodes against
    0.3 s of arithmetic.
    """
    lo, hi = window
    dim = len(shape)
    period = hi - lo + 1
    axes = [_axis_colours(n, lo, hi, a < len(periodic) and periodic[a]) for a, n in enumerate(shape)]
    counts = [int(k) for _, k, _ in axes]
    colours = [jnp.asarray(c) for c, _, _ in axes]
    tables = [jnp.asarray(o) for _, _, o in axes]
    idx = [jnp.asarray(a) for a in np.indices(shape)]
    n_col = int(np.prod(counts))

    carry = init
    for g in range(nf):  # the seeded field is static: nf is small, and it keeps the scatter's index static

        def body(c, carry, g=g):
            # int32 / bool intermediates: one index and one flag per node, on a grid that can hold
            # millions of them (int64 and float masks cost 8 bytes a node each)
            seed = jnp.ones(shape, bool)
            w = jnp.zeros(shape, dtype=jnp.int32)
            live = jnp.ones(shape, bool)
            for a in range(dim):
                stride = int(np.prod(counts[a + 1 :]))
                ca = (c // stride) % counts[a]
                seed = seed & (colours[a][idx[a]] == ca)
                oa = tables[a][ca][idx[a]]
                live = live & (oa != _NONE)
                w = w * period + jnp.where(oa == _NONE, 0, oa - lo).astype(jnp.int32)
            x = jnp.zeros((nf,) + shape, dtype).at[g].set(seed.astype(dtype))
            out = jnp.asarray(matvec(x.reshape(-1))).reshape((nf,) + shape)
            return update(carry, g, out, w, live.astype(dtype))

        carry = jax.jit(lambda carry, body=body: jax.lax.fori_loop(0, n_col, body, carry))(carry)
    return carry


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

    per_full = tuple(periodic) + (False,) * (dim - len(periodic))
    for cand in WINDOWS if window is None else (window,):
        S = _probe_window(matvec, shape, nf, cand, per_full, dtype)
        if not verify:
            return cand, S
        got = apply_stencil(S, cand, v.reshape(nf, *shape), per_full).reshape(-1)
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


def probe_reduced(matvec, shape, nf=1, *, window, periodic=(), dtype=None):
    """The two reductions of the stencil a multigrid level needs, **without storing it**:

    * ``d[f, i] = Σ_g Σ_o |a_{fg}(i, o)|`` -- the ℓ¹ row weight of every row;
    * ``strength[w] = mean_i Σ_{f,g} |a_{fg}(i, o_w)|`` -- how strongly the operator couples along each
      offset of the window.

    The full stencil of a fine level is ``nf²·W`` numbers per node (1.2 GB for a 2-D five-point operator at
    16.8M nodes); these two are ``nf`` per node and ``W`` in total. Same colours, same cost as :func:`probe`.
    """
    shape = tuple(int(s) for s in shape)
    dim = len(shape)
    dtype = jnp.result_type(float) if dtype is None else dtype
    per_full = tuple(periodic) + (False,) * (dim - len(periodic))
    init = (jnp.zeros((nf,) + shape, dtype), jnp.zeros(len(offsets(*window, dim)), dtype))

    def update(carry, _g, out, w, live):
        d, strength = carry
        contrib = jnp.abs(out) * live
        return d + contrib, strength.at[w.reshape(-1)].add(contrib.sum(axis=0).reshape(-1))

    d, strength = _accumulate_colours(matvec, shape, nf, window, per_full, dtype, init, update)
    return d, strength / float(np.prod(shape))


def _probe_window(matvec, shape, nf, window, periodic, dtype):
    """``S[f, g, i, w]``: every coefficient of every row, folded over the colours."""
    dim = len(shape)
    n_off = len(offsets(*window, dim))
    idx = tuple(jnp.asarray(a) for a in np.indices(shape))
    init = jnp.zeros((nf, nf) + shape + (n_off,), dtype)

    def update(S, g, out, w, live):
        for f in range(nf):
            S = S.at[(f, g) + idx + (w,)].add(out[f] * live)
        return S

    return _accumulate_colours(matvec, shape, nf, window, periodic, dtype, init, update)
