"""Operator-dependent multigrid on a structured grid, built from the operator itself.

A V-cycle built from the grid alone preconditions one operator: the constant-coefficient ``-Δ``. So a variable coefficient, a reaction term, an advection term, an anisotropy,
a flux row or a second field all get a preconditioner for a *different* problem than the one being
solved. This module builds the hierarchy from the operator's own coefficients instead:

1. read the fine operator's per-node stencil (:mod:`lattice`);
2. coarsen the axes that the operator actually couples strongly, by a factor of two;
3. form the coarse operator **variationally**, ``A_c = Pᵀ A P`` (Galerkin), by probing ``Pᵀ ∘ A ∘ P`` on
   the coarse lattice -- so the coarse levels inherit whatever the fine operator is, including its
   boundary rows;
4. smooth with ℓ¹-Jacobi, whose weights are read off the same stencil;
5. factorise the coarsest level once.

This is black-box multigrid (J. E. Dendy, *J. Comput. Phys.* 48 (1982) 366) with a probed operator in
place of an assembled one. The Galerkin coarse operator is the variational one (Trottenberg, Oosterlee &
Schüller, *Multigrid*, 2001, §2.3), which keeps the V-cycle symmetric when ``A`` is symmetric, so CG
stays valid. ℓ¹-Jacobi (Baker, Falgout, Kolev & Yang, *SIAM J. Sci. Comput.* 33 (2011) 2864) has no
damping parameter to choose and never diverges on a symmetric positive-definite operator.

Nothing here knows which PDE produced the operator: a row with no coefficients at all (a Dirichlet
degree of freedom the caller has eliminated) is passed through, and the number of levels follows the grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .lattice import apply_stencil, offsets, probe, probe_reduced, shift

#: Strength below which an axis is left uncoarsened (semi-coarsening): an axis whose couplings are weaker
#: than this fraction of the strongest axis' does not carry the error a coarse grid must represent. The
#: classical strong-connection threshold of Ruge & Stüben (*Multigrid Methods*, SIAM 1987, §4.2).
STRENGTH = 0.25
#: An axis of fewer than this many nodes cannot be halved (two cells at least).
MIN_NODES = 4
#: Where the hierarchy ends: a level of at most this many unknowns is factorised densely instead of being
#: coarsened further. Two costs meet here, and neither is a guess about the problem:
#:
#: * memory -- the factor is baked into the compiled V-cycle, and 2048 unknowns is 32 MB in float64;
#: * setup -- every level costs a compilation (measured: 0.65 s each on an RTX 3070, whatever its size),
#:   while a dense solve of 2048 unknowns costs one factorisation and a back-substitution per cycle.
#:
#: A level ABOVE it that cannot be coarsened (a convection stall) is smoothed instead of solved, which
#: weakens the preconditioner and changes no answer.
DENSE_MAX = 2048


def _axis_nodes(n: int):
    """The fine indices of the coarse nodes on an axis of ``n`` nodes.

    Every other node, with the last cell **merged** into its neighbour when the cell count is odd: 8 nodes
    (7 cells) give coarse nodes 0, 2, 4, 7, so the last coarse cell spans three fine cells rather than
    leaving a one-cell cell that would survive every level while its neighbours doubled.
    """
    cells = n - 1
    if cells <= 1:
        return np.arange(n)
    idx = list(range(0, n, 2))
    if cells % 2:  # odd cell count: merge the last (one-cell) coarse cell into its neighbour
        idx = idx[:-1]
        if idx[-1] != n - 1:
            idx.append(n - 1)
    elif idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray(idx, dtype=int)


def _axis_transfer(n: int):
    """``(coarse_fine_index, gather, weight)`` for linear interpolation along one axis.

    ``gather`` and ``weight`` are ``(n, 2)``: fine node ``i`` is ``Σ_k weight[i, k] · x_c[gather[i, k]]``,
    exact on linear functions for any (also non-uniform) coarse spacing.
    """
    coarse = _axis_nodes(n)
    gather = np.zeros((n, 2), dtype=int)
    weight = np.zeros((n, 2))
    right = np.searchsorted(coarse, np.arange(n), side="left")
    for i in range(n):
        j = int(right[i])
        if j < len(coarse) and coarse[j] == i:  # a coarse node: injected
            gather[i] = (j, j)
            weight[i] = (1.0, 0.0)
            continue
        a, b = j - 1, min(j, len(coarse) - 1)
        span = coarse[b] - coarse[a]
        gather[i] = (a, b)
        weight[i] = ((coarse[b] - i) / span, (i - coarse[a]) / span)
    return coarse, jnp.asarray(gather), jnp.asarray(weight)


def _prolong(x, transfers, axes):
    """Interpolate ``x`` (``(nf, *coarse_shape)``) onto the fine grid, one axis at a time."""
    out = x
    for axis, (_, gather, weight) in zip(axes, transfers):
        ax = axis + 1
        g = jnp.take(out, gather[:, 0], axis=ax) * jnp.expand_dims(
            weight[:, 0], tuple(range(ax)) + tuple(range(ax + 1, out.ndim))
        )
        h = jnp.take(out, gather[:, 1], axis=ax) * jnp.expand_dims(
            weight[:, 1], tuple(range(ax)) + tuple(range(ax + 1, out.ndim))
        )
        out = g + h
    return out


def _restrict(y, transfers, axes):
    """``Pᵀ y``: the transpose of :func:`_prolong`, one axis at a time (a weighted scatter-add)."""
    out = y
    for axis, (coarse, gather, weight) in zip(axes, transfers):
        ax = axis + 1
        n_c = len(coarse)
        acc = jnp.zeros(out.shape[:ax] + (n_c,) + out.shape[ax + 1 :], out.dtype)
        for k in range(2):
            w = jnp.expand_dims(weight[:, k], tuple(range(ax)) + tuple(range(ax + 1, out.ndim)))
            acc = _scatter_add_axis(acc, gather[:, k], out * w, ax)
        out = acc
    return out


def _scatter_add_axis(acc, index, src, axis):
    """``acc[..., index[i], ...] += src[..., i, ...]`` along ``axis``."""
    moved_src = jnp.moveaxis(src, axis, 0)
    moved_acc = jnp.moveaxis(acc, axis, 0)
    moved_acc = moved_acc.at[index].add(moved_src)
    return jnp.moveaxis(moved_acc, 0, axis)


def _strong_axes(strength, window, shape, dim, periodic=()):
    """The axes an operator couples strongly enough to coarsen: those whose pure-axis coupling is at least
    :data:`STRENGTH` of the strongest axis'. An axis of fewer than 4 nodes is never coarsened.

    ``strength`` is one number per window offset (:func:`lattice.probe_reduced`). Coarsening only the strong
    axes is semi-coarsening: on a 100:1 anisotropy the weak axis carries no error a coarse grid could
    represent, and coarsening it anyway is what makes a V-cycle diverge there.

    When no strong axis can be halved any more, the weak ones are coarsened instead, so the hierarchy always
    runs down to a few dozen unknowns: without that, a strongly anisotropic operator would stop at a level as
    long as its weak axis, and the "coarsest" solve would be the size of the problem.
    """
    offs = offsets(*window, dim)
    per_axis = np.zeros(dim)
    st = np.asarray(strength)
    for w, o in enumerate(offs):
        nz = [a for a, oa in enumerate(o) if oa != 0]
        if len(nz) == 1:
            per_axis[nz[0]] += float(st[w])
    width = window[1] - window[0] + 1
    # A periodic axis must keep at least the stencil's width, or a row's window wraps onto itself and the
    # operator has no stencil left to read.
    halvable = tuple(
        a
        for a in range(dim)
        if shape[a] >= MIN_NODES and (not (a < len(periodic) and periodic[a]) or len(_axis_nodes(shape[a])) >= width)
    )
    top = per_axis.max() if per_axis.size else 0.0
    if top <= 0:
        return halvable
    strong = tuple(a for a in halvable if per_axis[a] >= STRENGTH * top)
    return strong or halvable


def _strength_of(S, window, dim):
    """:func:`_strong_axes`'s input, from a stored stencil."""
    return jnp.abs(S).sum(axis=(0, 1)).reshape(-1, len(offsets(*window, dim))).mean(axis=0)


def _symmetric_stencil(S, window, dim):
    """Is the stencil symmetric, ``a_{fg}(i, o) == a_{gf}(i + o, −o)``? Then the operator is, and ℓ¹-Jacobi
    cannot amplify on it (Baker et al. 2011), so the power iteration below is not needed."""
    offs = offsets(*window, dim)
    tol = float(np.sqrt(np.finfo(np.asarray(S).dtype).eps))
    scale = float(jnp.max(jnp.abs(S))) or 1.0
    for w, o in enumerate(offs):
        back = offs.index(tuple(-oa for oa in o)) if all(window[0] <= -oa <= window[1] for oa in o) else None
        if back is None:
            if float(jnp.max(jnp.abs(S[..., w]))) > tol * scale:
                return False
            continue
        for f in range(S.shape[0]):
            for g in range(S.shape[1]):
                shifted = shift(S[g, f, ..., back][None], o)[0]  # a_{gf}(i + o, −o)
                if float(jnp.max(jnp.abs(S[f, g, ..., w] - shifted))) > tol * scale:
                    return False
    return True


def _smoother_amplifies(apply_fn, inv, shape, nf, iters=20, seed=0):
    """Does ℓ¹-Jacobi **grow** the error on this level: is ``ρ(I − D⁻¹A) > 1``?

    A coarse grid is only worth building while its own smoother still damps. Coarsening doubles ``h``, and
    for a convection term that doubles the cell Péclet number ``|b|h/ε``: past the classical limit of 2 the
    central-difference operator loses its M-matrix structure (Patankar, *Numerical Heat Transfer and Fluid
    Flow*, 1980, §5.2) and a point smoother amplifies instead of damping. Measured on an advection problem,
    ρ goes 0.997, 0.972, **1.060**, 1.117 down the levels, and a V-cycle that uses those levels diverges
    (factor 1.97); stopping where ρ first exceeds 1 and solving that level exactly is what works.

    This is measured, by power iteration on the smoother's own iteration matrix, rather than inferred from a
    stencil pattern: a *geometric* artefact can break diagonal dominance while leaving a perfectly good
    smoother (the merged odd cell of a 26-node axis is 12% short of dominant, and smooths at ρ = 0.94).
    """
    v = jnp.asarray(np.random.default_rng(seed).standard_normal((nf,) + shape))
    v = v / jnp.linalg.norm(v)
    rho = 0.0
    for _ in range(iters):
        av = apply_fn(v)
        w = v - (inv * av if nf == 1 else jnp.einsum("...fg,g...->f...", inv, av))
        rho = jnp.linalg.norm(w)
        v = w / jnp.maximum(rho, jnp.finfo(v.dtype).tiny)
    return float(rho) > 1.0


def _l1_weights(S):
    """ℓ¹-Jacobi weights ``d_i = Σ_j |a_ij|`` per row, from a stored stencil. A row with no coefficients at
    all (a degree of freedom the caller eliminated) gets 0, which the smoother reads as "leave it alone"."""
    return jnp.abs(S).sum(axis=(1, -1))  # over the input fields and the offsets -> (nf, *shape)


def _centre_block(S, window, dim):
    """The node's own ``nf x nf`` coupling ``a_{fg}(i, 0)``, which the point-block smoother inverts."""
    return S[..., offsets(*window, dim).index((0,) * dim)]


def _smoother_inverse(d, block):
    """``D⁻¹`` of the point-block ℓ¹ smoother, per node: the node's own block, with the rest of each row's
    ℓ¹ weight added to its diagonal.

    For one field this is ℓ¹-Jacobi exactly. For several it inverts the coupling a point smoother would
    otherwise ignore -- two fields that exchange strongly at the same node (a reaction) are a ``2x2`` solve,
    not two independent scalings. Rows that carry nothing keep an identity row, so the smoother leaves them.
    """
    nf = d.shape[0]
    if nf == 1:
        inv = jnp.where(d[0] > 0, 1.0 / jnp.where(d[0] > 0, d[0], 1.0), 0.0)
        return inv[None]
    off = d - jnp.abs(block).sum(axis=1)  # each row's weight outside its own node
    D = jnp.moveaxis(block, (0, 1), (-2, -1))  # (*shape, nf, nf)
    D = D + jnp.moveaxis(off, 0, -1)[..., None] * jnp.eye(nf, dtype=d.dtype)
    live = jnp.moveaxis(d, 0, -1).sum(axis=-1) > 0
    D = jnp.where(live[..., None, None], D, jnp.eye(nf, dtype=d.dtype))
    return jnp.where(live[..., None, None], jnp.linalg.inv(D), 0.0)


def _dense_from_stencil(S, window, shape, nf, periodic):
    """The coarsest level as a dense matrix, read straight off its stencil (one entry per node and offset)
    rather than by differencing it ``n`` times. The indices are host-side; the values may be traced."""
    dim = len(shape)
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    A = jnp.zeros((nf * n, nf * n), S.dtype)
    for w, o in enumerate(offsets(*window, dim)):
        src, live = idx, np.ones(shape, bool)
        for a, oa in enumerate(o):
            if oa == 0:
                continue
            src = np.roll(src, -oa, axis=a)
            if not (a < len(periodic) and periodic[a]):  # what wrapped is not a coupling
                keep = np.ones(shape[a], bool)
                keep[slice(shape[a] - oa, None) if oa > 0 else slice(0, -oa)] = False
                live &= keep.reshape((1,) * a + (-1,) + (1,) * (dim - a - 1))
        rows, cols, mask = idx.reshape(-1), src.reshape(-1), jnp.asarray(live.reshape(-1), S.dtype)
        for f in range(nf):
            for g in range(nf):
                A = A.at[f * n + rows, g * n + cols].add(S[f, g, ..., w].reshape(-1) * mask)
    return A


def _periodic_reduction(matvec, shape, nf, periodic):
    """Work on the **unique** nodes of a periodic axis.

    A periodic ``jno.fdm`` grid carries the duplicate node ``x = L ≡ x = 0``, tied by a row ``u[L] − u[0]``
    that couples two nodes a whole axis apart: no finite-difference window holds it, so the operator has no
    stencil to read. Injecting ``u[L] = u[0]`` satisfies that row identically and leaves the interior rows,
    which already wrap, so the reduced operator on the unique nodes is the one to coarsen.

    Returns ``(reduced_matvec, reduced_shape, inject, extract)``.
    """
    keep = tuple(n - 1 if p else n for n, p in zip(shape, periodic))

    def inject(xu):  # (nf, *keep) -> (nf, *shape), the duplicate node equal to the first
        x = xu
        for a, p in enumerate(periodic):
            if p:
                x = jnp.concatenate([x, jnp.take(x, jnp.asarray([0]), axis=a + 1)], axis=a + 1)
        return x

    def extract(y):  # (nf, *shape) -> (nf, *keep)
        return y[(slice(None),) + tuple(slice(0, k) for k in keep)]

    def reduced(v):
        x = inject(jnp.asarray(v).reshape((nf,) + keep))
        return extract(jnp.asarray(matvec(x.reshape(-1))).reshape((nf,) + shape)).reshape(-1)

    return reduced, keep, inject, extract


def build(matvec, shape, *, nf=1, periodic=(), dtype=None, n_pre=2, n_post=2, window=None):
    """An operator-dependent V-cycle ``M⁻¹: r -> e`` for ``matvec`` on the lattice ``shape``.

    Returns ``(apply, n_levels)``. ``apply`` takes and returns the flat blocked vector ``matvec`` uses
    (``nf`` blocks of ``prod(shape)``, C order within a block). Setup costs one stencil probe per level.

    The **fine** level keeps ``matvec`` itself and stores only its ℓ¹ weights, one number per unknown: its
    full stencil would be ``nf²·W`` numbers per node, 1.2 GB for a 2-D five-point operator at 16.8M nodes.
    Coarse levels, each at least twice smaller, store theirs.
    """
    full_shape = tuple(int(s) for s in shape)
    dim = len(full_shape)
    dtype = jnp.result_type(float) if dtype is None else dtype
    per = tuple(periodic) + (False,) * (dim - len(periodic))
    if any(per):  # coarsen the unique nodes; the duplicate node's tie row has no stencil
        matvec, shape, inject, extract = _periodic_reduction(matvec, full_shape, nf, per)
    else:
        shape, inject, extract = full_shape, (lambda x: x), (lambda y: y)
    if window is None:  # find the window once, on the fine operator, then reduce rather than store it
        window, _ = probe(matvec, shape, nf, periodic=per, dtype=dtype, seed=1)
    d0, strength0, block0 = probe_reduced(matvec, shape, nf, window=window, periodic=per, dtype=dtype)

    levels = [
        {
            "shape": shape,
            "apply": lambda x: matvec(x.reshape(-1)).reshape((nf,) + shape),
            "d": d0,
            "block": block0,
            "strength": strength0,
            "window": window,
            "periodic": per,
        }
    ]
    while True:
        top = levels[-1]
        sh = top["shape"]
        if nf * int(np.prod(sh)) <= DENSE_MAX:
            break  # small enough to factorise: cheaper than another level's setup
        axes = _strong_axes(top["strength"], top["window"], sh, dim, top["periodic"])
        if not axes:  # no axis can be halved: this is the coarsest level
            break
        transfers = [_axis_transfer(sh[a]) for a in axes]
        coarse_shape = tuple(len(transfers[axes.index(a)][0]) if a in axes else sh[a] for a in range(dim))
        top["axes"], top["transfers"] = axes, transfers
        fine_apply, fine_per = top["apply"], top["periodic"]

        @jax.jit  # one compiled program per level: the probe applies it once per colour
        def coarse_matvec(v, fine_apply=fine_apply, transfers=transfers, axes=axes, coarse_shape=coarse_shape):
            x = _prolong(v.reshape((nf,) + coarse_shape), transfers, axes)
            return _restrict(fine_apply(x), transfers, axes).reshape(-1)

        cper = tuple(p and a in axes or p for a, p in enumerate(fine_per))  # periodicity survives coarsening
        cwin, cS = probe(coarse_matvec, coarse_shape, nf, periodic=cper, dtype=dtype)
        cd, cblock = _l1_weights(cS), _centre_block(cS, cwin, dim)
        amplifies = not _symmetric_stencil(cS, cwin, dim) and _smoother_amplifies(
            lambda x, cS=cS, cwin=cwin, cper=cper: apply_stencil(cS, cwin, x, cper),
            _smoother_inverse(cd, cblock),
            coarse_shape,
            nf,
        )
        if amplifies and nf * int(np.prod(sh)) <= DENSE_MAX:
            break  # no smoother works on that level, and this one is small enough to solve exactly
        levels.append(
            {
                "shape": coarse_shape,
                "apply": jax.jit(lambda x, cS=cS, cwin=cwin, cper=cper: apply_stencil(cS, cwin, x, cper)),
                "d": cd,
                "block": cblock,
                "strength": _strength_of(cS, cwin, dim),
                "window": cwin,
                "periodic": cper,
                "S": cS,
            }
        )
        if coarse_shape == sh:
            break

    # The coarsest level, factorised once. Coarsening runs until no axis can be halved, so this is a few
    # dozen unknowns however large the fine grid is -- there is no size threshold anywhere. A row the caller
    # eliminated (a Dirichlet degree of freedom) carries nothing, so it gets an identity row and the V-cycle
    # leaves it where it was.
    last = levels[-1]
    n_c = nf * int(np.prod(last["shape"]))
    if n_c > DENSE_MAX:
        from ..logger import get_logger

        get_logger().info(
            f"multigrid: the coarsest level has {n_c} unknowns, above the {DENSE_MAX} this factorises densely "
            f"({8 * DENSE_MAX**2 / 1e6:.0f} MB), so it is smoothed instead of solved. Coarsening stopped there "
            "because halving further would leave an operator a point smoother cannot smooth (a cell Péclet "
            "number above 2, or an indefinite operator)."
        )
    lu = None
    if n_c <= DENSE_MAX:
        if "S" in last:
            dense = _dense_from_stencil(last["S"], last["window"], last["shape"], nf, last["periodic"])
        else:  # a grid too small to coarsen at all: difference the operator itself
            dense = jax.jacfwd(lambda v: matvec(v))(jnp.zeros(n_c, dtype))
        live = jnp.abs(dense).sum(axis=1) > 0
        lu = jax.scipy.linalg.lu_factor(jnp.where(live[:, None] & live[None, :], dense, jnp.eye(n_c, dtype=dtype)))

    # The per-level stencils and weights are handed to the compiled V-cycle as ARGUMENTS, not closed over:
    # a closed-over array becomes a constant inside the executable, and at 4.2M nodes loading that executable
    # ran the card out of memory ("Failed to load in-memory CUBIN").
    weights = tuple(_smoother_inverse(lev["d"], lev["block"]) for lev in levels)
    stencils = tuple(lev.get("S") for lev in levels[1:])

    def smooth(lev, x, r, state):
        inv = state[0][lev]
        res = r - level_apply(lev, x, state)
        if nf == 1:
            return x + inv * res
        # (*shape, nf, nf) @ (*shape, nf) -> one small solve per node
        applied = jnp.einsum("...fg,g...->f...", inv, res)
        return x + applied

    def level_apply(lev, x, state):
        if lev == 0:
            return jnp.asarray(matvec(x.reshape(-1))).reshape((nf,) + levels[0]["shape"])
        S = state[1][lev - 1]
        return apply_stencil(S, levels[lev]["window"], x, levels[lev]["periodic"])

    def coarsest(r, state):
        if state[2] is not None:
            return jax.scipy.linalg.lu_solve(state[2], r.reshape(-1)).reshape((nf,) + levels[-1]["shape"])
        x = jnp.zeros_like(r)  # too large to factorise: smooth it instead (linear, so the V-cycle stays linear)
        for _ in range(n_pre + n_post):
            x = smooth(len(levels) - 1, x, r, state)
        return x

    def vcycle(lev, r, state):
        if lev == len(levels) - 1:
            return coarsest(r, state)
        x = jnp.zeros_like(r)
        for _ in range(n_pre):
            x = smooth(lev, x, r, state)
        axes, transfers = levels[lev]["axes"], levels[lev]["transfers"]
        resid = r - level_apply(lev, x, state)
        x = x + _prolong(vcycle(lev + 1, _restrict(resid, transfers, axes), state), transfers, axes)
        for _ in range(n_post):
            x = smooth(lev, x, r, state)
        return x

    @jax.jit
    def _apply(r_flat, state):
        r = extract(jnp.asarray(r_flat).reshape((nf,) + full_shape))
        return inject(vcycle(0, r, state)).reshape(-1)

    state = (weights, stencils, lu)
    return (lambda r_flat: _apply(r_flat, state)), len(levels)
