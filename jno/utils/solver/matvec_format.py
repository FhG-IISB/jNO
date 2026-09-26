"""Which storage applies a sparse operator inside jNO's iterative solvers: CSR (cuSPARSE) or split COO.

Neither wins everywhere, and which one wins depends on the operator AND the machine. Measured on one
RTX 3070 (float64), through a whole Jacobi-BiCGStab solve and its gradient w.r.t. the operator values:
CSR was 1.45x faster forward and 1.10x faster backward on a 27-nonzeros-per-row 3-D stencil (97k
DOF), tied on 300k-DOF Laplacians (5-7 per row), and 1.25x SLOWER on a 69k-DOF 3-D Laplacian, where
cuSPARSE's fixed per-call cost dominates a 10 ms solve. So the choice is MEASURED, per operator, on the
device that runs it -- never a threshold tuned on one card:

* the real operator is timed whenever it is concrete (the solver entry points see it before tracing),
* an operator only ever seen traced is decided on a banded stand-in of the same shape and nonzeros per
  row, generated on the device,
* the decision is cached per (platform, device kind, size class, dtype) (see `_key`) and logged once.

``jno.setup(matvec_format="csr" | "coo" | "auto")`` (or ``[jno] matvec_format``) overrides it.

Both formats are exact, differentiable in the operator values and the vector, and vmappable (the CSR
batching rules are in :mod:`jno.utils.solver.sparse_batching`).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

_FORMATS = ("auto", "csr", "coo")
_FORMAT = "auto"

#: measured decisions, keyed on (platform, device kind, size class, dtype) -- see `_key`
_DECISIONS: dict = {}

_PRODUCTS_PER_TIMING = 20  # products per timed program: in-loop cost, dispatch amortised


def set_matvec_format(fmt: str) -> None:
    """``jno.setup(matvec_format=...)``: ``"auto"`` (measure per operator), ``"csr"`` or ``"coo"``."""
    global _FORMAT
    if fmt not in _FORMATS:
        raise ValueError(f"jno.setup(matvec_format={fmt!r}): expected one of {_FORMATS}.")
    if fmt != _FORMAT:
        _FORMAT = fmt
        # read while a solve is TRACED, and JAX caches compiled/batched programs: without this a change
        # after the first solve would be silently ignored (the lu_stack lesson)
        jax.clear_caches()


def _key(A):
    """The decision's key: the device, and the operator's SIZE CLASS rather than its exact size.

    Rows and columns to the nearest power of two, stored entries per row to the nearest half-octave. An
    exact key re-measured every operator an adaptive loop produces -- each remesh has a new size -- and a
    measurement is two compilations: measured ~0.5 s per new operator, which made a cold h-adaptive march
    3.9 s slower (15.8 -> 19.8 s) and a cold steady h-adaptive solve 1.7 s slower (8.5 -> 10.2 s), for no
    gain at those sizes. Operators within a factor ~1.4 of each other in size and density share one
    measurement, taken on the first real operator of the class; where they straddle a crossover the two
    formats cost about the same, so a shared decision gives little away.
    """
    import math

    dev = jax.devices()[0]
    n, m = (int(s) for s in A.shape)
    per_row = max(int(A.nse), 1) / max(n, 1)
    return (
        dev.platform,
        dev.device_kind,
        round(math.log2(max(n, 1))),
        round(math.log2(max(m, 1))),
        jnp.dtype(A.dtype).name,
        round(2 * math.log2(per_row)),
    )


def _is_concrete(A) -> bool:
    return not any(isinstance(x, jax.core.Tracer) for x in (A.data, A.indices))


def csr_parts(A):
    """``(data, indices, indptr)`` of ``A`` (a 2-D BCOO) as CSR, ready for cuSPARSE.

    Handles what jNO's operators can contain: duplicates (CSR products sum them, like BCOO), unsorted
    triplets (an uncompressed parametric assembly: stably sorted by row), and out-of-bound padding from
    ``sum_duplicates`` (turned into explicit zeros in the last row, which keeps a sorted operator sorted
    without a second sort). Call it OUTSIDE a solver loop -- it is O(nnz), or O(nnz log nnz) unsorted.
    """
    n, m = A.shape
    r, c, d = A.indices[:, 0], A.indices[:, 1], A.data
    valid = (r >= 0) & (r < n) & (c >= 0) & (c < m)
    d = jnp.where(valid, d, jnp.zeros((), d.dtype))
    r = jnp.where(valid, r, n - 1).astype(jnp.int32)
    c = jnp.where(valid, c, 0).astype(jnp.int32)
    if not getattr(A, "indices_sorted", False):
        order = jnp.argsort(r, stable=True)
        r, c, d = r[order], c[order], d[order]
    indptr = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(jnp.bincount(r, length=n)).astype(jnp.int32)])
    return d, c, indptr


def csr_matvec(parts, shape, transpose=False):
    from jax.experimental.sparse import csr as _csr

    d, c, p = parts

    def mv(v):
        v = jnp.asarray(v)
        if v.ndim == 1:
            return _csr._csr_matvec(d, c, p, v, shape=shape, transpose=transpose)
        return _csr._csr_matmat(d, c, p, v, shape=shape, transpose=transpose)

    return mv


def _banded_standin(A):
    """Same shape and nonzeros per row as ``A``, a band generated on the device (for traced operators)."""
    import jax.experimental.sparse as jsp

    n, m = A.shape
    per_row = max(1, min(m, int(round(int(A.nse) / max(n, 1)))))
    r = jnp.arange(n, dtype=jnp.int32)[:, None]
    start = (r.astype(jnp.int64) * (m - per_row) // max(n - 1, 1)).astype(jnp.int32)
    cols = (start + jnp.arange(per_row, dtype=jnp.int32)[None, :]).reshape(-1)
    rows = jnp.repeat(jnp.arange(n, dtype=jnp.int32), per_row)
    return jsp.BCOO((jnp.ones(cols.size, A.dtype), jnp.stack([rows, cols], 1)), shape=A.shape, indices_sorted=True)


def _time_products(mv, v):
    # The carry is the INPUT vector, made to depend on each product through a scalar: that works for a
    # rectangular operator (an AMG prolongation/restriction -- feeding the output back only fits a square
    # one) and still stops XLA from hoisting a loop-invariant product out of the loop. The extra
    # reduction is the same for both formats.
    def body(_, x):
        return x + jnp.asarray(1e-30, x.dtype) * jnp.sum(mv(x))

    f = jax.jit(lambda x: jax.lax.fori_loop(0, _PRODUCTS_PER_TIMING, body, x))
    jax.block_until_ready(f(v))
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(f(v))
        best = min(best, time.perf_counter() - t0)
    return best / _PRODUCTS_PER_TIMING


def choose(A, *, log: bool = True) -> str:
    """``"csr"`` or ``"coo"`` for the 2-D BCOO ``A`` -- the override, else the measured faster one."""
    if _FORMAT != "auto":
        return _FORMAT
    key = _key(A)
    if key in _DECISIONS:
        return _DECISIONS[key]
    from ..logger import get_logger
    from .linear import _split_coo_matvec

    real = _is_concrete(A)
    try:
        with jax.ensure_compile_time_eval():
            B = A if real else _banded_standin(A)
            v = jnp.ones(B.shape[1], B.dtype)
            t_coo = _time_products(_split_coo_matvec(B, False), v)
            t_csr = _time_products(csr_matvec(csr_parts(B), B.shape), v)
        fmt = "csr" if t_csr < t_coo else "coo"
        if log:
            get_logger().info(
                f"jNO sparse operator {B.shape[0]:,}x{B.shape[1]:,} ({int(A.nse):,} nnz, {key[4]}) on {key[1]}: "
                f"CSR {1e6 * t_csr:.1f} us vs COO {1e6 * t_coo:.1f} us per product "
                f"({'measured on the operator' if real else 'measured on a banded stand-in of its size'}) -> "
                f"{fmt.upper()}. Override with jno.setup(matvec_format=...)."
            )
    except Exception as exc:  # noqa: BLE001 -- a failed measurement must not break the solve
        fmt = "coo"
        get_logger().warning(
            f"jNO could not time CSR against COO for a {A.shape} operator ({type(exc).__name__}: {exc}); "
            "using COO (split indices). Override with jno.setup(matvec_format=...)."
        )
    _DECISIONS[key] = fmt
    return fmt


def prime(*operators) -> None:
    """Decide the format for CONCRETE operators now, on their real sparsity pattern.

    Solver entry points call this before tracing, so the traced solve finds a decision measured on the
    real operator rather than on a stand-in. Anything that is not a concrete plain BCOO is ignored."""
    from .linear import _plain_bcoo

    if _FORMAT != "auto":
        return
    for A in operators:
        if _plain_bcoo(A) and _is_concrete(A):
            choose(A)


def prepare(A, *, log: bool = True) -> dict:
    """EAGER: the measured format for the concrete BCOO ``A`` and its arrays, as a pytree of arrays --
    ``{"csr": (data, indices, indptr)}`` or ``{"coo": (data, rows, cols)}``.

    For operators that are frozen once and applied many times from inside a trace (the AMG hierarchy):
    the conversion happens here, once, instead of being staged into every compiled program."""
    fmt = choose(A, log=log)
    with jax.ensure_compile_time_eval():
        if fmt == "csr":
            return {"csr": csr_parts(A)}
        return {"coo": (A.data, A.indices[:, 0], A.indices[:, 1])}


def apply_prepared(prep: dict, shape, v):
    """``A @ v`` from :func:`prepare`'s arrays (``shape`` is ``A``'s)."""
    if "csr" in prep:
        return csr_matvec(prep["csr"], tuple(shape))(v)
    d, r, c = prep["coo"]
    v = jnp.asarray(v)
    prod = d.reshape(d.shape + (1,) * (v.ndim - 1)) * v.at[c].get(mode="fill", fill_value=0)
    return jnp.zeros((shape[0],) + v.shape[1:], prod.dtype).at[r].add(prod, mode="drop")
