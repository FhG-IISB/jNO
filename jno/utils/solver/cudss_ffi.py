"""NVIDIA cuDSS as an XLA custom call: factorise once, solve INSIDE a compiled program.

Every other sparse direct solve in jNO that keeps a factorisation between calls does it in a host
callback (`jax.pure_callback`): each apply stops the GPU stream, copies the vector to the host, runs
Python and copies the result back. Inside a Krylov loop that is one stall per iteration -- measured on
a welded PEEC preconditioner, ~2 ms an apply, a few hundred per design step.

Here the factorisation lives on the device in a numbered SLOT, filled from the host by
:func:`factor` (once per matrix), and :func:`solve` is an XLA FFI call that applies it without the
host: 0.25 ms an apply (same preconditioner, 200 applies in a `fori_loop`). cuDSS runs on the slot's
own stream, ordered against XLA's by events -- binding a cuDSS handle to XLA's stream call by call
returned wrong solves after the first.

The extension is compiled on first use with ``nvcc`` against the cuDSS shipped in the
``nvidia-cudss`` wheel and JAX's FFI headers, and cached by source hash. If anything is missing -- no
GPU, no ``nvcc``, no cuDSS -- :func:`available` is False and callers keep their host path.

Each slot carries a row/column scaling ``r``, ``c`` (an equilibration done by the caller), so
:func:`solve` returns ``x = c * B^{-1} (r * b)`` for ``B = diag(r) A diag(c)``; with ``r = c = 1`` it
is the plain solve. Complex double only, which is what the PEEC operators are.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "_ext" / "cudss_ffi.cu"
_LIB = None
_TRIED = False


def _paths():
    import jax

    site = Path(jax.__file__).resolve().parents[1]
    nv = site / "nvidia" / "cu12"
    return site / "jaxlib" / "include", nv / "include", nv / "lib"


def _nvcc():
    for c in (os.environ.get("NVCC"), "nvcc", "/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if c and (Path(c).exists() or subprocess.run(["which", c], capture_output=True).returncode == 0):
            return c
    return None


def _load():
    """Build (if needed), load and register the extension; None when it cannot be had."""
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    try:
        import jax

        if jax.default_backend() != "gpu" or not _SRC.exists():
            return None
        inc_jax, inc_nv, lib_nv = _paths()
        if not (inc_nv / "cudss.h").exists() or not (lib_nv / "libcudss.so.0").exists():
            return None
        key = hashlib.sha256(_SRC.read_bytes() + str(inc_jax).encode()).hexdigest()[:16]
        cache = Path(os.environ.get("JNO_EXT_CACHE", Path.home() / ".cache" / "jno" / "ext"))
        so = cache / f"libjno_cudss_{key}.so"
        if not so.exists():
            nvcc = _nvcc()
            if nvcc is None:
                return None
            cache.mkdir(parents=True, exist_ok=True)
            tmp = so.with_suffix(f".{os.getpid()}.tmp")
            cmd = [
                nvcc,
                "-O2",
                "-std=c++17",
                "-shared",
                "-Xcompiler",
                "-fPIC",
                f"-I{inc_jax}",
                f"-I{inc_nv}",
                str(_SRC),
                "-o",
                str(tmp),
                f"-L{lib_nv}",
                "-l:libcudss.so.0",
                "-Xlinker",
                f"-rpath={lib_nv}",
            ]
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                return None
            tmp.replace(so)  # atomic: a concurrent builder never loads a half-written library
        lib = ctypes.cdll.LoadLibrary(str(so))
        lib.jno_cudss_factor.argtypes = (
            [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64] + [ctypes.c_void_p] * 5 + [ctypes.c_int]
        )
        lib.jno_cudss_factor.restype = ctypes.c_int
        lib.jno_cudss_free.argtypes = [ctypes.c_int64]
        jax.ffi.register_ffi_target("jno_cudss_solve", jax.ffi.pycapsule(lib.jno_cudss_solve), platform="CUDA")
        _LIB = lib
    except Exception:  # noqa: BLE001 -- optional acceleration: any failure means the host path
        _LIB = None
    return _LIB


def available() -> bool:
    """True when the device solve can be used in this process (a GPU, nvcc on first use, cuDSS)."""
    return _load() is not None


def equilibrate(M, sweeps: int = 10):
    """Ruiz row/column equilibration to unit max-norm: ``(B, r, c)`` with ``B = diag(r) M diag(c)``.

    cuDSS pivots statically, so a matrix with a zero diagonal block -- a saddle point, a circuit's
    KCL rows -- factors to garbage as given; equilibrated, with cuDSS's diagonal matching, it factors.
    """
    import scipy.sparse as sp

    B = sp.csr_matrix(M).astype(complex)
    n = B.shape[0]
    r, c = np.ones(n), np.ones(n)
    for _ in range(sweeps):
        rn = np.sqrt(np.asarray(abs(B).max(axis=1).todense()).ravel())
        cn = np.sqrt(np.asarray(abs(B).max(axis=0).todense()).ravel())
        rn[rn == 0] = 1.0
        cn[cn == 0] = 1.0
        B = sp.diags(1.0 / rn) @ B @ sp.diags(1.0 / cn)
        r, c = r / rn, c / cn
    return sp.csr_matrix(B), r, c


def factor(slot: int, B, r=None, c=None, matching: int = 5) -> None:
    """Factorise the CSR matrix ``B`` into ``slot`` on the device (host call, synchronous)."""
    lib = _load()
    if lib is None:
        raise RuntimeError("jno: the cuDSS device solve is not available in this process.")
    B = B.tocsr()
    B.sort_indices()
    n = B.shape[0]
    ptr = np.ascontiguousarray(B.indptr, dtype=np.int32)
    idx = np.ascontiguousarray(B.indices, dtype=np.int32)
    val = np.ascontiguousarray(B.data, dtype=np.complex128)
    r = np.ascontiguousarray(np.ones(n) if r is None else r, dtype=np.float64)
    c = np.ascontiguousarray(np.ones(n) if c is None else c, dtype=np.float64)
    st = lib.jno_cudss_factor(
        int(slot),
        n,
        int(B.nnz),
        ptr.ctypes.data,
        idx.ctypes.data,
        val.ctypes.data,
        r.ctypes.data,
        c.ctypes.data,
        int(matching),
    )
    if st != 0:
        raise RuntimeError(f"jno: cuDSS factorisation failed (status {st}).")


def free(slot: int) -> None:
    lib = _load()
    if lib is not None:
        lib.jno_cudss_free(int(slot))


def solve(b, slot: int):
    """``x = c * B^{-1} (r * b)`` for the factorisation in ``slot`` -- an XLA op, usable under jit."""
    import jax

    # "sequential": a batch (jax.vmap over a vjp -- several adjoints of one solve) is solved one
    # right-hand side after another against the same factorisation; the kernel takes one vector
    return jax.ffi.ffi_call("jno_cudss_solve", jax.ShapeDtypeStruct(b.shape[-1:], b.dtype), vmap_method="sequential")(
        b, slot=np.int64(slot)
    )
